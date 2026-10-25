#pragma once

// include/strata/core/maxwell_experts.hpp - the Maxwell (sm_50, e.g. Tesla M10) expert tier.
//
// WHAT THIS IS: a static expert tier on one or more Maxwell GPUs, modelled on RemoteExperts
// (include/strata/core/remote_experts.hpp) but deliberately simpler.  Where RemoteExperts owns an
// ExpertCache with profile filling and adaptive swaps, MaxwellExperts holds a FIXED set of experts,
// copied once at startup from the host arena into the Maxwell device's memory, and computes them with
// the scalar Maxwell kernel (include/strata/kernels/cuda/expert_maxwell.hpp) instead of moe_grouped_s2.
//
// WHY FIXED, NOT CACHED: the M10s are a REPLACEMENT for the CPU pool, not a cache for the 3060.  The 3060
// keeps its own expert cache (unchanged); any expert NOT in the 3060's cache and NOT on an M10 still goes
// to the CPU.  The M10s serve the experts they hold.  A fixed assignment is correct because an M10 never
// needs to evict: it holds what fits, the CPU takes the rest.
//
// MULTI-DEVICE: one MaxwellExperts instance per M10 (up to 4).  Each holds a disjoint subset of
// (layer, expert) pairs.  Expert computation is embarrassingly parallel (no cross-expert dependencies),
// so the split needs no synchronisation between devices.
//
// KIND: entries served by this tier are tagged kind 3 in the dispatch (0 VRAM, 1 PCIe, 2 peer/remote,
// -1 CPU).  The CPU pool already skips kind >= 0; finish() writes the rows like RemoteExperts does.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <cuda_runtime.h>

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cuda/expert_maxwell.hpp"   // expert_maxwell_geo (the blob geometry)

namespace strata::core {

class MaxwellExperts {
public:
    MaxwellExperts() = default;
    ~MaxwellExperts();
    MaxwellExperts(const MaxwellExperts&) = delete;
    MaxwellExperts& operator=(const MaxwellExperts&) = delete;

    /// Fill this device with the (layer, expert) pairs in `assign` that fit in the card's free memory
    /// minus `reserve_mib`.  Blobs are copied once from the host arena (`src_blob(layer, expert)`).
    /// Returns false with `err` set when the device cannot be used.
    ///
    /// `src_blob` is a callback returning the host pointer of one expert's 1,382,400-byte blob.
    ///
    /// SPLIT_MODE_TENSOR (`tp_cards` == 4): one physical rank of the routed-expert TP tier.
    /// MaxwellMetaTier owns/orchestrates these ranks. Every card takes the WHOLE `assign` list and
    /// stores a TENSOR SHARD of every expert instead of whole experts: gate/up rows [64*blk_base,
    /// 64*(blk_base+blks)) and down blocks [blk_base, blk_base+blks) of the SC_D 64-element blocks, the
    /// slice rotating by layer (`(tp_card + layer) % tp_cards`) so the {3,3,2,2}-block split of 10 over
    /// 4 cards rotates evenly (48 layers % 4 == 0). Byte totals still vary with per-layer formats and
    /// expert counts. The four shards sum to the full blob; every
    /// routed entry is then computed by ALL cards concurrently, each writing a PARTIAL output row the
    /// consumer sums in rank order.  `tp_cards` == 1 (default) keeps the whole-expert layout, and the
    /// whole function is byte-identical to the previous behaviour.
    bool open(int device,
              const std::vector<std::pair<int32_t, int32_t>>& assign,
              const uint8_t* (*src_blob)(int64_t layer, int64_t expert, void* ctx), void* ctx,
              int64_t n_layers, int64_t n_expert, int reserve_mib,
              std::string& err, int tp_card = 0, int tp_cards = 1);
    void close();
    bool valid() const { return device_ >= 0; }
    int device() const { return device_; }

    /// Whether this device holds (layer, expert).
    bool owns(int64_t layer, int64_t expert) const;

    /// How many of the layer's routed entries (n_tok * k) this device owns.  Cheap, read-only: used by the
    /// dispatcher to begin the busiest device FIRST (LPT), so the longest chain starts earliest.
    int64_t count_owned(int64_t layer, const int32_t* ids, int64_t n_tok, int64_t k) const;

    /// Start this device's share of one layer: entries with kind[i] == -1 whose expert this device owns.
    /// `acts` are the engine's ALREADY-QUANTISED activations (one ActQ per token, same objects the CPU pool
    /// uses) - the GPU consumes exactly the same int8/scales/hx, so there is zero quantisation divergence.
    /// `dst_mapped` (the token path: the verifier's mapped host rows) is not null when the destination is
    /// device-accessible: the down stage is then launched HERE, scattering each row straight into its
    /// destination while the caller keeps working - no D2H copy, no host scatter.  Null for the prefill, whose
    /// destination is only known at finish_to (the down then launches there, into the staging block).
    /// `rows_f16` (v2 only, opt-in): the down stage writes each row as H fp16 halves PACKED INTO ITS OWN
    /// fp32 row slot (the first H/2 floats) instead of the full-width fp32 row - half the uplink bytes.  The
    /// verifier's row copy converts them back (it publishes the per-entry `kind16` flags that mark them), and
    /// the per-group destination offset keeps working because the fp16 rows ride the same `dst_mapped`.
    /// E1: hand this tier its mapped completion flag (the verifier allocates four).  When set, the
    /// decode down kernel raises it and (mode 2) the host stops waiting for the M10s.
    void set_completion_flag(uint32_t* f) { comp_flag_ = f; }
    /// Routed-TP wire option: the down writes fp16 partial rows and the D2H moves half
    /// the bytes (the reduce converts back exactly). Set before the first begin().
    void set_tp_wire16(bool on) { tp_wire16_ = on; }
    /// Meta packs the window's activations ONCE and shares them: begin() then bulk-copies
    /// the packed block instead of re-packing per rank (four times the same bytes today).
    void set_shared_acts(const uint8_t* packed) { shared_acts_ = packed; }
    bool begin(int64_t layer, const strata::kernels::cpu::ActQ* acts, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, float* dst_mapped, bool rows_f16, std::string& err);
    /// Wait for this device's staged rows and write them into `out` (row i = entry i, n_embd floats) - the
    /// mapped destination handed to begin().  Only the sync remains here.
    bool finish(float* out, std::string& err);
    /// Like finish(), but entry i's row goes to `dst_rows[(row_of_entry[i] - row_base) * n_embd]` - the
    /// prefill's grouped `Dm` tail block (a HOST staging buffer, one H2D copies it to the device).  For the
    /// token path (`row_of_entry` null) row i = entry i.  The down stage launches here and the rows are
    /// copied to the caller's rows.  `accumulate` (SPLIT_MODE_TENSOR): the rows are ADDED into `dst_rows`
    /// instead of copied - each card contributes its partial and the cards finish in rank order, so the
    /// accumulated sum is deterministic; the caller zeroes the block first.
    bool finish_to(float* dst_rows, const int32_t* row_of_entry, int32_t row_base, int64_t n_embd,
                   std::string& err, bool accumulate = false);
    /// The two-phase prefill's launch half (2026-10-23): launch THIS card's down now (cheap, async);
    /// the caller then finish_to()s the cards in rank order, whose sync/D2H/scatter-add order - and
    /// so the accumulated block - is bit-identical to the single-phase form.  The four cards' downs
    /// overlap like their gups instead of serialising behind each other's finishes.
    bool prefill_launch_down(std::string& err);
    /// STRATA_MW_DRED (the prefill's device-side reduce): the caller peer-copies this card's down
    /// partials on the primary GPU and reduces them there, so the completes become enqueue-only.
    /// These expose what that arm needs: the partial rows' owner ids, the device buffer, the
    /// down-complete event, and the registration of the caller's copy-done event (the next window's
    /// down waits it - d_out_ is single-buffered).
    const std::vector<int32_t>& owned_rows() const { return owned_row_; }

    int64_t resident() const { return resident_; }
    int64_t computed() const { return computed_.load(std::memory_order_relaxed); }
    double gib() const { return gib_; }
    /// SPLIT_MODE_TENSOR: this card holds tensor shards of the resident set (not whole experts); its
    /// output rows are PARTIAL sums the caller must reduce across cards in rank order.
    bool tensor() const { return tp_cards_ > 1; }
    int64_t n_tok_max() const { return n_tok_max_; }   ///< the sub-batch window (STRATA_MW_NTM; the prefill's MW_SUB follows it)
    /// E1: the tier's service counter (the value the last launch/host-raise left on comp_seq_).  The
    /// dispatcher reads it to pick the double-buffered staging parity of the UPCOMING service - its
    /// ring value is seq()+1, the same value the verifier publishes as this step's expected sequence.
    uint32_t seq() const { return comp_seq_; }
    /// STRATA_MW_PERSIST: whether open() armed this card's resident decode kernel.  The driver wires
    /// the E1 completion flags whenever any tier is persist-armed (the kernel raises them whatever
    /// STRATA_MW_GPUFLAG says - mode 0 then behaves like mode 1, the host wait kept).
    bool persist_armed() const { return persist_armed_; }
    /// The (layer, expert) pairs actually resident on this device (subset of the assignment that fit).
    /// Used by the driver to exclude them from the RAM complement (no duplication).
    const std::vector<std::pair<int32_t, int32_t>>& held_pairs() const { return held_; }

private:
    int device_ = -1;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int64_t resident_ = 0;
    // SPLIT_MODE_TENSOR: this card's rank and the tier's card count (1 = whole-expert mode).
    int tp_card_ = 0, tp_cards_ = 1;
    std::atomic<int64_t> computed_{0};
    double gib_ = 0.0;

    // A native pack's per-layer blob geometry (open() builds it): each layer's experts are contiguous
    // in their own region of d_blobs_ (its blobs differ in size from other layers'), the gate/up rows
    // keep their RAW GGUF block layouts for the IQ kernel (o_gate/o_up/gu_row), and the down rows are
    // converted to the S2 planar form at d_codes_off/d_scales_off.  Empty = the canonical Q2_0 pack:
    // the flat store, the compile-time S2 geometry (the 42/s2_ defaults).
    struct LayerGeo {
        int gu_type = 42;
        size_t base = 0;                                        ///< the region's offset inside d_blobs_
        size_t blob_bytes = (size_t) strata::kernels::cuda::expert_maxwell_geo::BLOB;
        size_t o_gate = 0, o_up = 0, gu_row = 0;                ///< the raw GGUF gate/up rows (i-quant layers)
        size_t d_codes_off = strata::kernels::cuda::expert_maxwell_geo::O_D_CODES;
        size_t d_scales_off = strata::kernels::cuda::expert_maxwell_geo::O_D_SCALES;
        // SPLIT_MODE_TENSOR: this card's slice of the layer (gate/up rows and down blocks, rotated by
        // layer).  Defaults = the full range, so every non-TP launch passes the plain geometry.
        int tp_row_base = 0, tp_rows = strata::kernels::cuda::expert_maxwell_geo::FF;
        int tp_blk_base = 0, tp_blks = strata::kernels::cuda::expert_maxwell_geo::SC_D;
        size_t d_row_stride = strata::kernels::cuda::expert_maxwell_geo::ROW_D;
        size_t d_scale_stride = (size_t) strata::kernels::cuda::expert_maxwell_geo::SC_D * 2;
    };
    std::vector<LayerGeo> geo_;   ///< per layer (native packs); empty = the canonical flat store
    const LayerGeo& geo_of(int64_t layer) const {
        static const LayerGeo canonical{};
        return geo_.empty() ? canonical : geo_[(size_t) layer];
    }

    // device memory: one contiguous blob store + activation/output/scratch.
    // P3.1 batched: ff/faq/out buffers are per-ENTRY (cap_entries), not per-expert, so all stage1
    // launches queue without intermediate sync, one D2H collects all ff, one H2D uploads all ffq,
    // all stage2 launch, one D2H collects results.  4 transfers + 2 syncs per layer, not per expert.
    uint8_t* d_blobs_ = nullptr;    // resident_ blobs: flat (canonical) or per-layer regions (native)
    // P6.x: ONE contiguous staging region per device: [packed acts][job table].  The host packs the acts at
    // offset 0 and the job table right after (n_tok * ACTS_STRIDE), the device mirrors the layout, so a
    // layer costs ONE async upload instead of two (measured: a CUDA call is ~4 us in this VM).
    uint8_t* d_up_ = nullptr;       // the merged device staging (the MAPPED host pointer when zero-copy)
    uint8_t* h_up_ = nullptr;       // its pinned host twin
    size_t up_bytes_ = 0;
    bool zerocopy_ = false;         // P7: the kernels read acts+jobs from the mapped pinned staging
    // E1: the GPU-raised doorbell.  comp_flag_ is the verifier's mapped tier flag; the down kernel's
    // last block raises it to the monotonic comp_seq_ (the verifier publishes the same count as the
    // expected value each step).  d_ctr_ is the in-stream arrival counter; pending_raise_ covers a
    // service whose jobs list came out empty.  Mode (STRATA_MW_GPUFLAG): 0 off, 1 flags+multi-wait
    // with the host sync kept, 2 also drops the decode-path sync (double-buffered staging).
    uint32_t* comp_flag_ = nullptr;
    bool tp_wire16_ = false;        // routed-TP delivery: fp16 partial rows on the wire
    const uint8_t* shared_acts_ = nullptr;   // Meta's once-packed activation block (null: pack per rank)
    uint32_t comp_seq_ = 0;
    uint32_t pending_raise_ = 0;
    uint32_t* d_ctr_ = nullptr;
    uint8_t* d_up2_ = nullptr;      // the second staging set (mode 2: layer l+1 stages while l reads)
    uint8_t* h_up2_ = nullptr;
    uint8_t* d_acts_ = nullptr;     // = d_up_ (packed ActQ: [q per token][scale per token][hx per token])
    uint8_t* d_aq_ = nullptr;       // = d_acts_ (int8 codes, H per token)
    float* d_ascale_ = nullptr;     // = d_acts_ + n_tok_max*H
    float* d_ahx_ = nullptr;        // = d_ascale_ + n_tok_max*(H/32)*4
    size_t acts_bytes_ = 0;         // packed size for n_tok_max tokens
    uint8_t* h_acts_ = nullptr;     // PINNED host staging for the packed upload (one async memcpy per layer)
    int32_t* d_jobs_ = nullptr;     // device job table: pairs of (blob slot, token) for the batch kernels
    int32_t* h_jobs_ = nullptr;     // PINNED host staging for the job table
    int8_t* d_faq_ = nullptr;       // per-entry FF-wide quantised intermediate
    float* d_fascale_ = nullptr;
    float* d_fahx_ = nullptr;
    float* d_ff_ = nullptr;         // per-entry FF floats, SwiGLU scratch
    float* d_out_ = nullptr;        // per-entry H-wide outputs
    float* h_out_ = nullptr;        // pinned host staging for results
    int64_t cap_entries_ = 0;

    // (layer, expert) -> index into d_blobs_, or -1
    std::vector<int32_t> res_;
    // the pairs actually resident (for held_pairs(): RAM complement exclusion)
    std::vector<std::pair<int32_t, int32_t>> held_;
    // per-begin state: which entries this device serves this layer
    std::vector<int32_t> owned_idx_;   // entry i -> blob index, or -1
    std::vector<int32_t> owned_row_;   // dense list of entry indices served
    int64_t n_tok_max_ = 0;

    // P3.3: 4 streams + events per device: several jobs in flight hide the M10's latency (the SMs are
    // latency-bound), measured 1.9x on four jobs versus a single default stream.
    cudaStream_t streams_[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t ev_up_ = nullptr, ev_q_ = nullptr;
    cudaEvent_t ev_d2h_done_ = nullptr;   // SPLIT_MODE_TENSOR: the previous layer's partial D2H completed (d_out_ reuse)
    cudaEvent_t ev_gup_[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t ev_down_[4] = {nullptr, nullptr, nullptr, nullptr};

    // S0 (STRATA_MW_LAYERPROF): this card's own truth - the kernel span measured on its stream (events,
    // gup->down) and the distinct-expert bytes the span moves, to read against the ~83 GB/s DRAM floor.
    // Counters only; nothing about the service changes, and all of it stays inert unless the env is set.
    cudaEvent_t s0_a_ = nullptr, s0_b_ = nullptr;
    cudaEvent_t s0_m_ = nullptr;   // S0 mid-stamp: after gup, before down - which stage eats the span
    double s0_layers_ = 0, s0_jobs_ = 0, s0_distinct_ = 0, s0_mb_ = 0, s0_krn_ms_ = 0, s0_sync_ms_ = 0;
    double s0_gup_ms_ = 0, s0_dn_ms_ = 0;
    bool s0_deferred_ = false;     // E1 mode 2: the span is reaped at the next begin (see begin_chain)
    void s0_report();              // the MWPROF m10 line; prints and resets when 1440 layers accrue

    // P3.4 (rev 3, misurato): la catena begin/finish gira INLINE sul thread chiamante.  Il worker persistente
    // per device (condvar nella prima prova, atomic-wait nella seconda) e' stato misurato: i
    // cudaDeviceSynchronize concorrenti dei 4 device si serializzano nel driver, e l'handoff per layer
    // costava ~150 us/layer in piu' della catena inline (29,3 vs 22,2 ms/finestra).  Il costo vero e' dentro
    // il singolo finish (sync + copia + scatter): quello va ridotto, non parallelizzato.
    // I w_* portano il lavoro da begin() a finish_to().
    int64_t w_layer_ = 0, w_n_tok_ = 0, w_k_ = 0;
    const strata::kernels::cpu::ActQ* w_acts_ = nullptr;
    const int32_t* w_ids_ = nullptr;
    float* w_dst_ = nullptr;
    const int32_t* w_row_of_ = nullptr;
    int32_t w_row_base_ = 0;
    int64_t w_n_embd_ = 0;
    bool w_dst_mapped_ = false;   // the destination is device-accessible: the down kernel writes it directly
    bool w_rows_f16_ = false;     // the down writes fp16 rows packed into the fp32 row slots, see begin()
    bool w_accumulate_ = false;   // SPLIT_MODE_TENSOR (prefill): ADD the rows into w_dst_ instead of copying
    bool down_launched_ = false;  // the down stage of this layer already ran at begin() time
    // STRATA_MW_MT: the run/entry tables the begin arm built in the staging (device pointers, valid
    // until the next begin overwrites the staging).  The prefill's down (finish_chain) consumes them.
    const int32_t* d_runs_mt_ = nullptr;
    const int32_t* d_ent_mt_ = nullptr;
    int mt_nruns_ = 0;
    cudaEvent_t ev_pf_ad2h_ = nullptr;                 // STRATA_MW_AD2H: the completes' pre-enqueued D2H
    bool ad2h_armed_ = false;                          // this window's D2H was armed at launch_down
    bool begin_chain(std::string& err);
    bool finish_chain(std::string& err);

    // STRATA_MW_PERSIST (see include/strata/kernels/cuda/expert_maxwell.hpp): the resident decode
    // kernel.  The decode dispatch becomes pinned-ring writes + one doorbell store - zero CUDA API
    // calls per layer - and the batched launches/syncs (and with them the visit-order poison) apply
    // only to the non-persist paths.  `w_persisted_` carries "this service went through the ring"
    // from begin_chain to finish_chain; the prefill (finish_to) and close() halt the kernel first:
    // Maxwell has no preemption, so nothing else may share the card with it.
    bool persist_armed_ = false;
    bool persist_live_ = false;
    bool w_persisted_ = false;
    uint32_t persist_nseq_ = 0;                // the RELATIVE doorbell counter (reset at every launch)
    uint32_t* d_seq_ = nullptr;                // the DEVICE doorbell (the copy engine rings it)
    volatile uint32_t* h_halt_ = nullptr;      // the MAPPED stop word (a host store; the kernel polls it)
    const volatile uint32_t* d_halt_ = nullptr;
    strata::kernels::cuda::PersistParams* d_pparams_ = nullptr;
    strata::kernels::cuda::PersistCtl* d_pctl_ = nullptr;
    bool persist_launch(std::string& err);
    void persist_halt();

    bool check(cudaError_t e, const char* what, std::string& err);
};

}  // namespace strata::core
