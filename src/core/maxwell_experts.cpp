// src/core/maxwell_experts.cpp - the Maxwell (sm_50) expert tier.  See the header for the design.
//
// DATA FLOW for one layer:
//   begin(): for each routed entry with kind==-1 owned by this device, dequantise the host ActQ into xq,
//            copy xq to the device, launch gate+up (stage1) for each owned expert, requantise ff on the
//            host, copy ffq, launch down (stage2).  Results are staged in h_out_.
//   finish(): the launches already synchronised in begin(); copy staged rows into `out`.
//
// This is deliberately synchronous (no stream overlap with the CPU pool) for the proof: correctness first,
// overlap later.  The CPU pool and the M10 work sequentially, so the M10 still replaces CPU work 1:1.
#include "strata/core/maxwell_experts.hpp"
#include "strata/core/maxwell_tp_plan.hpp"
#include "strata/core/peer_experts.hpp"

#include "strata/kernels/cuda/expert_maxwell.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>
#include <cuda.h>   // cuStreamWriteValue32: the persist doorbell rides the copy engine's stream

#include "ggml.h"
#include "ggml-cpu.h"   // ggml_cpu_init (the round-trip gate's to_float tables)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <random>
#include <vector>

namespace strata::core {
namespace {

using namespace strata::kernels::cuda::expert_maxwell_geo;
using strata::kernels::cuda::s2_expert_maxwell_gup_async;
using strata::kernels::cuda::s2_expert_maxwell_down_async;
using strata::kernels::cuda::s2_expert_maxwell_gup_batch_async;
using strata::kernels::cuda::s2_expert_maxwell_down_batch_async;
using strata::kernels::cuda::s2_expert_maxwell_down_batch_f16_async;
using strata::kernels::cuda::s2_expert_maxwell_quant_ff_async;
using strata::kernels::cuda::s2_expert_maxwell_gup_quant_batch_async;
using strata::kernels::cuda::iq_expert_maxwell_gup_quant_batch_async;

namespace {
// E1: STRATA_MW_GPUFLAG - 0 off; 1 the GPU-raised tier flags + the graph's multi-wait (host sync kept);
// 2 also drops the decode-path sync (the staging double-buffers).  See WORKLOG 2026-10-12.
int gpuflag_mode() {
    static const int m = [] { const char* e = std::getenv("STRATA_MW_GPUFLAG"); return e ? std::atoi(e) : 0; }();
    return m;
}
}
using strata::kernels::cuda::s2_expert_maxwell_v2;
using strata::kernels::cuda::s2_expert_maxwell_sync;
using strata::kernels::cuda::mw_tp_raise_flag_async;

constexpr int H = 2560;
constexpr int FF = 640;

// S0: whether the M10 tier's own counters run (STRATA_MW_LAYERPROF - the same flag the host-side layer
// print uses, so one env arms the whole per-layer truth).  Read once.
static bool s0_env() {
    static const bool on = std::getenv("STRATA_MW_LAYERPROF") != nullptr;
    return on;
}

// RAII device scope, like RemoteExperts::DeviceScope.
struct DeviceScope {
    int prev = -1;
    bool ok = false;
    std::string err;
    explicit DeviceScope(int device) {
        cudaError_t e = cudaGetDevice(&prev);
        if (e != cudaSuccess) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Maxwell DeviceScope: cudaGetDevice failed: %s",
                          cudaGetErrorString(e));
            err = buf;
            return;
        }
        e = cudaSetDevice(device);
        if (e != cudaSuccess) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Maxwell DeviceScope: cudaSetDevice(%d) failed: %s (prev=%d)",
                          device, cudaGetErrorString(e), prev);
            err = buf;
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (ok) cudaSetDevice(prev); }
};

}  // namespace

MaxwellExperts::~MaxwellExperts() { close(); }

bool MaxwellExperts::check(cudaError_t e, const char* what, std::string& err) {
    if (e != cudaSuccess) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): %s: %s", device_, what,
                      cudaGetErrorString(e));
        err = buf;
        return false;
    }
    return true;
}

bool MaxwellExperts::open(int device,
                          const std::vector<std::pair<int32_t, int32_t>>& assign,
                          const uint8_t* (*src_blob)(int64_t layer, int64_t expert, void* ctx), void* ctx,
                          int64_t n_layers, int64_t n_expert, int reserve_mib,
                          std::string& err, int tp_card, int tp_cards) {
    close();
    if ((tp_cards != 1 && tp_cards != 4) || tp_card < 0 || tp_card >= tp_cards) {
        err = "MaxwellExperts: tensor layout requires four ranks (or one whole-expert device)";
        return false;
    }
    tp_card_ = tp_card;
    tp_cards_ = tp_cards < 1 ? 1 : tp_cards;
    // STRATA_MW_BLOCKSYNC (opt-in): put this M10's context in BLOCKING-sync mode before anything
    // creates it - cudaStreamSynchronize then sleeps on the device's interrupt instead of spinning
    // in the driver.  The S0 campaign found the tier's per-card downs poisoned in SERVICE order
    // (~+55 us per earlier-served card, 80 -> 240 us; STRATA_MW_REV flipped the fast card with the
    // order): the earlier finishes' sync-SPINS clog the path later cards' kernel completions take
    // through this VM's driver/hypervisor.  A sleeping sync does not spin; the wake costs a few
    // tens of us per finish instead.  Must precede the context's creation (hence here, first).
    static const bool blocksync = [] {
        const char* e = std::getenv("STRATA_MW_BLOCKSYNC");
        const bool on = e != nullptr && std::atoi(e) != 0;
        if (on) std::fprintf(stderr, "strata maxwell: blocking sync on the M10 contexts (STRATA_MW_BLOCKSYNC)\n");
        return on;
    }();
    if (blocksync) {
        cudaSetDevice(device);
        const cudaError_t f = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        if (f != cudaSuccess)
            std::fprintf(stderr, "strata maxwell: blocking sync refused on CUDA%d (%s); spinning stays\n",
                         device, cudaGetErrorString(f));
    }
    // GUARD: this tier runs the expert encodings its kernels take - the Q2_0 (S2) form, and a native
    // pack's IQ2_S / IQ2_XXS / IQ1_M gate/up over the Q2_0 down (the IQ2_XS file's layers; the down is
    // converted to the S2 planar form at open()).  Anything else (an encoding a future pack may carry)
    // would be read with the wrong geometry - plausible garbage rows, silently, never a fault.  Refuse
    // instead, naming the first offending layer.  42 is ggml's Q2_0 (tools/gguf_reader.py).
    {
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (lay.native) {
            for (size_t L = 0; L < lay.fmt.size(); ++L) {
                const auto& f = lay.fmt[L];
                if (f.d_type != 42 ||
                    (f.gu_type != 42 && f.gu_type != 22 && f.gu_type != 16 && f.gu_type != 29)) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "MaxwellExperts (CUDA%d): layer %zu keeps its experts as gate/up ggml type "
                                  "%d and down type %d; this tier takes Q2_0 (42), and IQ2_S (22) / IQ2_XXS "
                                  "(16) / IQ1_M (29) gate/up over a Q2_0 down",
                                  device, L, f.gu_type, f.d_type);
                    err = buf;
                    return false;
                }
                // SPLIT_MODE_TENSOR: the shard builder and the sliced kernels cover the native i-quant
                // gate/up path; an S2 (42) gate/up layer in tensor mode would silently take the full-range
                // gup on every card.  Refuse it - the production IQ2_XS pack has none.
                if (tp_cards_ > 1 && f.gu_type == 42) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "MaxwellExperts (CUDA%d): layer %zu is S2 (42) gate/up; the tensor split "
                                  "takes the native i-quant layers (IQ2_S/IQ2_XXS/IQ1_M)",
                                  device, L);
                    err = buf;
                    return false;
                }
            }
        } else if (tp_cards_ > 1) {
            err = "MaxwellExperts: the tensor split needs a native pack (the canonical Q2_0 store is refused)";
            return false;
        }
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.err; return false; }

    // how much fits?  A canonical pack's blobs are all BLOB bytes; a native pack's differ per layer
    // (the IQ2_XS file: 1,510,400 / 1,305,600 / 1,177,600 B per expert, by its three gate/up encodings),
    // so the budget is taken in the layer's own blob bytes.
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) return false;
    const size_t reserve_b = (size_t) reserve_mib * 1024 * 1024;
    const size_t usable = (free_b > reserve_b) ? (free_b - reserve_b) : 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    // SPLIT_MODE_TENSOR: the SC_D (=10) 64-element blocks split across the cards at BLOCK granularity
    // (block granularity keeps every shared fp16 down scale whole on one card): slot s of tp_cards takes
    // q + (s < r) blocks.  With 4 cards that is {3,3,2,2}; the layers rotate the slot
    // ((tp_card + layer) % tp_cards). Rotation balances the number of large slices, not exact bytes:
    // per-layer formats and the profile's number of resident experts vary. The four shard sizes SUM
    // exactly to the full store; each card's actual total must fit independently.
    const auto blob_of = [&](int64_t layer) -> uint64_t {
        if (tp_cards_ <= 1) return lay.native ? lay.blob_bytes(layer) : (uint64_t) BLOB;
        const auto& f = lay.fmt[(size_t) layer];
        // shard = 2 * (64*blks) gate/up rows (gu_row bytes each) + H rows * blks * (16 codes + 2 scale)
        return maxwell_tp::shard_bytes(maxwell_tp::slice(tp_card_, (int) layer), f.gu_row);
    };

    // take experts in assignment order until the device is full
    std::vector<std::pair<int32_t, int32_t>> mine;
    size_t bytes = 0;
    for (const auto& p : assign) {
        if (bytes + blob_of(p.first) > usable) {
            // SPLIT_MODE_TENSOR: every card must hold the SAME set (the four partials of an entry are
            // summed by rank; a card that dropped an expert would contribute garbage).  The shards sum
            // to 1/tp_cards of the whole-expert bytes, so a list that fit before fits now - if it does
            // not, the layout is wrong for this rig: refuse instead of trimming.
            if (tp_cards_ > 1) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "MaxwellExperts (CUDA%d, tensor split): the assignment does not fit "
                              "(%.2f GiB of shards > %.2f GiB usable) - refusing rather than splitting "
                              "the resident set across cards",
                              device, bytes / 1e9 + blob_of(p.first) / 1e9, usable / 1e9);
                err = buf;
                return false;
            }
            break;
        }
        mine.push_back(p);
        bytes += blob_of(p.first);
    }
    if (mine.empty()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): no experts fit (%.1f GiB free, %.1f GiB reserved)",
                      device, free_b / 1e9, reserve_b / 1e9);
        err = buf;
        return false;
    }

    device_ = device;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    res_.assign((size_t) n_layers * (size_t) n_expert, -1);
    if (const char* dbg = std::getenv("STRATA_MW_TP_DEBUG"))
        if (std::atoi(dbg))
            std::fprintf(stderr, "TPDBG open CUDA%d: tp_cards=%d assign=%zu mine=%zu bytes=%.2f GiB\n",
                         device, tp_cards_, assign.size(), mine.size(), bytes / 1e9);

    // THE STORE: a native pack keeps each layer's experts in their own contiguous region (the blobs
    // differ in size per layer, so a flat slot*BLOB walk cannot address them); a canonical pack keeps
    // today's flat store.  geo_ carries each region's base and the launch geometry; res_ then holds the
    // slot WITHIN the layer (native) or the flat slot (canonical).
    if (lay.native) {
        std::vector<int64_t> count((size_t) n_layers, 0);
        for (const auto& p : mine) ++count[(size_t) p.first];
        geo_.assign((size_t) n_layers, LayerGeo{});
        size_t off = 0;
        for (int64_t L = 0; L < n_layers; ++L) {
            const auto& f = lay.fmt[(size_t) L];
            LayerGeo& g = geo_[(size_t) L];
            g.gu_type = f.gu_type;
            g.base = off;
            g.blob_bytes = (size_t) f.bytes;
            if (f.gu_type != 42) {   // the i-quant layer: raw GGUF gate/up rows, the down converted
                g.o_gate = 0;
                g.o_up = f.up_off;                       // gate rows first, up rows second (as the source)
                g.gu_row = f.gu_row;
                g.d_codes_off = f.down_off;               // the down region, de-interleaved to S2 planar
                g.d_scales_off = f.down_off + (size_t) H * ROW_D;
            }
            // SPLIT_MODE_TENSOR: this card's slice of the layer - the block range rotated by layer, the
            // matching gate/up row range (64 rows per block), and the COMPACT shard geometry that replaces
            // the full-blob offsets above.  tp_cards == 1 keeps the full-range defaults (the branch below
            // never fires and the store is byte-identical to the whole-expert layout).
            if (tp_cards_ > 1) {
                const auto part = maxwell_tp::slice(tp_card_, (int) L);
                g.tp_blk_base = part.block_base;
                g.tp_blks = part.blocks;
                g.tp_row_base = part.row_base;
                g.tp_rows = part.rows;
                g.o_gate = 0;
                g.o_up = (size_t) g.tp_rows * (size_t) f.gu_row;
                g.gu_row = f.gu_row;
                g.d_row_stride = (size_t) 16 * g.tp_blks;
                g.d_scale_stride = (size_t) 2 * g.tp_blks;
                g.d_codes_off = (size_t) 2 * (size_t) g.tp_rows * (size_t) f.gu_row;
                g.d_scales_off = g.d_codes_off + (size_t) H * g.d_row_stride;
                g.blob_bytes = g.d_scales_off + (size_t) H * g.d_scale_stride;
            }
            off += (size_t) count[(size_t) L] * (size_t) g.blob_bytes;
        }
    }
    // allocate the blob store on the device
    if (const char* dbg = std::getenv("STRATA_MW_TP_DEBUG"))
        if (std::atoi(dbg))
            std::fprintf(stderr, "TPDBG CUDA%d: geo built (%zu layers), store %.2f GiB\n",
                         device, geo_.size(), bytes / 1e9);
    if (!check(cudaMalloc(&d_blobs_, bytes), "cudaMalloc blobs", err)) { close(); return false; }

    // Convert each blob on copy.  The source blob for a native pack is INTERLEAVED (18 bytes per
    // 64-weight block: fp16 scale + 16 codes, rows at gu_row/d_row strides); the S2 kernels read PLANAR.
    // A Q2_0 layer (native or canonical) is converted to the S2 planar layout as before; an i-quant
    // layer keeps its gate/up rows EXACTLY as the GGUF stores them (the kernel reads the raw blocks -
    // an i-quant row cannot be re-expressed in the S2 form, tools/iq_pack.py) and converts only its
    // Q2_0 down rows to the S2 planar form, inside the blob, at the layer's own offsets.
    std::vector<int32_t> per_layer_slot(lay.native ? (size_t) n_layers : 0, 0);
    size_t staging_bytes = lay.native ? (size_t) lay.max_blob : (size_t) BLOB;
    if (tp_cards_ > 1) {   // the shard is always <= the full blob; take the largest layer's shard
        staging_bytes = 0;
        for (int64_t L = 0; L < n_layers; ++L) staging_bytes = std::max(staging_bytes, geo_[(size_t) L].blob_bytes);
    }
    std::vector<uint8_t> staging(staging_bytes);
    for (size_t i = 0; i < mine.size(); ++i) {
        const int64_t L = mine[i].first;
        const uint8_t* src = src_blob(L, mine[i].second, ctx);
        if (src == nullptr) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): source blob missing for (%d,%d)",
                          device, mine[i].first, mine[i].second);
            err = buf;
            close();
            return false;
        }
        size_t blob_bytes = (size_t) BLOB, dst_off = i * (size_t) BLOB;
        int32_t slot = (int32_t) i;
        if (lay.native) {
            const auto& f = lay.fmt[(size_t) L];
            const LayerGeo& g = geo_[(size_t) L];
            // SPLIT_MODE_TENSOR: the stored blob is the SHARD - its own size, and the region's slots
            // stride by it (geo_.base was accumulated with g.blob_bytes, and the copy below must read
            // exactly the shard bytes the staging holds, never the full f.bytes).
            blob_bytes = tp_cards_ > 1 ? g.blob_bytes : (size_t) f.bytes;
            slot = per_layer_slot[(size_t) L]++;
            dst_off = g.base + (size_t) slot * blob_bytes;
            if (f.gu_type == 42) {
                // native Q2_0 -> S2 planar: de-interleave each row's 18-byte blocks (as before)
                uint8_t* dst = staging.data();
                // S2 planar offsets (must match expert_maxwell_geo)
                const size_t ROW_GU = 640, ROW_D = 160, SC_GU = 40, SC_D = 10;
                const size_t O_D_CODES = 2ull * 640 * ROW_GU;
                const size_t O_GU_SCALES = O_D_CODES + 2560ull * ROW_D;
                const size_t O_D_SCALES = O_GU_SCALES + 2ull * 640 * SC_GU * 2;
                // gate + up rows
                for (int r = 0; r < 640; ++r) {
                    for (int half = 0; half < 2; ++half) {   // 0 = gate, 1 = up
                        const uint8_t* srow = src + (half == 0 ? 0 : f.up_off) + (size_t) r * f.gu_row;
                        uint8_t* dcode = dst + ((size_t) (2 * r + half)) * ROW_GU;
                        uint8_t* dscale = dst + O_GU_SCALES + ((size_t) (2 * r + half)) * SC_GU * 2;
                        for (int b = 0; b < 40; ++b) {
                            const uint8_t* blk = srow + (size_t) b * 18;
                            dscale[b * 2] = blk[0];
                            dscale[b * 2 + 1] = blk[1];
                            std::memcpy(dcode + b * 16, blk + 2, 16);
                        }
                    }
                }
                // down rows
                for (int r = 0; r < 2560; ++r) {
                    const uint8_t* srow = src + f.down_off + (size_t) r * f.d_row;
                    uint8_t* dcode = dst + O_D_CODES + (size_t) r * ROW_D;
                    uint8_t* dscale = dst + O_D_SCALES + (size_t) r * SC_D * 2;
                    for (int b = 0; b < 10; ++b) {
                        const uint8_t* blk = srow + (size_t) b * 18;
                        dscale[b * 2] = blk[0];
                        dscale[b * 2 + 1] = blk[1];
                        std::memcpy(dcode + b * 16, blk + 2, 16);
                    }
                }
            } else {
                // the i-quant layer: gate/up rows VERBATIM (the raw GGUF blocks the kernel decodes),
                // the Q2_0 down rows de-interleaved into the S2 planar form at the blob's own offsets
                if (tp_cards_ > 1) {
                    // SPLIT_MODE_TENSOR: the SHARD - this card's gate/up row range verbatim, and only
                    // its down blocks de-interleaved per row, all COMPACT at the shard's own offsets.
                    // The other cards take the complementary ranges; the shards sum to the full blob.
                    for (int lr = 0; lr < g.tp_rows; ++lr) {
                        const int64_t r = (int64_t) g.tp_row_base + lr;
                        std::memcpy(staging.data() + (size_t) lr * f.gu_row,
                                    src + (size_t) r * f.gu_row, (size_t) f.gu_row);
                        std::memcpy(staging.data() + g.o_up + (size_t) lr * f.gu_row,
                                    src + f.up_off + (size_t) r * f.gu_row, (size_t) f.gu_row);
                    }
                    for (int r = 0; r < H; ++r) {
                        const uint8_t* srow = src + f.down_off + (size_t) r * f.d_row;
                        uint8_t* dcode = staging.data() + g.d_codes_off + (size_t) r * g.d_row_stride;
                        uint8_t* dscale = staging.data() + g.d_scales_off + (size_t) r * g.d_scale_stride;
                        for (int lb = 0; lb < g.tp_blks; ++lb) {
                            const uint8_t* blk = srow + (size_t) (g.tp_blk_base + lb) * 18;
                            dscale[lb * 2] = blk[0];
                            dscale[lb * 2 + 1] = blk[1];
                            std::memcpy(dcode + (size_t) lb * 16, blk + 2, 16);
                        }
                    }
                } else {
                std::memcpy(staging.data(), src, f.down_off);
                for (int r = 0; r < H; ++r) {
                    const uint8_t* srow = src + f.down_off + (size_t) r * f.d_row;
                    uint8_t* dcode = staging.data() + g.d_codes_off + (size_t) r * ROW_D;
                    uint8_t* dscale = staging.data() + g.d_scales_off + (size_t) r * SC_D * 2;
                    for (int b = 0; b < SC_D; ++b) {
                        const uint8_t* blk = srow + (size_t) b * 18;
                        dscale[b * 2] = blk[0];
                        dscale[b * 2 + 1] = blk[1];
                        std::memcpy(dcode + b * 16, blk + 2, 16);
                    }
                }
                }
            }
        } else {
            // already S2 planar (canonical pack): direct copy
            std::memcpy(staging.data(), src, BLOB);
        }
        if (!check(cudaMemcpy((uint8_t*) d_blobs_ + dst_off, staging.data(), blob_bytes,
                              cudaMemcpyHostToDevice),
                   "copy blob", err)) { close(); return false; }
        res_[(size_t) L * (size_t) n_expert + (size_t) mine[i].second] = slot;
        held_.push_back(mine[i]);
    }
    resident_ = (int64_t) mine.size();
    gib_ = (double) bytes / (1024.0 * 1024.0 * 1024.0);

    // activation buffers: int8 codes + fp32 scales + hx per chunk, for H-wide and FF-wide acts.
    // n_tok_max/cap_entries bound the window.  P3.1: 8 tokens was the decode window; the prefill feeds whole
    // prompt chunks, so the window is 64 tokens (640 entries) and callers sub-batch larger chunks.
    n_tok_max_ = 64;
    cap_entries_ = 64 * 10;
    // P3.2: ONE packed ActQ buffer per device, tokens contiguous ([q H][scale][hx] each, ACTS_STRIDE), so a
    // layer's window uploads only n_tok * ACTS_STRIDE bytes - not the whole 64-token buffer (measured: the
    // full 205 KB copy cost 0.073 ms per device per layer on the M10's slow host link).
    acts_bytes_ = (size_t) n_tok_max_ * (size_t) ACTS_STRIDE;
    const size_t ce = (size_t) cap_entries_;
    const size_t jobs_bytes = ce * 3 * sizeof(int32_t);
    // P6.x (measured): ONE contiguous staging region [packed acts][job table] so a layer costs ONE async
    // upload instead of two.  A CUDA call is ~4 us in this VM; 2 calls -> 1 per device per layer.
    up_bytes_ = acts_bytes_ + jobs_bytes;
    // P7 (2026-10-12, MEASURED AND REJECTED as the default): zero-copy staging - the pinned buffer
    // allocated MAPPED, the kernels reading acts+jobs straight from host memory.  It removes the
    // per-layer cudaMemcpyAsync (~4 us of API x 4 devices) but the mapped reads are UNCACHED system
    // memory on sm_50: the kernel stalls on PCIe latency and the engine measured 28.2/34.1/33.8 tok/s
    // vs 40.0/51.0/46.7 (-30%).  OFF by default; STRATA_MW_ZEROCOPY=1 restores it for reference.
    static const bool zerocopy = [] {
        const char* e = std::getenv("STRATA_MW_ZEROCOPY");
        return e != nullptr && std::atoi(e) != 0;
    }();
    zerocopy_ = zerocopy;    if (zerocopy) {
        if (!check(cudaHostAlloc((void**) &h_up_, up_bytes_,
                                 cudaHostAllocMapped | (::strata::core::maxwell_portable() ? cudaHostAllocPortable : 0)),
                   "cudaHostAlloc mapped staging", err)) { close(); return false; }
        if (!check(cudaHostGetDevicePointer((void**) &d_up_, h_up_, 0), "map staging", err)) {
            close(); return false;
        }
        std::fprintf(stderr, "strata maxwell: acts+jobs staging is ZERO-COPY (mapped pinned; measured "
                             "-30%% tok/s on sm_50 - uncached sysmem reads; reference only)\n");
    } else {
        if (!check(cudaMalloc(&d_up_, up_bytes_), "cudaMalloc staging", err)) { close(); return false; }
        if (!check(cudaMallocHost((void**) &h_up_, up_bytes_), "cudaMallocHost staging", err)) { close(); return false; }
        if (gpuflag_mode() >= 2) {   // E1 mode 2: the second staging set (layer l+1 stages while l reads)
            if (!check(cudaMalloc(&d_up2_, up_bytes_), "cudaMalloc staging2", err)) { close(); return false; }
            if (!check(cudaMallocHost((void**) &h_up2_, up_bytes_), "cudaMallocHost staging2", err)) { close(); return false; }
        }
    }
    if (gpuflag_mode() >= 1 && !check(cudaMalloc(&d_ctr_, 4), "cudaMalloc comp ctr", err)) { close(); return false; }
    if (d_ctr_ != nullptr) { cudaMemset(d_ctr_, 0, 4); cudaGetLastError(); }
    d_acts_ = d_up_;                 // the packed acts live at offset 0
    h_acts_ = h_up_;
    d_jobs_ = (int32_t*) (d_up_ + acts_bytes_);   // re-based per layer in begin_chain (right after the acts)
    h_jobs_ = (int32_t*) (h_up_ + acts_bytes_);
    d_aq_ = d_acts_;                 // legacy pointers: the batch kernels index by token now
    d_ascale_ = nullptr;
    d_ahx_ = nullptr;
    if (!check(cudaMalloc(&d_faq_, ce * FF), "cudaMalloc faq", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_fascale_, ce * (FF/32) * sizeof(float)), "cudaMalloc fascale", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_fahx_, ce * (FF/32) * sizeof(float)), "cudaMalloc fahx", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_ff_, ce * FF * sizeof(float)), "cudaMalloc ff", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_out_, ce * H * sizeof(float)), "cudaMalloc out", err)) { close(); return false; }
    if (!check(cudaMallocHost((void**) &h_out_, ce * H * sizeof(float)), "cudaMallocHost out", err)) { close(); return false; }

    // P3.3: per-device streams and events for the overlap DAG
    for (int s = 0; s < 4; ++s) {
        if (!check(cudaStreamCreateWithFlags(&streams_[s], cudaStreamNonBlocking), "stream", err)) { close(); return false; }
        if (!check(cudaEventCreateWithFlags(&ev_gup_[s], cudaEventDisableTiming), "event", err)) { close(); return false; }
        if (!check(cudaEventCreateWithFlags(&ev_down_[s], cudaEventDisableTiming), "event", err)) { close(); return false; }
    }
    if (!check(cudaEventCreateWithFlags(&ev_up_, cudaEventDisableTiming), "event", err)) { close(); return false; }
    if (!check(cudaEventCreateWithFlags(&ev_q_, cudaEventDisableTiming), "event", err)) { close(); return false; }
    if (!check(cudaEventCreateWithFlags(&ev_d2h_done_, cudaEventDisableTiming), "event", err)) { close(); return false; }

    owned_idx_.assign((size_t) cap_entries_, -1);
    // P3.4 (rev 3, misurato): the chain runs inline on the caller's thread.  The persistent per-device worker
    // (condvars first, atomic wait/notify second) measured SLOWER: the four devices' cudaDeviceSynchronize
    // calls serialise inside the driver, so parallelising them buys nothing, and the per-layer handoff added
    // ~150 us/layer (29.3 vs 22.2 ms/window).  What pays is shrinking the single finish: the down stage now
    // launches there and scatters straight into the mapped destination (no D2H, no host memcpy).
    //
    // THE ROUND-TRIP GATE (native packs): the conversion above is checked on the real data before this
    // tier answers a token.  The failure mode it exists to catch is the one the first IQ2_XS attempt died
    // of and never isolated: a wrong offset or row stride in the src->blob conversion decodes PLAUSIBLE
    // garbage - never a fault, a model that answers nonsense words instead.  One held expert of each
    // distinct gate/up encoding runs the WHOLE chain on the device (the converted blob, the tier's own
    // packed ActQ, its own launch path) against the raw source blob dequantized by ggml and dotted in
    // fp64.  The tolerance is the act + FF quantisation noise the 3060's own native path carries
    // (~1e-2); a conversion bug is ~1.0 and cannot pass it.
    if (!geo_.empty()) {
        ggml_cpu_init();   // the to_float tables (idempotent; native_fmt already did this)
        int seen[4] = {0, 0, 0, 0};   // by gate/up type: 42, 22, 16, 29
        double worst = 0.0;
        for (const auto& pr : held_) {
            const int64_t L = pr.first, e = pr.second;
            const auto& f = lay.fmt[(size_t) L];
            const int ti = f.gu_type == 42 ? 0 : (f.gu_type == 22 ? 1 : (f.gu_type == 16 ? 2 : 3));
            if (ti < 0 || seen[ti]) continue;
            seen[ti] = 1;
            const uint8_t* src = src_blob(L, e, ctx);
            if (src == nullptr) {
                err = "MaxwellExperts: the round-trip gate lost its source blob";
                close();
                return false;
            }
            const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
            const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
            std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
            for (int64_t r = 0; r < FF; ++r) {
                tg->to_float(src + (size_t) r * f.gu_row, G.data() + (size_t) r * H, H);
                tg->to_float(src + f.up_off + (size_t) r * f.gu_row, U.data() + (size_t) r * H, H);
            }
            for (int64_t r = 0; r < H; ++r)
                td->to_float(src + f.down_off + (size_t) r * f.d_row, D.data() + (size_t) r * FF, FF);
            // the activation: deterministic, the engine's own ActQ (what the dispatch hands the tier)
            std::vector<float> x(H);
            std::mt19937 rng(1234 + (int) L * 31 + (int) e);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);
            strata::kernels::cpu::ActQ a;
            strata::kernels::cpu::act_quant_any(x.data(), (int) H, a);
            // the reference: the float expert (fp64 dots, the SwiGLU in fp32 as the kernels have it).
            // SPLIT_MODE_TENSOR: the reference is THIS CARD's partial - the same fp64 contraction
            // restricted to the shard's blocks (the gate/up SwiGLU stays full-width on the host: the
            // four cards' partials must sum to the full reference, and each is gated against its own).
            const LayerGeo& gg = geo_[(size_t) L];
            const int c0 = tp_cards_ > 1 ? gg.tp_blk_base * 64 : 0;
            const int c1 = tp_cards_ > 1 ? (gg.tp_blk_base + gg.tp_blks) * 64 : (int) FF;
            std::vector<float> ref(H), h((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) {
                    g += (double) G[(size_t) r * H + i] * x[i];
                    u += (double) U[(size_t) r * H + i] * x[i];
                }
                h[(size_t) r] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = c0; i < c1; ++i) o += (double) D[(size_t) r * FF + i] * h[i];
                ref[(size_t) r] = (float) o;
            }
            // the device: one job through the CONVERTED blob, the tier's own launch path
            const LayerGeo& g = geo_[(size_t) L];
            const int32_t slot = res_[(size_t) L * (size_t) n_expert_ + (size_t) e];
            const size_t nc4 = (size_t) (H / 32) * 4;
            std::memcpy(h_acts_, a.q, H);
            std::memcpy(h_acts_ + H, a.scale, nc4);
            std::memcpy(h_acts_ + H + nc4, a.hx, nc4);
            h_jobs_ = (int32_t*) (h_up_ + (size_t) 1 * ACTS_STRIDE);   // one token, re-based as begin_chain
            d_jobs_ = (int32_t*) (d_up_ + (size_t) 1 * ACTS_STRIDE);
            h_jobs_[0] = slot; h_jobs_[1] = 0; h_jobs_[2] = 0;
            if (!check(cudaMemcpyAsync(d_up_, h_up_, (size_t) 1 * ACTS_STRIDE + 3 * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, streams_[0]), "round-trip upload", err)) {
                close();
                return false;
            }
            if (g.gu_type == 42) {
                if (s2_expert_maxwell_v2()) {
                    s2_expert_maxwell_gup_quant_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_acts_,
                                                            d_faq_, d_fascale_, d_fahx_, streams_[0]);
                } else {
                    s2_expert_maxwell_gup_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_acts_, d_ff_,
                                                      streams_[0]);
                    s2_expert_maxwell_quant_ff_async(d_ff_, d_faq_, d_fascale_, d_fahx_, FF / 32,
                                                     streams_[0]);
                }
            } else {
                iq_expert_maxwell_gup_quant_batch_async(g.gu_type, d_blobs_ + g.base, g.blob_bytes,
                                                        d_jobs_, 1, d_acts_, g.o_gate, g.o_up, g.gu_row,
                                                        d_faq_, d_fascale_, d_fahx_, nullptr, streams_[0],
                                                        g.tp_row_base, g.tp_rows);
            }
            s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_faq_, d_fascale_, d_fahx_,
                                               d_out_, g.blob_bytes, g.d_codes_off, g.d_scales_off,
                                               streams_[0], nullptr, 0, nullptr,
                                               g.tp_blk_base, g.tp_blks, g.d_row_stride, g.d_scale_stride);
            if (!check(cudaStreamSynchronize(streams_[0]), "round-trip sync", err)) { close(); return false; }
            if (!check(cudaMemcpy(h_out_, d_out_, (size_t) H * sizeof(float), cudaMemcpyDeviceToHost),
                       "round-trip download", err)) { close(); return false; }
            double num = 0, den = 0;
            for (int64_t r = 0; r < H; ++r) {
                num += std::fabs((double) h_out_[r] - ref[(size_t) r]);
                den += std::fabs((double) ref[(size_t) r]);
            }
            const double rv = num / (den + 1e-30);
            worst = std::max(worst, rv);
            std::fprintf(stderr, "MaxwellExperts (CUDA%d): round-trip gate layer %lld expert %lld "
                                 "(%s): rel %.2e vs the float reference\n",
                         device_, (long long) L, (long long) e,
                         ggml_type_name((ggml_type) f.gu_type), rv);
            std::fflush(stderr);
            if (!std::isfinite(rv) || rv > 3e-2) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "MaxwellExperts (CUDA%d): the round-trip gate FAILED on layer %lld "
                               "expert %lld (%s): rel %.2e is non-finite or > 3e-2 - refusing",
                              device_, (long long) L, (long long) e,
                              ggml_type_name((ggml_type) f.gu_type), rv);
                err = buf;
                close();
                return false;
            }
        }
        std::fprintf(stderr, "MaxwellExperts (CUDA%d): round-trip gate PASSED (worst rel %.2e, "
                             "%lld blobs converted)\n", device_, worst, (long long) resident_);
        std::fflush(stderr);
    }

    // ---- STRATA_MW_PERSIST: arm the resident decode kernel -----------------------------------
    // Conditions: the env asks for it; the whole-expert layout (no tensor split); no zero-copy; a
    // native pack whose every layer is an i-quant the persistent kernel's switch covers (22/16/29 -
    // a 42/S2 gate/up layer or the canonical pack declines: this model's packs are all-i-quant);
    // the default gup geometry (STRATA_MW_IQ_SPLIT 4, no T256), which is what <4,128> instantiates.
    static const bool persist_env = [] {
        const char* e = std::getenv("STRATA_MW_PERSIST");
        const bool on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (on) std::fprintf(stderr, "strata maxwell: PERSISTENT decode tier (STRATA_MW_PERSIST)\n");
        return on;
    }();
    if (persist_env) {
        bool ok = !zerocopy_ && tp_cards_ == 1 && !geo_.empty();
        if (ok) {
            static const int iq_split_env = [] {
                const char* e = std::getenv("STRATA_MW_IQ_SPLIT");
                return e ? std::atoi(e) : 4;
            }();
            static const bool t256_env = [] {
                const char* e = std::getenv("STRATA_MW_IQ_T256");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            ok = iq_split_env == 4 && !t256_env;
        }
        if (ok)
            for (const auto& g : geo_)
                if (g.gu_type != 22 && g.gu_type != 16 && g.gu_type != 29) { ok = false; break; }
        if (!ok) {
            std::fprintf(stderr, "strata maxwell (CUDA%d): PERSIST declined (needs whole-expert, no "
                                 "zero-copy, an all-i-quant native pack, SPLIT 4 / 128 threads)\n", device_);
        } else {
            if (d_up2_ == nullptr && !check(cudaMalloc(&d_up2_, up_bytes_), "cudaMalloc staging2", err)) {
                close();
                return false;
            }
            if (!check(cudaMalloc(&d_seq_, 4), "cudaMalloc persist seq", err)) { close(); return false; }
            if (!check(cudaHostAlloc((void**) &h_halt_, 4,
                                     cudaHostAllocMapped | (::strata::core::maxwell_portable() ? cudaHostAllocPortable : 0)),
                       "cudaHostAlloc persist halt", err)) { close(); return false; }
            {
                void* dh = nullptr;
                if (!check(cudaHostGetDevicePointer(&dh, (void*) h_halt_, 0), "map persist halt", err)) {
                    close();
                    return false;
                }
                d_halt_ = (const volatile uint32_t*) dh;
            }
            *h_halt_ = 0;
            if (!check(cudaMalloc(&d_pctl_, sizeof(*d_pctl_)), "cudaMalloc persist ctl", err)) { close(); return false; }
            if (!check(cudaMalloc(&d_pparams_, sizeof(*d_pparams_)), "cudaMalloc persist params", err)) { close(); return false; }
            // the params image and the doorbell word are written by persist_launch() before every
            // launch (the completion flag is handed over by set_completion_flag only AFTER open)
            persist_armed_ = true;
            std::fprintf(stderr, "strata maxwell (CUDA%d): persistent tier armed (DMA dispatch + device "
                                 "doorbell; STRATA_MW_GPUFLAG=%d, recommended 2)\n",
                         device_, gpuflag_mode());
        }
    }
    return true;
}

void MaxwellExperts::close() {
    if (device_ < 0) return;
    persist_halt();   // the resident kernel must retire before anything it touches is freed
    DeviceScope scope(device_);
    for (int s = 0; s < 4; ++s) {
        if (streams_[s]) { cudaStreamDestroy(streams_[s]); streams_[s] = nullptr; }
        if (ev_gup_[s]) { cudaEventDestroy(ev_gup_[s]); ev_gup_[s] = nullptr; }
        if (ev_down_[s]) { cudaEventDestroy(ev_down_[s]); ev_down_[s] = nullptr; }
    }
    if (ev_up_) { cudaEventDestroy(ev_up_); ev_up_ = nullptr; }
    if (ev_q_) { cudaEventDestroy(ev_q_); ev_q_ = nullptr; }
    if (ev_d2h_done_) { cudaEventDestroy(ev_d2h_done_); ev_d2h_done_ = nullptr; }
    if (d_blobs_) { cudaFree(d_blobs_); d_blobs_ = nullptr; }
    if (d_up_ && !zerocopy_) { cudaFree(d_up_); }   // zero-copy: d_up_ IS h_up_'s mapping
    d_up_ = nullptr;
    if (d_up2_) { cudaFree(d_up2_); d_up2_ = nullptr; }
    if (h_up2_) { cudaFreeHost(h_up2_); h_up2_ = nullptr; }
    if (d_ctr_) { cudaFree(d_ctr_); d_ctr_ = nullptr; }
    if (h_up_) { cudaFreeHost(h_up_); h_up_ = nullptr; }
    d_acts_ = nullptr; h_acts_ = nullptr; d_jobs_ = nullptr; h_jobs_ = nullptr;
    d_aq_ = nullptr; d_ascale_ = nullptr; d_ahx_ = nullptr;
    if (d_faq_) { cudaFree(d_faq_); d_faq_ = nullptr; }
    if (d_fascale_) { cudaFree(d_fascale_); d_fascale_ = nullptr; }
    if (d_fahx_) { cudaFree(d_fahx_); d_fahx_ = nullptr; }
    if (d_ff_) { cudaFree(d_ff_); d_ff_ = nullptr; }
    if (d_out_) { cudaFree(d_out_); d_out_ = nullptr; }
    if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    if (s0_a_ != nullptr) { cudaEventDestroy(s0_a_); s0_a_ = nullptr; }
    if (s0_b_ != nullptr) { cudaEventDestroy(s0_b_); s0_b_ = nullptr; }
    if (s0_m_ != nullptr) { cudaEventDestroy(s0_m_); s0_m_ = nullptr; }
    if (d_pctl_) { cudaFree(d_pctl_); d_pctl_ = nullptr; }
    if (d_pparams_) { cudaFree(d_pparams_); d_pparams_ = nullptr; }
    if (d_seq_) { cudaFree(d_seq_); d_seq_ = nullptr; }
    if (h_halt_) { cudaFreeHost((void*) h_halt_); h_halt_ = nullptr; }
    d_halt_ = nullptr;
    persist_armed_ = false;
    persist_live_ = false;
    w_persisted_ = false;
    device_ = -1;
    resident_ = 0;
    gib_ = 0.0;
    res_.clear();
    held_.clear();
}

// STRATA_MW_PERSIST: (re)launch the resident kernel.  The params are refreshed each time - the
// completion flag is handed over by set_completion_flag AFTER open(), and a halt/relaunch cycle
// around a prefill must pick up the same words.
bool MaxwellExperts::persist_launch(std::string& err) {
    if (comp_flag_ == nullptr) {
        err = "persist: no completion flag wired (this path does not arm the E1 flags)";
        return false;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.err; return false; }
    strata::kernels::cuda::PersistParams pp{};
    pp.blobs = d_blobs_;
    pp.d_up[0] = d_up_;
    pp.d_up[1] = d_up2_;
    pp.faq = d_faq_;
    pp.fasc = d_fascale_;
    pp.fahx = d_fahx_;
    pp.comp_flag = comp_flag_;
    pp.d_seq = d_seq_;
    pp.halt = d_halt_;
    if (!check(cudaMemcpy(d_pparams_, &pp, sizeof(pp), cudaMemcpyHostToDevice), "copy persist params", err))
        return false;
    if (!check(cudaMemset(d_seq_, 0, 4), "reset persist seq", err)) return false;
    persist_nseq_ = 0;   // the doorbell counts relative services: a fresh kernel serves 1, 2, ...
    const int blocks = strata::kernels::cuda::maxwell_persist_launch(d_pparams_, d_pctl_, streams_[2], err);
    if (blocks <= 0) return false;
    persist_live_ = true;
    static thread_local int said_blocks = -1;
    if (blocks != said_blocks) {
        said_blocks = blocks;
        std::fprintf(stderr, "strata maxwell (CUDA%d): persistent tier resident (%d blocks)\n", device_, blocks);
    }
    return true;
}

// The stop word is a HOST STORE to the mapped halt word - the resident kernel owns its stream, so
// a stream-ordered write could never reach it (it would queue behind the kernel and both would
// wait for each other).  The kernel's poll loop reads the word periodically; a service mid-flight
// finishes first (its rows and flag raise are harmless - the window's expected values moved on).
void MaxwellExperts::persist_halt() {
    if (!persist_live_ || h_halt_ == nullptr) return;
    persist_live_ = false;
    {
        DeviceScope scope(device_);
        if (scope.ok) {
            __atomic_store_n((volatile uint32_t*) h_halt_, 1u, __ATOMIC_SEQ_CST);
            cudaStreamSynchronize(streams_[2]);
            cudaStreamSynchronize(streams_[0]);   // any in-flight service H2D retires too
            *h_halt_ = 0;                          // re-arm for the next launch
        }
    }
}

bool MaxwellExperts::owns(int64_t layer, int64_t expert) const {
    if (device_ < 0 || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] >= 0;
}

int64_t MaxwellExperts::count_owned(int64_t layer, const int32_t* ids, int64_t n_tok, int64_t k) const {
    if (device_ < 0 || layer < 0 || layer >= n_layers_) return 0;
    int64_t c = 0;
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        if (res_[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0) ++c;
    }
    return c;
}

bool MaxwellExperts::begin_chain(std::string& err) {
    const auto bt0 = std::chrono::steady_clock::now();
    const int64_t layer = w_layer_;
    const strata::kernels::cpu::ActQ* acts = w_acts_;
    const int32_t* ids = w_ids_;
    const int64_t n_tok = w_n_tok_, k = w_k_;
    const int32_t* kind = nullptr;   // collect every entry this device owns
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.err; return false; }
    const auto bt1 = std::chrono::steady_clock::now();
    owned_row_.clear();

    // P3.1 BATCHED, three phases per layer (not per expert):
    //   Phase 1: upload all ActQ + launch ALL stage1 (async, one sync at the end)
    //   Phase 2: ONE D2H of all ff, requant all on host, ONE H2D of all ffq
    //   Phase 3: launch ALL stage2 (async, one sync), ONE D2H of all results

    // ---- collect owned entries first (no GPU work yet) ----
    struct Job { int32_t slot; int64_t tok; int32_t row; };
    // thread_local to avoid per-layer allocation (the token path allocates nothing)
    static thread_local std::vector<Job> jobs;
    jobs.clear();
    for (int64_t i = 0; i < n_tok * k; ++i) {
        if (kind && kind[i] != -1 && kind[i] != 3) continue;
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const int32_t slot = res_[(size_t) layer * (size_t) n_expert_ + (size_t) e];
        if (slot < 0) continue;
        jobs.push_back({slot, i / k, (int32_t) i});
    }
    // One packing routine for BOTH dispatch paths (the persist ring slot and the legacy staging):
    // [acts: n_tok x ACTS_STRIDE][job table: (slot, tok, dst) x njobs] at `to`; returns the bytes
    // written.  `tp_compact` writes the COMPACT JOB SLOT as dst (SPLIT_MODE_TENSOR: the partial
    // block is one contiguous region the copy engine D2Hs whole); non-TP writes the entry index
    // (the down body's scatter key into the verifier's rows).
    const auto pack_staging = [&](uint8_t* to, size_t njobs, const Job* jobs_l, bool tp_compact) {
        const size_t nc4 = (size_t) (H / 32) * 4;
        if (shared_acts_ != nullptr) {
            // Meta packed the window once: one bulk copy per rank instead of the per-token loop x4.
            std::memcpy(to, shared_acts_, (size_t) n_tok * ACTS_STRIDE);
        } else
        for (int64_t t = 0; t < n_tok; ++t) {
            const auto& A = acts[t];
            uint8_t* at = to + (size_t) t * ACTS_STRIDE;
            std::memcpy(at, A.q, H);
            std::memcpy(at + H, A.scale, nc4);
            std::memcpy(at + H + nc4, A.hx, nc4);
        }
        int32_t* hj = (int32_t*) (to + (size_t) n_tok * ACTS_STRIDE);
        for (size_t j = 0; j < njobs; ++j) {
            hj[j * 3] = jobs_l[j].slot;
            hj[j * 3 + 1] = (int32_t) jobs_l[j].tok;
            hj[j * 3 + 2] = tp_compact ? (int32_t) j : jobs_l[j].row;
        }
        return (size_t) n_tok * ACTS_STRIDE + njobs * 3 * sizeof(int32_t);
    };

    // ---- STRATA_MW_PERSIST: the service is [pack pinned][ONE H2D][ONE stream-written doorbell]. ----
    // Both calls are issued while every M10 sits idle at its doorbell (no down window open to
    // poison); the copy engine does the staging (an in-kernel mapped pull measured ~2 ms/layer).
    // Packed layout [PersistHdr][acts][jobs]; the DEVICE doorbell counts RELATIVE services per
    // launch - the kernel serves them STRICTLY IN ORDER (the split window queues up to two, and
    // serving the newest would skip the one between: its flag would never rise) - while the hdr's
    // ring field carries the GLOBAL service value the verifier's multi-wait expects.  EMPTY
    // services (njobs 0) ride the same doorbell: every counted service is one the kernel raises.
    if (persist_armed_ && w_dst_mapped_ && !w_rows_f16_) {
        if (!persist_live_ && !persist_launch(err)) return false;
        const uint32_t ring = comp_seq_ + 1;
        const uint32_t nseq = ++persist_nseq_;
        uint8_t* hsel = (nseq & 1u) ? h_up2_ : h_up_;
        uint8_t* dsel = (nseq & 1u) ? d_up2_ : d_up_;
        namespace mk = strata::kernels::cuda;
        const size_t staged = pack_staging(hsel + mk::PERSIST_HDR_BYTES, jobs.size(), jobs.data(), false);
        {
            const LayerGeo& g = geo_of(layer);
            mk::PersistHdr* hd = (mk::PersistHdr*) hsel;
            hd->njobs = (uint32_t) jobs.size();
            hd->n_tok = (uint32_t) n_tok;
            hd->gu_type = (uint32_t) g.gu_type;
            hd->ring = ring;
            hd->tp_row_base = g.tp_row_base;
            hd->tp_rows = g.tp_rows;
            hd->tp_blk_base = g.tp_blk_base;
            hd->tp_blks = g.tp_blks;
            hd->blob_base = g.base;
            hd->blob_stride = g.blob_bytes;
            hd->o_gate = g.o_gate;
            hd->o_up = g.o_up;
            hd->gu_row = g.gu_row;
            hd->d_codes_off = g.d_codes_off;
            hd->d_scales_off = g.d_scales_off;
            hd->d_row_stride = g.d_row_stride;
            hd->d_scale_stride = g.d_scale_stride;
            hd->dst = w_dst_;
        }
        const size_t bytes = mk::PERSIST_HDR_BYTES + staged;
        if (!check(cudaMemcpyAsync(dsel, hsel, bytes, cudaMemcpyHostToDevice, streams_[0]),
                   "persist H2D", err)) return false;
        if (CUDA_SUCCESS != cuStreamWriteValue32((CUstream) streams_[0], (CUdeviceptr) (uintptr_t) d_seq_,
                                                 nseq, 0)) {
            err = "persist: the doorbell write failed";
            return false;
        }
        comp_seq_ = ring;
        for (size_t j = 0; j < jobs.size(); ++j) {
            owned_row_.push_back(jobs[j].row);
            computed_.fetch_add(1, std::memory_order_relaxed);
        }
        down_launched_ = true;   // the kernel's down phase covers it; finish must not re-stage
        w_persisted_ = true;
        return true;
    }
    if (jobs.empty()) return true;
    if ((int64_t) jobs.size() > cap_entries_) {
        err = "MaxwellExperts: more owned entries than cap_entries";
        return false;
    }
    // P3.1 stats: histogram of jobs/layer to validate batching effectiveness
    {
        static thread_local long hist[16] = {};
        static thread_local long n_calls = 0;
        if (jobs.size() < 16) ++hist[jobs.size()];
        if (++n_calls % 20000 == 0) {
            std::fprintf(stderr, "Maxwell STAT jobs/layer histogram (n=%ld):", n_calls);
            for (int i = 0; i < 16; ++i) if (hist[i]) std::fprintf(stderr, " [%d]=%ld", i, hist[i]);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
    }

    // ---- ONE packed upload (acts + job table, contiguous) then ONE launch per stage: the whole layer's
    //      jobs are processed by the two batched kernels (gup -> quant -> down).  No events, no per-job
    //      launches; finish() is the single wait per layer. ----
    // S0 (STRATA_MW_LAYERPROF): count what this card serves - jobs, DISTINCT experts (the bytes the span
    // moves are distinct x blob_bytes; duplicate entries re-read the same blob) - and start the device
    // span measurement.  The events ride the same stream the kernels do, so the elapsed is the card's
    // own gup->down occupancy, independent of when the host got around to us.
    const bool s0 = s0_env();
    if (s0) {
        int distinct = 0;
        for (size_t a = 0; a < jobs.size(); ++a) {
            bool first = true;
            for (size_t b = 0; b < a; ++b)
                if (jobs[b].slot == jobs[a].slot) { first = false; break; }
            if (first) ++distinct;
        }
        s0_layers_ += 1.0; s0_jobs_ += (double) jobs.size(); s0_distinct_ += (double) distinct;
        s0_mb_ += (double) distinct * (double) geo_of(layer).blob_bytes / 1048576.0;
        if (s0_a_ == nullptr) { cudaEventCreate(&s0_a_); cudaEventCreate(&s0_b_); cudaEventCreate(&s0_m_); }
        else if (s0_deferred_) {
            // E1 mode 2 left the previous layer's span unharvested (no sync on that path).  The stream
            // is the same, so this layer's kernels queue BEHIND that down: by now its events are
            // settled, and the elapsed can be reaped right before the new span arms.
            float el = 0.0f, eg = 0.0f, ed = 0.0f;
            cudaEventElapsedTime(&el, s0_a_, s0_b_);
            cudaEventElapsedTime(&eg, s0_a_, s0_m_);
            cudaEventElapsedTime(&ed, s0_m_, s0_b_);
            s0_krn_ms_ += (double) el; s0_gup_ms_ += (double) eg; s0_dn_ms_ += (double) ed;
            if (s0_layers_ >= 1440.0) s0_report();
            s0_deferred_ = false;
        }
        cudaEventRecord(s0_a_, streams_[0]);
    }
    size_t staged = 0;
    {
        // re-base the job table right after the packed acts so ONE async memcpy covers both (P6.x).
        // E1 mode 2: alternate the two staging sets (layer l+1 stages while layer l's kernels still
        // read set 0) - the selection is the SERVICE parity, matching the launch sequence.
        uint8_t* hsel = h_up_;
        uint8_t* dsel = d_up_;
        if (gpuflag_mode() >= 2 && d_up2_ != nullptr && (comp_seq_ & 1u)) { hsel = h_up2_; dsel = d_up2_; }
        h_jobs_ = (int32_t*) (hsel + (size_t) n_tok * ACTS_STRIDE);
        d_jobs_ = (int32_t*) (dsel + (size_t) n_tok * ACTS_STRIDE);
        h_acts_ = hsel;
        d_acts_ = dsel;
        staged = pack_staging(hsel, jobs.size(), jobs.data(), tp_cards_ > 1);
    }
    // ONE async upload for the acts + the job table (contiguous staging); finish() is the only wait.
    // Zero-copy: the mapped staging needs no upload at all - the kernel reads it over PCIe.
    if (!zerocopy_ &&
        !check(cudaMemcpyAsync(d_acts_, h_acts_, staged, cudaMemcpyHostToDevice, streams_[0]),
               "copy acts+jobs", err)) return false;
    // v2 (default): gate+up and the FF quantisation in ONE launch - the fused kernel's block covers exactly
    // one 32-value chunk of its job, so the separate quant launch (~4 us of API per device per layer in this
    // VM, plus the kernel off the chain) disappears.  v1 keeps the two-launch pair.  A native pack's
    // i-quant layers (IQ2_S/IQ2_XXS/IQ1_M) take the IQ kernel - it is fused by construction, and the
    // STRATA_S2_V2 arm is the S2 form's geometry A/B, not a choice the i-quant path offers.
    const LayerGeo& g = geo_of(layer);
    if (g.gu_type == 42) {
        if (s2_expert_maxwell_v2()) {
            s2_expert_maxwell_gup_quant_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_acts_,
                                                    d_faq_, d_fascale_, d_fahx_, streams_[0]);
        } else {
            s2_expert_maxwell_gup_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_acts_, d_ff_,
                                              streams_[0]);
            s2_expert_maxwell_quant_ff_async(d_ff_, d_faq_, d_fascale_, d_fahx_,
                                             (int) (jobs.size() * (size_t) (FF / 32)), streams_[0]);
        }
    } else {
        iq_expert_maxwell_gup_quant_batch_async(g.gu_type, d_blobs_ + g.base, g.blob_bytes, d_jobs_,
                                                (int) jobs.size(), d_acts_, g.o_gate, g.o_up, g.gu_row,
                                                d_faq_, d_fascale_, d_fahx_, nullptr, streams_[0],
                                                g.tp_row_base, g.tp_rows);
    }
    // The TOKEN path hands begin() the mapped destination: its down stage launches HERE and scatters each row
    // straight into `w_dst_` (the job table already carries every row's destination), so it runs while the
    // caller finishes the layer - no D2H copy and no host scatter.  The prefill's destination is only known at
    // finish_to, so ITS down launches there, into the staging block.
    // S0: the mid-stamp splits the card's span into gup (the i-quant dequant + dot) and down (the S2 form).
    if (s0 && s0_a_ != nullptr) cudaEventRecord(s0_m_, streams_[0]);
    down_launched_ = false;
    if (w_dst_mapped_) {
        // STRATA_MW_DOWN_LOCAL_TIMING (measurement only, garbage output): the down launches HERE as
        // usual but writes card-local d_out_ instead of the mapped rows - nothing delivers the rows,
        // the generation is garbage, and the [mid,end] span then shows the down kernel's time with
        // the mapped-write path OUT of the picture.  If the visit-order stagger (83 -> 240+ us/L)
        // disappears here, the mapped destination itself is the contention, not the kernel.
        static const bool dlt = [] {
            const char* e = std::getenv("STRATA_MW_DOWN_LOCAL_TIMING");
            const bool on = e != nullptr && std::atoi(e) != 0;
            if (on) std::fprintf(stderr, "strata maxwell: TIMING-ONLY down to card-local d_out_ "
                                         "(STRATA_MW_DOWN_LOCAL_TIMING; the output will be garbage)\n");
            return on;
        }();
        float* const dst_rows = dlt ? d_out_ : w_dst_;
        // E1: in GPU-flag mode the decode down launch carries the completion flag - the LAST block
        // raises it once this tier's rows are visible, and (mode 2) the host never syncs this layer.
        // fp16 rows take the no-tail variant, so the flag stays off there (the sync path is kept).
        const bool e1 = comp_flag_ != nullptr && !w_rows_f16_ && gpuflag_mode() >= 1;
        const uint32_t ring = e1 ? (comp_seq_ + 1) : 0;
        if (w_rows_f16_) {
            // fp16 rows (opt-in): H halves packed into the fp32 row's own slot (the first H/2 floats), so
            // the per-group destination offset stays whatever `w_dst_` already carries.  Half the bytes
            // cross the uplink; the verifier's row copy converts them back (its kind16 flags mark them).
            // (No tensor-split form: TP clears rows_f16 in begin().)
            s2_expert_maxwell_down_batch_f16_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_faq_,
                                                   d_fascale_, d_fahx_, (uint16_t*) dst_rows, g.blob_bytes,
                                                   g.d_codes_off, g.d_scales_off, streams_[0]);
        } else if (tp_cards_ > 1 && !dlt) {
            // SPLIT_MODE_TENSOR delivery: the down writes this card's PARTIAL rows to CARD-LOCAL VRAM -
            // no mapped stores from the kernel (four concurrent downs writing mapped host memory
            // measured 3x stretched, the LOCAL_DOWN shared-endpoint effect) - then the COPY ENGINE moves
            // the compact partial block to the host partial area on a second stream, and a one-block
            // kernel raises the E1 doorbell after it.  The down itself carries no tail; the ring
            // advances with the raise.
            // The block is moved in CHUNKS: each chunk's D2H only waits for its own downs, so the wire
            // hides under the remaining down work (the copy engine does not contend with the kernels -
            // measured in the LOCAL_DOWN campaign).  STRATA_MW_TP_WIRE16=1 halves the bytes further:
            // the down writes fp16 partials and the 3060-side reduce converts them back exactly.
            if (ev_d2h_done_ != nullptr) cudaStreamWaitEvent(streams_[0], ev_d2h_done_, 0);  // d_out_ reuse
            const int njobs = (int) jobs.size();
            const size_t row_b = tp_wire16_ ? (size_t) H * 2 : (size_t) H * 4;
            int chunk = (njobs + 3) / 4;
            if (chunk < 1) chunk = 1;
            for (int j0 = 0, c = 0; j0 < njobs; j0 += chunk, ++c) {
                const int nj = std::min(chunk, njobs - j0);
                if (tp_wire16_)
                    s2_expert_maxwell_down_batch_f16_async(d_blobs_ + g.base, d_jobs_ + (size_t) j0 * 3, nj,
                                                           d_faq_ + (size_t) j0 * FF, d_fascale_ + (size_t) j0 * (FF / 32),
                                                           d_fahx_ + (size_t) j0 * (FF / 32), (uint16_t*) d_out_,
                                                           g.blob_bytes, g.d_codes_off, g.d_scales_off, streams_[0],
                                                           g.tp_blk_base, g.tp_blks, g.d_row_stride, g.d_scale_stride);
                else
                    s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_ + (size_t) j0 * 3, nj,
                                                       d_faq_ + (size_t) j0 * FF, d_fascale_ + (size_t) j0 * (FF / 32),
                                                       d_fahx_ + (size_t) j0 * (FF / 32), d_out_,
                                                       g.blob_bytes, g.d_codes_off, g.d_scales_off, streams_[0],
                                                       nullptr, 0, nullptr,
                                                       g.tp_blk_base, g.tp_blks, g.d_row_stride, g.d_scale_stride);
                cudaEventRecord(ev_down_[c & 3], streams_[0]);
                cudaStreamWaitEvent(streams_[1], ev_down_[c & 3], 0);
                cudaMemcpyAsync((uint8_t*) w_dst_ + (size_t) j0 * row_b,
                                (const uint8_t*) d_out_ + (size_t) j0 * row_b,
                                (size_t) nj * row_b, cudaMemcpyDeviceToHost, streams_[1]);
            }
            if (e1) {
                mw_tp_raise_flag_async(comp_flag_, ring, streams_[1]);
                cudaEventRecord(ev_d2h_done_, streams_[1]);
                comp_seq_ = ring;
            }
        } else {
            s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_faq_,
                                               d_fascale_, d_fahx_, dst_rows, g.blob_bytes, g.d_codes_off,
                                               g.d_scales_off, streams_[0],
                                               e1 ? comp_flag_ : nullptr, ring, e1 ? d_ctr_ : nullptr,
                                               g.tp_blk_base, g.tp_blks, g.d_row_stride, g.d_scale_stride);
        }
        if (e1 && tp_cards_ > 1 && !dlt) { /* the ring advanced with the raise above */ }
        else if (e1) comp_seq_ = ring;
        down_launched_ = true;
    }
    for (size_t j = 0; j < jobs.size(); ++j) {
        owned_row_.push_back(jobs[j].row);
        computed_.fetch_add(1, std::memory_order_relaxed);
    }
    if (std::getenv("STRATA_MAXWELL_TIMING") != nullptr) {
        const auto bt2 = std::chrono::steady_clock::now();
        static thread_local double a_set = 0, a_all = 0;
        static thread_local long n = 0;
        a_set += std::chrono::duration<double, std::milli>(bt1 - bt0).count();
        a_all += std::chrono::duration<double, std::milli>(bt2 - bt0).count();
        if (++n % 2000 == 0)
            std::fprintf(stderr, "Maxwell TIMING begin  dev=%d n=%ld setdev=%.1f us total=%.1f us\n",
                         device_, n, 1000.0 * a_set / (double) n, 1000.0 * a_all / (double) n);
    }
    return true;
}

bool MaxwellExperts::begin(int64_t layer, const strata::kernels::cpu::ActQ* acts, const int32_t* ids,
                           int64_t n_tok, int64_t k, const int32_t* /*kind*/, float* dst_mapped,
                           bool rows_f16, std::string& err) {
    (void) err;
    if (device_ < 0) return true;
    if (n_tok > n_tok_max_ || n_tok * k > cap_entries_) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): window too large (%lld tok x %lld)",
                      device_, (long long) n_tok, (long long) k);
        err = buf;
        return false;
    }
    w_layer_ = layer; w_acts_ = acts; w_ids_ = ids; w_n_tok_ = n_tok; w_k_ = k;
    w_dst_ = dst_mapped;
    w_persisted_ = false;   // set by begin_chain's persist branch; never carried across services
    // STRATA_MW_LOCAL_DOWN (opt-in experiment, S0): route the down stage through card-local d_out_ +
    // one pinned D2H + a host row scatter, instead of the down kernel's writes into the mapped host
    // rows.  The S0 events show the mapped-write downs slow down in card-visit order (83 -> 240 us/L
    // for the same work) whenever they overlap the other cards' downs - a shared-endpoint effect on
    // this VM (the P3.4 note saw the same serialization on concurrent syncs), not bandwidth (the f16
    // rows arm halved the bytes and moved nothing).  The rows land in the same buffer with the same
    // values, only later and by the host's hand - bitwise-neutral by construction.
    static const bool local_down = [] {
        const char* e = std::getenv("STRATA_MW_LOCAL_DOWN");
        const bool on = e != nullptr && std::atoi(e) != 0;
        if (on) std::fprintf(stderr, "strata maxwell: down via card-local staging + D2H (STRATA_MW_LOCAL_DOWN)\n");
        return on;
    }();
    w_dst_mapped_ = dst_mapped != nullptr && !local_down;
    w_rows_f16_ = rows_f16 && w_dst_mapped_;   // fp16 applies to the mapped token path only
    if (w_rows_f16_ && tp_cards_ > 1) {
        // SPLIT_MODE_TENSOR: the fp16 row form has no tensor-split variant (yet) - the partials ride the
        // plain fp32 slots.  One note, then silently plain.
        static const bool said = [] {
            std::fprintf(stderr, "strata maxwell: tensor split ignores STRATA_MW_ROWS_F16 (partials stay fp32)\n");
            return true;
        }();
        (void) said;
        w_rows_f16_ = false;
    }
    w_row_of_ = nullptr;
    w_row_base_ = 0;
    w_n_embd_ = H;
    w_accumulate_ = false;
    return begin_chain(err);
}

void MaxwellExperts::s0_report() {
    const double mb_l = s0_mb_ / s0_layers_, ideal_us = mb_l * 12.05;   // MB/(83 GB/s) in us
    const double krn_us = 1000.0 * s0_krn_ms_ / s0_layers_;
    std::fprintf(stderr, "MWPROF m10 CUDA%d layers %.0f: jobs/L %.2f distinct/L %.2f, %.3f MB/L, "
                         "ideal %.0f us/L, kernel %.0f us/L (slack %.2fx) [gup %.0f + down %.0f], "
                         "sync-wait %.0f us/L\n",
                 device_, s0_layers_, s0_jobs_ / s0_layers_, s0_distinct_ / s0_layers_, mb_l,
                 ideal_us, krn_us, ideal_us > 0 ? krn_us / ideal_us : 0.0,
                 1000.0 * s0_gup_ms_ / s0_layers_, 1000.0 * s0_dn_ms_ / s0_layers_,
                 1000.0 * s0_sync_ms_ / s0_layers_);
    std::fflush(stderr);
    s0_layers_ = s0_jobs_ = s0_distinct_ = s0_mb_ = s0_krn_ms_ = s0_sync_ms_ = 0;
    s0_gup_ms_ = s0_dn_ms_ = 0;
}

bool MaxwellExperts::finish(float* out, std::string& err) {
    (void) out;   // the destination was handed to begin(); only the wait and the scatter-less finish remain
    return finish_chain(err);
}

bool MaxwellExperts::finish_to(float* dst_rows, const int32_t* row_of_entry, int32_t row_base, int64_t n_embd,
                               std::string& err, bool accumulate) {
    if (device_ < 0) return true;
    persist_halt();   // Maxwell has no preemption: the prefill's launches need the whole card
    w_dst_ = dst_rows; w_row_of_ = row_of_entry; w_row_base_ = row_base; w_n_embd_ = n_embd;
    w_dst_mapped_ = false;
    w_accumulate_ = accumulate;
    return finish_chain(err);   // the prefill's down stage launches here, where its destination is known
}

bool MaxwellExperts::finish_chain(std::string& err) {
    // STRATA_MW_PERSIST first: the service (empty ones included - they rode the doorbell too) is
    // already complete (or completing) inside the resident kernel; the legacy empty-raise below
    // must not also fire for it (two writers could regress the monotonic flag word).
    if (w_persisted_) {
        w_persisted_ = false;
        if (comp_flag_ != nullptr && gpuflag_mode() < 2) {
            const uint32_t want = comp_seq_;
            while (__atomic_load_n(comp_flag_, __ATOMIC_ACQUIRE) < want) { /* spin: the kernel is finishing */ }
        }
        return true;
    }
    // E1: a decode service this tier sat out still advances the completion sequence - the verifier's
    // expected value counts every service, so the flag must rise here (a mapped write, one fence).
    if (owned_row_.empty()) {
        if (comp_flag_ != nullptr && w_dst_mapped_ && (gpuflag_mode() >= 1 || persist_armed_)) {
            __atomic_store_n(comp_flag_, comp_seq_ + 1, __ATOMIC_SEQ_CST);
            comp_seq_ += 1;
        }
        return true;
    }
    const auto ft0 = std::chrono::steady_clock::now();
    // THIS device, explicitly: without the scope the sync below lands on whatever device the caller had
    // current (the primary GPU, whose graph is waiting for this very host step: a deadlock).
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.err; return false; }
    const auto ft1 = std::chrono::steady_clock::now();
    // The down stage: launched at begin() for the token path (see there).  For the prefill it launches NOW,
    // into the staging block, and the host copies the rows out below.  One wait per device per layer.
    if (!down_launched_) {
        for (size_t j = 0; j < owned_row_.size(); ++j) h_jobs_[j * 3 + 2] = (int32_t) j;
        if (!zerocopy_ &&
            !check(cudaMemcpyAsync(d_jobs_, h_jobs_, owned_row_.size() * 3 * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams_[0]), "copy jobs dst", err)) return false;
        const LayerGeo& g = geo_of(w_layer_);
        s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, (int) owned_row_.size(), d_faq_,
                                           d_fascale_, d_fahx_, d_out_, g.blob_bytes, g.d_codes_off,
                                           g.d_scales_off, streams_[0], nullptr, 0, nullptr,
                                           g.tp_blk_base, g.tp_blks, g.d_row_stride, g.d_scale_stride);
        down_launched_ = true;
    }
    // S0 (STRATA_MW_LAYERPROF): the span ends after the down launch, whichever path launched it; the
    // elapsed is harvested past the sync, where the events have settled.  s0_sync_ accumulates how long
    // the host stood AT the sync - what the chain pays beyond the card's own kernel occupancy.
    const bool s0 = s0_env() && s0_a_ != nullptr;
    const auto s0_w0 = std::chrono::steady_clock::now();
    if (s0) cudaEventRecord(s0_b_, streams_[0]);
    // E1 mode 2: the decode path does not wait - the down kernel's last block raises the tier flag,
    // the 3060's graph multi-waits on it, and the NEXT service's staging uses the other buffer set.
    // The prefill (non-mapped destination) still syncs: its rows are read on the host.
    if (w_dst_mapped_ && comp_flag_ != nullptr && !w_rows_f16_ && gpuflag_mode() >= 2) {
        s0_deferred_ = s0;          // the span is reaped at the next begin (the stream orders it)
        w_dst_mapped_ = false;      // the service is complete; the scatter was in-kernel
        owned_row_.clear();
        return true;
    }
    // STRATA_MW_GENTLE_SYNC (opt-in): poll this card's stream with short host sleeps instead of
    // cudaStreamSynchronize's driver spin.  The S0/REV campaign proved the spin is poison on this VM:
    // whichever card is served later runs its down kernel slower (~+55 us per earlier finish, the
    // flip test STRATA_MW_REV moved the fast card with the order) - the spinning finishes clog the
    // completion path the other cards' kernels need.  A sleeping poll does not spin; the wake lands
    // within one quantum (30 us) of the true end.  (cudaDeviceScheduleBlockingSync - STRATA_MW_BLOCKSYNC -
    // is the flag-level form of the same idea; this stack's prefill then sees 0 MiB free, so it stays
    // off and the poll form is the one that runs.)
    static const bool gentle = [] {
        const char* e = std::getenv("STRATA_MW_GENTLE_SYNC");
        const bool on = e != nullptr && std::atoi(e) != 0;
        if (on) std::fprintf(stderr, "strata maxwell: gentle stream sync - 30 us polls, no driver spin "
                                     "(STRATA_MW_GENTLE_SYNC)\n");
        return on;
    }();
    if (gentle) {
        for (;;) {
            const cudaError_t q = cudaStreamQuery(streams_[0]);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) { err = "finish query: " + std::string(cudaGetErrorString(q)); return false; }
            std::this_thread::sleep_for(std::chrono::microseconds(30));
        }
    } else if (!check(cudaStreamSynchronize(streams_[0]), "finish sync", err)) return false;
    const auto ft2 = std::chrono::steady_clock::now();
    // S0: the verdict line.  ideal = MB/layer over 83 GB/s (the per-card DRAM floor, M10-TEST-FINAL);
    // slack = kernel/ideal - 1.0x is the bandwidth floor, anything above is recoverable elsewhere.
    if (s0) {
        float el = 0.0f, eg = 0.0f, ed = 0.0f;
        cudaEventElapsedTime(&el, s0_a_, s0_b_);
        cudaEventElapsedTime(&eg, s0_a_, s0_m_);
        cudaEventElapsedTime(&ed, s0_m_, s0_b_);
        s0_krn_ms_ += (double) el;
        s0_gup_ms_ += (double) eg;
        s0_dn_ms_ += (double) ed;
        s0_sync_ms_ += std::chrono::duration<double, std::milli>(ft2 - s0_w0).count();
        if (s0_layers_ >= 1440.0) s0_report();
        s0_a_ = nullptr;   // the next begin_chain arms a fresh span (the event objects are reused)
    }
    if (const char* e = std::getenv("STRATA_MAXWELL_TIMING")) {
        static thread_local double a_set = 0, a_sync = 0;
        static thread_local long n = 0;
        a_set += std::chrono::duration<double, std::milli>(ft1 - ft0).count();
        a_sync += std::chrono::duration<double, std::milli>(ft2 - ft1).count();
        if (++n % 2000 == 0)
            std::fprintf(stderr, "Maxwell TIMING finish dev=%d n=%ld setdev=%.1f us sync=%.1f us\n",
                         device_, n, 1000.0 * a_set / (double) n, 1000.0 * a_sync / (double) n);
    }
    if (w_dst_mapped_) { owned_row_.clear(); return true; }
    if (!check(cudaMemcpy(h_out_, d_out_, owned_row_.size() * (size_t) w_n_embd_ * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy results batch", err)) return false;
    for (size_t s = 0; s < owned_row_.size(); ++s) {
        const int32_t i = owned_row_[s];
        const int32_t row = w_row_of_ ? w_row_of_[i] - w_row_base_ : i;
        float* dst = w_dst_ + (size_t) row * (size_t) w_n_embd_;
        const float* src = h_out_ + s * (size_t) w_n_embd_;
        if (w_accumulate_)   // SPLIT_MODE_TENSOR: this card's partial ADDED in rank order (deterministic)
            for (int64_t e = 0; e < w_n_embd_; ++e) dst[e] += src[e];
        else
            std::memcpy(dst, src, (size_t) w_n_embd_ * sizeof(float));
    }
    return true;
}

}  // namespace strata::core
