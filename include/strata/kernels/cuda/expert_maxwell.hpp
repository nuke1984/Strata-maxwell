// include/strata/kernels/cuda/expert_maxwell.hpp - Maxwell (sm_50) proof-of-concept for the S2 expert kernel.
//
// WHAT THIS IS: a port, to Maxwell hardware (cc 5.0, e.g. a Tesla M10), of the S2 expert computation the CPU
// does today (strata::kernels::cpu::s2_expert_scalar, src/kernels/cpu/expert.cpp:629).  It reads the SAME
// planar FP32-scale S2 blob layout the CPU reads (see include/strata/kernels/cpu/expert.hpp), with the same
// integer bias (code - 1) and the same FP32 evaluation contract.  It is NOT a copy of llama.cpp's MMVQ:
// llama.cpp's kernel navigates the GGUF `block_q2_0` format (interleaved, fp16 scale); this navigates
// Strata's pack format (planar, fp32 scale).
//
// WHY A SEPARATE FILE: the engine's normal CUDA kernels (src/kernels/cuda/*.cu) are compiled for sm_75+ and
// freely use features Maxwell lacks (__syncwarp/_sync intrinsics, mma/tensor cores, cp.async, __half math).
// Maxwell is a deliberately constrained target, so its kernel lives apart and is compiled for sm_50 only.
// The goal of this proof is CORRECTNESS on the format, not speed: the layout is scalar (one thread per
// output row, no shared memory) so a parity failure can only be a decode bug, never a race.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

namespace strata::kernels::cuda {

/// Byte geometry of the blob (exported so the test can build a valid blob without duplicating constants).
namespace expert_maxwell_geo {
    inline constexpr int H = 2560;
    inline constexpr int FF = 640;
    inline constexpr int QK = 64;
    inline constexpr int ROW_GU = H * 2 / 8;
    inline constexpr int ROW_D = FF * 2 / 8;
    inline constexpr int SC_GU = H / QK;
    inline constexpr int SC_D = FF / QK;
    // one expert = gate + up + down rows.  Each 64-weight block is BB=18 bytes (2-byte fp16 scale + 16
    // code bytes), exactly as in include/strata/kernels/cpu/expert.hpp.  Total = 1,382,400 bytes.
    inline constexpr size_t BLOB = (2ull * FF * (H / QK) + (size_t) H * (FF / QK)) * 18;
    inline constexpr size_t O_GU_CODES = 0;
    inline constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
    inline constexpr size_t O_GU_SCALES = O_D_CODES + 1ull * H * ROW_D;
    // scales are fp16 (2 bytes each), exactly as in the CPU header (SC_GU * 2, SC_D * 2)
    inline constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
    // one token's packed ActQ for the batch kernels: [q H][scale (H/32)*4][hx (H/32)*4] contiguous, so the
    // per-layer upload copies only n_tok * ACTS_STRIDE bytes instead of the whole 64-token window.
    inline constexpr int ACTS_STRIDE = H + 2 * (H / 32) * 4;
}

/// One S2 expert on a Maxwell device: `out(H) = down( swiglu( gate(x), up(x) ) )`, with the EXACT
/// integer-dot + hx-correction contract of the CPU pool (row_dot in src/kernels/cpu/expert.cpp).
///
/// `d_blob` : one 1,382,400-byte expert in the pack's planar layout (device pointer).
/// `d_aq/d_ascale/d_ahx` : the quantised activation (int8 codes, fp32 scales, hx per 32-elem chunk).
/// `d_out`  : device buffer with room for H + FF floats (first H = output, next FF = SwiGLU scratch).
/// Synchronises before returning.
void s2_expert_maxwell(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                       const float* d_ahx, float* d_out);

/// Stage-split API: each stage takes the quantised activation of its own width (H for gate/up,
/// FF for down).  A parity test feeds both sides identical ActQ and a mismatch can only be a decode bug.
void s2_expert_maxwell_gup(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                           const float* d_ahx, float* d_ff);
void s2_expert_maxwell_down(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                            const float* d_ahx, float* d_out);
/// P3.1 async launches without sync (same kernels, caller synchronises once per batch).  `stream` 0 = the
/// legacy default stream; pass a non-default stream to overlap several jobs on one device.
void s2_expert_maxwell_gup_async(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                                 const float* d_ahx, float* d_ff, cudaStream_t stream = 0);
void s2_expert_maxwell_down_async(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                                  const float* d_ahx, float* d_out, cudaStream_t stream = 0);
/// Intermediate quantisation ON THE DEVICE (no host round trip): FF-wide SwiGLU floats -> int8 codes +
/// fp32 scale + hx, bit-exact with the CPU's act_quant_q8_1_avx2 for finite inputs.  `nchunks` is the total
/// number of 32-element chunks (entries * FF/32); d_ff/d_q are contiguous per entry.  Async, no sync.
void s2_expert_maxwell_quant_ff_async(const float* d_ff, int8_t* d_q, float* d_scale, float* d_hx,
                                      int nchunks, cudaStream_t stream = 0);
/// Batched stages: ONE launch for the whole layer's jobs.  `d_jobs` is njobs pairs of (blob slot, token
/// index) in device memory; `acts_base` is the packed per-token ActQ ([q H][scale][hx] at ACTS_STRIDE each),
/// and the per-job buffers are contiguous (d_ff + j*FF, d_out + j*H, ...).
void s2_expert_maxwell_gup_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                       const uint8_t* acts_base, float* d_ff, cudaStream_t stream = 0);
/// v2 ONLY: gate+up AND the FF-wide quantisation in ONE launch.  The v2 kernel's block covers exactly one
/// 32-value quant chunk of its job (FF/32 = 20 chunks per job, PER_BLOCK = 32), so the SwiGLU output is
/// quantised in the epilogue with the SAME shuffle tree as s2_expert_maxwell_quant_ff_async - bit-identical,
/// and the caller drops the separate quant_ff launch (measured: ~4 us of API per device per layer in this VM
/// plus the quant kernel off the chain).  Callers use this when s2_expert_maxwell_v2() is true and fall back
/// to gup_batch + quant_ff otherwise.
void s2_expert_maxwell_gup_quant_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                             const uint8_t* acts_base, int8_t* d_faq, float* d_fasc,
                                             float* d_fahx, cudaStream_t stream = 0);
/// A native pack's i-quant layer's gate/up + the SAME fused FF quantisation, for the blobs the tier
/// converts at open() (see MaxwellExperts): the gate/up rows keep their RAW GGUF block layouts at
/// `o_gate`/`o_up`, `gu_row` bytes per row, `blob_stride` bytes per blob (a native pack's blob size is
/// per layer).  `acts_base` is the same packed ActQ the S2 call reads: ActQ.q is block_q8_1.qs bit for
/// bit and the kernel rounds each chunk's scale to the d the device q8_1 quantizer stores, so the
/// numbers match the 3060's native expert path.  `gu_type`: 22 IQ2_S, 16 IQ2_XXS, 29 IQ1_M; anything
/// else refuses.  The FF image (faq/fasc/fahx) is bit-identical to the S2 path's, so the down stage
/// consumes it unchanged.
void iq_expert_maxwell_gup_quant_batch_async(int gu_type, const uint8_t* d_blobs, size_t blob_stride,
                                             const int32_t* d_jobs, int njobs, const uint8_t* acts_base,
                                             size_t o_gate, size_t o_up, size_t gu_row,
                                             int8_t* d_faq, float* d_fasc, float* d_fahx,
                                             float* d_ff_dbg = nullptr, cudaStream_t stream = 0,
                                             int row_base = 0, int rows = expert_maxwell_geo::FF);
/// STRATA_MW_MT (the prefill arm, 2026-10-23): the multi-token gup - one RUN (an expert, up to 8
/// adjacent entries) per block pass; each (row,k) weight element is hoisted once (load + codebook +
/// sign work) and applied to every entry's act, the dp4a order and float tail verbatim the per-entry
/// vecdots, so every (expert, token) result is bit-identical.  `d_runs` = [nruns x 4] (slot, count,
/// entries_off, unused), `d_entries` = [total x 2] (tok, entry); the down still consumes the standard
/// job table, unchanged.
void iq_expert_maxwell_gup_quant_mt_async(int gu_type, const uint8_t* d_blobs, size_t blob_stride,
                                          const int32_t* d_runs, const int32_t* d_entries, int nruns,
                                          const uint8_t* acts_base, size_t o_gate, size_t o_up, size_t gu_row,
                                          int8_t* d_faq, float* d_fasc, float* d_fahx,
                                          float* d_ff_dbg = nullptr, cudaStream_t stream = 0,
                                          int row_base = 0, int rows = expert_maxwell_geo::FF);
/// SPLIT_MODE_TENSOR (the Meta tier): the same fused gup+quant kernel restricted to this card's SLICE of
/// the gate/up rows - `row_base`/`rows` select the slice (the blob stores exactly those rows, compact).
/// The dots, the SwiGLU and the quant tree run the same code per row, so each row's value is bit-identical
/// to the full-blob kernel's; the faq/fasc/fahx chunks keep the GLOBAL layout (chunk j*(FF/32)+row/32),
/// so the card's down stage and the 4-way partial sum need no remapping.  Default (0, FF) = the full range.
void s2_expert_maxwell_down_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                        const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                        float* d_out, size_t blob_stride = expert_maxwell_geo::BLOB,
                                        size_t d_codes_off = expert_maxwell_geo::O_D_CODES,
                                        size_t d_scales_off = expert_maxwell_geo::O_D_SCALES,
                                        cudaStream_t stream = 0,
                                        volatile uint32_t* mflag = nullptr, const uint32_t ring = 0,
                                        uint32_t* ctr = nullptr,
                                        int blk_base = 0, int blks = expert_maxwell_geo::SC_D,
                                        size_t d_row_stride = expert_maxwell_geo::ROW_D,
                                        size_t d_scale_stride = (size_t) expert_maxwell_geo::SC_D * 2);
/// v2 ONLY: the down stage with fp16 rows (half the uplink bytes when the destination is mapped host
/// memory; the verifier converts them back during its row copy).  `d_out16` holds H halves per entry.
void s2_expert_maxwell_down_batch_f16_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                            const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                            uint16_t* d_out16, size_t blob_stride = expert_maxwell_geo::BLOB,
                                            size_t d_codes_off = expert_maxwell_geo::O_D_CODES,
                                            size_t d_scales_off = expert_maxwell_geo::O_D_SCALES,
                                            cudaStream_t stream = 0,
                                            int blk_base = 0, int blks = expert_maxwell_geo::SC_D,
                                            size_t d_row_stride = expert_maxwell_geo::ROW_D,
                                            size_t d_scale_stride = (size_t) expert_maxwell_geo::SC_D * 2);
void s2_expert_maxwell_sync();

/// SPLIT_MODE_TENSOR delivery: raise the tier's mapped completion flag to `ring` on `stream`, ordered
/// after whatever the stream executed before (the D2H of the card's partial block).  The same E1
/// doorbell contract the down kernel's last block carries on the mapped-destination path.
void mw_tp_raise_flag_async(volatile uint32_t* mflag, uint32_t ring, cudaStream_t stream = 0);
/// Whether the v2 batch kernels are enabled (STRATA_S2_V2, default on): the caller picks the fused
/// gup_quant call when true, the v1 gup_batch + quant_ff pair otherwise.
bool s2_expert_maxwell_v2();

// ---- STRATA_MW_PERSIST: the self-service tier ------------------------------------------------
//
// One PERSISTENT kernel per card runs the decode services; the host's per-layer dispatch becomes
// pure pinned-memory writes (acts + job table + header into a 2-slot ring, then one doorbell store)
// - ZERO CUDA API calls per layer per card, so the visit-order "poison" (every host call during the
// down window stretches the cards served after it) cannot occur and the two per-layer launches
// disappear.  The kernel polls the mapped doorbell (block 0 only; the broadcast rides device
// memory so the uplink stays quiet), stages the service's acts+jobs from the mapped slot into
// device memory ONCE and in bulk (P7's lesson honoured: no mapped reads inside any dot), grid-
// barriers on a hand-rolled generation counter (the grid is sized to the kernel's own occupancy,
// so every block is resident and the barrier cannot deadlock - sm_50 has no cooperative launch),
// runs the SAME gup and down bodies the batched kernels run, and after a final system-wide fence
// block 0 raises the E1 tier completion flag exactly like the batched down's last block did.
struct PersistParams {          // device memory, written once before the launch
    const uint8_t* blobs;       ///< d_blobs_ (the layer region rides the per-service header)
    uint8_t* d_up[2];           ///< the two device staging sets (parity per service)
    int8_t* faq;                ///< the FF quant image (gup writes, down reads)
    float* fasc;
    float* fahx;
    volatile uint32_t* comp_flag;   ///< the verifier's mapped tier flag (E1 contract)
    volatile uint32_t* d_seq;       ///< DEVICE doorbell: the copy engine writes the service number
                                    ///< after the service's H2D (cuStreamWriteValue32); 0xFFFFFFFF = halt
    const volatile uint32_t* halt;  ///< MAPPED stop word (a host store - the ONLY way to reach a kernel
                                    ///< that owns its stream; checked periodically by the poll loop)
};
struct PersistCtl {             // device memory, zeroed before every launch
    volatile int service;       ///< block 0's broadcast of the newest doorbell value (-1 = stop)
    unsigned bar_count;         ///< the grid barrier's arrival count
    volatile unsigned bar_gen;  ///< ..and its generation
};
static_assert(sizeof(PersistCtl) % 4 == 0);
/// One service's header, the FIRST 128 bytes of the packed staging the H2D copies (so it lands in
/// device memory with the acts and the job table in ONE transfer).  Host-written, kernel-read.
struct PersistHdr {
    uint32_t njobs, n_tok, gu_type, ring;               ///< the E1 flag value this service raises
    int tp_row_base, tp_rows, tp_blk_base, tp_blks;     ///< non-TP: (0, FF, 0, SC_D)
    uint32_t pad0[4];
    size_t blob_base, blob_stride, o_gate, o_up, gu_row;
    size_t d_codes_off, d_scales_off, d_row_stride, d_scale_stride;
    float* dst;                                         ///< the service's mapped rows base
};
static_assert(sizeof(PersistHdr) == 128);
/// The device staging layout: [PersistHdr][acts n_tok*ACTS_STRIDE][job table njobs*12].
inline constexpr size_t PERSIST_HDR_BYTES = sizeof(PersistHdr);
inline constexpr uint32_t PERSIST_HALT = 0xFFFFFFFFu;   ///< the device doorbell's stop value

/// Launch the persistent kernel (grid = its own occupancy x SMs) on `stream`.  The ctl is zeroed
/// in-stream first.  Returns the block count (0 + err on failure).
int maxwell_persist_launch(const PersistParams* d_params, PersistCtl* d_ctl, cudaStream_t stream, std::string& err);

}  // namespace strata::kernels::cuda
