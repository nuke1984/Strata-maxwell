// src/kernels/cuda/expert_maxwell.cu - the Maxwell (sm_50) S2 expert kernel, proof of concept.
//
// See include/strata/kernels/cuda/expert_maxwell.hpp for the contract.  This file is compiled for sm_50 ONLY
// (see CMakeLists.txt): it must not reference any instruction Maxwell lacks.
//
// Maxwell constraints honoured here:
//   * no __syncwarp / *_sync warp intrinsics (cc >= 7.0) - we use ONLY legacy __shfl_down (cc >= 3.0)
//   * no mma / ldmatrix / tensor cores (cc >= 7.0) - all MACs are integer IMAD / vmad
//   * no dp4a (cc >= 6.1) - we use PTX vmad.s32.s32.s32 for the signed 4-byte dot (llama.cpp's m10-final.patch)
//   * no cp.async (cc >= 8.0) - plain global loads
//   * no __half arithmetic (cc < 5.3) - scales are FP32 in memory, decoded with plain float ops
#include "strata/kernels/cuda/expert_maxwell.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The i-quant block layouts, the codebook grids and the iq1m scale union, exactly as the engine's own
// i-quant kernels take them (src/kernels/cuda/iq_kernels.cu includes the same file the same way): the
// grids are __device__ tables defined by GGML_COMMON_IMPL_CUDA and land in this sm_50 translation unit.
// MIT license, third_party/ggml (see its LICENSE and VERSION.txt).
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "strata/core/maxwell_tp_plan.hpp"

namespace strata::kernels::cuda {
namespace {

using namespace expert_maxwell_geo;

// fp32 -> fp16, round-to-nearest-even.  Hand-rolled: Maxwell has no fp16 arithmetic (the CUDA intrinsic
// is a stub there, verified on the M10: it compiles but returns zero).  Denormals flush to zero (an
// absolute error below 6e-5, irrelevant for expert rows); inf/nan map to inf/nan.
__device__ __forceinline__ uint16_t f2h_rn(float x) {
    const uint32_t u = __float_as_uint(x);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t ex = (u >> 23) & 0xFFu;
    const uint32_t m = u & 0x7FFFFFu;
    if (ex == 0xFFu) return (uint16_t) (sign | 0x7C00u | (m ? 0x200u : 0u));
    const int32_t e = (int32_t) ex - 127 + 15;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// Maxwell has no native __dp4a.  PTX vmad.s32.s32.s32 does a signed 4-byte dot product with byte selectors -
// the same primitive llama.cpp's m10-final.patch added to ggml_cuda_dp4a for the Tesla M10.  On cc >= 6.1 we
// use the hardware instruction; elsewhere (host-side compile paths) a scalar fallback keeps the code valid.
__device__ __forceinline__ int dp4a_s32(const int a, const int b, const int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, c);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 500)
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
#else
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// Decode the 16 packed codes of one S2 block and expand to int8 with bias -1.
// Codes are packed 4 per byte LSB-first (element i at byte i/4, bits (i%4)*2).  One 16-byte block is 4 int32,
// each holding 4 2-bit codes.  Returns the codes as 4 int32 with one int8 code per byte, already biased.
__device__ __forceinline__ void unpack_s2_block(const uint8_t* codes, int w8[4]) {
    const int32_t* c32 = (const int32_t*) codes;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = c32[j];
        // spread 2-bit fields into byte lanes: byte k of the int32 = (q >> (k*2)) & 3, then subtract 1
        const int w = ((q >> 0) & 3) | (((q >> 2) & 3) << 8) | (((q >> 4) & 3) << 16) | (((q >> 6) & 3) << 24);
        w8[j] = w - 0x01010101;   // subtract 1 from each byte lane (the bias)
    }
}

// fp16 (IEEE 754 half) -> fp32.  Maxwell has no __half arithmetic, so scales stored as fp16 in the
// pack blob are widened with integer ops.  Same conversion as strata::fp16_to_fp32 on the host.
__device__ __forceinline__ float fp16_to_fp32_dev(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            m &= 0x3FF;
            f = sign | ((uint32_t) (e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return *(const float*) &f;
}

// Read one fp16 scale from the blob (2-byte stride, exactly as the CPU's h2f(scales + 2*b)).
__device__ __forceinline__ float read_scale(const uint8_t* scales, int b) {
    uint16_t h = (uint16_t) scales[b * 2] | ((uint16_t) scales[b * 2 + 1] << 8);
    return fp16_to_fp32_dev(h);
}

// The quantised activation as the GPU sees it: int8 codes + fp32 scales + hx correction per 32-elem
// chunk.  Mirrors strata::kernels::cpu::ActQ (q, scale, hx) but as separate device pointers so the kernel
// signature stays flat.  nchunks = width / 32.
struct ActQDev {
    const int8_t* q;      // nchunks*32 int8 codes
    const float* scale;   // nchunks fp32 scales
    const float* hx;      // nchunks hx correction (scale * sum(q))
    int nchunks;
};

// Integer dot of 8 S2 bytes (32 codes) against 32 int8 activations: sum(code * q), exact.
// 8 bytes = 2 int32; each int32 holds 16 2-bit codes (4 per byte).  NO bias (handled by hx).
// Uses dp4a_s32 (vmad on Maxwell) for the signed byte dots, exactly like llama.cpp's MMVQ.
__device__ __forceinline__ int dot32_s2(int c, const int8_t* q) {
    const int32_t* q32 = (const int32_t*) q;
    int sumi = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int shift = g * 8;   // 4 codes per group, 8 bits per group
        const int cc = (c >> shift) & 0xFFFF;
        // expand 4 2-bit codes to 4 bytes
        const int w = ((cc >> 0) & 3) | (((cc >> 2) & 3) << 8) | (((cc >> 4) & 3) << 16) | (((cc >> 6) & 3) << 24);
        sumi = dp4a_s32(q32[g], w, sumi);
    }
    return sumi;
}

// One 64-weight block (16 bytes of codes) = two 32-elem chunks: ONE int4 load instead of two int32 ones.
// 16 bytes per load halves the sectors the scattered row walk fetches (a 32-byte sector then serves 16
// useful bytes, not 4); the arithmetic is identical.
__device__ __forceinline__ void block_pair_s2(const uint8_t* codes, const int8_t* q, int& dot0, int& dot1) {
    const int4 c4 = *(const int4*) codes;
    dot0 = dot32_s2(c4.x, q) + dot32_s2(c4.y, q + 16);
    dot1 = dot32_s2(c4.z, q + 32) + dot32_s2(c4.w, q + 48);
}

// One row dot, exactly as row_dot in src/kernels/cpu/expert.cpp:233:
//   sum_b [ d_w * d_x * sum(code*q) ] - sum_b [ d_w * hx ]
// The first sum is over 64-weight blocks (each spanning two 32-elem activation chunks); the correction
// uses the precomputed hx per chunk.  All integer partial sums are exact; FP32 rounding matches the
// scalar CPU path (one FMA per chunk, scalar correction accumulated separately).
__device__ __forceinline__ float row_dot_s2(const uint8_t* codes, const uint8_t* scales, const ActQDev& a,
                                           int nblocks) {
    float acc = 0.f;
    float corr = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = read_scale(scales, b);
        // block b covers activation chunks 2b and 2b+1 (64 weights = two 32-elem chunks), ONE int4 load
        int dot0 = 0, dot1 = 0;
        block_pair_s2(codes + b * 16, a.q + (2 * b) * 32, dot0, dot1);
        acc += d * a.scale[2 * b] * (float) dot0;
        acc += d * a.scale[2 * b + 1] * (float) dot1;
        corr += d * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    return acc - corr;
}

// STAGE 1: gate + up -> SwiGLU, one thread per FF row.  Uses row_dot_s2 (integer exact + hx),
// exactly as the CPU pool does.  Writes the intermediate to `ff` (a device buffer).
__global__ void expert_s2_stage1(const uint8_t* __restrict__ blob, const int8_t* __restrict__ aq,
                                 const float* __restrict__ ascale, const float* __restrict__ ahx,
                                 float* __restrict__ ff) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= FF) return;
    ActQDev a{aq, ascale, ahx, H / 32};
    const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
    const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
    const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
    const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
    const float sg = row_dot_s2(gc, gs, a, SC_GU);
    const float su = row_dot_s2(uc, us, a, SC_GU);
    ff[r] = (sg / (1.f + __expf(-sg))) * su;   // SwiGLU
}

// STAGE 1, latency-hiding variant: SPLIT threads cooperate on one FF row.  The one-thread-per-row version
// runs 640 threads = ONE wave on the M10's five SMs, so its serial 40-block walk is exposed global-load
// latency (measured 0.14 ms/expert); splitting the blocks across 4 threads gives 2560 threads (20 blocks)
// and cuts it several-fold.  The partial sums are combined in shared memory: FP32 addition order differs
// from the serial CPU reference only in the last bits (tree reduction), like any parallel reduction.
template <int SPLIT>
__global__ void expert_s2_stage1_mt(const uint8_t* __restrict__ blob, const int8_t* __restrict__ aq,
                                    const float* __restrict__ ascale, const float* __restrict__ ahx,
                                    float* __restrict__ ff) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int row = blockIdx.x * PER_BLOCK + (int) (threadIdx.x / SPLIT);
    const int part = (int) (threadIdx.x % SPLIT);
    if (row >= FF) return;
    ActQDev a{aq, ascale, ahx, H / 32};
    const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * row) * ROW_GU;
    const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * row) * SC_GU * 2;
    const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * row + 1) * ROW_GU;
    const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * row + 1) * SC_GU * 2;
    float accg = 0.f, corg = 0.f, accu = 0.f, coru = 0.f;
    for (int b = part; b < SC_GU; b += SPLIT) {
        const float dg = read_scale(gs, b);
        int d0 = 0, d1 = 0;
        block_pair_s2(gc + b * 16, a.q + (2 * b) * 32, d0, d1);
        accg += dg * a.scale[2 * b] * (float) d0;
        accg += dg * a.scale[2 * b + 1] * (float) d1;
        corg += dg * (a.hx[2 * b] + a.hx[2 * b + 1]);
        const float du = read_scale(us, b);
        int e0 = 0, e1 = 0;
        block_pair_s2(uc + b * 16, a.q + (2 * b) * 32, e0, e1);
        accu += du * a.scale[2 * b] * (float) e0;
        accu += du * a.scale[2 * b + 1] * (float) e1;
        coru += du * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    __shared__ float4 sh[128];
    sh[threadIdx.x] = make_float4(accg, corg, accu, coru);
    __syncthreads();
    if (part == 0) {
        float ag = 0.f, cg = 0.f, au = 0.f, cu = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float4 v = sh[threadIdx.x + s];
            ag += v.x; cg += v.y; au += v.z; cu += v.w;
        }
        const float sg = ag - cg;
        const float su = au - cu;
        ff[row] = (sg / (1.f + __expf(-sg))) * su;
    }
}

// STAGE 2: down -> out, one thread per H row.  Same exact computation.
__global__ void expert_s2_stage2(const uint8_t* __restrict__ blob, const int8_t* __restrict__ aq,
                                 const float* __restrict__ ascale, const float* __restrict__ ahx,
                                 float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= H) return;
    ActQDev a{aq, ascale, ahx, FF / 32};
    const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
    const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
    out[r] = row_dot_s2(dc, ds, a, SC_D);
}

// Batched job descriptor: one expert of one token.  The host uploads an array of these, and the batch
// kernels below process ALL of a layer's jobs in ONE launch per stage, so the per-job launch overhead
// disappears and the whole layer's work fills the SMs.
struct MaxJob { int32_t slot; int32_t tok; int32_t dst; };

// Batched stage1: one launch for every job of the layer; grid covers njobs*FF rows, SPLIT threads per row.
template <int SPLIT>
__global__ void expert_s2_stage1_batch(const uint8_t* __restrict__ blobs, const MaxJob* __restrict__ jobs,
                                       int njobs, const uint8_t* __restrict__ acts_base,
                                       float* __restrict__ ff) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int idx = blockIdx.x * PER_BLOCK + (int) (threadIdx.x / SPLIT);
    if (idx >= njobs * FF) return;
    const int j = idx / FF;
    const int row = idx - j * FF;
    const int part = (int) (threadIdx.x % SPLIT);
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * BLOB;
    const uint8_t* at = acts_base + (size_t) jd.tok * ACTS_STRIDE;   // [q H][scale][hx]
    ActQDev a{(const int8_t*) at, (const float*) (at + H), (const float*) (at + H + (H / 32) * 4), H / 32};
    const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * row) * ROW_GU;
    const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * row) * SC_GU * 2;
    const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * row + 1) * ROW_GU;
    const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * row + 1) * SC_GU * 2;
    float accg = 0.f, corg = 0.f, accu = 0.f, coru = 0.f;
    for (int b = part; b < SC_GU; b += SPLIT) {
        const float dg = read_scale(gs, b);
        int d0 = 0, d1 = 0;
        block_pair_s2(gc + b * 16, a.q + (2 * b) * 32, d0, d1);
        accg += dg * a.scale[2 * b] * (float) d0;
        accg += dg * a.scale[2 * b + 1] * (float) d1;
        corg += dg * (a.hx[2 * b] + a.hx[2 * b + 1]);
        const float du = read_scale(us, b);
        int e0 = 0, e1 = 0;
        block_pair_s2(uc + b * 16, a.q + (2 * b) * 32, e0, e1);
        accu += du * a.scale[2 * b] * (float) e0;
        accu += du * a.scale[2 * b + 1] * (float) e1;
        coru += du * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    __shared__ float4 sh[128];
    sh[threadIdx.x] = make_float4(accg, corg, accu, coru);
    __syncthreads();
    if (part == 0) {
        float ag = 0.f, cg = 0.f, au = 0.f, cu = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float4 v = sh[threadIdx.x + s];
            ag += v.x; cg += v.y; au += v.z; cu += v.w;
        }
        const float sg = ag - cg;
        const float su = au - cu;
        ff[(size_t) j * FF + row] = (sg / (1.f + __expf(-sg))) * su;
    }
}

// Batched stage2: one launch for every job of the layer; grid covers njobs*H rows, SPLIT threads per row.
// `blob_stride`/`d_codes_off`/`d_scales_off` place the blob and its down planes: the S2 (canonical) blob
// passes the geo constants, a converted native blob its own per-layer offsets (the launchers default them).
template <int SPLIT>
__global__ void expert_s2_stage2_batch(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                       size_t d_codes_off, size_t d_scales_off,
                                       const MaxJob* __restrict__ jobs,
                                       int njobs, const int8_t* __restrict__ faq,
                                       const float* __restrict__ fasc, const float* __restrict__ fahx,
                                       float* __restrict__ out) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int idx = blockIdx.x * PER_BLOCK + (int) (threadIdx.x / SPLIT);
    if (idx >= njobs * H) return;
    const int j = idx / H;
    const int row = idx - j * H;
    const int part = (int) (threadIdx.x % SPLIT);
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * blob_stride;
    ActQDev a{faq + (size_t) j * FF, fasc + (size_t) j * (FF / 32), fahx + (size_t) j * (FF / 32), FF / 32};
    const uint8_t* dc = blob + d_codes_off + (size_t) row * ROW_D;
    const uint8_t* ds = blob + d_scales_off + (size_t) row * SC_D * 2;
    float acc = 0.f, corr = 0.f;
    for (int b = part; b < SC_D; b += SPLIT) {
        const float d = read_scale(ds, b);
        int d0 = 0, d1 = 0;
        block_pair_s2(dc + b * 16, a.q + (2 * b) * 32, d0, d1);
        acc += d * a.scale[2 * b] * (float) d0;
        acc += d * a.scale[2 * b + 1] * (float) d1;
        corr += d * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    __shared__ float2 sh[128];
    sh[threadIdx.x] = make_float2(acc, corr);
    __syncthreads();
    if (part == 0) {
        float a0 = 0.f, c0 = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float2 v = sh[threadIdx.x + s];
            a0 += v.x; c0 += v.y;
        }
        out[(size_t) jd.dst * H + row] = a0 - c0;
    }
}

// ---------------------------------------------------------------------------------------------------------
// v2 (misurato): the v1 batch kernels issue ~640 SCALAR global act loads per row (16 int32 per 64-weight
// block, see dot32_s2) and chain the four vmad of each dot serially; the SASS shows the result: 810
// instructions with ~1000 unpack ops, 56 regs, DEPBAR stalls, ~34% of the measured VRAM bandwidth.
// v2 fixes exactly that:
//   * the block's rows all belong to ONE job (FF=640 is a multiple of PER_BLOCK), so the token's ActQ is
//     staged ONCE per block in shared memory and read with 128-bit LDS (no global latency in the loop);
//   * the integer dot uses FOUR independent vmad chains: integer adds are exact, so the result is
//     BIT-IDENTICAL to v1 (only the reduction tree of the integer sum changes);
//   * the weight codes stay 128-bit global loads (already vectorized).
// Select with STRATA_S2_V2=1 (default off while measuring).
__device__ __forceinline__ int expand4_s2(int b8) {
    return ((b8 >> 0) & 3) | (((b8 >> 2) & 3) << 8) | (((b8 >> 4) & 3) << 16) | (((b8 >> 6) & 3) << 24);
}

__device__ __forceinline__ int dot16_s2_ilp(int c, const int4 q) {
    const int s0 = dp4a_s32(q.x, expand4_s2(c & 0xFF), 0);
    const int s1 = dp4a_s32(q.y, expand4_s2((c >> 8) & 0xFF), 0);
    const int s2 = dp4a_s32(q.z, expand4_s2((c >> 16) & 0xFF), 0);
    const int s3 = dp4a_s32(q.w, expand4_s2((c >> 24) & 0xFF), 0);
    return (s0 + s1) + (s2 + s3);
}

// v2b: same dot, but the 2-bit -> byte-lane expansion comes from a 256-entry table in shared memory
// (one LDS instead of ~6 ALU ops per 4 codes).  The table is built once per block (256 words, ~0.07
// ops/weight amortised) and the entries are the same values expand4_s2 computes.
__device__ __forceinline__ int dot16_s2_lut(int c, const int4 q, const uint32_t* __restrict__ lut) {
    const int s0 = dp4a_s32(q.x, (int) lut[c & 0xFF], 0);
    const int s1 = dp4a_s32(q.y, (int) lut[(c >> 8) & 0xFF], 0);
    const int s2 = dp4a_s32(q.z, (int) lut[(c >> 16) & 0xFF], 0);
    const int s3 = dp4a_s32(q.w, (int) lut[(c >> 24) & 0xFF], 0);
    return (s0 + s1) + (s2 + s3);
}

// same, with the code block ALREADY loaded (software pipelining)
__device__ __forceinline__ void block_pair_s2_pre(const int4 c4, const int8_t* __restrict__ sq,
                                                  const uint32_t* __restrict__ lut, int& dot0, int& dot1) {
    const int4 q0 = *(const int4*) (sq);
    const int4 q1 = *(const int4*) (sq + 16);
    const int4 q2 = *(const int4*) (sq + 32);
    const int4 q3 = *(const int4*) (sq + 48);
    dot0 = dot16_s2_lut(c4.x, q0, lut) + dot16_s2_lut(c4.y, q1, lut);
    dot1 = dot16_s2_lut(c4.z, q2, lut) + dot16_s2_lut(c4.w, q3, lut);
}

__device__ __forceinline__ void block_pair_s2_lut(const uint8_t* __restrict__ codes,
                                                  const int8_t* __restrict__ sq,
                                                  const uint32_t* __restrict__ lut, int& dot0, int& dot1) {
    const int4 c4 = *(const int4*) codes;
    block_pair_s2_pre(c4, sq, lut, dot0, dot1);
}

__device__ __forceinline__ void block_pair_s2_smem(const uint8_t* __restrict__ codes,
                                                   const int8_t* __restrict__ sq, int& dot0, int& dot1) {
    const int4 c4 = *(const int4*) codes;
    const int4 q0 = *(const int4*) (sq);
    const int4 q1 = *(const int4*) (sq + 16);
    const int4 q2 = *(const int4*) (sq + 32);
    const int4 q3 = *(const int4*) (sq + 48);
    dot0 = dot16_s2_ilp(c4.x, q0) + dot16_s2_ilp(c4.y, q1);
    dot1 = dot16_s2_ilp(c4.z, q2) + dot16_s2_ilp(c4.w, q3);
}

template <int SPLIT>
__global__ void expert_s2_stage1_batch_v2(const uint8_t* __restrict__ blobs, const MaxJob* __restrict__ jobs,
                                          int njobs, const uint8_t* __restrict__ acts_base,
                                          float* __restrict__ ff) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(FF % PER_BLOCK == 0, "v2 stages one job per block: FF must be a multiple of PER_BLOCK");
    const int row0 = blockIdx.x * PER_BLOCK;
    if (row0 >= njobs * FF) return;
    const int j = row0 / FF;                    // every row of this block is in job j (FF % PER_BLOCK == 0)
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * BLOB;
    const uint8_t* at = acts_base + (size_t) jd.tok * ACTS_STRIDE;
    __shared__ __align__(16) int8_t sq[H];
    __shared__ float ssc[H / 32];
    __shared__ float shx[H / 32];
    __shared__ uint32_t slut[256];
    {
        const int4* srcq = (const int4*) at;
        int4* dstq = (int4*) sq;
        for (int i = (int) threadIdx.x; i < H / 16; i += 128) dstq[i] = srcq[i];
        const float* ssrc = (const float*) (at + H);
        for (int i = (int) threadIdx.x; i < H / 32; i += 128) {
            ssc[i] = ssrc[i];
            shx[i] = ssrc[H / 32 + i];
        }
        for (int i = (int) threadIdx.x; i < 256; i += 128) slut[i] = (uint32_t) expand4_s2(i);
    }
    __syncthreads();
    const int idx = row0 + (int) (threadIdx.x / SPLIT);
    if (idx >= njobs * FF) return;
    const int row = idx - j * FF;
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * row) * ROW_GU;
    const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * row) * SC_GU * 2;
    const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * row + 1) * ROW_GU;
    const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * row + 1) * SC_GU * 2;
    float accg = 0.f, corg = 0.f, accu = 0.f, coru = 0.f;
    // software pipeline: the next iteration's code blocks are loaded BEFORE the current dot runs, so the
    // global latency overlaps the vmad work (the measured gap between the pure-load pattern and the kernel).
    int4 ng = make_int4(0,0,0,0), nu = make_int4(0,0,0,0);
    if (part < SC_GU) { ng = *(const int4*) (gc + part * 16); nu = *(const int4*) (uc + part * 16); }
    for (int b = part; b < SC_GU; b += SPLIT) {
        const int4 cg4 = ng, cu4 = nu;
        const int nb = b + SPLIT;
        if (nb < SC_GU) { ng = *(const int4*) (gc + nb * 16); nu = *(const int4*) (uc + nb * 16); }
        const float dg = read_scale(gs, b);
        int d0 = 0, d1 = 0;
        block_pair_s2_pre(cg4, sq + (2 * b) * 32, slut, d0, d1);
        accg += dg * ssc[2 * b] * (float) d0;
        accg += dg * ssc[2 * b + 1] * (float) d1;
        corg += dg * (shx[2 * b] + shx[2 * b + 1]);
        const float du = read_scale(us, b);
        int e0 = 0, e1 = 0;
        block_pair_s2_pre(cu4, sq + (2 * b) * 32, slut, e0, e1);
        accu += du * ssc[2 * b] * (float) e0;
        accu += du * ssc[2 * b + 1] * (float) e1;
        coru += du * (shx[2 * b] + shx[2 * b + 1]);
    }
    __shared__ float4 sh[128];
    sh[threadIdx.x] = make_float4(accg, corg, accu, coru);
    __syncthreads();
    if (part == 0) {
        float ag = 0.f, cg = 0.f, au = 0.f, cu = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float4 v = sh[threadIdx.x + s];
            ag += v.x; cg += v.y; au += v.z; cu += v.w;
        }
        const float sg = ag - cg;
        const float su = au - cu;
        ff[(size_t) j * FF + row] = (sg / (1.f + __expf(-sg))) * su;
    }
}

// The v2 down body, extracted so the persistent kernel (STRATA_MW_PERSIST) can run the SAME code
// per virtual block (the batched launch below = one body call on blockIdx.x; the persistent kernel
// grid-strides it).  `smem` carries the body's scratch: one 2.9 KB union, laid out at offset 0 -
// the arithmetic is the inline version's, untouched.
template <int SPLIT, bool F16>
__device__ __forceinline__ void s2_stage2_v2_body(const int vb, char* smem,
                                                  const uint8_t* __restrict__ blobs, size_t blob_stride,
                                                  size_t d_codes_off, size_t d_scales_off,
                                                  const MaxJob* __restrict__ jobs,
                                                  int njobs, const int8_t* __restrict__ faq,
                                                  const float* __restrict__ fasc, const float* __restrict__ fahx,
                                                  float* __restrict__ out,
                                                  int blk_base, int blks,
                                                  size_t d_row_stride, size_t d_scale_stride) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(H % PER_BLOCK == 0, "v2 stages one job per block: H must be a multiple of PER_BLOCK");
    int8_t* sq = (int8_t*) smem;                                        // FF bytes
    float* ssc = (float*) (smem + FF);                                  // FF/32
    float* shx = ssc + FF / 32;                                         // FF/32
    uint32_t* slut = (uint32_t*) (smem + FF + 2 * (FF / 32) * 4);       // 256 entries
    float2* sh = (float2*) (slut + 256);                                // 128 entries
    const int row0 = vb * PER_BLOCK;
    if (row0 >= njobs * H) return;
    const int j = row0 / H;
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * blob_stride;
    {
        const int4* srcq = (const int4*) (faq + (size_t) j * FF);
        int4* dstq = (int4*) sq;
        for (int i = (int) threadIdx.x; i < FF / 16; i += 128) dstq[i] = srcq[i];
        const float* sc = fasc + (size_t) j * (FF / 32);
        const float* hx = fahx + (size_t) j * (FF / 32);
        for (int i = (int) threadIdx.x; i < FF / 32; i += 128) {
            ssc[i] = sc[i];
            shx[i] = hx[i];
        }
        for (int i = (int) threadIdx.x; i < 256; i += 128) slut[i] = (uint32_t) expand4_s2(i);
    }
    __syncthreads();
    const int idx = row0 + (int) (threadIdx.x / SPLIT);
    if (idx >= njobs * H) return;
    const int row = idx - j * H;
    const int part = (int) (threadIdx.x % SPLIT);
    // SPLIT_MODE_TENSOR: the contraction runs over this card's BLOCKS [blk_base, blk_base+blks) of the
    // SC_D 64-element blocks; the blob's down region stores exactly those blocks per row, compact
    // (d_row_stride/d_scale_stride bytes per row).  Full range (0, SC_D, ROW_D, SC_D*2) = the plain call.
    const uint8_t* dc = blob + d_codes_off + (size_t) row * d_row_stride;
    const uint8_t* ds = blob + d_scales_off + (size_t) row * d_scale_stride;
    float acc = 0.f, corr = 0.f;
    for (int b = blk_base + part; b < blk_base + blks; b += SPLIT) {
        const int lb = b - blk_base;                    // the block's place in this card's compact shard
        const float d = read_scale(ds, lb);
        int d0 = 0, d1 = 0;
        block_pair_s2_lut(dc + lb * 16, sq + (2 * b) * 32, slut, d0, d1);
        acc += d * ssc[2 * b] * (float) d0;
        acc += d * ssc[2 * b + 1] * (float) d1;
        corr += d * (shx[2 * b] + shx[2 * b + 1]);
    }
    sh[threadIdx.x] = make_float2(acc, corr);
    __syncthreads();
    if (part == 0) {
        float a0 = 0.f, c0 = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float2 v = sh[threadIdx.x + s];
            a0 += v.x; c0 += v.y;
        }
        const float r = a0 - c0;
        if (F16) ((uint16_t*) out)[(size_t) jd.dst * H + row] = f2h_rn(r);   // half the uplink bytes
        else     out[(size_t) jd.dst * H + row] = r;
    }
}

// kPersistSmem (below) needs the down body's scratch total; keep the two in one place.
constexpr int kS2DownSmem = FF + 2 * (FF / 32) * 4 + 256 * 4 + 128 * (int) sizeof(float2);

template <int SPLIT, bool F16 = false>
__global__ void expert_s2_stage2_batch_v2(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                          size_t d_codes_off, size_t d_scales_off,
                                          const MaxJob* __restrict__ jobs,
                                          int njobs, const int8_t* __restrict__ faq,
                                          const float* __restrict__ fasc, const float* __restrict__ fahx,
                                          float* __restrict__ out,
                                          volatile uint32_t* __restrict__ mflag, const uint32_t ring,
                                          uint32_t* __restrict__ ctr,
                                          int blk_base, int blks,
                                          size_t d_row_stride, size_t d_scale_stride) {
    __shared__ __align__(16) char smem[kS2DownSmem];
    s2_stage2_v2_body<SPLIT, F16>((int) blockIdx.x, smem, blobs, blob_stride, d_codes_off, d_scales_off,
                                  jobs, njobs, faq, fasc, fahx, out, blk_base, blks, d_row_stride,
                                  d_scale_stride);
    // E1 (2026-10-12): the GPU-raised doorbell.  When wired (mflag != nullptr), the LAST block of this
    // launch makes its rows visible system-wide and raises the tier's mapped completion flag to `ring`;
    // the 3060's graph multi-waits on the four tier flags plus the host's main flag, and the host never
    // waits for the M10s.  The arrival counter resets IN-STREAM (this tier's next down launch cannot
    // start before this kernel retires), so it is already 0 for the next layer.  The grid is sized
    // exactly (H % PER_BLOCK == 0), so every block reaches this tail and the count closes.
    if (mflag != nullptr) {
        __threadfence_system();                       // every thread: its rows visible system-wide
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned old = atomicAdd(ctr, 1u);
            if (old == gridDim.x - 1) {
                *mflag = ring;
                __threadfence_system();
                *ctr = 0;
            }
        }
    }
}

// ---- STRATA_MW_MT (down): the multi-token S2 down, the same run treatment the gup got ------------------
//
// The measured waste (2026-10-23 prefill anatomy): the v2 down re-reads and re-expands the expert's
// down code rows once per (expert, TOKEN) - the same multiplicity tax the gup had.  This kernel serves
// a RUN (one expert, up to E_MAX adjacent entries, the table the gup arm already built and left in the
// staging) per block-group: each (row, block)'s 16 code bytes are loaded and LUT-expanded ONCE into
// payloads, then dotted against EVERY entry's staged faq row.  The integer dots are exact (integer
// adds), the per-entry float tail and the SPLIT reduction order are verbatim s2_stage2_v2_body's, so
// every (expert, token) row is bit-identical.  dst comes from the standard job table at the entry's
// SORTED POSITION (the prefill fixup sets dst = position; the table is order-agnostic either way).
template <int SPLIT = 2, int THREADS = 128, int E_MAX = 4>
__global__ void expert_s2_stage2_mt_batch(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                          size_t d_codes_off, size_t d_scales_off,
                                          const int32_t* __restrict__ runs,      // [nruns x 4]: slot, count, entries_off, unused
                                          const int32_t* __restrict__ entries,   // [total x 2]: tok, sorted position
                                          const MaxJob* __restrict__ jobs,
                                          const int8_t* __restrict__ faq, const float* __restrict__ fasc,
                                          const float* __restrict__ fahx, float* __restrict__ out,
                                          int blk_base, int blks,
                                          size_t d_row_stride, size_t d_scale_stride) {
    constexpr int PER_BLOCK = THREADS / SPLIT;
    static_assert(H % PER_BLOCK == 0, "the MT down stages one run's rows per block, as the v2 one");
    __shared__ __align__(16) int8_t sq[E_MAX][FF];
    __shared__ __align__(16) float ssc[E_MAX][FF / 32];
    __shared__ __align__(16) float shx[E_MAX][FF / 32];
    __shared__ __align__(16) uint32_t slut[256];
    __shared__ __align__(16) float2 sh[E_MAX][THREADS];
    const int rpb = H / PER_BLOCK;                     // row-blocks per run
    const int run = (int) (blockIdx.x / rpb);
    const int vb = (int) (blockIdx.x % rpb);
    const int slot = runs[run * 4 + 0];
    const int E = runs[run * 4 + 1];
    const int ent0 = runs[run * 4 + 2];
    if (E <= 0 || E > E_MAX) return;
    const uint8_t* blob = blobs + (size_t) slot * blob_stride;
    {
        for (int i = (int) threadIdx.x; i < 256; i += THREADS) slut[i] = (uint32_t) expand4_s2(i);
        for (int e = 0; e < E; ++e) {
            const int pos = entries[(ent0 + e) * 2 + 1];
            const int4* srcq = (const int4*) (faq + (size_t) pos * FF);
            int4* dstq = (int4*) sq[e];
            for (int i = (int) threadIdx.x; i < FF / 16; i += THREADS) dstq[i] = srcq[i];
            const float* sc = fasc + (size_t) pos * (FF / 32);
            const float* hx = fahx + (size_t) pos * (FF / 32);
            for (int i = (int) threadIdx.x; i < FF / 32; i += THREADS) {
                ssc[e][i] = sc[i];
                shx[e][i] = hx[i];
            }
        }
    }
    __syncthreads();
    const int row = vb * PER_BLOCK + (int) (threadIdx.x / SPLIT);
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* dc = blob + d_codes_off + (size_t) row * d_row_stride;
    const uint8_t* ds = blob + d_scales_off + (size_t) row * d_scale_stride;
    float acc[E_MAX], corr[E_MAX];
#pragma unroll
    for (int e = 0; e < E_MAX; ++e) { acc[e] = 0.f; corr[e] = 0.f; }
    for (int b = blk_base + part; b < blk_base + blks; b += SPLIT) {
        const int lb = b - blk_base;                    // the block's place in this card's compact shard
        // ONE hoist per (row, block): the 16 code bytes load + LUT-expand once, amortised over E
        const int4 c4 = *(const int4*) (dc + lb * 16);
        const float d = read_scale(ds, lb);
        int p[16];
        p[0]  = (int) slut[(c4.x >> 0) & 0xFF];   p[1]  = (int) slut[(c4.x >> 8) & 0xFF];
        p[2]  = (int) slut[(c4.x >> 16) & 0xFF];  p[3]  = (int) slut[(c4.x >> 24) & 0xFF];
        p[4]  = (int) slut[(c4.y >> 0) & 0xFF];   p[5]  = (int) slut[(c4.y >> 8) & 0xFF];
        p[6]  = (int) slut[(c4.y >> 16) & 0xFF];  p[7]  = (int) slut[(c4.y >> 24) & 0xFF];
        p[8]  = (int) slut[(c4.z >> 0) & 0xFF];   p[9]  = (int) slut[(c4.z >> 8) & 0xFF];
        p[10] = (int) slut[(c4.z >> 16) & 0xFF];  p[11] = (int) slut[(c4.z >> 24) & 0xFF];
        p[12] = (int) slut[(c4.w >> 0) & 0xFF];   p[13] = (int) slut[(c4.w >> 8) & 0xFF];
        p[14] = (int) slut[(c4.w >> 16) & 0xFF];  p[15] = (int) slut[(c4.w >> 24) & 0xFF];
#pragma unroll
        for (int e = 0; e < E_MAX; ++e) {
            if (e >= E) break;
            const int8_t* qe = sq[e] + (2 * b) * 32;
            const int4 q0 = *(const int4*) (qe);
            const int4 q1 = *(const int4*) (qe + 16);
            const int4 q2 = *(const int4*) (qe + 32);
            const int4 q3 = *(const int4*) (qe + 48);
            int d0 = (dp4a_s32(q0.x, p[0], 0) + dp4a_s32(q0.y, p[1], 0)) + (dp4a_s32(q0.z, p[2], 0) + dp4a_s32(q0.w, p[3], 0));
            d0 += (dp4a_s32(q1.x, p[4], 0) + dp4a_s32(q1.y, p[5], 0)) + (dp4a_s32(q1.z, p[6], 0) + dp4a_s32(q1.w, p[7], 0));
            int d1 = (dp4a_s32(q2.x, p[8], 0) + dp4a_s32(q2.y, p[9], 0)) + (dp4a_s32(q2.z, p[10], 0) + dp4a_s32(q2.w, p[11], 0));
            d1 += (dp4a_s32(q3.x, p[12], 0) + dp4a_s32(q3.y, p[13], 0)) + (dp4a_s32(q3.z, p[14], 0) + dp4a_s32(q3.w, p[15], 0));
            acc[e] += d * ssc[e][2 * b] * (float) d0;
            acc[e] += d * ssc[e][2 * b + 1] * (float) d1;
            corr[e] += d * (shx[e][2 * b] + shx[e][2 * b + 1]);
        }
    }
#pragma unroll
    for (int e = 0; e < E_MAX; ++e) {
        if (e >= E) break;
        sh[e][threadIdx.x] = make_float2(acc[e], corr[e]);
    }
    __syncthreads();
    if (part == 0) {
#pragma unroll
        for (int e = 0; e < E_MAX; ++e) {
            if (e >= E) break;
            float a0 = 0.f, c0 = 0.f;
            for (int s = 0; s < SPLIT; ++s) {
                const float2 v = sh[e][threadIdx.x + s];
                a0 += v.x; c0 += v.y;
            }
            const int pos = entries[(ent0 + e) * 2 + 1];
            out[(size_t) jobs[pos].dst * H + row] = a0 - c0;
        }
    }
}

// v2 + FUSED quantisation: identical to expert_s2_stage1_batch_v2 except that the SwiGLU output is
// quantised in the epilogue instead of being written to ff.  With SPLIT=4 the block covers exactly one
// 32-value quant chunk of its job (PER_BLOCK=32, FF/32=20 chunks per job), and the epilogue runs the SAME
// code as expert_s2_quant_ff over the block's 32 values (one warp, the same shuffle tree), so faq/fasc/fahx
// are BIT-IDENTICAL to the separate quant kernel and the caller drops that launch entirely.
template <int SPLIT>
__global__ void expert_s2_stage1_batch_v2q(const uint8_t* __restrict__ blobs, const MaxJob* __restrict__ jobs,
                                           int njobs, const uint8_t* __restrict__ acts_base,
                                           int8_t* __restrict__ faq, float* __restrict__ fasc,
                                           float* __restrict__ fahx) {
    static_assert(SPLIT == 4, "the fused quant needs the block to cover exactly one 32-value chunk");
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(FF % PER_BLOCK == 0, "v2 stages one job per block: FF must be a multiple of PER_BLOCK");
    const int row0 = blockIdx.x * PER_BLOCK;
    if (row0 >= njobs * FF) return;
    const int j = row0 / FF;                    // every row of this block is in job j (FF % PER_BLOCK == 0)
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * BLOB;
    const uint8_t* at = acts_base + (size_t) jd.tok * ACTS_STRIDE;
    __shared__ __align__(16) int8_t sq[H];
    __shared__ float ssc[H / 32];
    __shared__ float shx[H / 32];
    __shared__ uint32_t slut[256];
    {
        const int4* srcq = (const int4*) at;
        int4* dstq = (int4*) sq;
        for (int i = (int) threadIdx.x; i < H / 16; i += 128) dstq[i] = srcq[i];
        const float* ssrc = (const float*) (at + H);
        for (int i = (int) threadIdx.x; i < H / 32; i += 128) {
            ssc[i] = ssrc[i];
            shx[i] = ssrc[H / 32 + i];
        }
        for (int i = (int) threadIdx.x; i < 256; i += 128) slut[i] = (uint32_t) expand4_s2(i);
    }
    __syncthreads();
    const int idx = row0 + (int) (threadIdx.x / SPLIT);
    if (idx >= njobs * FF) return;
    const int row = idx - j * FF;
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * row) * ROW_GU;
    const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * row) * SC_GU * 2;
    const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * row + 1) * ROW_GU;
    const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * row + 1) * SC_GU * 2;
    float accg = 0.f, corg = 0.f, accu = 0.f, coru = 0.f;
    int4 ng = make_int4(0,0,0,0), nu = make_int4(0,0,0,0);
    if (part < SC_GU) { ng = *(const int4*) (gc + part * 16); nu = *(const int4*) (uc + part * 16); }
    for (int b = part; b < SC_GU; b += SPLIT) {
        const int4 cg4 = ng, cu4 = nu;
        const int nb = b + SPLIT;
        if (nb < SC_GU) { ng = *(const int4*) (gc + nb * 16); nu = *(const int4*) (uc + nb * 16); }
        const float dg = read_scale(gs, b);
        int d0 = 0, d1 = 0;
        block_pair_s2_pre(cg4, sq + (2 * b) * 32, slut, d0, d1);
        accg += dg * ssc[2 * b] * (float) d0;
        accg += dg * ssc[2 * b + 1] * (float) d1;
        corg += dg * (shx[2 * b] + shx[2 * b + 1]);
        const float du = read_scale(us, b);
        int e0 = 0, e1 = 0;
        block_pair_s2_pre(cu4, sq + (2 * b) * 32, slut, e0, e1);
        accu += du * ssc[2 * b] * (float) e0;
        accu += du * ssc[2 * b + 1] * (float) e1;
        coru += du * (shx[2 * b] + shx[2 * b + 1]);
    }
    __shared__ float4 sh[128];
    __shared__ float sx[32];
    sh[threadIdx.x] = make_float4(accg, corg, accu, coru);
    __syncthreads();
    if (part == 0) {
        float ag = 0.f, cg = 0.f, au = 0.f, cu = 0.f;
        for (int s = 0; s < SPLIT; ++s) {
            const float4 v = sh[threadIdx.x + s];
            ag += v.x; cg += v.y; au += v.z; cu += v.w;
        }
        const float sg = ag - cg;
        const float su = au - cu;
        sx[threadIdx.x / SPLIT] = (sg / (1.f + __expf(-sg))) * su;   // row-group order == chunk lane order
    }
    __syncthreads();
    // ONE warp quantises the block's chunk: the exact body of expert_s2_quant_ff (same tree, same roundings)
    if (threadIdx.x < 32) {
        const int lane = (int) threadIdx.x;
        const int chunk = (int) (j * (FF / 32) + (row0 % FF) / PER_BLOCK);
        const float v = sx[lane];
        float av = fabsf(v);
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) av = fmaxf(av, __shfl_down(av, s));
        const float amax = __shfl(av, 0);
        const float sc = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = sc > 0.f ? 1.f / sc : 0.f;
        const float t = __fmul_rn(v, inv);
        const float r = __fadd_rn(t, (t >= 0.f) ? 0.5f : -0.5f);
        int qi = __float2int_rz(r);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        faq[(size_t) chunk * 32 + lane] = (int8_t) qi;
        int ssum = qi;
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) ssum += __shfl_down(ssum, s);
        if (lane == 0) {
            fasc[chunk] = sc;
            fahx[chunk] = __fmul_rn(sc, (float) ssum);
        }
    }
}

// v2 is the DEFAULT (measured: ~30% faster kernels, parity-bit-identical, window -8.9% on the engine);
// STRATA_S2_V2=0 restores the v1 kernels for an A/B.
inline bool s2_v2_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_S2_V2");
        return e == nullptr || e[0] == '\0' || e[0] != '0';
    }();
    return on;
}

// A/B hooks for the v2 launch geometry (threads cooperating on one row).  The defaults are the measured
// best; the env overrides exist so a sweep can rerun without editing/rebuilding the constants.
inline int s2_split_gup() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_S2_SPLIT_GU");
        const int x = e ? std::atoi(e) : 4;
        return (x == 2 || x == 4 || x == 8) ? x : 4;
    }();
    return v;
}
inline int s2_split_down() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_S2_SPLIT_D");
        const int x = e ? std::atoi(e) : 2;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 2;
    }();
    return v;
}

// ---------------------------------------------------------------------------------------------------------
// IQ gate/up (a native pack's i-quant layers).  An i-quant row cannot be re-expressed in the S2 pack
// form (tools/iq_pack.py refuses it), so the blob's gate/up rows keep their RAW GGUF block layouts and
// the dot products below are the engine's own (src/kernels/cuda/iq_kernels.cu, transcribed from
// llama.cpp's vecdotq.cuh), with exactly one change per body: ggml_cuda_dp4a -> dp4a_s32 (the PTX vmad
// the S2 kernels above already use; llama.cpp's m10-final.patch made the same swap).  The activations
// are the tier's staged ActQ (the same packed upload the S2 kernels read): ActQ.q IS block_q8_1.qs bit
// for bit (act_quant_q8_1 and quantize_q8_1_rows compute the same codes - amax/127, +-0.5 away from
// zero, truncate, clamp +-127), so each q8_1 block is rebuilt in shared memory with
// d = f2h_rn(scale): the same d bits the device q8_1 quantizer stores, the #606 clamp (|d| past 65504)
// keeping it finite.  The s half of ds is left zero: none of these three dots reads it (IQ1_M takes its
// delta sum from the qs bytes themselves, sumy below).
//
// fp16 -> fp32 by bits (h2f_dev/low2f_dev), NOT the __half2float intrinsic: __float2half was measured on
// this M10 compiling to a stub that returns zero (the f2h_rn comment above), so no fp16 conversion
// intrinsic is trusted here; fp16_to_fp32_dev is bit-identical to the host's strata::fp16_to_fp32.
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ float h2f_dev(half h) {
    return fp16_to_fp32_dev(*reinterpret_cast<const uint16_t*>(&h));
}
__device__ __forceinline__ float low2f_dev(half2 h) {
    return fp16_to_fp32_dev(reinterpret_cast<const uint16_t*>(&h)[0]);
}

// ---- the dot products (vecdotq.cuh): ggml_cuda_dp4a -> dp4a_s32, __half2float -> h2f_dev,
// __low2float -> low2f_dev; every other line verbatim from src/kernels/cuda/iq_kernels.cu.
__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs, const uint2* __restrict__ grid_sh) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = grid_sh[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = dp4a_s32(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = dp4a_s32(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = h2f_dev(bq2->d) * low2f_dev(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs, const uint64_t* __restrict__ grid_sh) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (grid_sh + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = dp4a_s32(grid_l, u0, sumi0);
            sumi0 = dp4a_s32(grid_h, u1, sumi0);
        } else {
            sumi1 = dp4a_s32(grid_l, u0, sumi1);
            sumi1 = dp4a_s32(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = h2f_dev(bq2->d) * low2f_dev(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs, const uint32_t* __restrict__ grid_sh) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = grid_sh[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = dp4a_s32(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = dp4a_s32(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = dp4a_s32(u0, 0x01010101, sumy);
        sumy = dp4a_s32(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = h2f_dev(scale.f16) * low2f_dev(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// The formats of the i-quant layers this tier takes (as iq_kernels.cu's Fmt<>): qk = values per block,
// ipb = dot calls per block, step = the iqs stride between calls.  22 = IQ2_S, 16 = IQ2_XXS, 29 = IQ1_M
// (the IQ2_XS model file's three gate/up encodings; down is Q2_0 everywhere, the S2 kernel above).
template <int TY> struct IqFmt;
template <> struct IqFmt<22> {
    static constexpr int qk = 256, ipb = 8, step = 2;
    static constexpr int grid_bytes = 1024 * 8;   // iq2s_grid: uint64 x 1024
    __device__ static const void* grid_dev() { return (const void*) iq2s_grid; }
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const void* gsh) {
        return vec_dot_iq2_s_q8_1(v, y, kbx, iqs, (const uint64_t*) gsh);
    }
};
template <> struct IqFmt<16> {
    static constexpr int qk = 256, ipb = 8, step = 2;
    static constexpr int grid_bytes = 256 * 8;   // iq2xxs_grid: uint2 x 256
    __device__ static const void* grid_dev() { return (const void*) iq2xxs_grid; }
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const void* gsh) {
        return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs, (const uint2*) gsh);
    }
};
template <> struct IqFmt<29> {
    static constexpr int qk = 256, ipb = 8, step = 1;
    static constexpr int grid_bytes = 2048 * 4;   // iq1s_grid_gpu: uint32 x 2048
    __device__ static const void* grid_dev() { return (const void*) iq1s_grid_gpu; }
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const void* gsh) {
        return vec_dot_iq1_m_q8_1(v, y, kbx, iqs, (const uint32_t*) gsh);
    }
};

// The IQ gup kernel: the same batched geometry as expert_s2_stage1_batch_v2q (one job's 32 rows per
// CUDA block, SPLIT=4 threads per row, the SwiGLU output quantised in the SAME epilogue - the same
// shuffle tree, the same roundings - so the down stage's faq/fasc/fahx are bit-identical to the S2
// path's and the S2 down kernel consumes them unchanged).  The differences are exactly three: the
// blob's gate/up rows are raw GGUF blocks at o_gate/o_up with the layer's gu_row stride (not the S2
// planes), the row walk is lane-strided DOT CALLS (k = part; k < nb*ipb; k += SPLIT, kbx = k/ipb,
// iqs = step*(k%ipb), exactly the engine's row_dot), and there is no hx correction term (the i-quant
// grids carry their own signs; the act's d rides inside the dot).
//
// THREADS (default 128, opt-in 256 via STRATA_MW_IQ_T256 with SPLIT 2/4): the occupancy arm.  Each
// block stages sact (3.2 KB) + the codebook grid (2-8 KB) whatever its size, so at 32 rows per
// 128-thread block the shared-memory ceiling holds only ~5-8 blocks per SM - 31-50% occupancy, too
// few resident loads to hide the M10's DRAM latency (S0: the gup stage measured 6x off the 83 GB/s
// floor).  Doubling the rows per block halves the blocks AND the redundant act+grid staging while
// doubling resident threads per SM; the per-output summation order is untouched (the SPLIT meet and
// the chunk tree walk the same sequence), so the image stays bit-identical - a launch-geometry
// change only, exactly like the SPLIT sweep it generalizes.
// The IQ gup body, extracted like the down body above: the batched launch = one call on blockIdx.x,
// the persistent kernel (STRATA_MW_PERSIST) grid-strides the SAME code.  `smem` carries the union:
// [sact (H/32 blocks)][align 16][sx PER_BLOCK floats][align 16][sgrid grid_bytes].  The arithmetic
// is the inline version's, untouched - bit-identical by construction.
// The IQ gup body's shared-memory layout, ONE source of truth for the body, the batched wrapper
// and the persistent kernel's union buffer: [sact (H/32 blocks)][align 16][sx PER_BLOCK*MR floats]
// [align 16][sgrid: XXS 2 KB, S/IQ1_M 8 KB].  MR (rows per thread-group, STRATA_MW_IQ_MR) widens
// only the sx region - one slot per row of the block's PER_BLOCK*MR rows.
template <int TY, int SPLIT, int THREADS, int MR = 1>
struct IqV2qSmem {
    static_assert(MR == 1 || MR == 2 || MR == 4 || MR == 5, "STRATA_MW_IQ_MR takes 1, 2, 4 or 5");
    static constexpr int per_block = THREADS / SPLIT;
    static constexpr int sact = ((H / 32) * (int) sizeof(block_q8_1) + 15) & ~15;
    static constexpr int sx = (per_block * MR * 4 + 15) & ~15;
    static constexpr int grid = (TY == 16 ? 2048 : 8192);   // F::grid_bytes
    static constexpr int total = sact + sx + grid;
};

// MR (STRATA_MW_IQ_MR, default 1): each SPLIT-lane group interleaves MR row-pairs in registers -
// the block covers PER_BLOCK*MR rows of one job, so the sact/sgrid staging amortises MR-fold and
// each act smem read serves 2*MR dots instead of 2.  BIT-EXACT by construction: every row keeps
// its k-set and order (k = part, part+SPLIT, ...), its dot sequence, its SPLIT shuffle tree and
// its row-major sx/chunk slot; the r-loop only interleaves independent accumulators.  The weight
// path (codes load + codebook lookup) is per-row and does NOT amortise - MR's win is bounded to
// the act/index/staging share (the SASS of the MR=1 loop puts that near 10%).
template <int TY, int SPLIT, int THREADS, int MR = 1>
__device__ __forceinline__ void iq_stage1_v2q_body(const int vb, char* smem,
                                                   const uint8_t* __restrict__ blobs, size_t blob_stride,
                                                   const MaxJob* __restrict__ jobs, int njobs,
                                                   const uint8_t* __restrict__ acts_base,
                                                   size_t o_gate, size_t o_up, size_t gu_row,
                                                   int8_t* __restrict__ faq, float* __restrict__ fasc,
                                                   float* __restrict__ fahx, float* __restrict__ ff_dbg,
                                                   int row_base, int rows_slice) {
    using F = IqFmt<TY>;
    using L = IqV2qSmem<TY, SPLIT, THREADS, MR>;
    constexpr int PER_BLOCK = L::per_block;
    constexpr int ROWS = PER_BLOCK * MR;
    block_q8_1* sact = (block_q8_1*) smem;
    float* sx = (float*) (smem + L::sact);
    uint8_t* sgrid = (uint8_t*) (smem + L::sact + L::sx);
    // SPLIT is the sweep knob (STRATA_MW_IQ_SPLIT, default 4).  The fused quant epilogue below walks
    // PER_BLOCK/32 chunks, so any SPLIT that divides 32 keeps the chunk tree bit-identical: with 4 it
    // compiles to exactly the one-chunk body this kernel shipped with.
    static_assert(SPLIT == 1 || SPLIT == 2 || SPLIT == 4, "the IQ gup kernel takes SPLIT 1, 2 or 4");
    // SPLIT_MODE_TENSOR: `rows_slice` is this card's gate/up row count (its slice of FF; a multiple of
    // PER_BLOCK for every shard the planner emits).  The blob stores exactly those rows, compact, so the
    // mapping is the full kernel's with row -> row_base + local; every per-row value stays bit-identical.
    static_assert(FF % PER_BLOCK == 0, "the IQ kernel stages one job per block, as the S2 one");
    const int row0 = vb * ROWS;
    if (row0 >= njobs * rows_slice) return;
    const int j = row0 / rows_slice;         // every row of this block is in job j (rows_slice % ROWS == 0)
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * blob_stride;
    const uint8_t* at = acts_base + (size_t) jd.tok * ACTS_STRIDE;
    {
        // the job's token's activation as q8_1 blocks: ActQ.q verbatim (8 int32 per block, the act is
        // 4-aligned here), d = f2h_rn(scale) with the #606 clamp, s = 0 (unused by these dots).
        const int* src = (const int*) at;
        const float* sc = (const float*) (at + H);
        for (int k = (int) threadIdx.x; k < H / 32; k += THREADS) {
            int* dst = (int*) &sact[k].qs[0];
            for (int w = 0; w < 8; ++w) dst[w] = src[k * 8 + w];
            float d = sc[k];
            d = fabsf(d) > 65504.f ? copysignf(65504.f, d) : d;
            uint16_t* ds = (uint16_t*) &sact[k].ds;
            ds[0] = f2h_rn(d);
            ds[1] = 0;
        }
    }
    // the codebook grids staged once per block: the dots hit them at data-dependent indices, and a
    // shared copy turns those latency-exposed global reads (~21 GB/s measured, 18% of the ceiling)
    // into shared lookups.  2 KB (IQ2_XXS) or 8 KB (IQ2_S / IQ1_M); staged coalesced.
    {
        const int4* g = (const int4*) F::grid_dev();
        int4* d = (int4*) sgrid;
#pragma unroll
        for (int i = (int) threadIdx.x; i < F::grid_bytes / 16; i += THREADS) d[i] = g[i];
    }
    __syncthreads();
    const int idx = row0 + (int) (threadIdx.x / SPLIT) * MR;   // this group's first row (never out of range)
    const int row = idx - j * rows_slice;                       // the group's LOCAL first row (compact shard)
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* gr[MR];
    const uint8_t* ur[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        gr[r] = blob + o_gate + (size_t) (row + r) * gu_row;
        ur[r] = blob + o_up + (size_t) (row + r) * gu_row;
    }
    const int nb = H / F::qk;                // 256-value weight blocks per row (10)
    float accg[MR], accu[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { accg[r] = 0.f; accu[r] = 0.f; }
    for (int k = part; k < nb * F::ipb; k += SPLIT) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* sa = sact + kbx * (F::qk / 32);   // ONE act pointer per k: all 2*MR dots share it
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            accg[r] += F::dot(gr[r], sa, kbx, iqs, sgrid);
            accu[r] += F::dot(ur[r], sa, kbx, iqs, sgrid);
        }
    }
    // the SPLIT partial sums meet by warp shuffles, in the SAME sequential order the shared-buffer
    // reduction used (s = 0, 1, ... SPLIT-1), so the image stays bit-identical; a part-group never
    // crosses a warp boundary (SPLIT divides 32).  EVERY lane shuffles (a shfl under divergent
    // control is undefined); each lane of a group converges on its group's sequential sum, and only
    // the part == 0 leader's copy is consumed.  One KB of shared saved.
    {
        const int lane = (int) (threadIdx.x & 31);
        const unsigned full = 0xFFFFFFFFu;
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            float ag = accg[r], au = accu[r];
#pragma unroll
            for (int s = 1; s < SPLIT; ++s) {
                ag += __shfl_sync(full, accg[r], (lane + s) & 31);
                au += __shfl_sync(full, accu[r], (lane + s) & 31);
            }
            if (part == 0) {
                const float sxv = (ag / (1.f + __expf(-ag))) * au;   // row-group order == chunk lane order
                sx[(threadIdx.x / SPLIT) * MR + r] = sxv;
                if (ff_dbg != nullptr) ff_dbg[(size_t) j * FF + (size_t) (row_base + row + r)] = sxv;   // the parity tool reads the pre-quant dots (global row index)
            }
        }
    }
    __syncthreads();
    // ONE warp quantises the block's chunks (PER_BLOCK*MR/32 of them; exactly one at SPLIT=4, MR=1):
    // the exact body of expert_s2_stage1_batch_v2q's epilogue (same tree, same roundings) per chunk,
    // so faq/fasc/fahx are bit-identical to the S2 path's whatever SPLIT or MR is.
    if (threadIdx.x < 32) {
        const int lane = (int) threadIdx.x;
        constexpr int CHUNKS = ROWS / 32;
        const int chunk0 = (int) (j * (FF / 32) + (size_t) (row_base + row0 % rows_slice) / 32);   // GLOBAL chunk index (the faq layout is entry-major, full FF)
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
        const int chunk = chunk0 + c;
        const float v = sx[c * 32 + lane];
        float av = fabsf(v);
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) av = fmaxf(av, __shfl_down(av, s));
        const float amax = __shfl(av, 0);
        const float sc = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = sc > 0.f ? 1.f / sc : 0.f;
        const float t = __fmul_rn(v, inv);
        const float r = __fadd_rn(t, (t >= 0.f) ? 0.5f : -0.5f);
        int qi = __float2int_rz(r);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        faq[(size_t) chunk * 32 + lane] = (int8_t) qi;
        int ssum = qi;
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) ssum += __shfl_down(ssum, s);
        if (lane == 0) {
            fasc[chunk] = sc;
            fahx[chunk] = __fmul_rn(sc, (float) ssum);
        }
        }
    }
}

template <int TY, int SPLIT, int THREADS = 128, int MR = 1>
__global__ void expert_iq_stage1_batch_v2q(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                           const MaxJob* __restrict__ jobs, int njobs,
                                           const uint8_t* __restrict__ acts_base,
                                           size_t o_gate, size_t o_up, size_t gu_row,
                                           int8_t* __restrict__ faq, float* __restrict__ fasc,
                                           float* __restrict__ fahx, float* __restrict__ ff_dbg,
                                           int row_base, int rows_slice) {
    __shared__ __align__(16) char smem[IqV2qSmem<TY, SPLIT, THREADS, MR>::total];
    iq_stage1_v2q_body<TY, SPLIT, THREADS, MR>((int) blockIdx.x, smem, blobs, blob_stride, jobs, njobs,
                                               acts_base, o_gate, o_up, gu_row, faq, fasc, fahx, ff_dbg,
                                               row_base, rows_slice);
}

// ---- STRATA_MW_PERSIST: the self-service tier's kernel ---------------------------------------
//
// One resident kernel per card serves the whole decode stream.  Per service: block 0 polls the
// mapped doorbell and broadcasts the service number through DEVICE memory (the other blocks never
// touch the uplink while waiting); every block then stages the service's acts+jobs from the mapped
// ring slot into device memory in ONE bulk cooperative copy (volatile reads: uncached sysmem on
// sm_50, and the slot is host-rewritten two services later); a generation grid barrier orders the
// phases (the grid never exceeds the kernel's own occupancy, so all blocks are resident and the
// barrier cannot deadlock - sm_50 has no cooperative launch); the gup and down phases grid-stride
// the SAME bodies the batched kernels run, bit-identical by construction; and after a final
// system-wide fence block 0 raises the tier's E1 completion flag, the exact contract the batched
// down's last block carried.  The host's per-layer dispatch is pinned writes + one doorbell store:
// zero CUDA API calls per layer per card - the visit-order poison has no call to pay it with.
__device__ __forceinline__ void persist_grid_barrier(PersistCtl* c) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned gen = *(volatile unsigned*) &c->bar_gen;
        if (atomicAdd(&c->bar_count, 1u) == gridDim.x - 1u) {
            c->bar_count = 0;
            __threadfence();
            *(volatile unsigned*) &c->bar_gen = gen + 1u;
        } else {
            while (*(volatile unsigned*) &c->bar_gen == gen) {}
        }
    }
    __syncthreads();
}

template <int SPLIT, int THREADS>
__global__ void __launch_bounds__(THREADS) maxwell_persist_kernel(const PersistParams* __restrict__ pp,
                                                                  PersistCtl* c) {
    constexpr int kGupSmem = IqV2qSmem<22, SPLIT, THREADS>::total > IqV2qSmem<29, SPLIT, THREADS>::total
                                 ? IqV2qSmem<22, SPLIT, THREADS>::total
                                 : IqV2qSmem<29, SPLIT, THREADS>::total;
    constexpr int kSmem = kGupSmem > kS2DownSmem ? kGupSmem : kS2DownSmem;
    __shared__ __align__(16) char smem[kSmem];
    __shared__ int sh_s;
    const volatile uint32_t* seq = pp->d_seq;   // DEVICE doorbell: L2 poll, no PCIe on this path
    constexpr int kGupPer = THREADS / SPLIT;    // 32 at the armed <4, 128>
    constexpr int kDownPer = 64;                // the down body's PER_BLOCK at SPLIT 2
    // Services are served STRICTLY IN ORDER (seen+1, never "the newest"): under the split window
    // (P6) the host can legitimately run TWO services ahead - B(l)'s ring fires before A(l)'s
    // multi-wait - so two doorbells can queue, and serving the newest would SKIP the one between:
    // its rows never compute and its tier flag never rises (the hang of the first compound arm).
    // The doorbell counts RELATIVE per launch (1, 2, ...); an EMPTY service (njobs 0) is delivered
    // the same way, so every service the verifier counts is one the kernel raises.
    unsigned seen = 0;
    for (;;) {
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int s = 0;
            uint32_t polls = 0;
            for (;;) {
                const uint32_t v = *seq;           // L2 poll: the fast path, no PCIe
                if (v == PERSIST_HALT) { s = -1; break; }
                if (v > seen) { s = (int) seen + 1; break; }   // serve the NEXT, not the newest
                // the stop word is a host store to mapped memory (a stream write cannot reach this
                // kernel: IT owns its stream, and a queued write would wait for it to exit); the
                // periodic mapped read keeps the idle poll loop PCIe-quiet
                if ((++polls & 1023u) == 0 && *pp->halt) { s = -1; break; }
            }
            __threadfence();
            *(volatile int*) &c->service = s;
        }
        if (threadIdx.x == 0) {
            int s;
            while ((s = *(volatile int*) &c->service) == (int) seen) {}
            sh_s = s;
        }
        __syncthreads();
        const int s = sh_s;
        if (s < 0) return;
        seen = (unsigned) s;
        // the staging [hdr|acts|jobs] already landed in d_up[parity] via the H2D the copy engine
        // ran before writing the doorbell - the header is a device read, like everything else
        const uint8_t* dslot = pp->d_up[seen & 1u];
        const PersistHdr* hp = (const PersistHdr*) dslot;
        const PersistHdr h = *hp;
        const size_t acts_bytes = (size_t) h.n_tok * ACTS_STRIDE;
        const uint8_t* acts_base = dslot + PERSIST_HDR_BYTES;
        const MaxJob* jobs = (const MaxJob*) (acts_base + acts_bytes);
        const uint8_t* blobs = pp->blobs + h.blob_base;
        persist_grid_barrier(c);   // all blocks see the staged bytes before anyone consumes them
        // the switch OUTSIDE the virtual-block walk: gu_type is per-service, and a switch inside the
        // loop made the register allocator carry every branch's pressure across iterations (126
        // registers, 4 blocks/SM - the batched kernel needs ~8).  One loop per branch keeps the
        // live ranges clean.
        const int gup_blocks = (int) (((size_t) h.njobs * (size_t) h.tp_rows + kGupPer - 1) / kGupPer);
        switch (h.gu_type) {   // arm time guarantees 22/16/29 (native pack, no S2/42 layer)
            case 22:
                for (int vb = (int) blockIdx.x; vb < gup_blocks; vb += gridDim.x)
                    iq_stage1_v2q_body<22, SPLIT, THREADS>(vb, smem, blobs, h.blob_stride, jobs, (int) h.njobs, acts_base, h.o_gate, h.o_up, h.gu_row, pp->faq, pp->fasc, pp->fahx, nullptr, h.tp_row_base, h.tp_rows);
                break;
            case 16:
                for (int vb = (int) blockIdx.x; vb < gup_blocks; vb += gridDim.x)
                    iq_stage1_v2q_body<16, SPLIT, THREADS>(vb, smem, blobs, h.blob_stride, jobs, (int) h.njobs, acts_base, h.o_gate, h.o_up, h.gu_row, pp->faq, pp->fasc, pp->fahx, nullptr, h.tp_row_base, h.tp_rows);
                break;
            case 29:
                for (int vb = (int) blockIdx.x; vb < gup_blocks; vb += gridDim.x)
                    iq_stage1_v2q_body<29, SPLIT, THREADS>(vb, smem, blobs, h.blob_stride, jobs, (int) h.njobs, acts_base, h.o_gate, h.o_up, h.gu_row, pp->faq, pp->fasc, pp->fahx, nullptr, h.tp_row_base, h.tp_rows);
                break;
            default:
                break;
        }
        persist_grid_barrier(c);
        const int down_blocks = (int) (((size_t) h.njobs * (size_t) H + kDownPer - 1) / kDownPer);
        for (int vb = (int) blockIdx.x; vb < down_blocks; vb += gridDim.x)
            s2_stage2_v2_body<2, false>(vb, smem, blobs, h.blob_stride, h.d_codes_off, h.d_scales_off,
                                        jobs, (int) h.njobs, pp->faq, pp->fasc, pp->fahx, h.dst,
                                        h.tp_blk_base, h.tp_blks, h.d_row_stride, h.d_scale_stride);
        // the rows are system-visible, the whole grid passes one barrier, then block 0 rings the bell
        __threadfence_system();
        __syncthreads();
        persist_grid_barrier(c);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *pp->comp_flag = h.ring;
            __threadfence_system();
        }
    }
}

}  // namespace

bool s2_expert_maxwell_v2() { return s2_v2_enabled(); }

// SPLIT_MODE_TENSOR delivery: a one-block kernel ordered on the D2H copy's stream, AFTER the copy engine
// moved the card's partial rows into host memory - it makes them visible system-wide and raises the tier's
// mapped completion flag to `ring`: the same contract the down kernel's last block carries on the mapped
// path, moved behind the copy engine (the in-kernel mapped stores measured 3x stretched when four cards
// issue them concurrently - the LOCAL_DOWN shared-endpoint effect).
__global__ void mw_tp_raise_flag_kernel(volatile uint32_t* mflag, const uint32_t ring) {
    if (threadIdx.x == 0) {
        __threadfence_system();          // the D2H's data is host-visible before the doorbell
        *mflag = ring;
        __threadfence_system();
    }
}

void mw_tp_raise_flag_async(volatile uint32_t* mflag, uint32_t ring, cudaStream_t stream) {
    mw_tp_raise_flag_kernel<<<1, 32, 0, stream>>>(mflag, ring);
}

void s2_expert_maxwell(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                       const float* d_ahx, float* d_out) {
    float* d_ff = d_out + H;
    const int threads = 128;
    expert_s2_stage1<<<(FF + threads - 1) / threads, threads>>>(d_blob, d_aq, d_ascale, d_ahx, d_ff);
    cudaDeviceSynchronize();
    // NOTE: stage 2 needs the FF-wide requantised activation, which the caller computes from d_ff.
    // This combined entry point is for single-call use; the split API below is preferred.
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_expert_maxwell: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// Stage-split API: the activation quantisation lives on the host (same contract as the CPU).
void s2_expert_maxwell_gup(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                           const float* d_ahx, float* d_ff) {
    const int threads = 128;
    expert_s2_stage1<<<(FF + threads - 1) / threads, threads>>>(d_blob, d_aq, d_ascale, d_ahx, d_ff);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_expert_maxwell_gup: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// P3.1 async launches (no sync): the caller batches N experts then synchronises once.
void s2_expert_maxwell_gup_async(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                                 const float* d_ahx, float* d_ff, cudaStream_t stream) {
    constexpr int SPLIT = 4;
    constexpr int PER_BLOCK = 128 / SPLIT;
    expert_s2_stage1_mt<SPLIT><<<(FF + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blob, d_aq, d_ascale, d_ahx, d_ff);
}

void s2_expert_maxwell_down_async(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                                  const float* d_ahx, float* d_out, cudaStream_t stream) {
    const int threads = 128;
    expert_s2_stage2<<<(H + threads - 1) / threads, threads, 0, stream>>>(d_blob, d_aq, d_ascale, d_ahx, d_out);
}

void s2_expert_maxwell_sync() {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_expert_maxwell_sync: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// Intermediate quantisation ON THE DEVICE: replaces the host's act_quant_q8_1_avx2 round trip (D2H of ff,
// CPU quant, H2D of q/scales/hx) with one kernel.  Bit-exact with that CPU contract for finite inputs:
//   chunk of 32 -> amax = max|x| ; s = amax>0 ? amax/127 : 0 ; inv = s>0 ? 1/s : 0
//   t = x*inv (one fp32 rounding) ; r = t + (t>=0 ? 0.5 : -0.5) ; q = trunc(r) clamped to [-127,127]
//   scale = s ; hx = s * sum(q)
// __fmul_rn/__fadd_rn keep the two roundings separate (no FMA contraction), matching the CPU exactly.
__global__ void expert_s2_quant_ff(const float* __restrict__ ff, int8_t* __restrict__ q,
                                   float* __restrict__ scale, float* __restrict__ hx, int nchunks) {
    const int warp_in_block = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int g = blockIdx.x * ((int) blockDim.x >> 5) + warp_in_block;   // one warp per 32-elem chunk
    if (g >= nchunks) return;
    const float v = ff[(size_t) g * 32 + lane];
    float av = fabsf(v);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) av = fmaxf(av, __shfl_down(av, s));
    const float amax = __shfl(av, 0);
    const float sc = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = sc > 0.f ? 1.f / sc : 0.f;
    const float t = __fmul_rn(v, inv);
    const float r = __fadd_rn(t, (t >= 0.f) ? 0.5f : -0.5f);
    int qi = __float2int_rz(r);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[(size_t) g * 32 + lane] = (int8_t) qi;
    int ssum = qi;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) ssum += __shfl_down(ssum, s);
    if (lane == 0) {
        scale[g] = sc;
        hx[g] = __fmul_rn(sc, (float) ssum);
    }
}

void s2_expert_maxwell_quant_ff_async(const float* d_ff, int8_t* d_q, float* d_scale, float* d_hx,
                                      int nchunks, cudaStream_t stream) {
    if (nchunks <= 0) return;
    const int threads = 128;                 // 4 warps = 4 chunks per block
    const int per_block = threads / 32;
    const int blocks = (nchunks + per_block - 1) / per_block;
    expert_s2_quant_ff<<<blocks, threads, 0, stream>>>(d_ff, d_q, d_scale, d_hx, nchunks);
}

// Batched stage entry points: ONE launch per stage for the whole layer.  `d_jobs` is njobs pairs of
// (blob slot, token index) in device memory; d_ff/d_faq/d_fascale/d_fahx/d_out are per-job contiguous.
void s2_expert_maxwell_gup_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                       const uint8_t* acts_base, float* d_ff, cudaStream_t stream) {
    if (njobs <= 0) return;
    // SPLIT sweep re-run on the v2 kernels (2026-10-08, warm-clock protocol, njobs 5/6/12): the current
    // geometry is the measured best -- gate/up SPLIT=4: 102/119/221 us vs 115/135/257 (SPLIT=8) and
    // 170/190/341 (SPLIT=2); down SPLIT=2: 58/68/127 us vs 75/88/167 (SPLIT=4).  The kernels move
    // ~52 GB/s of the M10's 75.7 GB/s streaming peak (microbench), not the "~90% of theoretical" an
    // earlier note claimed.  STRATA_S2_SPLIT_GU / STRATA_S2_SPLIT_D re-run the sweep without edits.
    constexpr int SPLIT = 4;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * FF;
    if (s2_v2_enabled()) {   // STRATA_S2_V2=1: acts staged in smem, 128-bit LDS, 4 independent vmad chains
        const int sp = s2_split_gup();
        if (sp == 2) {
            expert_s2_stage1_batch_v2<2><<<(rows + 64 - 1) / 64, 128, 0, stream>>>(
                d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
        } else if (sp == 8) {
            expert_s2_stage1_batch_v2<8><<<(rows + 16 - 1) / 16, 128, 0, stream>>>(
                d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
        } else {
            expert_s2_stage1_batch_v2<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
                d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
        }
        return;
    }
    expert_s2_stage1_batch<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
}

void s2_expert_maxwell_down_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                        const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                        float* d_out, size_t blob_stride, size_t d_codes_off,
                                        size_t d_scales_off, cudaStream_t stream,
                                        volatile uint32_t* mflag, const uint32_t ring, uint32_t* ctr,
                                        int blk_base, int blks, size_t d_row_stride, size_t d_scale_stride) {
    if (njobs <= 0) return;
    // SPLIT_MODE_TENSOR: a sliced call (blks != SC_D) is a v2 feature (the smem-staged act + the E1 tail);
    // the v1 fallback answers full-range calls only, which is all the non-TP paths issue.
    if (blks != SC_D && !s2_v2_enabled()) {
        std::fprintf(stderr, "strata maxwell: a tensor-split down needs the v2 kernels (STRATA_S2_V2)\n");
        std::exit(1);
    }
    constexpr int SPLIT = 2;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * H;
    if (s2_v2_enabled()) {
        const int sp = s2_split_down();
        if (sp == 1) {
            expert_s2_stage2_batch_v2<1><<<(rows + 127) / 128, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr, blk_base, blks, d_row_stride, d_scale_stride);
        } else if (sp == 4) {
            expert_s2_stage2_batch_v2<4><<<(rows + 32 - 1) / 32, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr, blk_base, blks, d_row_stride, d_scale_stride);
        } else if (sp == 8) {
            expert_s2_stage2_batch_v2<8><<<(rows + 16 - 1) / 16, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr, blk_base, blks, d_row_stride, d_scale_stride);
        } else {
            expert_s2_stage2_batch_v2<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr, blk_base, blks, d_row_stride, d_scale_stride);
        }
        return;
    }
    (void) mflag; (void) ring; (void) ctr;   // the v1 path has no completion tail
    expert_s2_stage2_batch<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
        d_faq, d_fasc, d_fahx, d_out);
}

// STRATA_MW_MT down launcher (the prefill arm): one launch for the whole run table the gup arm built
// and left in the staging.  Same geometry contract as s2_expert_maxwell_down_batch_async (per-layer
// offsets, the card's block range).  E_MAX MUST match the gup MT launcher's and the host's run-width
// cap (all three are 4 today).
void s2_expert_maxwell_down_mt_async(const uint8_t* d_blobs, size_t blob_stride, size_t d_codes_off,
                                     size_t d_scales_off, const int32_t* d_runs, const int32_t* d_entries,
                                     int nruns, const int32_t* d_jobs,
                                     const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                     float* d_out, cudaStream_t stream,
                                     int blk_base, int blks, size_t d_row_stride, size_t d_scale_stride) {
    if (nruns <= 0) return;
    // The runs table is a SHARED contract with the gup: the host builds runs of width <= 4
    // (E_MAX=4 everywhere - the E=2/8 forms measured negative and did not port to 0.2.6).
    constexpr int SPLIT = 2, THREADS = 128;
    constexpr int PER_BLOCK = THREADS / SPLIT;
    const int blocks = nruns * (H / PER_BLOCK);
    expert_s2_stage2_mt_batch<SPLIT, THREADS, 4><<<blocks, THREADS, 0, stream>>>(
        d_blobs, blob_stride, d_codes_off, d_scales_off, d_runs, d_entries, (const MaxJob*) d_jobs,
        d_faq, d_fasc, d_fahx, d_out, blk_base, blks, d_row_stride, d_scale_stride);
}

// v2 ONLY: the down stage with fp16 rows - half the uplink bytes when the destination is mapped host
// memory.  d_out16 holds H halves per entry; the verifier converts them back during its row copy.  v1 has
// no fp16 variant, so the caller must not request it with STRATA_S2_V2=0.
void s2_expert_maxwell_down_batch_f16_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                            const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                            uint16_t* d_out16, size_t blob_stride, size_t d_codes_off,
                                            size_t d_scales_off, cudaStream_t stream,
                                            int blk_base, int blks, size_t d_row_stride, size_t d_scale_stride) {
    if (njobs <= 0) return;
    constexpr int SPLIT = 2;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * H;
    expert_s2_stage2_batch_v2<SPLIT, true><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
        d_faq, d_fasc, d_fahx, (float*) d_out16, nullptr, 0, nullptr,   // fp16 takes no completion tail
        blk_base, blks, d_row_stride, d_scale_stride);
}

void s2_expert_maxwell_gup_quant_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                             const uint8_t* acts_base, int8_t* d_faq, float* d_fasc,
                                             float* d_fahx, cudaStream_t stream) {
    if (njobs <= 0) return;
    constexpr int SPLIT = 4;   // the fusion requires the block to cover exactly one 32-value chunk
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * FF;
    expert_s2_stage1_batch_v2q<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_faq, d_fasc, d_fahx);
}

// STRATA_MW_IQ_MR dispatch: one template hop so the launcher's switch stays one-level.  The MR=1
// instantiation is byte-identical to the pre-MR kernel (same template args, same smem layout).
template <int TY, int SP, int THR>
static void launch_iq_mr(int mr, int blocks, cudaStream_t stream, const uint8_t* d_blobs, size_t blob_stride,
                         const MaxJob* d_jobs, int njobs, const uint8_t* acts_base, size_t o_gate, size_t o_up,
                         size_t gu_row, int8_t* d_faq, float* d_fasc, float* d_fahx, float* d_ff_dbg,
                         int row_base, int rows_slice) {
    switch (mr) {
        case 5:  expert_iq_stage1_batch_v2q<TY, SP, THR, 5><<<blocks, THR, 0, stream>>>(
                     d_blobs, blob_stride, d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                     d_faq, d_fasc, d_fahx, d_ff_dbg, row_base, rows_slice); break;
        case 4:  expert_iq_stage1_batch_v2q<TY, SP, THR, 4><<<blocks, THR, 0, stream>>>(
                     d_blobs, blob_stride, d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                     d_faq, d_fasc, d_fahx, d_ff_dbg, row_base, rows_slice); break;
        case 2:  expert_iq_stage1_batch_v2q<TY, SP, THR, 2><<<blocks, THR, 0, stream>>>(
                     d_blobs, blob_stride, d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                     d_faq, d_fasc, d_fahx, d_ff_dbg, row_base, rows_slice); break;
        default: expert_iq_stage1_batch_v2q<TY, SP, THR, 1><<<blocks, THR, 0, stream>>>(
                     d_blobs, blob_stride, d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                     d_faq, d_fasc, d_fahx, d_ff_dbg, row_base, rows_slice); break;
    }
}

void iq_expert_maxwell_gup_quant_batch_async(int gu_type, const uint8_t* d_blobs, size_t blob_stride,
                                             const int32_t* d_jobs, int njobs, const uint8_t* acts_base,
                                             size_t o_gate, size_t o_up, size_t gu_row,
                                             int8_t* d_faq, float* d_fasc, float* d_fahx,
                                             float* d_ff_dbg, cudaStream_t stream,
                                             int row_base, int rows_slice) {
    if (njobs <= 0) return;
    // STRATA_MW_IQ_SPLIT: the gup geometry sweep knob (default 4, the fused design's original value;
    // 2 and 1 put more rows per block - fewer blocks, more dot work per thread).  The epilogue walks
    // PER_BLOCK/32 chunks, so every value's faq/fasc/fahx image is bit-identical (maxwell_iq_parity
    // gates each of them).
    static const int split = [] {
        const char* e = std::getenv("STRATA_MW_IQ_SPLIT");
        const int v = e != nullptr ? std::atoi(e) : 4;
        return (v == 1 || v == 2 || v == 4) ? v : 4;
    }();
    static const bool split_said = [] {
        if (split != 4)
            std::fprintf(stderr, "strata maxwell: IQ gup SPLIT %d (STRATA_MW_IQ_SPLIT; 4 is the default)\n", split);
        return true;
    }();
    (void) split_said;
    // STRATA_MW_IQ_T256 (opt-in, default off): the 256-thread occupancy arm - see the kernel's THREADS
    // note.  PER_BLOCK must still divide FF (640), so the 256-thread form takes SPLIT 2 or 4 only.
    static const bool t256 = [] {
        const char* e = std::getenv("STRATA_MW_IQ_T256");
        const bool on = e != nullptr && e[0] != '\0' && e[0] != '0';
        if (on)
            std::fprintf(stderr, "strata maxwell: IQ gup 256-thread blocks (STRATA_MW_IQ_T256)\n");
        return on;
    }();
    const bool wide = t256 && split >= 2;
    // STRATA_MW_IQ_MR (opt-in, default 1 = the shipped geometry): MR row-pairs per thread-group -
    // the staging amortises MR-fold and the act reads serve 2*MR dots; bit-exact (see the body).
    // A slice whose rows_slice is not a multiple of PER_BLOCK*MR (a tensor-mode shard, e.g. 160
    // rows with MR=4) silently falls back to MR=1: correctness never depends on the knob.
    static const int mr_want = [] {
        const char* e = std::getenv("STRATA_MW_IQ_MR");
        const int v = e != nullptr ? std::atoi(e) : 1;
        const int ok = (v == 1 || v == 2 || v == 4 || v == 5) ? v : 1;
        if (ok != 1)
            std::fprintf(stderr, "strata maxwell: IQ gup MR %d rows/group (STRATA_MW_IQ_MR; 1 is the default)\n", ok);
        return ok;
    }();
    if (!::strata::core::maxwell_tp::iq_slice_valid(row_base, rows_slice, split, wide)) {
        std::fprintf(stderr, "strata maxwell: IQ slice [%d,%d) is incompatible with SPLIT=%d THREADS=%d\n",
                     row_base, row_base + rows_slice, split, wide ? 256 : 128);
        std::exit(1);
    }
    const int pb = (wide ? 256 : 128) / split;
    const int mr = (rows_slice % (pb * mr_want) == 0) ? mr_want : 1;
    const int tot_rows = njobs * rows_slice;   // SPLIT_MODE_TENSOR: the card's slice of every job
    const int blocks = (tot_rows + pb * mr - 1) / (pb * mr);
#define STRATA_LAUNCH_IQ(TY, SP, THR)                                                              \
    launch_iq_mr<TY, SP, THR>(mr, blocks, stream, d_blobs, blob_stride, (const MaxJob*) d_jobs,    \
                              njobs, acts_base, o_gate, o_up, gu_row, d_faq, d_fasc, d_fahx,        \
                              d_ff_dbg, row_base, rows_slice)
    switch (gu_type) {
        case 22:   // IQ2_S
            if (wide) { if (split == 4) STRATA_LAUNCH_IQ(22, 4, 256); else STRATA_LAUNCH_IQ(22, 2, 256); }
            else if (split == 4) STRATA_LAUNCH_IQ(22, 4, 128);
            else if (split == 2) STRATA_LAUNCH_IQ(22, 2, 128);
            else STRATA_LAUNCH_IQ(22, 1, 128);
            break;
        case 16:   // IQ2_XXS
            if (wide) { if (split == 4) STRATA_LAUNCH_IQ(16, 4, 256); else STRATA_LAUNCH_IQ(16, 2, 256); }
            else if (split == 4) STRATA_LAUNCH_IQ(16, 4, 128);
            else if (split == 2) STRATA_LAUNCH_IQ(16, 2, 128);
            else STRATA_LAUNCH_IQ(16, 1, 128);
            break;
        case 29:   // IQ1_M
            if (wide) { if (split == 4) STRATA_LAUNCH_IQ(29, 4, 256); else STRATA_LAUNCH_IQ(29, 2, 256); }
            else if (split == 4) STRATA_LAUNCH_IQ(29, 4, 128);
            else if (split == 2) STRATA_LAUNCH_IQ(29, 2, 128);
            else STRATA_LAUNCH_IQ(29, 1, 128);
            break;
        default:
            std::fprintf(stderr, "iq_expert_maxwell_gup_quant_batch_async: ggml type %d is not an "
                                 "i-quant this tier's kernels take (22 IQ2_S, 16 IQ2_XXS, 29 IQ1_M)\n",
                         gu_type);
            std::exit(1);
    }
#undef STRATA_LAUNCH_IQ
}

// ---- STRATA_MW_MT: the multi-token gup (the prefill arm, 2026-10-23) ---------------------------
//
// The measured problem (2026-10-23 prefill anatomy): the per-entry kernel loads and unpacks each
// weight element once per (expert, TOKEN) - at prompt chunks the same expert serves several tokens,
// so the weight-side instruction stream (the SASS: ~36% unpack, ~11% math) repeats per token.  This
// kernel serves a RUN (one expert, up to E_MAX adjacent entries) per block pass: each (row, k)
// weight element is loaded, codebook-looked-up and sign-expanded ONCE into 8 int payloads, then
// dotted against EVERY entry's act - the dp4a order and the float tail per entry are verbatim the
// per-entry vecdots above, so every (expert, token) result is bit-identical.  Acts live in shared
// memory per run ([E_MAX][H/32] blocks, the same q8_1 rebuild), so the amortised weight stream is
// the only change.
namespace {

template <int TY> struct IqMt;

template <> struct IqMt<16> {   // IQ2_XXS
    struct Pack { int p[8]; int ls; float d; };
    __device__ static Pack hoist(const void* v, int kbx, int iqs, const uint8_t* sgrid) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) v + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const uint2* grid_sh = (const uint2*) sgrid;
        Pack w;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_sh[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            w.p[k0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            w.p[k0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        w.ls = aux32 >> 27 | 1;
        w.d = h2f_dev(bq2->d);
        return w;
    }
    // apply takes the act as (qs pointer, ds-derived float): the staged form passes the smem block's
    // bytes and low2f(ds); the GACT form passes the raw act bytes and the SAME f2h_rn->low2f round trip
    // computed from the float scale - identical values, so both paths share one arithmetic body.
    __device__ static float apply(const Pack& w, const uint8_t* qs, float dys) {
        int sumi = 0;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const int u0 = get_int_b4(qs, k0 + 0);
            sumi = dp4a_s32(w.p[k0 + 0], u0, sumi);
            const int u1 = get_int_b4(qs, k0 + 1);
            sumi = dp4a_s32(w.p[k0 + 1], u1, sumi);
        }
        sumi = sumi * w.ls / 8;
        const float d = w.d * dys;
        return d * sumi;
    }
    __device__ static int blk(int iqs) { return iqs / 2; }   // the q8_1 block index within the k-block
};

template <> struct IqMt<22> {   // IQ2_S
    struct Pack { int p[8]; int ls0, ls1; float d; };
    __device__ static Pack hoist(const void* v, int kbx, int iqs, const uint8_t* sgrid) {
        const block_iq2_s* bq2 = (const block_iq2_s*) v + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_sh = (const uint64_t*) sgrid;
        Pack w;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_sh + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            w.p[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            w.p[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        w.ls0 = bq2->scales[iqs / 2] & 0x0F;
        w.ls1 = bq2->scales[iqs / 2] >> 4;
        w.d = h2f_dev(bq2->d);
        return w;
    }
    __device__ static float apply(const Pack& w, const uint8_t* qs, float dys) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(qs, l0 + 0);
            const int u1 = get_int_b4(qs, l0 + 1);
            if (l0 < 4) {
                sumi0 = dp4a_s32(w.p[l0 + 0], u0, sumi0);
                sumi0 = dp4a_s32(w.p[l0 + 1], u1, sumi0);
            } else {
                sumi1 = dp4a_s32(w.p[l0 + 0], u0, sumi1);
                sumi1 = dp4a_s32(w.p[l0 + 1], u1, sumi1);
            }
        }
        const int sumi = (sumi0 * w.ls0 + sumi1 * w.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = w.d * dys;
        return d * sumi;
    }
    __device__ static int blk(int iqs) { return iqs / 2; }
};

template <> struct IqMt<29> {   // IQ1_M
    struct Pack { int p[8]; float delta[2], d; int sc0, sc1; };
    __device__ static Pack hoist(const void* v, int kbx, int iqs, const uint8_t* sgrid) {
        const block_iq1_m* bq1 = (const block_iq1_m*) v + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_sh = (const uint32_t*) sgrid;
        Pack w;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_sh[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            w.p[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            w.p[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            w.delta[l0 / 4] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        w.d = h2f_dev(scale.f16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        w.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        w.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return w;
    }
    __device__ static float apply(const Pack& w, const uint8_t* qs, float dys) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(qs, l0 + 0);
            const int u1 = get_int_b4(qs, l0 + 1);
            sumi[l0 / 4] = dp4a_s32(w.p[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = dp4a_s32(w.p[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = dp4a_s32(u0, 0x01010101, sumy);
            sumy = dp4a_s32(u1, 0x01010101, sumy);
            sumf[l0 / 4] += w.delta[l0 / 4] * sumy;
        }
        const float d = w.d * dys;
        return d * ((sumi[0] + sumf[0]) * w.sc0 + (sumi[1] + sumf[1]) * w.sc1);
    }
    __device__ static int blk(int iqs) { return iqs; }
};

// The MT kernel's geometry: one run (one expert, E <= E_MAX adjacent entries) x PER_BLOCK rows per
// block; the standard job table is NOT consumed here (the down still uses it downstream).
template <int TY, int SPLIT = 4, int THREADS = 128, int E_MAX = 8>
__global__ void expert_iq_stage1_mt_batch(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                          const int32_t* __restrict__ runs,      // [nruns x 4]: slot, count, entries_off, unused
                                          const int32_t* __restrict__ entries,   // [total x 2]: tok, entry (the global entry index)
                                          const uint8_t* __restrict__ acts_base,
                                          size_t o_gate, size_t o_up, size_t gu_row,
                                          int8_t* __restrict__ faq, float* __restrict__ fasc,
                                          float* __restrict__ fahx, float* __restrict__ ff_dbg,
                                          int row_base, int rows_slice) {
    using F = IqFmt<TY>;
    constexpr int PER_BLOCK = THREADS / SPLIT;
    constexpr int ACTS_B = H / 32;
    static_assert(FF % PER_BLOCK == 0, "the MT gup stages one run's rows per block, as the S2 one");
    __shared__ __align__(16) block_q8_1 sact[E_MAX * ACTS_B];
    __shared__ __align__(16) float sx[E_MAX * PER_BLOCK];
    __shared__ __align__(16) uint8_t sgrid[F::grid_bytes];
    const int rpb = FF / PER_BLOCK;                     // row-blocks per run
    const int run = (int) (blockIdx.x / rpb);
    const int vb = (int) (blockIdx.x % rpb);
    const int slot = runs[run * 4 + 0];
    const int E = runs[run * 4 + 1];
    const int ent0 = runs[run * 4 + 2];
    if (E <= 0 || E > E_MAX || vb * PER_BLOCK >= rows_slice) return;
    const uint8_t* blob = blobs + (size_t) slot * blob_stride;
    {
        const int4* g = (const int4*) F::grid_dev();
        int4* d = (int4*) sgrid;
#pragma unroll
        for (int i = (int) threadIdx.x; i < F::grid_bytes / 16; i += THREADS) d[i] = g[i];
        const int total = E * ACTS_B;
        for (int i = (int) threadIdx.x; i < total; i += THREADS) {
            const int e = i / ACTS_B, k = i % ACTS_B;
            const int32_t tok = entries[(ent0 + e) * 2 + 0];
            const uint8_t* at = acts_base + (size_t) tok * ACTS_STRIDE;
            const int* src = (const int*) at;
            const float* sc = (const float*) (at + H);
            int* dst = (int*) &sact[e * ACTS_B + k].qs[0];
#pragma unroll
            for (int w = 0; w < 8; ++w) dst[w] = src[k * 8 + w];
            float dd = sc[k];
            dd = fabsf(dd) > 65504.f ? copysignf(65504.f, dd) : dd;
            uint16_t* ds = (uint16_t*) &sact[e * ACTS_B + k].ds;
            ds[0] = f2h_rn(dd);
            ds[1] = 0;
        }
    }
    __syncthreads();
    const int row0 = vb * PER_BLOCK;
    const int row_l = row0 + (int) (threadIdx.x / SPLIT);   // this card's LOCAL row (the blob is the compact shard)
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* gr = blob + o_gate + (size_t) row_l * gu_row;
    const uint8_t* ur = blob + o_up + (size_t) row_l * gu_row;
    const int nb = H / F::qk;
    float accg[E_MAX], accu[E_MAX];
#pragma unroll
    for (int e = 0; e < E_MAX; ++e) { accg[e] = 0.f; accu[e] = 0.f; }
    for (int k = part; k < nb * F::ipb; k += SPLIT) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        // ONE hoist per side per k: the weight codes load + codebook + sign work, amortised over E
        const auto wg = IqMt<TY>::hoist(gr, kbx, iqs, sgrid);
        const auto wu = IqMt<TY>::hoist(ur, kbx, iqs, sgrid);
        const block_q8_1* sab = sact + kbx * (F::qk / 32);
#pragma unroll
        for (int e = 0; e < E_MAX; ++e) {
            if (e >= E) break;
            const block_q8_1* sa = sab + e * ACTS_B + IqMt<TY>::blk(iqs);
            const float dys = low2f_dev(sa->ds);
            accg[e] += IqMt<TY>::apply(wg, (const uint8_t*) sa->qs, dys);
            accu[e] += IqMt<TY>::apply(wu, (const uint8_t*) sa->qs, dys);
        }
    }
    {
        const int lane = (int) (threadIdx.x & 31);
        const unsigned full = 0xFFFFFFFFu;
#pragma unroll
        for (int e = 0; e < E_MAX; ++e) {
            if (e >= E) break;
            float ag = accg[e], au = accu[e];
#pragma unroll
            for (int s = 1; s < SPLIT; ++s) {
                ag += __shfl_sync(full, accg[e], (lane + s) & 31);
                au += __shfl_sync(full, accu[e], (lane + s) & 31);
            }
            if (part == 0) {
                const float sxv = (ag / (1.f + __expf(-ag))) * au;
                sx[e * PER_BLOCK + threadIdx.x / SPLIT] = sxv;
                if (ff_dbg != nullptr) {
                    const int entry = entries[(ent0 + e) * 2 + 1];
                    ff_dbg[(size_t) entry * FF + (size_t) (row_base + row_l)] = sxv;
                }
            }
        }
    }
    __syncthreads();
    // the quant epilogue: one warp per 32-value chunk of the block's rows (PER_BLOCK/32 chunks - one
    // at the 128x4/256x8 geometries, two at 256x4).  The shuffle tree and rounding are verbatim the
    // S2 quant kernel's, so faq/fasc/fahx stay bit-identical whichever geometry launched.
    static_assert(PER_BLOCK % 32 == 0, "the MT gup quant epilogue covers whole 32-value chunks");
    if (threadIdx.x < 32 * (PER_BLOCK / 32)) {
        const int cw = (int) (threadIdx.x / 32);   // this warp's chunk within the block
        const int lane = (int) (threadIdx.x % 32);
#pragma unroll
        for (int e = 0; e < E_MAX; ++e) {
            if (e >= E) break;
            const int entry = entries[(ent0 + e) * 2 + 1];
            const int chunk0 = (int) ((size_t) entry * (FF / 32) + (size_t) (row_base + (row0 + cw * 32) % rows_slice) / 32);
            const float v = sx[e * PER_BLOCK + cw * 32 + lane];
            float av = fabsf(v);
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) av = fmaxf(av, __shfl_down(av, s));
            const float amax = __shfl(av, 0);
            const float scc = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = scc > 0.f ? 1.f / scc : 0.f;
            const float t = __fmul_rn(v, inv);
            const float r = __fadd_rn(t, (t >= 0.f) ? 0.5f : -0.5f);
            int qi = __float2int_rz(r);
            qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
            faq[(size_t) chunk0 * 32 + lane] = (int8_t) qi;
            int ssum = qi;
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) ssum += __shfl_down(ssum, s);
            if (lane == 0) {
                fasc[chunk0] = scc;
                fahx[chunk0] = __fmul_rn(scc, (float) ssum);
            }
        }
    }
}

}  // namespace

// The MT launcher: one launch per layer for the whole run table (the down still consumes the
// standard job table, unchanged).
void iq_expert_maxwell_gup_quant_mt_async(int gu_type, const uint8_t* d_blobs, size_t blob_stride,
                                          const int32_t* d_runs, const int32_t* d_entries, int nruns,
                                          const uint8_t* acts_base, size_t o_gate, size_t o_up, size_t gu_row,
                                          int8_t* d_faq, float* d_fasc, float* d_fahx, float* d_ff_dbg,
                                          cudaStream_t stream, int row_base, int rows_slice) {
    if (nruns <= 0) return;
    // STRATA_MW_MT_T (geometry, from the 2026-10-26 campaign): "256x4" (eight warps, 64 rows/block,
    // arithmetic identical to 128x4 - the shipped form); anything else = "128x4" (four warps,
    // 32 rows/block).  The other measured geometries (256x8, the acts-in-global GACT form, and the
    // E=2/E=8 run-width variants) measured negative and were not carried into the 0.2.6 port.
    static const bool g256 = [] {
        const char* e = std::getenv("STRATA_MW_MT_T");
        return e != nullptr && strncmp(e, "256x4", 5) == 0;
    }();
    const int thr = g256 ? 256 : 128;
    const int per_block = g256 ? 64 : 32;
    const int blocks = nruns * (FF / per_block);
#define STRATA_LAUNCH_MT_G(TY, SP, THR)                                                        \
    expert_iq_stage1_mt_batch<TY, SP, THR, 4><<<blocks, THR, 0, stream>>>(                      \
        d_blobs, blob_stride, d_runs, d_entries, acts_base, o_gate, o_up, gu_row,               \
        d_faq, d_fasc, d_fahx, d_ff_dbg, row_base, rows_slice)
    switch (gu_type) {
        case 22:   // IQ2_S
            if (g256) STRATA_LAUNCH_MT_G(22, 4, 256);
            else STRATA_LAUNCH_MT_G(22, 4, 128);
            break;
        case 16:   // IQ2_XXS
            if (g256) STRATA_LAUNCH_MT_G(16, 4, 256);
            else STRATA_LAUNCH_MT_G(16, 4, 128);
            break;
        case 29:   // IQ1_M
            if (g256) STRATA_LAUNCH_MT_G(29, 4, 256);
            else STRATA_LAUNCH_MT_G(29, 4, 128);
            break;
        default:
            std::fprintf(stderr, "iq_expert_maxwell_gup_quant_mt_async: ggml type %d is not an "
                                 "i-quant this tier's kernels take\n", gu_type);
            std::exit(1);
    }
#undef STRATA_LAUNCH_MT_G
}

// STRATA_MW_PERSIST: launch the resident kernel.  The grid is the kernel's OWN occupancy x SMs -
// never more - so the in-kernel generation barrier is deadlock-free by construction.  The ctl is
// zeroed in-stream first (service 0, barrier 0/0).  Returns the block count, or 0 with err set.
int maxwell_persist_launch(const PersistParams* d_params, PersistCtl* d_ctl, cudaStream_t stream,
                           std::string& err) {
    using namespace strata::kernels::cuda;
    if (cudaMemsetAsync((void*) d_ctl, 0, sizeof(PersistCtl), stream) != cudaSuccess) {
        err = "persist: ctl memset failed";
        return 0;
    }
    int per_sm = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, maxwell_persist_kernel<4, 128>, 128, 0);
    if (e != cudaSuccess || per_sm <= 0) {
        err = std::string("persist: occupancy query failed: ") + cudaGetErrorString(e ? e : cudaErrorInvalidValue);
        return 0;
    }
    int sms = 0;
    int dev = -1;   // the CURRENT device (the caller scoped it): ordinal 0 is the 3060 elsewhere
    e = cudaGetDevice(&dev);
    if (e == cudaSuccess)
        e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (e != cudaSuccess || sms <= 0) {
        err = std::string("persist: SM count query failed: ") + cudaGetErrorString(e ? e : cudaErrorInvalidValue);
        return 0;
    }
    const int blocks = per_sm * sms;
    maxwell_persist_kernel<4, 128><<<blocks, 128, 0, stream>>>(d_params, d_ctl);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("persist: launch failed: ") + cudaGetErrorString(e);
        return 0;
    }
    return blocks;
}

void s2_expert_maxwell_down(const uint8_t* d_blob, const int8_t* d_aq, const float* d_ascale,
                            const float* d_ahx, float* d_out) {
    const int threads = 128;
    expert_s2_stage2<<<(H + threads - 1) / threads, threads>>>(d_blob, d_aq, d_ascale, d_ahx, d_out);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_expert_maxwell_down: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels::cuda
