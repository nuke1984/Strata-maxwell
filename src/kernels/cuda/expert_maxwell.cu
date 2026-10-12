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

template <int SPLIT, bool F16 = false>
__global__ void expert_s2_stage2_batch_v2(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                          size_t d_codes_off, size_t d_scales_off,
                                          const MaxJob* __restrict__ jobs,
                                          int njobs, const int8_t* __restrict__ faq,
                                          const float* __restrict__ fasc, const float* __restrict__ fahx,
                                          float* __restrict__ out,
                                          volatile uint32_t* __restrict__ mflag, const uint32_t ring,
                                          uint32_t* __restrict__ ctr) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(H % PER_BLOCK == 0, "v2 stages one job per block: H must be a multiple of PER_BLOCK");
    const int row0 = blockIdx.x * PER_BLOCK;
    if (row0 >= njobs * H) return;
    const int j = row0 / H;
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * blob_stride;
    __shared__ __align__(16) int8_t sq[FF];
    __shared__ float ssc[FF / 32];
    __shared__ float shx[FF / 32];
    __shared__ uint32_t slut[256];
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
    const uint8_t* dc = blob + d_codes_off + (size_t) row * ROW_D;
    const uint8_t* ds = blob + d_scales_off + (size_t) row * SC_D * 2;
    float acc = 0.f, corr = 0.f;
    for (int b = part; b < SC_D; b += SPLIT) {
        const float d = read_scale(ds, b);
        int d0 = 0, d1 = 0;
        block_pair_s2_lut(dc + b * 16, sq + (2 * b) * 32, slut, d0, d1);
        acc += d * ssc[2 * b] * (float) d0;
        acc += d * ssc[2 * b + 1] * (float) d1;
        corr += d * (shx[2 * b] + shx[2 * b + 1]);
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
        const float r = a0 - c0;
        if (F16) ((uint16_t*) out)[(size_t) jd.dst * H + row] = f2h_rn(r);   // half the uplink bytes
        else     out[(size_t) jd.dst * H + row] = r;
    }
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
template <int TY, int SPLIT>
__global__ void expert_iq_stage1_batch_v2q(const uint8_t* __restrict__ blobs, size_t blob_stride,
                                           const MaxJob* __restrict__ jobs, int njobs,
                                           const uint8_t* __restrict__ acts_base,
                                           size_t o_gate, size_t o_up, size_t gu_row,
                                           int8_t* __restrict__ faq, float* __restrict__ fasc,
                                           float* __restrict__ fahx, float* __restrict__ ff_dbg) {
    using F = IqFmt<TY>;
    // SPLIT is the sweep knob (STRATA_MW_IQ_SPLIT, default 4).  The fused quant epilogue below walks
    // PER_BLOCK/32 chunks, so any SPLIT that divides 32 keeps the chunk tree bit-identical: with 4 it
    // compiles to exactly the one-chunk body this kernel shipped with.
    static_assert(SPLIT == 1 || SPLIT == 2 || SPLIT == 4, "the IQ gup kernel takes SPLIT 1, 2 or 4");
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(FF % PER_BLOCK == 0, "the IQ kernel stages one job per block, as the S2 one");
    const int row0 = blockIdx.x * PER_BLOCK;
    if (row0 >= njobs * FF) return;
    const int j = row0 / FF;                 // every row of this block is in job j (FF % PER_BLOCK == 0)
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * blob_stride;
    const uint8_t* at = acts_base + (size_t) jd.tok * ACTS_STRIDE;
    __shared__ __align__(16) block_q8_1 sact[H / 32];
    {
        // the job's token's activation as q8_1 blocks: ActQ.q verbatim (8 int32 per block, the act is
        // 4-aligned here), d = f2h_rn(scale) with the #606 clamp, s = 0 (unused by these dots).
        const int* src = (const int*) at;
        const float* sc = (const float*) (at + H);
        for (int k = (int) threadIdx.x; k < H / 32; k += 128) {
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
    __shared__ __align__(16) uint8_t sgrid[F::grid_bytes];
    {
        const int4* g = (const int4*) F::grid_dev();
        int4* d = (int4*) sgrid;
#pragma unroll
        for (int i = (int) threadIdx.x; i < F::grid_bytes / 16; i += 128) d[i] = g[i];
    }
    __syncthreads();
    const int idx = row0 + (int) (threadIdx.x / SPLIT);   // never out of range (FF % PER_BLOCK == 0)
    const int row = idx - j * FF;
    const int part = (int) (threadIdx.x % SPLIT);
    const uint8_t* gr = blob + o_gate + (size_t) row * gu_row;
    const uint8_t* ur = blob + o_up + (size_t) row * gu_row;
    const int nb = H / F::qk;                // 256-value weight blocks per row (10)
    float accg = 0.f, accu = 0.f;
    for (int k = part; k < nb * F::ipb; k += SPLIT) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        accg += F::dot(gr, sact + kbx * (F::qk / 32), kbx, iqs, sgrid);
        accu += F::dot(ur, sact + kbx * (F::qk / 32), kbx, iqs, sgrid);
    }
    __shared__ float sx[PER_BLOCK];
    // the SPLIT partial sums meet by warp shuffles, in the SAME sequential order the shared-buffer
    // reduction used (s = 0, 1, ... SPLIT-1), so the image stays bit-identical; a part-group never
    // crosses a warp boundary (SPLIT divides 32).  EVERY lane shuffles (a shfl under divergent
    // control is undefined); each lane of a group converges on its group's sequential sum, and only
    // the part == 0 leader's copy is consumed.  One KB of shared saved.
    {
        const int lane = (int) (threadIdx.x & 31);
        const unsigned full = 0xFFFFFFFFu;
        float ag = accg, au = accu;
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            ag += __shfl_sync(full, accg, (lane + s) & 31);
            au += __shfl_sync(full, accu, (lane + s) & 31);
        }
        if (part == 0) {
            const float sxv = (ag / (1.f + __expf(-ag))) * au;   // row-group order == chunk lane order
            sx[threadIdx.x / SPLIT] = sxv;
            if (ff_dbg != nullptr) ff_dbg[(size_t) j * FF + row] = sxv;   // the parity tool reads the pre-quant dots
        }
    }
    __syncthreads();
    // ONE warp quantises the block's chunks (PER_BLOCK/32 of them; exactly one at SPLIT=4): the exact
    // body of expert_s2_stage1_batch_v2q's epilogue (same tree, same roundings) per chunk, so
    // faq/fasc/fahx are bit-identical to the S2 path's whatever SPLIT is.
    if (threadIdx.x < 32) {
        const int lane = (int) threadIdx.x;
        constexpr int CHUNKS = PER_BLOCK / 32;
        const int chunk0 = (int) (j * (FF / 32) + (row0 % FF) / 32);
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

}  // namespace

bool s2_expert_maxwell_v2() { return s2_v2_enabled(); }

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
                                        volatile uint32_t* mflag, const uint32_t ring, uint32_t* ctr) {
    if (njobs <= 0) return;
    constexpr int SPLIT = 2;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * H;
    if (s2_v2_enabled()) {
        const int sp = s2_split_down();
        if (sp == 1) {
            expert_s2_stage2_batch_v2<1><<<(rows + 127) / 128, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr);
        } else if (sp == 4) {
            expert_s2_stage2_batch_v2<4><<<(rows + 32 - 1) / 32, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr);
        } else if (sp == 8) {
            expert_s2_stage2_batch_v2<8><<<(rows + 16 - 1) / 16, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr);
        } else {
            expert_s2_stage2_batch_v2<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
                d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
                d_faq, d_fasc, d_fahx, d_out, mflag, ring, ctr);
        }
        return;
    }
    (void) mflag; (void) ring; (void) ctr;   // the v1 path has no completion tail
    expert_s2_stage2_batch<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
        d_faq, d_fasc, d_fahx, d_out);
}

// v2 ONLY: the down stage with fp16 rows - half the uplink bytes when the destination is mapped host
// memory.  d_out16 holds H halves per entry; the verifier converts them back during its row copy.  v1 has
// no fp16 variant, so the caller must not request it with STRATA_S2_V2=0.
void s2_expert_maxwell_down_batch_f16_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                            const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                            uint16_t* d_out16, size_t blob_stride, size_t d_codes_off,
                                            size_t d_scales_off, cudaStream_t stream) {
    if (njobs <= 0) return;
    constexpr int SPLIT = 2;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * H;
    expert_s2_stage2_batch_v2<SPLIT, true><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, blob_stride, d_codes_off, d_scales_off, (const MaxJob*) d_jobs, njobs,
        d_faq, d_fasc, d_fahx, (float*) d_out16, nullptr, 0, nullptr);   // fp16 takes no completion tail
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

void iq_expert_maxwell_gup_quant_batch_async(int gu_type, const uint8_t* d_blobs, size_t blob_stride,
                                             const int32_t* d_jobs, int njobs, const uint8_t* acts_base,
                                             size_t o_gate, size_t o_up, size_t gu_row,
                                             int8_t* d_faq, float* d_fasc, float* d_fahx,
                                             float* d_ff_dbg, cudaStream_t stream) {
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
    const int rows = njobs * FF;
    const int blocks = (rows + 128 / split - 1) / (128 / split);
    switch (gu_type) {
        case 22:   // IQ2_S
            if (split == 4) expert_iq_stage1_batch_v2q<22, 4><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else if (split == 2) expert_iq_stage1_batch_v2q<22, 2><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else expert_iq_stage1_batch_v2q<22, 1><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            break;
        case 16:   // IQ2_XXS
            if (split == 4) expert_iq_stage1_batch_v2q<16, 4><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else if (split == 2) expert_iq_stage1_batch_v2q<16, 2><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else expert_iq_stage1_batch_v2q<16, 1><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            break;
        case 29:   // IQ1_M
            if (split == 4) expert_iq_stage1_batch_v2q<29, 4><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else if (split == 2) expert_iq_stage1_batch_v2q<29, 2><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            else expert_iq_stage1_batch_v2q<29, 1><<<blocks, 128, 0, stream>>>(
                d_blobs, blob_stride, (const MaxJob*) d_jobs, njobs, acts_base, o_gate, o_up, gu_row,
                d_faq, d_fasc, d_fahx, d_ff_dbg);
            break;
        default:
            std::fprintf(stderr, "iq_expert_maxwell_gup_quant_batch_async: ggml type %d is not an "
                                 "i-quant this tier's kernels take (22 IQ2_S, 16 IQ2_XXS, 29 IQ1_M)\n",
                         gu_type);
            std::exit(1);
    }
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
