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

namespace strata::kernels::cuda {
namespace {

using namespace expert_maxwell_geo;

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
template <int SPLIT>
__global__ void expert_s2_stage2_batch(const uint8_t* __restrict__ blobs, const MaxJob* __restrict__ jobs,
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
    const uint8_t* blob = blobs + (size_t) jd.slot * BLOB;
    ActQDev a{faq + (size_t) j * FF, fasc + (size_t) j * (FF / 32), fahx + (size_t) j * (FF / 32), FF / 32};
    const uint8_t* dc = blob + O_D_CODES + (size_t) row * ROW_D;
    const uint8_t* ds = blob + O_D_SCALES + (size_t) row * SC_D * 2;
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

template <int SPLIT>
__global__ void expert_s2_stage2_batch_v2(const uint8_t* __restrict__ blobs, const MaxJob* __restrict__ jobs,
                                          int njobs, const int8_t* __restrict__ faq,
                                          const float* __restrict__ fasc, const float* __restrict__ fahx,
                                          float* __restrict__ out) {
    constexpr int PER_BLOCK = 128 / SPLIT;
    static_assert(H % PER_BLOCK == 0, "v2 stages one job per block: H must be a multiple of PER_BLOCK");
    const int row0 = blockIdx.x * PER_BLOCK;
    if (row0 >= njobs * H) return;
    const int j = row0 / H;
    const MaxJob jd = jobs[j];
    const uint8_t* blob = blobs + (size_t) jd.slot * BLOB;
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
    const uint8_t* dc = blob + O_D_CODES + (size_t) row * ROW_D;
    const uint8_t* ds = blob + O_D_SCALES + (size_t) row * SC_D * 2;
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
        out[(size_t) jd.dst * H + row] = a0 - c0;
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

}  // namespace

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
    // P3.5 (misurato sul device, griglia corretta: njobs 4/8/12/16): SPLIT=4 e' gia' al limite pratico.
    // SPLIT=8 e SPLIT=2 danno lo stesso tempo, SPLIT=16 peggiora (374 vs 271 us a njobs=12); T256 uguale a
    // T128; la mappatura a chunk consecutivi peggiora.  Insieme a SPLIT=2 del down (nessuna variante lo
    // batte), i kernel stanno al ~90% del throughput intero teorico della M10: non c'e' margine a questo livello.
    constexpr int SPLIT = 4;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * FF;
    if (s2_v2_enabled()) {   // STRATA_S2_V2=1: acts staged in smem, 128-bit LDS, 4 independent vmad chains
        expert_s2_stage1_batch_v2<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
            d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
        return;
    }
    expert_s2_stage1_batch<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, (const MaxJob*) d_jobs, njobs, acts_base, d_ff);
}

void s2_expert_maxwell_down_batch_async(const uint8_t* d_blobs, const int32_t* d_jobs, int njobs,
                                        const int8_t* d_faq, const float* d_fasc, const float* d_fahx,
                                        float* d_out, cudaStream_t stream) {
    if (njobs <= 0) return;
    constexpr int SPLIT = 2;
    constexpr int PER_BLOCK = 128 / SPLIT;
    const int rows = njobs * H;
    if (s2_v2_enabled()) {
        expert_s2_stage2_batch_v2<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
            d_blobs, (const MaxJob*) d_jobs, njobs, d_faq, d_fasc, d_fahx, d_out);
        return;
    }
    expert_s2_stage2_batch<SPLIT><<<(rows + PER_BLOCK - 1) / PER_BLOCK, 128, 0, stream>>>(
        d_blobs, (const MaxJob*) d_jobs, njobs, d_faq, d_fasc, d_fahx, d_out);
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
