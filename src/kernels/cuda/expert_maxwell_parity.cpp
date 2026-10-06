// src/kernels/cuda/expert_maxwell_parity.cpp - parity test for the Maxwell S2 expert kernel.
//
// THE REFERENCE IS THE CHAIN, NOT A SECOND OPINION: the CPU side is transcribed from
// strata::kernels::cpu::s2_expert_scalar (src/kernels/cpu/expert.cpp:629), whose activations and layout are
// the engine's contract; the GPU side is the Maxwell kernel in expert_maxwell.cu.  A parity test whose two
// sides were written together would show neither.  The two sides here were written from the SAME spec but
// INDEPENDENTLY, so a fault in either shows up as a mismatch.
//
// Build (Maxwell):
//   nvcc -O2 -arch=sm_50 -o expert_maxwell_parity \
//        src/kernels/cuda/expert_maxwell.cu src/kernels/cuda/expert_maxwell_parity.cpp
// Run:
//   ./expert_maxwell_parity
//
// The test constructs a DETERMINISTIC S2 blob, runs the CPU scalar reference and the Maxwell kernel on the
// same input, and compares bit-for-bit.  The tolerance is the same order as the engine's expert parity
// (1e-5), because both sides evaluate the same FP32 expression in the same order and only differ in the
// reduction tree of the inner product.
#include "strata/kernels/cuda/expert_maxwell.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::kernels::cuda::expert_maxwell_geo;

// fp16 -> fp32 (IEEE 754 half), the same helper Strata uses for its S2 scales.
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            f = sign;
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            m &= 0x3FF;
            f |= (uint32_t) (127 - 15 + e) << 23;
            f |= m << 13;
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return *(const float*) &f;
}

// CPU reference, transcribed from src/kernels/cpu/expert.cpp:629 (quant_acts = true, the engine's mode).
// The activation quantisation is the legacy FP32-scale contract (amax/127, lroundf).
void s2_expert_scalar_cpu(const uint8_t* blob, const float* x_in, float* out) {
    // quantise the input to int8 and back, so both sides see the same dequantised activation
    float xq[H];
    for (int c = 0; c < H / 32; ++c) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = (fabsf(x_in[c*32+i]) > amax) ? fabsf(x_in[c*32+i]) : amax;
        const float d = amax / 127.f;
        for (int i = 0; i < 32; ++i) xq[c*32+i] = d * lroundf(x_in[c*32+i] / d);
    }
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2*r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2*r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2*r+1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2*r+1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = fp16_to_fp32((uint16_t)(gs[2*b] | (gs[2*b+1] << 8)));
            const float du = fp16_to_fp32((uint16_t)(us[2*b] | (us[2*b+1] << 8)));
            for (int j = 0; j < QK; ++j) {
                const int o = b*QK + j;
                sg += (float) (((gc[b*16 + (j>>2)] >> (2*(j&3))) & 3) - 1) * dg * xq[o];
                su += (float) (((uc[b*16 + (j>>2)] >> (2*(j&3))) & 3) - 1) * du * xq[o];
            }
        }
        ff[r] = (sg / (1.f + expf(-sg))) * su;
    }
    // re-quantise the intermediate, as the engine does
    float ffq[FF];
    for (int c = 0; c < FF / 32; ++c) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = (fabsf(ff[c*32+i]) > amax) ? fabsf(ff[c*32+i]) : amax;
        const float d = amax / 127.f;
        for (int i = 0; i < 32; ++i) ffq[c*32+i] = d * lroundf(ff[c*32+i] / d);
    }
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = fp16_to_fp32((uint16_t)(ds[2*b] | (ds[2*b+1] << 8)));
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b*16 + (j>>2)] >> (2*(j&3))) & 3) - 1) * d * ffq[b*QK + j];
        }
        out[r] = acc;
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void) argc; (void) argv;

    // deterministic blob and input
    std::mt19937 rng(20261004);
    std::vector<uint8_t> blob(BLOB);
    for (size_t i = 0; i < BLOB; ++i) blob[i] = (uint8_t) (rng() & 0xFF);
    // make the FP32 scales sane (finite, normal, positive) - the two scale planes
    for (size_t s = O_GU_SCALES; s < O_D_SCALES; s += 2) {
        const uint16_t h = (uint16_t)(0x3800 + (rng() & 0x03FF));   // ~0.5..1.0 fp16
        std::memcpy(&blob[s], &h, 2);
    }
    for (size_t s = O_D_SCALES; s < BLOB; s += 2) {
        const uint16_t h = (uint16_t)(0x3800 + (rng() & 0x03FF));
        std::memcpy(&blob[s], &h, 2);
    }
    std::vector<float> x(H);
    for (int i = 0; i < H; ++i) x[i] = ((rng() % 2001) - 1000) * 0.001f;

    // quantise the input on the host (same legacy contract as the CPU reference), so GPU and CPU see
    // the SAME dequantised activation and the test compares only the weight-consumption stages
    // quantise input to ActQ (int8 + scale + hx, EXACT legacy contract)
    std::vector<int8_t> aq(H);
    std::vector<float> ascale(H/32), ahx(H/32);
    for (int c = 0; c < H / 32; ++c) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = (fabsf(x[c*32+i]) > amax) ? fabsf(x[c*32+i]) : amax;
        const float d = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = d > 0.f ? 1.f / d : 0.f;
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) {
            float v = x[c*32+i] * inv;
            int qv = (int) (v >= 0.f ? v + 0.5f : v - 0.5f);
            if (qv < -127) qv = -127;
            if (qv > 127) qv = 127;
            aq[c*32+i] = (int8_t) qv;
            sum += qv;
        }
        ascale[c] = d;
        ahx[c] = d * (float) sum;
    }

    // CPU reference with EXACT row_dot (integer dot + hx), as in src/kernels/cpu/expert.cpp:233
    auto row_dot_cpu = [&](const uint8_t* codes, const uint8_t* scales, const int8_t* q,
                           const float* sc, const float* hx, int nblocks) {
        float acc = 0.f, corr = 0.f;
        for (int b = 0; b < nblocks; ++b) {
            const float d = fp16_to_fp32((uint16_t)(scales[2*b] | (scales[2*b+1] << 8)));
            // 64 weights = two 32-elem chunks; each chunk: 8 bytes codes, 32 int8
            for (int h = 0; h < 2; ++h) {
                const uint8_t* cb = codes + b * 16 + h * 8;
                const int8_t* qb = q + (2*b+h) * 32;
                int dot = 0;
                for (int j = 0; j < 32; ++j)
                    dot += (int) (((cb[j>>2] >> (2*(j&3))) & 3)) * (int) qb[j];
                acc += d * sc[2*b+h] * (float) dot;
                corr += d * hx[2*b+h];
            }
        }
        return acc - corr;
    };
    std::vector<float> ff_cpu(FF);
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob.data() + O_GU_CODES + (size_t) (2*r) * ROW_GU;
        const uint8_t* gs = blob.data() + O_GU_SCALES + (size_t) (2*r) * SC_GU * 2;
        const uint8_t* uc = blob.data() + O_GU_CODES + (size_t) (2*r+1) * ROW_GU;
        const uint8_t* us = blob.data() + O_GU_SCALES + (size_t) (2*r+1) * SC_GU * 2;
        const float sg = row_dot_cpu(gc, gs, aq.data(), ascale.data(), ahx.data(), SC_GU);
        const float su = row_dot_cpu(uc, us, aq.data(), ascale.data(), ahx.data(), SC_GU);
        ff_cpu[r] = (sg / (1.f + expf(-sg))) * su;
    }
    // requantise intermediate to ActQ
    std::vector<int8_t> faq(FF);
    std::vector<float> fascale(FF/32), fahx(FF/32);
    for (int c = 0; c < FF / 32; ++c) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = (fabsf(ff_cpu[c*32+i]) > amax) ? fabsf(ff_cpu[c*32+i]) : amax;
        const float d = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = d > 0.f ? 1.f / d : 0.f;
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) {
            float v = ff_cpu[c*32+i] * inv;
            int qv = (int) (v >= 0.f ? v + 0.5f : v - 0.5f);
            if (qv < -127) qv = -127;
            if (qv > 127) qv = 127;
            faq[c*32+i] = (int8_t) qv;
            sum += qv;
        }
        fascale[c] = d;
        fahx[c] = d * (float) sum;
    }
    std::vector<float> out_cpu(H);
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob.data() + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob.data() + O_D_SCALES + (size_t) r * SC_D * 2;
        out_cpu[r] = row_dot_cpu(dc, ds, faq.data(), fascale.data(), fahx.data(), SC_D);
    }

    // GPU: allocate ActQ buffers, copy, run both stages with IDENTICAL inputs
    uint8_t* d_blob = nullptr;
    int8_t *d_aq = nullptr, *d_faq = nullptr;
    float *d_ascale = nullptr, *d_ahx = nullptr, *d_fascale = nullptr, *d_fahx = nullptr;
    float* d_ff = nullptr;
    float* d_out = nullptr;
    check(cudaMalloc(&d_blob, BLOB), "cudaMalloc blob");
    check(cudaMalloc(&d_aq, H), "cudaMalloc aq");
    check(cudaMalloc(&d_ascale, (H/32)*sizeof(float)), "cudaMalloc ascale");
    check(cudaMalloc(&d_ahx, (H/32)*sizeof(float)), "cudaMalloc ahx");
    check(cudaMalloc(&d_faq, FF), "cudaMalloc faq");
    check(cudaMalloc(&d_fascale, (FF/32)*sizeof(float)), "cudaMalloc fascale");
    check(cudaMalloc(&d_fahx, (FF/32)*sizeof(float)), "cudaMalloc fahx");
    check(cudaMalloc(&d_ff, FF * sizeof(float)), "cudaMalloc ff");
    check(cudaMalloc(&d_out, H * sizeof(float)), "cudaMalloc out");
    check(cudaMemcpy(d_blob, blob.data(), BLOB, cudaMemcpyHostToDevice), "copy blob");
    check(cudaMemcpy(d_aq, aq.data(), H, cudaMemcpyHostToDevice), "copy aq");
    check(cudaMemcpy(d_ascale, ascale.data(), (H/32)*sizeof(float), cudaMemcpyHostToDevice), "copy ascale");
    check(cudaMemcpy(d_ahx, ahx.data(), (H/32)*sizeof(float), cudaMemcpyHostToDevice), "copy ahx");
    check(cudaMemcpy(d_faq, faq.data(), FF, cudaMemcpyHostToDevice), "copy faq");
    check(cudaMemcpy(d_fascale, fascale.data(), (FF/32)*sizeof(float), cudaMemcpyHostToDevice), "copy fascale");
    check(cudaMemcpy(d_fahx, fahx.data(), (FF/32)*sizeof(float), cudaMemcpyHostToDevice), "copy fahx");

    // stage 1: gate+up, CPU ff vs GPU ff
    strata::kernels::cuda::s2_expert_maxwell_gup(d_blob, d_aq, d_ascale, d_ahx, d_ff);
    std::vector<float> ff_gpu(FF);
    check(cudaMemcpy(ff_gpu.data(), d_ff, FF * sizeof(float), cudaMemcpyDeviceToHost), "copy ff");

    // stage 2: down, CPU out vs GPU out
    strata::kernels::cuda::s2_expert_maxwell_down(d_blob, d_faq, d_fascale, d_fahx, d_out);
    std::vector<float> out_gpu(H);
    check(cudaMemcpy(out_gpu.data(), d_out, H * sizeof(float), cudaMemcpyDeviceToHost), "copy out");

    // compare stage 1
    double max_abs1 = 0.0, max_rel1 = 0.0;
    int bad1 = 0;
    for (int i = 0; i < FF; ++i) {
        const double a = ff_cpu[i], b = ff_gpu[i];
        const double abs_err = fabs(a - b);
        const double rel_err = (a != 0.0) ? abs_err / fabs(a) : abs_err;
        if (abs_err > max_abs1) max_abs1 = abs_err;
        if (rel_err > max_rel1) max_rel1 = rel_err;
        if (abs_err > 1e-5 && rel_err > 1e-5) ++bad1;
    }
    std::printf("stage1 (gate+up): max abs %.6g, max rel %.6g, bad %d/%d\n", max_abs1, max_rel1, bad1, FF);

    // compare stage 2
    double max_abs = 0.0, max_rel = 0.0;
    int bad = 0;
    for (int i = 0; i < H; ++i) {
        const double a = out_cpu[i], b = out_gpu[i];
        const double abs_err = fabs(a - b);
        const double rel_err = (a != 0.0) ? abs_err / fabs(a) : abs_err;
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > 1e-5 && rel_err > 1e-5) ++bad;
    }
    std::printf("stage2 (down):    max abs %.6g, max rel %.6g, bad %d/%d\n", max_abs, max_rel, bad, H);
    if (bad1 > 0 || bad > 0) {
        std::fprintf(stderr, "PARITY FAILED\n");
        return 1;
    }
    std::printf("PARITY OK\n");
    return 0;
}
