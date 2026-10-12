// src/kernels/cuda/iq_gup_bench.cpp - standalone timing harness for the Maxwell IQ gup kernel.
//
// WHY: the engine profiles (nsys, 2026-10-12) put the IQ gup at 81% of the M10 chain's busy time
// (~375 us/launch at ~6.6 jobs, ~21 GB/s against the ~83 GB/s practical ceiling).  This bench runs
// THE PRODUCTION KERNEL (iq_expert_maxwell_gup_quant_batch_async, STRATA_MW_IQ_SPLIT honoured) on
// REAL GGUF rows at decode-like job counts, so kernel changes iterate in seconds instead of the
// ~8-minute engine reload.  It measures ONLY: correctness stays maxwell_iq_parity's job.
//
// Burn first (the M10 idles at P8/135 MHz), then cudaEvent-time a loop of launches.
//
// Usage: iq_gup_bench <shard1.gguf> <layer> [njobs=6] [iters=300]
#include "strata/kernels/cuda/expert_maxwell.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace geo = strata::kernels::cuda::expert_maxwell_geo;
namespace cpu = strata::kernels::cpu;
using geo::H;
using geo::FF;

int main(int argc, char** argv) {
    ggml_cpu_init();
    if (argc < 3) {
        std::fprintf(stderr, "usage: iq_gup_bench <shard1.gguf> <layer> [njobs=6] [iters=300]\n");
        return 2;
    }
    const int layer = std::atoi(argv[2]);
    const int njobs = argc > 3 ? std::atoi(argv[3]) : 6;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 300;

    int m10 = -1, n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { std::fprintf(stderr, "no CUDA\n"); return 2; }
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) == cudaSuccess && p.major == 5) { m10 = d; break; }
    }
    if (m10 < 0) { std::fprintf(stderr, "iq_gup_bench: no cc 5.0 device\n"); return 2; }
    cudaSetDevice(m10);
    cudaFree(0);

    const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
    const strata::TensorInfo* tg = nullptr, * tu = nullptr;
    size_t at = 0;
    tg = model.find("blk." + std::to_string(layer) + ".ffn_gate_exps.weight", &at);
    const uint8_t* dg = tg ? model.shard(at).tensor_data(*tg) : nullptr;
    at = 0;
    tu = model.find("blk." + std::to_string(layer) + ".ffn_up_exps.weight", &at);
    const uint8_t* du = tu ? model.shard(at).tensor_data(*tu) : nullptr;
    if (!tg || !tu || !dg || !du) { std::fprintf(stderr, "layer %d: no gate/up tensors\n", layer); return 2; }
    const int gu_type = (int) tg->type;
    if (gu_type != 22 && gu_type != 16 && gu_type != 29) {
        std::fprintf(stderr, "layer %d: gate/up type %d not IQ2_S/IQ2_XXS/IQ1_M\n", layer, gu_type);
        return 2;
    }

    // blob layout per expert, as the tier assembles: [gate | up]  (the gup never reads the down)
    const size_t gu_row = 0;   // unused placeholder (f.up_off carries the layout)
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, 42, H, FF, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const size_t part = f.up_off;   // bytes of one role's rows for one expert

    const int experts_in_file = (int) (tg->elements() / ((uint64_t) H * (uint64_t) FF));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fs(0.25f, 2.5f);
    std::uniform_int_distribution<int> ib(-128, 127);

    // njobs blobs, each a different expert; jobs point slot i, token i
    std::vector<uint8_t> blobs((size_t) njobs * f.bytes);
    std::vector<int> picks;
    for (int j = 0; j < njobs; ++j) {
        const int e = (int) (rng() % (experts_in_file > 0 ? experts_in_file : 128));
        picks.push_back(e);
        std::memcpy(blobs.data() + (size_t) j * f.bytes, dg + (size_t) e * part, part);
        std::memcpy(blobs.data() + (size_t) j * f.bytes + part, du + (size_t) e * part, part);
    }
    // one ActQ per token (synthetic, timing only)
    const int as = (int) geo::ACTS_STRIDE;
    std::vector<uint8_t> acts((size_t) njobs * as);
    for (int c = 0; c < njobs; ++c) {
        uint8_t* at2 = acts.data() + (size_t) c * as;
        for (int i = 0; i < H; ++i) at2[i] = (uint8_t) ib(rng);
        float* sc = (float*) (at2 + H);
        for (int i = 0; i < H / 32; ++i) sc[i] = fs(rng);
        float* hx = (float*) (at2 + H + (size_t) (H / 32) * 4);
        for (int i = 0; i < H / 32; ++i) hx[i] = 0.f;
    }
    std::vector<int32_t> jobs((size_t) njobs * 3);
    for (int j = 0; j < njobs; ++j) { jobs[j * 3] = j; jobs[j * 3 + 1] = j; jobs[j * 3 + 2] = 0; }

    uint8_t *d_blobs = nullptr, *d_acts = nullptr;
    int32_t* d_jobs = nullptr;
    int8_t* d_faq = nullptr;
    float *d_fasc = nullptr, *d_fahx = nullptr, *d_ff = nullptr;
    cudaMalloc(&d_blobs, blobs.size());
    cudaMalloc(&d_acts, acts.size());
    cudaMalloc(&d_jobs, jobs.size() * 4);
    cudaMalloc(&d_faq, (size_t) njobs * FF);
    cudaMalloc(&d_fasc, (size_t) njobs * (FF / 32) * 4);
    cudaMalloc(&d_fahx, (size_t) njobs * (FF / 32) * 4);
    cudaMalloc(&d_ff, (size_t) njobs * FF * 4);
    cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acts, acts.data(), acts.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

    using strata::kernels::cuda::iq_expert_maxwell_gup_quant_batch_async;
    auto launch = [&]() {
        iq_expert_maxwell_gup_quant_batch_async(gu_type, d_blobs, f.bytes, d_jobs, njobs, d_acts,
                                                0, part, f.gu_row, d_faq, d_fasc, d_fahx, d_ff, st);
    };

    // burn ~2 s: the M10 sits at P8/135 MHz from idle
    const auto t0 = std::chrono::steady_clock::now();
    do { launch(); } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0);
    cudaStreamSynchronize(st);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e1, st);
    cudaStreamSynchronize(st);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double per_us = (double) ms * 1000.0 / iters;
    const double bytes = (double) njobs * 2.0 * (double) part;   // gate+up read per launch
    const double gbps = bytes / (per_us * 1000.0);
    const char* tn = gu_type == 22 ? "IQ2_S" : gu_type == 16 ? "IQ2_XXS" : "IQ1_M";
    std::printf("iq_gup_bench: layer %d (%s), %d jobs, %d iters: %.1f us/launch, %.1f GB/s "
                "(gate+up %.2f MB/launch)\n",
                layer, tn, njobs, iters, per_us, gbps, bytes / 1048576.0);
    (void) gu_row;
    return 0;
}
