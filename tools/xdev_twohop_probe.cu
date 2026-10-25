// xdev_twohop_probe - the completes-redesign foundation measurement, in the row-split regime.
//
// The DRED-era claims ("cross-device transfers fault") came from tensor-split/never-engaged arms;
// the ONE form proven in-engine was the TWO-HOP: M10 VRAM -> pinned staging OWNED BY THE 3060
// (D2H on the M10's stream, host event sync) -> H2D on the 3060's stream.  This probe measures
// that form standalone in the current (row-split) regime: correctness (checksums, many rounds,
// changing patterns) plus per-leg timing at the prefill completes' real sizes - the numbers the
// direct-into-Dm completes variant would need.  Deliberately does NOT test the historically
// crashing forms (direct peer copy, cross-device stream waits, foreign-pinned H2D): no current
// arm needs them, and the 2026-10-25 incident taught us a faulting op can wedge a die.
//
// Build:  nvcc -O2 -arch=sm_50 -o /tmp/opencode/xdev_twohop_probe tools/xdev_twohop_probe.cu
// Run:    /tmp/opencode/xdev_twohop_probe [rounds] [rows] [n]

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 200;
    const int rows = argc > 2 ? std::atoi(argv[2]) : 320;   // ~a window's owned rows
    const int n = argc > 3 ? std::atoi(argv[3]) : 3072;     // n_embd
    const size_t bytes = (size_t) rows * n * 4;

    // device 0 = the 3060 (staging owner + destination), 1-4 = the M10 dies
    float* d_src[4] = {}; cudaStream_t s_m10[4]; cudaEvent_t ev[4];
    float* h_stage = nullptr;   // pinned, allocated in the 3060's context (the proven two-hop form)
    float* d_dst = nullptr;     // the 3060-side destination (disjoint regions per card)
    cudaStream_t s_3060;

    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&s_3060, cudaStreamNonBlocking));
    CK(cudaHostAlloc(&h_stage, bytes, cudaHostAllocDefault));
    CK(cudaMalloc(&d_dst, bytes * 4));
    for (int d = 0; d < 4; ++d) {
        CK(cudaSetDevice(1 + d));
        CK(cudaStreamCreateWithFlags(&s_m10[d], cudaStreamNonBlocking));
        CK(cudaEventCreate(&ev[d]));
        CK(cudaMalloc(&d_src[d], bytes));
    }
    std::printf("buffers: %zu B per card; %d rounds\n", bytes, rounds);

    // fill each M10 source with a per-(card, round)-distinct pattern via the copy engine
    // (H2D of a host-generated pattern; keeps the probe free of custom kernels entirely)
    std::vector<float> pat(rows * n);

    double d2h_ms = 0, h2d_ms = 0, wait_ms = 0;
    long long mism = 0;
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < 4; ++i) {
            for (size_t k = 0; k < pat.size(); ++k) pat[k] = (float) ((r * 17 + i * 7 + k) & 0xFFFF);
            CK(cudaSetDevice(1 + i));
            CK(cudaMemcpyAsync(d_src[i], pat.data(), bytes, cudaMemcpyHostToDevice, s_m10[i]));
        }
        for (int i = 0; i < 4; ++i) { CK(cudaSetDevice(1 + i)); CK(cudaStreamSynchronize(s_m10[i])); }

        // the two-hop per card: D2H into the 3060-owned pinned staging, event; host waits; H2D on the 3060
        for (int i = 0; i < 4; ++i) {
            CK(cudaSetDevice(1 + i));
            double t0 = now_ms();
            CK(cudaMemcpyAsync(h_stage, d_src[i], bytes, cudaMemcpyDeviceToHost, s_m10[i]));
            CK(cudaEventRecord(ev[i], s_m10[i]));
            double t1 = now_ms();
            CK(cudaEventSynchronize(ev[i]));
            double t2 = now_ms();
            CK(cudaSetDevice(0));
            CK(cudaMemcpyAsync(d_dst + (size_t) i * rows * n, h_stage, bytes, cudaMemcpyHostToDevice, s_3060));
            double t3 = now_ms();
            d2h_ms += t1 - t0; wait_ms += t2 - t1; h2d_ms += t3 - t2;
            if (r % 50 == 49) {   // verify: read back the card's region and checksum against the pattern
                for (size_t k = 0; k < pat.size(); ++k) pat[k] = (float) ((r * 17 + i * 7 + k) & 0xFFFF);
                std::vector<float> back(rows * n);
                CK(cudaMemcpyAsync(back.data(), d_dst + (size_t) i * rows * n, bytes, cudaMemcpyDeviceToHost, s_3060));
                CK(cudaStreamSynchronize(s_3060));
                for (size_t k = 0; k < pat.size(); ++k) if (back[k] != pat[k]) { ++mism; break; }
            }
        }
    }
    std::printf("two-hop per card per round: D2H %.3f ms | event wait %.3f ms | H2D(3060) %.3f ms\n",
                d2h_ms / (4.0 * rounds), wait_ms / (4.0 * rounds), h2d_ms / (4.0 * rounds));
    std::printf("checks: %lld mismatches over %d rounds -> %s\n", mism, rounds, mism == 0 ? "CLEAN" : "CORRUPT");
    return mism == 0 ? 0 : 2;
}
