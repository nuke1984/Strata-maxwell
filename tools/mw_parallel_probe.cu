// mw_parallel_probe - are the four M10 dies executing in PARALLEL or SERIALIZED (vGPU timeslice)?
//
// The 2026-10-27 REV flip test showed the downs' execution follows the host's COMPLETION order,
// suggesting the vGPU layer may timeslice the four dies.  This probe measures it directly:
//   case 1: one die alone runs W ms of DRAM-bound work            -> wall T1
//   case 2: all four launched back-to-back, then synced in order  -> wall T4
//   T4 ~= T1   -> the dies overlap (the chain's 4x3.4ms staircase is NOT timeslicing)
//   T4 ~= 4*T1 -> the dies serialize (the chain is an irreducible sum; only overlap with the
//                 3060/host or less work can raise the rate - shapes the whole 250 plan)
//
// Build:  nvcc -O2 -o /tmp/opencode/mw_parallel_probe tools/mw_parallel_probe.cu
// Run:    /tmp/opencode/mw_parallel_probe [ms_per_die]

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

constexpr size_t BUF_MB = 256;     // per die, sized like a layer's expert blob traffic
constexpr int THREADS = 256;

__global__ void read_pass(const float* __restrict__ buf, size_t floats, int passes, float* sink) {
    float acc = 0.f;
    for (int p = 0; p < passes; ++p) {
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < floats;
             i += (size_t) gridDim.x * blockDim.x)
            acc += buf[i];
        buf = buf;   // the compiler must not hoist
    }
    if (acc == 12345.678f) *sink = acc;   // never true; keeps the reads live
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int target_ms = argc > 1 ? std::atoi(argv[1]) : 4;
    const int dev0 = 1, ndev = 4;      // the M10s
    float* buf[4] = {}; cudaStream_t st[4]; cudaEvent_t ev[4];
    const size_t floats = BUF_MB * 1024 * 1024 / 4;
    for (int d = 0; d < ndev; ++d) {
        CK(cudaSetDevice(dev0 + d));
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
        CK(cudaEventCreate(&ev[d]));
        CK(cudaMalloc(&buf[d], floats * 4));
        CK(cudaMemset(buf[d], 1, floats * 4));
    }
    // calibrate passes on die 0 to ~= target_ms
    int passes = 1;
    for (;;) {
        CK(cudaSetDevice(dev0));
        double t0 = now_ms();
        read_pass<<<5 * 6, THREADS, 0, st[0]>>>(buf[0], floats, passes, buf[0]);
        CK(cudaEventRecord(ev[0], st[0]));
        CK(cudaEventSynchronize(ev[0]));
        const double w = now_ms() - t0;
        if (w >= (double) target_ms * 0.9 || passes > 64) { std::printf("calibrated: %d passes = %.2f ms on one die\n", passes, w); break; }
        passes *= 2;
    }
    // case 1: one die, three quiet
    {
        double t0 = now_ms();
        CK(cudaSetDevice(dev0));
        read_pass<<<5 * 6, THREADS, 0, st[0]>>>(buf[0], floats, passes, buf[0]);
        CK(cudaEventRecord(ev[0], st[0]));
        CK(cudaEventSynchronize(ev[0]));
        std::printf("case1 one die alone:        %.2f ms\n", now_ms() - t0);
    }
    // warm + case 2: all four launched back-to-back (no host waits between), synced in order
    for (int rep = 0; rep < 3; ++rep) {
        double t0 = now_ms();
        double per[4] = {};
        for (int d = 0; d < ndev; ++d) {
            CK(cudaSetDevice(dev0 + d));
            read_pass<<<5 * 6, THREADS, 0, st[d]>>>(buf[d], floats, passes, buf[d]);
            CK(cudaEventRecord(ev[d], st[d]));
        }
        for (int d = 0; d < ndev; ++d) {
            CK(cudaSetDevice(dev0 + d));
            double s0 = now_ms();
            CK(cudaEventSynchronize(ev[d]));
            per[d] = now_ms() - s0;
        }
        const double wall = now_ms() - t0;
        std::printf("case2 all four, rep %d:     wall %.2f ms | sync waits %.2f/%.2f/%.2f/%.2f\n",
                    rep, wall, per[0], per[1], per[2], per[3]);
    }
    // case 3: same, but sync in REVERSE order (the REV phenomenon: does execution follow host attention?)
    {
        double t0 = now_ms();
        double per[4] = {};
        for (int d = 0; d < ndev; ++d) {
            CK(cudaSetDevice(dev0 + d));
            read_pass<<<5 * 6, THREADS, 0, st[d]>>>(buf[d], floats, passes, buf[d]);
            CK(cudaEventRecord(ev[d], st[d]));
        }
        for (int ii = ndev - 1; ii >= 0; --ii) {
            CK(cudaSetDevice(dev0 + ii));
            double s0 = now_ms();
            CK(cudaEventSynchronize(ev[ii]));
            per[ii] = now_ms() - s0;
        }
        std::printf("case3 reverse-order syncs:  wall %.2f ms | sync waits %.2f/%.2f/%.2f/%.2f\n",
                    now_ms() - t0, per[0], per[1], per[2], per[3]);
    }
    std::printf("verdict: case2 wall ~= case1 -> dies PARALLEL; ~= 4x case1 -> SERIALIZED (timeslice)\n");
    return 0;
}
