// mw_dred_repro - the STRATA_MW_DRED fault's standalone repro (2026-10-26 forensics, session 4).
//
// The engine's DRED arm crashes with an illegal memory access on MULTI-WINDOW prefill prompts on this
// vGPU stack (single-window requests pass and answer correctly); compute-sanitizer cannot run the
// engine (cuBLAS init fails under memcheck).  This harness replicates the arm's exact CUDA op
// sequence - four M10 contexts (devices 1-4) each running [upload -> kernel -> kernel -> D2H ->
// event record] per window, and the primary (device 0) running the DRED arm [host event syncs ->
// CPU stage -> H2D x4 -> reduce kernel] - with tiny dummy kernels instead of the real gup/down.
// No cuBLAS, no engine: memcheck names the faulting op.
//
// Build:  nvcc -O2 -g -o /tmp/opencode/mw_dred_repro tools/mw_dred_repro.cu
// Run:    compute-sanitizer --tool memcheck /tmp/opencode/mw_dred_repro [windows]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

constexpr int N = 2048;      // the row width (n_embd)
constexpr int ROWS = 512;    // a realistic multi-window window's job count

__global__ void dummy_write(float* dst, int rows, int n, float v) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* d = dst + (size_t) r * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = v + (float) r;
}

__global__ void dred_reduce(const float* p0, const float* p1, const float* p2, const float* p3,
                            const int32_t* map, float* dst, int64_t rows, int64_t n) {
    const int64_t r = (int64_t) blockIdx.x;
    if (r >= rows) return;
    float* d = dst + (int64_t) map[r] * n;
    const float* a = p0 + r * n; const float* b = p1 + r * n;
    const float* c = p2 + r * n; const float* e = p3 + r * n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        d[i] = ((a[i] + b[i]) + c[i]) + e[i];
}

int main(int argc, char** argv) {
    const int windows = argc > 1 ? std::atoi(argv[1]) : 11;   // the 2.7k prompt = 11
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::printf("devices: %d\n", ndev);
    if (ndev < 5) { std::fprintf(stderr, "need the 3060 + 4 M10s\n"); return 1; }

    // device 0 (the 3060): the partial slots, the map, the destination
    float* d_part[4] = {};
    int32_t* d_map = nullptr;
    float* d_dst = nullptr;
    float* h_stage[4] = {};
    CK(cudaSetDevice(0));
    for (int i = 0; i < 4; ++i) {
        CK(cudaMalloc(&d_part[i], (size_t) ROWS * N * 4));
        CK(cudaHostAlloc(&h_stage[i], (size_t) ROWS * N * 4, cudaHostAllocDefault));
    }
    CK(cudaMalloc(&d_map, (size_t) ROWS * 4));
    CK(cudaMalloc(&d_dst, (size_t) ROWS * N * 4));
    cudaStream_t cs;
    CK(cudaStreamCreate(&cs));

    // each M10: the staging (device), the out buffer, its pinned h_out_ (the engine allocates it in
    // the CARD's context - plain cudaMallocHost), its D2H event
    cudaStream_t m10s[4] = {};
    float *d_acts[4] = {}, *d_out[4] = {};
    float* h_out[4] = {};
    float* h_out2[4] = {};   // DRED's alternating second buffer
    cudaEvent_t ev_d2h[4] = {}, ev_down[4] = {};
    for (int i = 0; i < 4; ++i) {
        CK(cudaSetDevice(i + 1));
        CK(cudaStreamCreate(&m10s[i]));
        CK(cudaMalloc(&d_acts[i], (size_t) ROWS * N * 4));    // the staging stand-in
        CK(cudaMalloc(&d_out[i], (size_t) ROWS * N * 4));
        CK(cudaMallocHost(&h_out[i], (size_t) ROWS * N * 4));
        CK(cudaMallocHost(&h_out2[i], (size_t) ROWS * N * 4));
        CK(cudaEventCreateWithFlags(&ev_d2h[i], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_down[i], cudaEventDisableTiming));
    }

    int32_t map_h[ROWS];
    for (int r = 0; r < ROWS; ++r) map_h[r] = r;

    for (int w = 0; w < windows; ++w) {
        // begin: upload the "acts" (device-local, harmless)
        for (int i = 0; i < 4; ++i) {
            CK(cudaSetDevice(i + 1));
            CK(cudaMemcpyAsync(d_acts[i], h_out[i], (size_t) ROWS * N * 4, cudaMemcpyHostToDevice, m10s[i]));
            dummy_write<<<ROWS, 256, 0, m10s[i]>>>(d_acts[i], ROWS, N, (float) w);          // "gup"
        }
        // launch_down: the down kernel + the AD2H D2H into the (alternating) h_out_ + the events
        for (int i = 0; i < 4; ++i) {
            CK(cudaSetDevice(i + 1));
            dummy_write<<<ROWS, 256, 0, m10s[i]>>>(d_out[i], ROWS, N, (float) w + 0.5f);    // "down"
            CK(cudaEventRecord(ev_down[i], m10s[i]));
            float* dst = (w & 1) ? h_out2[i] : h_out[i];
            CK(cudaMemcpyAsync(dst, d_out[i], (size_t) ROWS * N * 4, cudaMemcpyDeviceToHost, m10s[i]));
            CK(cudaEventRecord(ev_d2h[i], m10s[i]));
        }
        // the DRED arm (on the primary): host event syncs, pure-CPU staging, H2D x4, map, reduce
        CK(cudaSetDevice(0));
        for (int i = 0; i < 4; ++i) {
            CK(cudaEventSynchronize(ev_d2h[i]));
            const float* src = (w & 1) ? h_out2[i] : h_out[i];
            std::memcpy(h_stage[i], src, (size_t) ROWS * N * 4);
            CK(cudaMemcpyAsync(d_part[i], h_stage[i], (size_t) ROWS * N * 4, cudaMemcpyHostToDevice, cs));
        }
        CK(cudaMemcpyAsync(d_map, map_h, (size_t) ROWS * 4, cudaMemcpyHostToDevice, cs));
        dred_reduce<<<ROWS, 256, 0, cs>>>(d_part[0], d_part[1], d_part[2], d_part[3], d_map, d_dst, ROWS, N);
        CK(cudaStreamSynchronize(cs));
        std::printf("window %d/%d done\n", w + 1, windows);
    }
    std::printf("PASS: %d windows, no fault\n", windows);
    return 0;
}
