// vmad_ffma_probe - settle the dp4a-form question for the M10's gup inner loop with a number.
//
// The web research (llama.cpp family analysis) claimed FFMA + byte-to-float beats "emulated dp4a"
// 3-4x - but that assumed the SCALAR fallback (c + a0*b0 + ...).  Our dp4a_s32 is 4x vmad.s32.s32.s32
// PTX (single-issue video ops).  This probe measures both forms at the apply loop's real shape:
// per (k, token): 8 dp4a-style 4-byte dots (32 vmad) vs the float path (per byte: 2-op mantissa
// expand of the weight byte + 2-op expand of the act byte + FFMA), plus the integer combine.
//
// Build:  nvcc -O2 -arch=sm_50 -o /tmp/opencode/vmad_ffma_probe tools/vmad_ffma_probe.cu
// Run:    /tmp/opencode/vmad_ffma_probe

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

__device__ __forceinline__ int dp4a_vmad(const int a, const int b, const int c) {
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
}

// the E-amortized shape of the real gup: one hoisted weight pack, E tokens' q8 acts
template <int E>
__global__ void k_vmad(const int* __restrict__ wp, const int* __restrict__ acts,
                       float* __restrict__ out, int n_k, int iters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) { out[tid] = 0.f; return; }   // one worker measures the stream cost; keep it simple
    const int w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
    const int w4 = wp[4], w5 = wp[5], w6 = wp[6], w7 = wp[7];
    long long acc = 0;
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < n_k; ++k) {
            const int a0 = acts[(k * E + 0) & 8191];
            int s = dp4a_vmad(w0, a0, 0);
            s = dp4a_vmad(w1, a0, s);
            s = dp4a_vmad(w2, a0, s);
            s = dp4a_vmad(w3, a0, s);
            s = dp4a_vmad(w4, a0, s);
            s = dp4a_vmad(w5, a0, s);
            s = dp4a_vmad(w6, a0, s);
            s = dp4a_vmad(w7, a0, s);
            acc += s;
        }
    }
    out[tid] = (float) (acc & 0xFFF);
}

__device__ __forceinline__ float b2f(unsigned int b) {   // byte (0..255) -> float, 2 int ops + 1 add
    return __int_as_float(0x4B000000 | b) - 8388608.0f;
}

template <int E>
__global__ void k_ffma(const int* __restrict__ wp, const int* __restrict__ acts,
                       float* __restrict__ out, int n_k, int iters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) { out[tid] = 0.f; return; }
    const int w[8] = { wp[0], wp[1], wp[2], wp[3], wp[4], wp[5], wp[6], wp[7] };
    float facc = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < n_k; ++k) {
            const int a0 = acts[(k * E + 0) & 8191];
            const unsigned char* a8 = (const unsigned char*) &a0;
            for (int j = 0; j < 8; ++j) {
                const unsigned char* w8 = (const unsigned char*) &w[j];
                float s = 0.f;
                s = fmaf(b2f((unsigned) w8[0]), b2f((unsigned) a8[0]), s);
                s = fmaf(b2f((unsigned) w8[1]), b2f((unsigned) a8[1]), s);
                s = fmaf(b2f((unsigned) w8[2]), b2f((unsigned) a8[2]), s);
                s = fmaf(b2f((unsigned) w8[3]), b2f((unsigned) a8[3]), s);
                facc += s;
            }
        }
    }
    out[tid] = facc;
}

// throughput form: EVERY thread works (the issue-rate truth, not one warp's latency)
template <int E>
__global__ void k_vmad_thr(const int* __restrict__ wp, const int* __restrict__ acts,
                           float* __restrict__ out, int n_k, int iters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int w0 = wp[tid & 7], w1 = wp[(tid + 1) & 7], w2 = wp[(tid + 2) & 7], w3 = wp[(tid + 3) & 7];
    const int w4 = wp[(tid + 4) & 7], w5 = wp[(tid + 5) & 7], w6 = wp[(tid + 6) & 7], w7 = wp[(tid + 7) & 7];
    long long acc = 0;
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < n_k; ++k) {
            const int a0 = acts[(k * E + tid) & 8191];
            int s = dp4a_vmad(w0, a0, 0);
            s = dp4a_vmad(w1, a0, s);
            s = dp4a_vmad(w2, a0, s);
            s = dp4a_vmad(w3, a0, s);
            s = dp4a_vmad(w4, a0, s);
            s = dp4a_vmad(w5, a0, s);
            s = dp4a_vmad(w6, a0, s);
            s = dp4a_vmad(w7, a0, s);
            acc += s;
        }
    }
    out[tid] = (float) (acc & 0xFFF);
}

template <int E>
__global__ void k_ffma_thr(const int* __restrict__ wp, const int* __restrict__ acts,
                           float* __restrict__ out, int n_k, int iters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int w[8] = { wp[tid & 7], wp[(tid + 1) & 7], wp[(tid + 2) & 7], wp[(tid + 3) & 7],
                       wp[(tid + 4) & 7], wp[(tid + 5) & 7], wp[(tid + 6) & 7], wp[(tid + 7) & 7] };
    float facc = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < n_k; ++k) {
            const int a0 = acts[(k * E + tid) & 8191];
            const unsigned char* a8 = (const unsigned char*) &a0;
            for (int j = 0; j < 8; ++j) {
                const unsigned char* w8 = (const unsigned char*) &w[j];
                float s = 0.f;
                s = fmaf(b2f((unsigned) w8[0]), b2f((unsigned) a8[0]), s);
                s = fmaf(b2f((unsigned) w8[1]), b2f((unsigned) a8[1]), s);
                s = fmaf(b2f((unsigned) w8[2]), b2f((unsigned) a8[2]), s);
                s = fmaf(b2f((unsigned) w8[3]), b2f((unsigned) a8[3]), s);
                facc += s;
            }
        }
    }
    out[tid] = facc;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    CK(cudaSetDevice(1));   // the first M10
    int *wp, *acts; float* out;
    CK(cudaMalloc(&wp, 32));  CK(cudaMalloc(&acts, 32768 * 4));  CK(cudaMalloc(&out, 256 * 4));
    CK(cudaMemset(acts, 0x21, 32768 * 4));
    int h[8] = { 0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C, 0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C };
    CK(cudaMemcpy(wp, h, 32, cudaMemcpyHostToDevice));
    const int n_k = 1024, iters = 5000;
    // throughput form: 5 SMs x 8 blocks x 256 threads = full die
    const int grid = 40, block = 256;
    float res;
    double t0 = now_ms();
    k_vmad_thr<1><<<grid, block>>>(wp, acts, out, n_k, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&res, out, 4, cudaMemcpyDeviceToHost));
    const double tv = now_ms() - t0;
    t0 = now_ms();
    k_ffma_thr<1><<<grid, block>>>(wp, acts, out, n_k, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&res, out, 4, cudaMemcpyDeviceToHost));
    const double tf = now_ms() - t0;
    const double macs = (double) grid * block * n_k * iters * 32.0;
    std::printf("throughput (40 blocks x 256 thr, %d k x %d iters x 32 MACs):\n", n_k, iters);
    std::printf("  vmad form: %8.1f ms   -> %7.1f G MAC/s\n", tv, macs / tv / 1e6);
    std::printf("  ffma form: %8.1f ms   -> %7.1f G MAC/s\n", tf, macs / tf / 1e6);
    std::printf("  ratio ffma/vmad: %.2fx   (res %.1f)\n", tv / tf, res);
    return 0;
}
