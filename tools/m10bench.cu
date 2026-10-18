// M10 overlap-penalty microbench: is the visit-order down stagger engine-internal or platform?
//   A: one host thread launches a ~150 us kernel on 4 M10s (staggered like the engine's begin loop),
//      per-card events, syncs in visit order  - the engine's exact call pattern
//   B: same launches, but synchronized start (all launched back-to-back after a barrier)
//   C: four host threads, one per card, each launches+syncs only its own device
//   D: single card alone (baseline duration)
// Kernel: reads ~6 MB of device memory, trivial math, writes device-local - no host interaction.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

__global__ void stream_kernel(const float4* __restrict__ src, float4* __restrict__ dst, int n4, int reps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 acc = src[i];
    for (int r = 0; r < reps; ++r) {                 // ~bytes-bound: keep it memory-streaming
        const float4 v = src[(i + r * 977) & (n4 - 1)];
        acc.x += v.x * 1e-9f; acc.y += v.y * 1e-9f; acc.z += v.z * 1e-9f; acc.w += v.w * 1e-9f;
    }
    dst[i] = acc;
}

struct Card {
    int dev; float4 *a = nullptr, *b = nullptr; cudaStream_t s; cudaEvent_t e0, e1;
    float4* h_stage = nullptr;   // pinned upload staging, like the tier's acts
    float4* d_stage = nullptr;
};

// the 3060's graph-park: spin hot on a mapped host flag (no __nanosleep pre-Volta; like the fused wait).
__global__ void spin_flag(volatile uint32_t* f) {
    while (*f == 0u) { }
}

// The engine's per-layer shape: pinned H2D upload -> kernel1 (gup-like, ~3x longer) -> mid event ->
// kernel2 (down-like) -> end event; launched staggered from one thread; synced in visit order.
__global__ void stream_kernel2(const float4* __restrict__ src, const float4* __restrict__ st,
                               float4* __restrict__ dst, int n4, int reps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 acc = src[i];
    for (int r = 0; r < reps; ++r) {
        const float4 v = src[(i + r * 977) & (n4 - 1)];
        acc.x += v.x * 1e-9f + st[r].x; acc.y += v.y * 1e-9f; acc.z += v.z * 1e-9f; acc.w += v.w * 1e-9f;
    }
    dst[i] = acc;
}

// down-shaped: ~240 blocks, tiny per-thread work, a 256-entry LUT prologue in shared (like the real
// kernel's expand4_s2 staging), serial float adds - the real stage's geometry and startup cost.
__global__ void down_shaped(const float4* __restrict__ src, const float4* __restrict__ st,
                            float4* __restrict__ dst, int n4) {
    __shared__ unsigned int slut[256];
    for (int i = (int) threadIdx.x; i < 256; i += 128) {
        unsigned int e = (unsigned int) (i * 2654435761u);       // an expand-ish cost
        e ^= e >> 13; e *= 1274126177u; e ^= e >> 16;
        slut[i] = e;
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 acc = src[i];
    float a = 0.f;
    for (int r = 0; r < 8; ++r) a += (float) slut[(i + r * 37) & 255] * 1e-9f;   // LUT lookups, serial adds
    acc.x += a + st[i & 3].x * 1e-9f;
    dst[i] = acc;
}


static double run_pair_mode(const char* name, std::vector<Card>& cs, int mode, int n4, int reps, int iters,
                            Card* primary = nullptr, volatile uint32_t* poll_flag = nullptr) {
    std::vector<double> k1(cs.size(), 0.0), k2(cs.size(), 0.0);
    std::vector<cudaEvent_t> mid(cs.size());
    int count = (int) cs.size();
    for (int c = 0; c < count; ++c) { CK(cudaSetDevice(cs[c].dev)); CK(cudaEventCreate(&mid[c])); }
    for (int it = 0; it < iters; ++it) {
        if (poll_flag != nullptr) *poll_flag = 0;          // the 3060 will spin until this rises
        if (mode == 4 && primary != nullptr) {             // the 3060's graph-park: a kernel spinning on
            spin_flag<<<1, 1, 0, primary->s>>>(poll_flag); // the mapped host flag, like the fused wait
            CK(cudaSetDevice(primary->dev));
        }
        for (int c = 0; c < count; ++c) {                 // begin loop: upload + gup + mid + down, staggered
            Card& k = cs[c];
            CK(cudaSetDevice(k.dev));
            CK(cudaMemcpyAsync(k.d_stage, k.h_stage, 4096, cudaMemcpyHostToDevice, k.s));
            CK(cudaEventRecord(k.e0, k.s));
            stream_kernel2<<<(n4 + 127) / 128, 128, 0, k.s>>>(k.a, k.d_stage, k.b, n4, reps);
            CK(cudaEventRecord(mid[c], k.s));
            if (mode == 2)                                 // down-shaped small-grid kernel2
                down_shaped<<<240, 128, 0, k.s>>>(k.a, k.d_stage, k.b, 240 * 128);
            else
                stream_kernel2<<<(n4 / 4 + 127) / 128, 128, 0, k.s>>>(k.a, k.d_stage, k.b, n4 / 4, reps);
            CK(cudaEventRecord(k.e1, k.s));
            if (mode == 0 || mode == 4) std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        for (int c = 0; c < count; ++c) {                 // finish loop: sync in visit order
            CK(cudaSetDevice(cs[c].dev));
            CK(cudaStreamSynchronize(cs[c].s));
        }
        if (poll_flag != nullptr) *poll_flag = 1;          // release the 3060's spin
        if (mode == 4 && primary != nullptr) CK(cudaStreamSynchronize(primary->s));
        for (int c = 0; c < count; ++c) {
            float a, b;
            cudaEventElapsedTime(&a, cs[c].e0, mid[c]);
            cudaEventElapsedTime(&b, mid[c], cs[c].e1);
            k1[c] += a; k2[c] += b;
        }
    }
    std::printf("  %-24s:", name);
    for (int c = 0; c < count; ++c)
        std::printf("  C%d [%5.0f + %5.0f]", cs[c].dev, 1000.0 * k1[c] / iters, 1000.0 * k2[c] / iters);
    std::printf("  us\n");
    return k2[count - 1] / iters;
}


static double run_mode(const char* name, std::vector<Card>& cs, int mode, int n4, int reps, int iters) {
    // mode 0: staggered one-thread (engine-like); 1: synced one-thread; 2: four threads; 3: card 0 alone
    std::vector<double> el(cs.size(), 0.0);
    int count = (mode == 3) ? 1 : (int) cs.size();
    for (int it = 0; it < iters; ++it) {
        if (mode == 2) {                              // four threads: each its own device, launch+sync
            std::vector<std::thread> ts;
            for (int c = 0; c < count; ++c) {
                ts.emplace_back([&, c] {
                    Card& k = cs[c];
                    cudaSetDevice(k.dev);
                    cudaEventRecord(k.e0, k.s);
                    stream_kernel<<<(n4 + 127) / 128, 128, 0, k.s>>>(k.a, k.b, n4, reps);
                    cudaEventRecord(k.e1, k.s);
                    cudaStreamSynchronize(k.s);
                });
            }
            for (auto& t : ts) t.join();
        } else {
            for (int c = 0; c < count; ++c) {         // launches from ONE thread, device-hopping
                Card& k = cs[c];
                cudaSetDevice(k.dev);
                cudaEventRecord(k.e0, k.s);
                stream_kernel<<<(n4 + 127) / 128, 128, 0, k.s>>>(k.a, k.b, n4, reps);
                cudaEventRecord(k.e1, k.s);
                if (mode == 0) std::this_thread::sleep_for(std::chrono::microseconds(20));  // engine begin stagger
            }
            for (int c = 0; c < count; ++c) {         // syncs in visit order, device-hopping (the engine's finish)
                cudaSetDevice(cs[c].dev);
                cudaStreamSynchronize(cs[c].s);
            }
        }
        for (int c = 0; c < count; ++c) { float ms; cudaEventElapsedTime(&ms, cs[c].e0, cs[c].e1); el[c] += ms; }
    }
    std::printf("  %-24s:", name);
    for (int c = 0; c < count; ++c) std::printf("  C%d %6.1f us", cs[c].dev, 1000.0 * el[c] / iters);
    std::printf("\n");
    return el[count - 1] / iters;
}

int main(int argc, char** argv) {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs;
    for (int i = 1; i < ndev && (int) devs.size() < 4; ++i) devs.push_back(i);   // CUDA1..4 = the M10s
    const int n4 = 1 << 20;                       // 16 MB of float4 = ~6 MB read with reps pattern
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    const int iters = 300;
    std::vector<Card> cs(devs.size() + 1);        // + CUDA0 = the 3060, for the spin_flag test
    cs[0].dev = 0;
    for (size_t c = 1; c < cs.size(); ++c) cs[c].dev = devs[c - 1];
    volatile uint32_t* poll_flag = nullptr;
    for (size_t c = 0; c < cs.size(); ++c) {
        CK(cudaSetDevice(cs[c].dev));
        if (c > 0) {
            CK(cudaMalloc(&cs[c].a, (size_t) n4 * 16)); CK(cudaMalloc(&cs[c].b, (size_t) n4 * 16));
            CK(cudaMemset(cs[c].a, 1, (size_t) n4 * 16));
            CK(cudaMalloc(&cs[c].d_stage, 4096)); CK(cudaMallocHost(&cs[c].h_stage, 4096));
        }
        if (c == 0) CK(cudaMallocHost((void**) &poll_flag, 64));   // the mapped flag the 3060 polls
        CK(cudaStreamCreate(&cs[c].s)); CK(cudaEventCreate(&cs[c].e0)); CK(cudaEventCreate(&cs[c].e1));
    }
    std::vector<Card> m10s(cs.begin() + 1, cs.end());
    std::printf("m10bench: %zu M10s + 3060 spinner, kernel reads ~%.0f MB, %d iters\n", m10s.size(),
                (double) n4 * 16 * (1 + reps * 0.25) / 1e6, iters);
    run_pair_mode("P1 staggered pair (engine)", m10s, 0, n4, reps, iters);
    run_pair_mode("P4 staggered + 3060 SPIN", m10s, 4, n4, reps, iters, &cs[0], poll_flag);
    run_pair_mode("P2 synced pair", m10s, 1, n4, reps, iters);
    run_mode("A one-thread staggered", cs, 0, n4, reps, iters);
    run_mode("B one-thread synced", cs, 1, n4, reps, iters);
    run_mode("C four threads", cs, 2, n4, reps, iters);
    return 0;
}
