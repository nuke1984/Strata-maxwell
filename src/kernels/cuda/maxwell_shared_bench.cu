// Standalone X1 screening kernels, NOT a serving path. Weights are expanded to
// FP16 at load; activations and accumulators are F32 (no native half ALU needed).
// This deliberately does not claim equivalence to production's Q8_1 arithmetic.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__device__ float weight(uint16_t h) {
    // Maxwell supports half->float conversion, not native half arithmetic.
    // The harness checks every half bit pattern against f16_bits.hpp first.
    float v;
    asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h));
    return v;
}
__global__ void conversions(float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) out[i] = weight((uint16_t) i);
}
__device__ float sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}

// One warp per row, four rows per block. All tokens share each weight load.
template<int T>
__global__ void gup(const uint16_t* __restrict__ gate, const uint16_t* __restrict__ up,
                    const float* __restrict__ x, float* __restrict__ ff, int h, int rows) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    float g[T] = {}, u[T] = {};
    if (row < rows) {
        for (int j = lane; j < h; j += 32) {
            const float wg = weight(__ldg(gate + (size_t) row * h + j));
            const float wu = weight(__ldg(up + (size_t) row * h + j));
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const float a = __ldg(x + (size_t) t * h + j);
                g[t] = fmaf(wg, a, g[t]);
                u[t] = fmaf(wu, a, u[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float sg = sum(g[t]), su = sum(u[t]);
            if (!lane) ff[(size_t) t * rows + row] = (sg / (1.0f + expf(-sg))) * su;
        }
    }
}

template<int T>
__global__ void down(const uint16_t* __restrict__ w, const float* __restrict__ ff,
                     float* __restrict__ out, int h, int rows) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    float a[T] = {};
    if (row < h) {
        for (int j = lane; j < rows; j += 32) {
            const float v = weight(__ldg(w + (size_t) row * rows + j));
            #pragma unroll
            for (int t = 0; t < T; ++t) a[t] = fmaf(v, __ldg(ff + (size_t) t * rows + j), a[t]);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float v = sum(a[t]);
            if (!lane) out[(size_t) t * h + row] = v;
        }
    }
}
}

void maxwell_shared_conversions(float* out) { conversions<<<256, 256>>>(out); }

void maxwell_shared_launch(const uint16_t* gate, const uint16_t* up, const uint16_t* wdown,
                            const float* x, float* ff, float* out, int h, int rows,
                            int tokens, cudaStream_t stream) {
    #define RUN(T) case T: \
        gup<T><<<(rows + 3) / 4, 128, 0, stream>>>(gate, up, x, ff, h, rows); \
        down<T><<<(h + 3) / 4, 128, 0, stream>>>(wdown, ff, out, h, rows); break
    switch (tokens) { RUN(1); RUN(2); RUN(4); }
    #undef RUN
}
