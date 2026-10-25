// The consumer is already running while the CPU publishes a new expected value
// and its main flag. Old rank flags must never release it, even for one poll.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T* mapped(size_t n) {
    T* p = nullptr;
    ck(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    return p;
}
int main() try {
    ck(cudaSetDevice(0));
    auto* words = mapped<uint32_t>(128);
    auto* data = mapped<float>(32);
    auto* list = mapped<int32_t>(8);
    const uint32_t* flags[4] = {words + 32, words + 48, words + 64, words + 80};
    const float* partials[4] = {data, data + 4, data + 8, data + 12};
    for (int i = 0; i < 16; ++i) data[i] = (float) i;
    list[0] = 1; list[1] = 0; list[2] = 0; list[3] = 0; list[4] = 0;
    words[96] = 0;   // Meta cancellation flag
    uint32_t* counter = nullptr;
    ck(cudaMalloc(&counter, sizeof(uint32_t))); ck(cudaMemset(counter, 0, sizeof(uint32_t)));
    cudaStream_t stream; cudaEvent_t done;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    int total = 0;
    for (int mode = 0; mode < 5; ++mode) {
        int early = 0;
        for (uint32_t i = 0; i < 1000; ++i) {
            auto* main = (volatile uint32_t*) words;
            auto* expected = (volatile uint32_t*) (words + 16);
            *main = 0; *expected = i;
            for (int r = 0; r < 4; ++r) ((volatile uint32_t*) words)[32 + r * 16] = i;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (mode == 0)
                strata::kernels::wait_mw(words, 1, words + 16, flags, stream);
            else if (mode == 1)
                strata::kernels::copy_rows_from_mapped_after_mw(words, 1, words + 16, flags,
                                                               data + 16, data, 1, 4, list + 1, list + 4, stream);
            else if (mode == 2)
                strata::kernels::sum_tp_rows_after_mw(words, 1, words + 16, flags,
                                                       data + 16, 1, 4, partials, list, list, stream);
            else
                strata::kernels::maxwell_meta_reduce(words, 1, words + 16, flags, partials, partials, list + 2, list + 3,
                    data + 16, data, 1, 4, list + 1, list + 4, mode == 3, false, words + 96, counter, words + 112, stream);
            if (mode == 4) strata::kernels::maxwell_meta_complete(words + 16, words + 112, stream);
            ck(cudaEventRecord(done, stream));
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            *expected = i + 1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *main = 1;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            const cudaError_t state = cudaEventQuery(done);
            if (state == cudaSuccess) ++early;
            else if (state != cudaErrorNotReady) ck(state);
            for (int r = 0; r < 4; ++r) ((volatile uint32_t*) words)[32 + r * 16] = i + 1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ck(cudaEventSynchronize(done));
            if (mode >= 3 && words[112] != i + 1) throw std::runtime_error("Meta consumption ticket differs");
        }
        std::printf("publish mode %d: %d early completions / 1000\n", mode, early);
        total += early;
    }
    ck(cudaEventDestroy(done)); ck(cudaStreamDestroy(stream));
    ck(cudaFree(counter)); ck(cudaFreeHost(words)); ck(cudaFreeHost(data)); ck(cudaFreeHost(list));
    if (total) throw std::runtime_error("consumer accepted stale expected sequence");
    std::puts("PASS: 5000 asynchronous expected/main publications gated by the NEW rank flags (including both Meta transports)");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
