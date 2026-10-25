// Real-GGUF placement parity: the unchanged gather kernels must produce identical
// F32 rows from mapped host storage and from STRATA_EMBED_VRAM=1 storage.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_head.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void placement(bool vram) {
#ifdef _WIN32
    _putenv_s("STRATA_EMBED_VRAM", vram ? "1" : "0");
#else
    setenv("STRATA_EMBED_VRAM", vram ? "1" : "0", 1);
#endif
}
template<class T> T* alloc(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
}

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::fprintf(stderr, "usage: native_embed_placement_test <model-shard1.gguf>\n");
        return 2;
    }
    check(cudaSetDevice(0));
    const auto shards = strata::gguf_split_paths(argv[1]);
    const strata::GgufModel model(shards);
    strata::core::ModelGeometry g;
    std::string err = strata::read_geometry(model.meta(), g);
    if (!err.empty()) throw std::runtime_error(err);
    strata::core::NativeEmbed host, device;
    placement(false);
    if (!host.load(shards, g.n_embd, g.n_vocab, err)) throw std::runtime_error(err);
    if (host.device_resident()) throw std::runtime_error("mapped-host baseline fell back to VRAM");
    placement(true);
    if (!device.load(shards, g.n_embd, g.n_vocab, err)) throw std::runtime_error(err);
    if (!device.device_resident()) throw std::runtime_error("requested VRAM placement is not resident");
    if (host.bytes() != device.bytes() || host.type() != device.type())
        throw std::runtime_error("embedding metadata changed with placement");
    constexpr int batch = 1024;
    auto* ids = alloc<int32_t>(batch);
    auto* a = alloc<float>((size_t) batch * g.n_embd);
    auto* b = alloc<float>((size_t) batch * g.n_embd);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<int32_t> tokens(batch);
    std::vector<float> ah((size_t) batch * g.n_embd), bh(ah.size());
    for (int64_t first = 0; first < g.n_vocab; first += batch) {
        const int n = (int) std::min<int64_t>(batch, g.n_vocab - first);
        for (int i = 0; i < n; ++i) tokens[i] = (int32_t) (first + i);
        check(cudaMemcpyAsync(ids, tokens.data(), (size_t) n * 4, cudaMemcpyHostToDevice, stream));
        host.gather_dev(ids, n, a, stream);
        device.gather_dev(ids, n, b, stream);
        check(cudaStreamSynchronize(stream));
        const size_t bytes = (size_t) n * g.n_embd * sizeof(float);
        check(cudaMemcpy(ah.data(), a, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(bh.data(), b, bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(ah.data(), bh.data(), bytes))
            throw std::runtime_error("gather_dev differs at batch " + std::to_string(first));
    }
    // Single-row path used outside the batched verifier, including the drafter.
    for (int64_t token : {int64_t(0), g.n_vocab / 2, g.n_vocab - 1}) {
        host.gather_one(token, a, stream);
        device.gather_one(token, b, stream);
        check(cudaStreamSynchronize(stream));
        const size_t bytes = (size_t) g.n_embd * sizeof(float);
        check(cudaMemcpy(ah.data(), a, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(bh.data(), b, bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(ah.data(), bh.data(), bytes)) throw std::runtime_error("gather_one differs");
    }
    std::printf("PASS: all %lld vocabulary rows x %lld F32 values bit-identical; gather_one PASS; %.2f MiB in VRAM\n",
                (long long) g.n_vocab, (long long) g.n_embd, (double) device.bytes() / 1048576.0);
    check(cudaStreamDestroy(stream));
    check(cudaFree(ids)); check(cudaFree(a)); check(cudaFree(b));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "native_embed_placement_test: %s\n", e.what());
    return 1;
}
