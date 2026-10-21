// X1 go/no-go harness: real shared-expert tensors, expanded FP16 TP on four
// Maxwell dies versus production native shared_expert_multi on the main GPU.
// No model-serving changes. CPU fp64 reference checks the new *float* contract;
// native Q8_1 differences are reported, never called bitwise parity.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void maxwell_shared_launch(const uint16_t*, const uint16_t*, const uint16_t*, const float*,
                            float*, float*, int, int, int, cudaStream_t);
void maxwell_shared_conversions(float*);
namespace {
using Clock = std::chrono::steady_clock;
using namespace strata::kernels;
constexpr int H = 2560, FF = 640, R = FF / 4;
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> T* alloc(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template<class T> void upload(T* p, const std::vector<T>& v) {
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
struct Tensor {
    const strata::TensorInfo* info;
    const uint8_t* data;
    std::vector<float> f32;
    std::vector<uint16_t> f16;
};
Tensor tensor(const strata::GgufModel& model, const std::string& name, int n, int rows) {
    size_t shard = 0;
    Tensor t{};
    t.info = model.find(name, &shard);
    if (!t.info || t.info->shape.empty() || t.info->shape[0] != (uint64_t) n ||
        t.info->elements() != (uint64_t) n * rows)
        throw std::runtime_error("missing/wrong shape: " + name);
    t.data = model.shard(shard).tensor_data(*t.info);
    const auto* traits = ggml_get_type_traits((ggml_type) t.info->type);
    if (!traits || !traits->to_float) throw std::runtime_error("no dequantizer: " + name);
    t.f32.resize((size_t) n * rows);
    traits->to_float(t.data, t.f32.data(), (int64_t) t.f32.size());
    for (float v : t.f32) t.f16.push_back(f16_from_f32(v));
    return t;
}
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        s += (double) b[i] * b[i];
    }
    return std::sqrt(d / std::max(s, 1e-30));
}
struct Card {
    int dev = -1;
    uint16_t *gate = nullptr, *up = nullptr, *down = nullptr;
    float *x = nullptr, *ff = nullptr, *out = nullptr, *host = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Card() {
        if (dev < 0) return;
        cudaSetDevice(dev);
        if (stream) cudaStreamSynchronize(stream);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(gate); cudaFree(up); cudaFree(down);
        cudaFree(x); cudaFree(ff); cudaFree(out);
        if (host) cudaFreeHost(host);
        if (stream) cudaStreamDestroy(stream);
    }
};
}

int main(int argc, char** argv) try {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: maxwell_shared_bench <shard1.gguf> <layer> [tokens=4 (1/2/4)] [iters=200]\n");
        return 2;
    }
    const int layer = std::stoi(argv[2]), tokens = argc > 3 ? std::stoi(argv[3]) : 4;
    const int iters = argc > 4 ? std::stoi(argv[4]) : 200;
    if (layer < 0 || (tokens != 1 && tokens != 2 && tokens != 4) || iters < 1)
        throw std::runtime_error("invalid layer/tokens/iters");
    ggml_cpu_init();
    int count = 0, main_dev = -1;
    check(cudaGetDeviceCount(&count));
    std::vector<int> devices;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, i));
        if (p.major == 5 && p.minor == 0) devices.push_back(i);
        if (p.major >= 7 && main_dev < 0) main_dev = i;
    }
    if (devices.size() != 4 || main_dev < 0) throw std::runtime_error("needs four sm_50 dies and one sm_70+ GPU");
    check(cudaSetDevice(devices[0]));
    auto* conv = alloc<float>(65536);
    maxwell_shared_conversions(conv);
    std::vector<float> conversion(65536);
    check(cudaMemcpy(conversion.data(), conv, conversion.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaFree(conv));
    for (int i = 0; i < 65536; ++i) {
        const float ref = f32_from_f16((uint16_t) i);
        if (std::isnan(ref) ? !std::isnan(conversion[i]) : std::memcmp(&ref, &conversion[i], 4) != 0)
            throw std::runtime_error("half->float conversion mismatch at " + std::to_string(i));
    }
    std::printf("sm_50 half->float: all 65536 bit patterns PASS (NaN payloads ignored)\n");
    const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = tensor(model, prefix + "ffn_gate_shexp.weight", H, FF);
    const auto up = tensor(model, prefix + "ffn_up_shexp.weight", H, FF);
    const auto down = tensor(model, prefix + "ffn_down_shexp.weight", FF, H);
    const auto gin = tensor(model, prefix + "ffn_gate_inp_shexp.weight", H, 1);
    if (gin.info->type != 30) throw std::runtime_error("scalar gate must be BF16");
    std::printf("layer=%d tokens=%d types=%u/%u/%u: FP16 float-contract prototype, NOT production parity\n",
                layer, tokens, gate.info->type, up.info->type, down.info->type);
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.2f);
    std::vector<float> input((size_t) tokens * H);
    for (auto& x : input) x = normal(rng);
    float* pinned = nullptr;
    check(cudaHostAlloc(&pinned, input.size() * sizeof(float), cudaHostAllocPortable));
    std::copy(input.begin(), input.end(), pinned);
    std::array<Card, 4> cards;
    for (int rank = 0; rank < 4; ++rank) {
        auto& c = cards[rank];
        c.dev = devices[rank];
        check(cudaSetDevice(c.dev));
        check(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&c.start)); check(cudaEventCreate(&c.stop));
        c.gate = alloc<uint16_t>((size_t) R * H); c.up = alloc<uint16_t>((size_t) R * H);
        c.down = alloc<uint16_t>((size_t) H * R);
        c.x = alloc<float>(input.size()); c.ff = alloc<float>((size_t) tokens * R);
        c.out = alloc<float>(input.size());
        check(cudaHostAlloc(&c.host, input.size() * sizeof(float), cudaHostAllocPortable));
        check(cudaMemcpy(c.gate, gate.f16.data() + (size_t) rank * R * H, (size_t) R * H * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(c.up, up.f16.data() + (size_t) rank * R * H, (size_t) R * H * 2, cudaMemcpyHostToDevice));
        std::vector<uint16_t> wd((size_t) H * R);
        for (int r = 0; r < H; ++r)
            std::copy_n(down.f16.data() + (size_t) r * FF + rank * R, R, wd.data() + (size_t) r * R);
        upload(c.down, wd); upload(c.x, input);
        // Both transfers are copy-engine operations; never read mapped acts on Maxwell.
        check(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
        check(cudaMemcpyAsync(c.x, pinned, input.size() * 4, cudaMemcpyHostToDevice, c.stream));
        maxwell_shared_launch(c.gate, c.up, c.down, c.x, c.ff, c.out, H, R, tokens, c.stream);
        check(cudaMemcpyAsync(c.host, c.out, input.size() * 4, cudaMemcpyDeviceToHost, c.stream));
        check(cudaStreamEndCapture(c.stream, &c.graph));
        check(cudaGraphInstantiate(&c.exec, c.graph, nullptr, nullptr, 0));
    }
    auto launch = [&](Card& c) {
        maxwell_shared_launch(c.gate, c.up, c.down, c.x, c.ff, c.out, H, R, tokens, c.stream);
        check(cudaGetLastError());
    };
    // Bounded burn queues; do not enqueue seconds of work without draining it.
    const auto burn = Clock::now();
    do {
        for (auto& c : cards) {
            check(cudaSetDevice(c.dev));
            for (int i = 0; i < 10; ++i) launch(c);
        }
        for (auto& c : cards) { check(cudaSetDevice(c.dev)); check(cudaStreamSynchronize(c.stream)); }
    } while (std::chrono::duration<double>(Clock::now() - burn).count() < 2.0);
    // Per-card event batches measure compute without hot-window host calls.
    // Cards measured individually: optimistic lower bound, NOT a concurrency claim.
    double max_compute_us = 0;
    for (auto& c : cards) {
        check(cudaSetDevice(c.dev));
        check(cudaEventRecord(c.start, c.stream));
        for (int i = 0; i < iters; ++i) launch(c);
        check(cudaEventRecord(c.stop, c.stream)); check(cudaEventSynchronize(c.stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, c.start, c.stop));
        const double us = ms * 1000.0 / iters;
        max_compute_us = std::max(max_compute_us, us);
        std::printf("CUDA%d compute %.2f us/layer\n", c.dev, us);
    }
    // End-to-end host-staged graph dispatch: all four are launched before waits.
    // Includes the existing VM's driver/sync penalty. CPU ordered sum is timed;
    // final delivery to the 3060 and its scalar sigmoid are NOT timed (optimistic).
    std::vector<float> assembled(input.size());
    const auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        for (auto& c : cards) { check(cudaSetDevice(c.dev)); check(cudaGraphLaunch(c.exec, c.stream)); }
        for (auto& c : cards) { check(cudaSetDevice(c.dev)); check(cudaStreamSynchronize(c.stream)); }
        for (size_t j = 0; j < assembled.size(); ++j)
            assembled[j] = ((cards[0].host[j] + cards[1].host[j]) + cards[2].host[j]) + cards[3].host[j];
    }
    const double roundtrip_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count() / iters;
    std::vector<float> reference(input.size());
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> ff(FF);
        for (int r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int j = 0; j < H; ++j) {
                g += (double) f32_from_f16(gate.f16[(size_t) r * H + j]) * input[(size_t) t * H + j];
                u += (double) f32_from_f16(up.f16[(size_t) r * H + j]) * input[(size_t) t * H + j];
            }
            ff[r] = g / (1.0 + std::exp(-g)) * u;
        }
        for (int r = 0; r < H; ++r) {
            double v = 0;
            for (int j = 0; j < FF; ++j) v += f32_from_f16(down.f16[(size_t) r * FF + j]) * ff[j];
            reference[(size_t) t * H + r] = (float) v;
        }
    }
    const double error = rel(assembled, reference);
    std::printf("FP16-weight/F32-activation chain vs fp64: relL2 %.3e (%s)\n", error, error < 2e-5 ? "PASS" : "FAIL");
    if (!(error < 2e-5)) return 1;
    // Production native path, same layer/input/token count, isolated on the main GPU.
    check(cudaSetDevice(main_dev));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    NativeSharedWeights nw;
    nw.gate_type = gate.info->type; nw.up_type = up.info->type; nw.down_type = down.info->type;
    auto native_weight = [&](const Tensor& t) {
        const size_t bytes = native_mmvq_weight_bytes(t.info->type,
                                                     (int) t.info->shape[0], (int) t.info->shape[1]);
        auto* d = alloc<uint8_t>(bytes);
        check(cudaMemcpy(d, t.data, bytes, cudaMemcpyHostToDevice));
        return d;
    };
    nw.gate_data = native_weight(gate); nw.up_data = native_weight(up); nw.down_data = native_weight(down);
    nw.q8_1 = alloc<uint8_t>(native_q8_1_bytes(H, tokens));
    auto* x = alloc<float>(input.size()); upload(x, input);
    auto* xbf = alloc<uint16_t>(input.size());
    auto* gi = alloc<uint16_t>(H);
    check(cudaMemcpy(gi, gin.data, H * 2, cudaMemcpyHostToDevice));
    auto* g = alloc<float>((size_t) tokens * FF); auto* u = alloc<float>((size_t) tokens * FF);
    auto* scalar = alloc<float>(tokens); auto* out = alloc<float>(input.size());
    shared_expert_set_native_bf16(true);
    auto native = [&] { shared_expert_multi(tokens, x, xbf, nw, gi, g, u, scalar, out, H, FF, stream); };
    const auto nburn = Clock::now();
    do {
        for (int i = 0; i < 10; ++i) native();
        check(cudaStreamSynchronize(stream));
    } while (std::chrono::duration<double>(Clock::now() - nburn).count() < 2.0);
    cudaGraph_t native_graph = nullptr;
    cudaGraphExec_t native_exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    native();
    check(cudaStreamEndCapture(stream, &native_graph));
    check(cudaGraphInstantiate(&native_exec, native_graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(native_exec, stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1));
    check(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; ++i) check(cudaGraphLaunch(native_exec, stream));
    check(cudaEventRecord(e1, stream)); check(cudaEventSynchronize(e1));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("RESULT main-native-graph %.2f us/L; TP-compute-max %.2f us/L; TP-host-staged %.2f us/L\n",
                ms * 1000.0 / iters, max_compute_us, roundtrip_us);
    std::vector<float> native_out(input.size());
    check(cudaMemcpy(native_out.data(), out, native_out.size() * 4, cudaMemcpyDeviceToHost));
    for (int t = 0; t < tokens; ++t) {
        double dot = 0;
        for (int j = 0; j < H; ++j) dot += (double) input[(size_t) t * H + j] * gin.f32[j];
        const float scale = (float) (1.0 / (1.0 + std::exp(-dot)));
        for (int j = 0; j < H; ++j) assembled[(size_t) t * H + j] *= scale;
    }
    std::printf("Float-contract TP vs native Q8_1 output: relL2 %.3e (diagnostic, not parity)\n",
                rel(assembled, native_out));
    std::printf("FP16 TP full-model weights: %.2f MiB/card (48 layers); no serving integration or end-to-end speed claim\n",
                48.0 * 3 * H * R * 2 / 1048576.0);
    check(cudaEventDestroy(e0)); check(cudaEventDestroy(e1));
    check(cudaGraphExecDestroy(native_exec)); check(cudaGraphDestroy(native_graph));
    check(cudaStreamDestroy(stream));
    for (auto p : {x, g, u, scalar, out}) check(cudaFree(p));
    check(cudaFree(xbf)); check(cudaFree(gi)); check(cudaFree(nw.q8_1));
    check(cudaFree(const_cast<void*>(nw.gate_data)));
    check(cudaFree(const_cast<void*>(nw.up_data)));
    check(cudaFree(const_cast<void*>(nw.down_data)));
    check(cudaFreeHost(pinned));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "maxwell_shared_bench: %s\n", e.what());
    return 1;
}
