// Routed-expert TP correctness only, not a speed benchmark. Uses the production
// loader/transport/flags/reduction on 4 M10s, against the unsliced Maxwell kernels.
// Requires the real IQ2_XS GGUF + native pack. Run with serving stopped.
#include "strata/core/maxwell_experts.hpp"
#include "strata/core/maxwell_meta.hpp"
#include "strata/core/maxwell_tp_plan.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace tp = strata::core::maxwell_tp;
namespace cpu = strata::kernels::cpu;
namespace mw = strata::kernels::cuda;
constexpr int H = 2560, FF = 640, CAP = 640, K = 10;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T* dev(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); return p; }
template<class T> T* host(size_t n) {
    T* p = nullptr; ck(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable)); return p;
}
struct Store {
    int per_layer = 1;
    std::vector<std::vector<uint8_t>> blob;
    std::vector<int> expert;
    static const uint8_t* get(int64_t l, int64_t e, void* ctx) {
        auto& s = *(Store*) ctx;
        if (l < 0 || l >= (int64_t) s.blob.size()) return nullptr;
        const int offset = ((int) e - s.expert[l] + 512) % 512;
        return offset < s.per_layer ? s.blob[l].data() + (size_t) offset * (s.blob[l].size() / s.per_layer) : nullptr;
    }
    static bool copy(int64_t l, int64_t e, uint8_t* dst, size_t bytes, void* ctx) {
        auto& s = *(Store*) ctx;
        const auto* src = get(l, e, ctx);
        if (!src || s.blob[l].size() / s.per_layer != bytes) return false;
        std::memcpy(dst, src, bytes); return true;
    }
};
std::vector<uint8_t> converted(const std::vector<uint8_t>& source, const cpu::NativeFmt& f, tp::Slice s, int expert = 0) {
    const auto* src = source.data() + (size_t) expert * f.bytes;
    const size_t codes = (size_t) 2 * s.rows * f.gu_row, scales = codes + (size_t) H * s.blocks * 16;
    std::vector<uint8_t> out(scales + (size_t) H * s.blocks * 2);
    for (int r = 0; r < s.rows; ++r) {
        std::memcpy(out.data() + (size_t) r * f.gu_row, src + (size_t) (s.row_base + r) * f.gu_row, f.gu_row);
        std::memcpy(out.data() + (size_t) (s.rows + r) * f.gu_row,
                    src + f.up_off + (size_t) (s.row_base + r) * f.gu_row, f.gu_row);
    }
    for (int r = 0; r < H; ++r)
        for (int b = 0; b < s.blocks; ++b) {
            const auto* p = src + f.down_off + (size_t) r * f.d_row + (s.block_base + b) * 18;
            std::memcpy(out.data() + codes + ((size_t) r * s.blocks + b) * 16, p + 2, 16);
            std::memcpy(out.data() + scales + ((size_t) r * s.blocks + b) * 2, p, 2);
        }
    return out;
}
// Independent assembly of the compact source blob. The actual 4-rank outputs
// come from MaxwellExperts::open's own conversion, not this helper.
struct Reference {
    int device;
    cudaStream_t stream{};
    uint8_t *blob{}, *acts{};
    int32_t* jobs{};
    int8_t* q{};
    float *scale{}, *hx{}, *ff{}, *out{};
    std::vector<int8_t> whole_q;
    std::vector<float> whole_scale, whole_hx, whole_ff;
    Reference(int d, size_t max_blob) : device(d) {
        ck(cudaSetDevice(d)); ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blob = dev<uint8_t>(max_blob); acts = dev<uint8_t>(64 * mw::expert_maxwell_geo::ACTS_STRIDE);
        jobs = dev<int32_t>(CAP * 3); q = dev<int8_t>(CAP * FF);
        scale = dev<float>(CAP * FF / 32); hx = dev<float>(CAP * FF / 32);
        ff = dev<float>(CAP * FF); out = dev<float>(CAP * H);
    }
    ~Reference() {
        cudaSetDevice(device); cudaStreamSynchronize(stream);
        for (void* p : std::array<void*, 8>{blob, acts, jobs, q, scale, hx, ff, out}) cudaFree(p);
        cudaStreamDestroy(stream);
    }
    std::vector<float> run(const std::vector<uint8_t>& raw, const cpu::NativeFmt& f,
                            const std::vector<cpu::ActQ>& a, const std::vector<int>& entries,
                            const std::vector<int32_t>& ids, int first_expert) {
        ck(cudaSetDevice(device));
        std::vector<uint8_t> full;
        const int experts = (int) (raw.size() / f.bytes);
        for (int e = 0; e < experts; ++e) {
            auto b = converted(raw, f, {0, 10, 0, 640}, e); full.insert(full.end(), b.begin(), b.end());
        }
        std::vector<uint8_t> packed(a.size() * mw::expert_maxwell_geo::ACTS_STRIDE);
        for (size_t t = 0; t < a.size(); ++t) {
            auto* p = packed.data() + t * mw::expert_maxwell_geo::ACTS_STRIDE;
            std::memcpy(p, a[t].q, H);
            std::memcpy(p + H, a[t].scale, (H / 32) * 4);
            std::memcpy(p + H + (H / 32) * 4, a[t].hx, (H / 32) * 4);
        }
        std::vector<int32_t> j(entries.size() * 3);
        for (size_t i = 0; i < entries.size(); ++i) {
            j[3 * i] = (ids[entries[i]] - first_expert + 512) % 512;
            j[3 * i + 1] = entries[i] / K; j[3 * i + 2] = (int) i;
        }
        ck(cudaMemcpyAsync(blob, full.data(), full.size(), cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(acts, packed.data(), packed.size(), cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(jobs, j.data(), j.size() * 4, cudaMemcpyHostToDevice, stream));
        const int n = (int) entries.size();
        mw::iq_expert_maxwell_gup_quant_batch_async(f.gu_type, blob, f.bytes, jobs, n, acts,
                                                   0, FF * f.gu_row, f.gu_row, q, scale, hx, ff, stream);
        const size_t codes = 2ull * FF * f.gu_row, scales = codes + H * 160;
        mw::s2_expert_maxwell_down_batch_async(blob, jobs, n, q, scale, hx, out, f.bytes, codes, scales, stream);
        ck(cudaStreamSynchronize(stream));
        std::vector<float> answer((size_t) n * H);
        if (n) ck(cudaMemcpy(answer.data(), out, answer.size() * 4, cudaMemcpyDeviceToHost));
        whole_q.resize((size_t) n * FF); whole_ff.resize((size_t) n * FF);
        whole_scale.resize((size_t) n * FF / 32); whole_hx.resize(whole_scale.size());
        if (n) {
            ck(cudaMemcpy(whole_q.data(), q, whole_q.size(), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(whole_ff.data(), ff, whole_ff.size() * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(whole_scale.data(), scale, whole_scale.size() * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(whole_hx.data(), hx, whole_hx.size() * 4, cudaMemcpyDeviceToHost));
        }
        return answer;
    }
    void intermediates(const std::vector<uint8_t>& raw, const cpu::NativeFmt& f, int layer, int njobs) {
        for (int rank = 0; rank < 4; ++rank) {
            const auto s = tp::slice(rank, layer);
            std::vector<uint8_t> compact;
            const size_t stride = tp::shard_bytes(s, f.gu_row);
            for (size_t e = 0; e < raw.size() / f.bytes; ++e) {
                auto b = converted(raw, f, s, (int) e); compact.insert(compact.end(), b.begin(), b.end());
            }
            ck(cudaMemcpyAsync(blob, compact.data(), compact.size(), cudaMemcpyHostToDevice, stream));
            mw::iq_expert_maxwell_gup_quant_batch_async(f.gu_type, blob, stride, jobs, njobs, acts,
                                                       0, (size_t) s.rows * f.gu_row, f.gu_row, q, scale, hx, ff,
                                                       stream, s.row_base, s.rows);
            ck(cudaStreamSynchronize(stream));
            std::vector<int8_t> iq(whole_q.size());
            std::vector<float> val(whole_ff.size()), sc(whole_scale.size()), corr(sc.size());
            ck(cudaMemcpy(iq.data(), q, iq.size(), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(val.data(), ff, val.size() * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(sc.data(), scale, sc.size() * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(corr.data(), hx, corr.size() * 4, cudaMemcpyDeviceToHost));
            for (int j = 0; j < njobs; ++j) {
                const size_t row = (size_t) j * FF + s.row_base, chunk = row / 32;
                if (std::memcmp(val.data() + row, whole_ff.data() + row, s.rows * 4)) {
                    for (int r = 0; r < s.rows; ++r)
                        if (std::memcmp(&val[row + r], &whole_ff[row + r], 4)) {
                            std::fprintf(stderr, "SwiGLU mismatch L%d rank%d job%d row%d: shard %.9g full %.9g\n",
                                         layer, rank, j, s.row_base + r, val[row + r], whole_ff[row + r]);
                            break;
                        }
                    throw std::runtime_error("SwiGLU shard differs");
                }
                require(!std::memcmp(iq.data() + row, whole_q.data() + row, s.rows), "quantized shard differs");
                require(!std::memcmp(sc.data() + chunk, whole_scale.data() + chunk, s.rows / 32 * 4), "scale shard differs");
                require(!std::memcmp(corr.data() + chunk, whole_hx.data() + chunk, s.rows / 32 * 4), "correction shard differs");
            }
        }
    }
};

// Fixed-routing tier benchmark, not end-to-end serving. One captured 48-layer
// consumer graph per backend. GPU->host ready markers replace the primary's
// attention/router work; the host polls memory, never CUDA APIs during rank work.
void fixed_benchmark(const Store& store, std::array<strata::core::MaxwellExperts, 4>& tiers,
    strata::core::MaxwellMetaTier& meta, int main_gpu, std::array<float*, 4>& partial,
    std::array<uint32_t*, 4>& flags, uint32_t* main_flag, uint32_t* expected,
    std::array<int32_t*, 2>& lists, float* rows, float* fallback, float* old_gpu,
    float* meta_gpu, int32_t* hits, cudaStream_t stream) {
    require(std::atoi(std::getenv("STRATA_MW_GPUFLAG")) == 2, "fixed benchmark requires GPUFLAG=2");
    ck(cudaSetDevice(main_gpu));
    struct Packet { std::vector<cpu::ActQ> acts; std::vector<int32_t> ids, entries; };
    std::array<Packet, 48> packets;
    for (int l = 0; l < 48; ++l) {
        auto& p = packets[l]; p.acts.resize(4); p.ids.resize(4 * K);
        std::mt19937 rng(9000 + l); std::normal_distribution<float> normal(0.f, 0.2f);
        for (int t = 0; t < 4; ++t) {
            std::vector<float> x(H); for (auto& v : x) v = normal(rng);
            cpu::act_quant_any(x.data(), H, p.acts[t]);
        }
        for (int i = 0; i < 4 * K; ++i) {
            p.ids[i] = (store.expert[l] + (i % K + K - i / K) % K) % 512;
            if (Store::get(l, p.ids[i], (void*) &store)) p.entries.push_back(i);
        }
    }
    hits[0] = 0;
    for (int i = 0; i < 4 * K; ++i)
        if (!Store::get(0, packets[0].ids[i], (void*) &store)) hits[1 + hits[0]++] = i;
    auto* ready = host<int32_t>(16); auto* done = host<int32_t>(16);
    auto* steps = dev<int32_t>(48);
    std::array<int32_t, 48> numbers{}; for (int i = 0; i < 48; ++i) numbers[i] = i + 1;
    ck(cudaMemcpyAsync(steps, numbers.data(), sizeof(numbers), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
    struct Graphs {
        std::array<cudaGraph_t, 2> graph{};
        std::array<cudaGraphExec_t, 2> exec{};
        ~Graphs() {
            for (auto p : exec) if (p) cudaGraphExecDestroy(p);
            for (auto p : graph) if (p) cudaGraphDestroy(p);
        }
    } graphs;
    const uint32_t* ff[4] = {flags[0], flags[1], flags[2], flags[3]};
    const float* pp[4] = {partial[0], partial[1], partial[2], partial[3]};
    std::string error;
    for (int variant = 0; variant < 2; ++variant) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < 48; ++l) {
            ck(cudaMemcpyAsync(ready, steps + l, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            if (variant) {
                require(meta.record_consume(stream, main_flag, l + 1, l, meta_gpu, fallback,
                                             4 * K, H, hits + 1, hits, error), error);
            } else {
                strata::kernels::sum_tp_rows_after_mw(main_flag, l + 1, expected + tp::expected_slot(l), ff,
                                                       rows, 4 * K, H, pp, lists[0], lists[1], stream);
                strata::kernels::wait_mw(main_flag, l + 1, expected + tp::expected_slot(l), ff, stream);
                strata::kernels::copy_rows_from_mapped_after(main_flag, l + 1, old_gpu, rows,
                                                             4 * K, H, hits + 1, hits, stream);
            }
        }
        ck(cudaMemcpyAsync(done, steps, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamEndCapture(stream, &graphs.graph[variant]));
        ck(cudaGraphInstantiate(&graphs.exec[variant], graphs.graph[variant], nullptr, nullptr, 0));
    }
    auto window = [&](int variant) {
        const auto start = std::chrono::steady_clock::now();
        *(volatile uint32_t*) main_flag = 0; *(volatile int32_t*) ready = 0; *(volatile int32_t*) done = 0;
        if (!variant)
            for (int l = 0; l < 48; ++l) expected[tp::expected_slot(l)] = tiers[0].seq() + l + 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ck(cudaGraphLaunch(graphs.exec[variant], stream));
        auto await = [&](volatile int32_t* word, int value) {
            unsigned spins = 0;
            while (*word < value)
                if ((++spins & 4095u) == 0 && std::chrono::steady_clock::now() - start > std::chrono::seconds(20))
                    throw std::runtime_error("fixed-routing benchmark timed out");
        };
        for (int l = 0; l < 48; ++l) {
            await(ready, l + 1);
            const auto& p = packets[l];
            if (variant) require(meta.decode(l, p.acts.data(), p.ids.data(), 4, K, error), error);
            else {
                const uint32_t next = tiers[0].seq() + 1;
                auto* list = lists[next & 1]; list[0] = (int) p.entries.size();
                std::copy(p.entries.begin(), p.entries.end(), list + 1);
                for (int r = 0; r < 4; ++r)
                    require(tiers[r].begin(l, p.acts.data(), p.ids.data(), 4, K, nullptr, partial[r], false, error), error);
                for (auto& rank : tiers) require(rank.finish(nullptr, error), error);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) main_flag = l + 1;
        }
        await(done, 1);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 48;
    };
    try {
        const auto burn = std::chrono::steady_clock::now();
        do { window(0); window(1); }
        while (std::chrono::steady_clock::now() - burn < std::chrono::seconds(2));
        for (int block = 0; block < 7; ++block) {
            const double a0 = window(0), b0 = window(1), b1 = window(1), a1 = window(0);
            std::printf("META_BENCH %d %.6f %.6f %.6f %.6f\n", block, a0, b0, b1, a1);
        }
        std::vector<float> old_result(4 * K * H), meta_result(old_result.size());
        ck(cudaMemcpy(old_result.data(), old_gpu, old_result.size() * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(meta_result.data(), meta_gpu, meta_result.size() * 4, cudaMemcpyDeviceToHost));
        const bool wire16 = std::getenv("STRATA_MW_META_WIRE16") != nullptr &&
                            std::atoi(std::getenv("STRATA_MW_META_WIRE16")) != 0;
        if (wire16) {
            double num = 0, den = 0;
            for (size_t i = 0; i < old_result.size(); ++i) {
                num += std::fabs((double) meta_result[i] - old_result[i]);
                den += std::fabs((double) old_result[i]);
            }
            require(std::isfinite(num) && num / std::max(den, 1e-30) < 5e-3, "fp16-wire bench result drifted beyond 5e-3");
        } else
        require(!std::memcmp(old_result.data(), meta_result.data(), old_result.size() * 4), "fixed-routing benchmark changed result bits");
    } catch (...) {
        meta.abort();
        for (int i = 0; i < (int) tp::expected_slots; ++i) expected[i] = 0;
        *(volatile uint32_t*) main_flag = UINT32_MAX;
        for (auto* f : flags) *(volatile uint32_t*) f = UINT32_MAX;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        cudaStreamSynchronize(stream);
        throw;
    }
    ck(cudaStreamSynchronize(stream));
    for (auto& p : graphs.exec) { ck(cudaGraphExecDestroy(p)); p = nullptr; }
    for (auto& p : graphs.graph) { ck(cudaGraphDestroy(p)); p = nullptr; }
    ck(cudaFree(steps)); ck(cudaFreeHost(ready)); ck(cudaFreeHost(done));
}

int main(int argc, char** argv) try {
    const bool bench_meta = argc == 4 && std::string(argv[3]) == "--bench-meta";
    const bool test_meta = bench_meta || (argc == 4 && std::string(argv[3]) == "--meta");
    if (argc != 3 && !test_meta) { std::fprintf(stderr, "usage: maxwell_tp_chain_test <shard1.gguf> <native-pack> [--meta|--bench-meta]\n"); return 2; }
    const char* flag = std::getenv("STRATA_MW_GPUFLAG");
    require(flag && (std::atoi(flag) == 1 || std::atoi(flag) == 2), "set STRATA_MW_GPUFLAG=1 or 2 before the test");
    ggml_cpu_init();
    strata::core::set_maxwell_portable(true);
    std::string error;
    require(cpu::expert_layout_load(argv[2], 48, 512, error), error);
    const auto& layout = cpu::expert_layout();
    require(layout.native && layout.fmt.size() == 48, "needs the 48-layer native layout");
    const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
    Store store;
    store.per_layer = test_meta ? 8 : 1;
    store.blob.resize(48); store.expert.resize(48);
    std::vector<std::pair<int32_t, int32_t>> assign;
    for (int l = 0; l < 48; ++l) {
        const auto& f = layout.fmt[l];
        const int e = (l * 17 + 7) % 512;
        store.expert[l] = e; store.blob[l].resize(f.bytes * store.per_layer);
        for (int x = 0; x < store.per_layer; ++x) assign.emplace_back(l, (e + x) % 512);
        for (int role = 0; role < 3; ++role) {
            const char* name = role == 0 ? "ffn_gate_exps.weight" : role == 1 ? "ffn_up_exps.weight" : "ffn_down_exps.weight";
            size_t shard = 0;
            const auto* tensor = model.find("blk." + std::to_string(l) + "." + name, &shard);
            require(tensor != nullptr, "missing expert tensor");
            const size_t bytes = role == 2 ? f.bytes - f.down_off : f.up_off;
            const size_t offset = role == 0 ? 0 : role == 1 ? f.up_off : f.down_off;
            for (int x = 0; x < store.per_layer; ++x)
                std::memcpy(store.blob[l].data() + (size_t) x * f.bytes + offset,
                            model.shard(shard).tensor_data(*tensor) + (size_t) ((e + x) % 512) * bytes, bytes);
        }
    }
    int ndev = 0; ck(cudaGetDeviceCount(&ndev));
    std::vector<int> dies;
    int main_gpu = -1;
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, d));
        if (p.major == 5 && p.minor == 0) dies.push_back(d);
        else if (p.major >= 7 && main_gpu < 0) main_gpu = d;
    }
    require(dies.size() == 4 && main_gpu >= 0, "needs four M10s and a main sm_70+ GPU");
    std::array<strata::core::MaxwellExperts, 4> tiers;
    ck(cudaSetDevice(main_gpu));
    std::array<float*, 4> partial{};
    std::array<uint32_t*, 4> flags{};
    for (int r = 0; r < 4; ++r) {
        partial[r] = host<float>((size_t) CAP * H);
        flags[r] = host<uint32_t>(16); std::fill_n(flags[r], 16, 0);
        require(tiers[r].open(dies[r], assign, Store::get, &store, 48, 512, 0, error, r, 4), error);
        require(tiers[r].resident() == (int64_t) assign.size() && tiers[r].held_pairs() == assign, "TP resident set changed");
        tiers[r].set_completion_flag(flags[r]);
    }
    ck(cudaSetDevice(main_gpu));
    auto* main_flag = host<uint32_t>(16);
    auto* expected = host<uint32_t>(tp::expected_slots);
    std::array<int32_t*, 2> lists{host<int32_t>(CAP + 1), host<int32_t>(CAP + 1)};
    auto* rows = host<float>((size_t) CAP * H);
    cudaStream_t stream{}; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    strata::core::MaxwellMetaTier meta;
    strata::core::MaxwellMetaTier::Config meta_config;
    float *meta_gpu = nullptr, *oracle_gpu = nullptr, *fallback = nullptr;
    int32_t* hits = nullptr;
    if (test_meta) {
        std::copy(dies.begin(), dies.end(), meta_config.devices.begin()); meta_config.main_device = main_gpu;
        meta_config.reserve_mib = 0;
        if (const char* v = std::getenv("STRATA_MW_META_DIRECT")) meta_config.direct_output = std::atoi(v) != 0;
        if (const char* v = std::getenv("STRATA_MW_META_HEAVY_FIRST")) meta_config.heavy_first = std::atoi(v) != 0;
        require(meta.open(meta_config, assign, Store::copy, &store, error), error);
        require(meta.resident() == (int64_t) assign.size() && meta.held_pairs() == assign, "Meta changed the routed expert assignment");
        ck(cudaSetDevice(main_gpu));
        meta_gpu = dev<float>((size_t) CAP * H); oracle_gpu = dev<float>((size_t) CAP * H);
        fallback = host<float>((size_t) CAP * H); hits = host<int32_t>(CAP + 1);
    }
    double worst = 0;
    int services = 0, empty = 0, prefill = 0;
    std::vector<float> repeat_rows;
    {
        Reference reference(dies[0], layout.max_blob * store.per_layer);
        auto check_case = [&](int l, int nt, bool no_jobs, bool prompt, bool check_images) {
            const auto& f = layout.fmt[l];
            std::mt19937 rng(1000 + l * 31 + nt);
            std::normal_distribution<float> normal(0.f, 0.2f);
            std::vector<float> input((size_t) nt * H);
            for (auto& x : input) x = normal(rng);
            std::vector<cpu::ActQ> a(nt);
            for (int t = 0; t < nt; ++t) cpu::act_quant_any(input.data() + (size_t) t * H, H, a[t]);
            std::vector<int32_t> ids((size_t) nt * K), mapping(ids.size());
            std::vector<int> entries;
            for (size_t i = 0; i < ids.size(); ++i) {
                // Ten distinct IDs per token, with the owned expert at a varying
                // rank; the same expert is reused ACROSS tokens, as in real routing.
                const int offset = ((int) (i % K) + K - (int) (i / K) % K) % K;
                ids[i] = (store.expert[l] + offset + (no_jobs ? K : 0)) % 512;
                mapping[i] = (int) i + 7;
                if (Store::get(l, ids[i], &store)) entries.push_back((int) i);
            }
            const auto answer = reference.run(store.blob[l], f, a, entries, ids, store.expert[l]);
            if (check_images && !entries.empty()) reference.intermediates(store.blob[l], f, l, (int) entries.size());
            ck(cudaSetDevice(main_gpu));
            std::fill_n(rows, (size_t) CAP * H, prompt ? 0.f : 5.25f);
            const uint32_t want = tiers[0].seq() + 1;
            const uint32_t ring = (uint32_t) (services % 96 + 1);   // exercise expected-slot wrapping beyond 64
            auto* list = lists[want & 1];
            list[0] = (int) entries.size();
            std::copy(entries.begin(), entries.end(), list + 1);
            for (int r = 0; r < 4; ++r)
                require(tiers[r].begin(l, a.data(), ids.data(), nt, K, nullptr, prompt ? nullptr : partial[r], false, error), error);
            for (int r = 0; r < 4; ++r) {
                if (prompt) require(tiers[r].finish_to(rows, mapping.data(), 7, H, error, true), error);
                else require(tiers[r].finish(rows, error), error);
            }
            ck(cudaSetDevice(main_gpu));
            if (!prompt) {
                expected[tp::expected_slot(ring - 1)] = want;
                *main_flag = ring;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                const uint32_t* ff[4] = {flags[0], flags[1], flags[2], flags[3]};
                const float* pp[4] = {partial[0], partial[1], partial[2], partial[3]};
                strata::kernels::sum_tp_rows_after_mw(main_flag, ring, expected + tp::expected_slot(ring - 1), ff,
                                                       rows, nt * K, H, pp, lists[0], lists[1], stream);
                ck(cudaStreamSynchronize(stream));
                ++services;
                if (no_jobs) ++empty;
                for (const auto& tier : tiers) require(tier.seq() == want, "rank service counters diverged");
            } else ++prefill;
            double num = 0, den = 0;
            for (size_t j = 0; j < entries.size(); ++j)
                for (int h = 0; h < H; ++h) {
                    const float value = rows[(size_t) entries[j] * H + h], ref = answer[j * H + h];
                    require(std::isfinite(value) && std::isfinite(ref), "non-finite TP/reference output");
                    num += std::fabs((double) value - ref); den += std::fabs((double) ref);
                    if (!prompt) {
                        const float cpu_sum = ((partial[0][j * H + h] + partial[1][j * H + h]) + partial[2][j * H + h]) + partial[3][j * H + h];
                        require(!std::memcmp(&value, &cpu_sum, sizeof(float)), "GPU reduction differs from fixed rank order");
                    }
                }
            for (size_t i = 0; i < ids.size(); ++i)
                if (!Store::get(l, ids[i], &store))
                    for (int h = 0; h < H; ++h) require(rows[i * H + h] == (prompt ? 0.f : 5.25f), "assembly overwrote an unowned row");
            const double rel = num / std::max(den, 1e-30);
            require(rel < 1e-5, "TP sum differs from the whole expert: " + std::to_string(rel));
            if (l == 8 && nt == 4 && !no_jobs && !prompt) {
                const size_t count = (size_t) nt * K * H;
                if (repeat_rows.empty()) repeat_rows.assign(rows, rows + count);
                else require(!std::memcmp(rows, repeat_rows.data(), count * sizeof(float)), "fixed-input TP repeat changed bits");
            }
            worst = std::max(worst, rel);
            if (test_meta) {
                const size_t count = (size_t) nt * K * H;
                if (prompt) {
                    std::vector<float> meta_rows(count, 0.f);
                    require(meta.prefill(l, a.data(), ids.data(), nt, K, meta_rows.data(), mapping.data(), 7, nt * K, error), error);
                    require(!std::memcmp(rows, meta_rows.data(), count * sizeof(float)), "Meta prefill changed rank-order output bits");
                } else {
                    ck(cudaSetDevice(main_gpu));
                    std::fill_n(fallback, (size_t) CAP * H, 5.25f);
                    // One unowned row is a primary-GPU hit; other unowned rows
                    // carry CPU/fallback data. Check both delivery rules exactly.
                    hits[0] = 1; hits[1] = -1;
                    for (size_t i = 0; i < ids.size(); ++i)
                        if (!Store::get(l, ids[i], &store)) { hits[1] = (int) i; break; }
                    strata::kernels::copy_rows_from_mapped(oracle_gpu, rows, nt * K, H, hits + 1, hits, stream);
                    require(meta.decode(l, a.data(), ids.data(), nt, K, error), error);
                    *main_flag = ring + 1;   // Meta ticket parity must be independent of the caller's ring
                    require(meta.record_consume(stream, main_flag, ring + 1, l, meta_gpu, fallback, nt * K, H,
                                                 hits + 1, hits, error), error);
                    ck(cudaStreamSynchronize(stream));
                    for (size_t i = 0; i < count; ++i) require(fallback[i] == 5.25f, "Meta modified the caller's fallback input");
                    std::vector<float> actual(count), oracle(count);
                    ck(cudaMemcpy(actual.data(), meta_gpu, count * sizeof(float), cudaMemcpyDeviceToHost));
                    ck(cudaMemcpy(oracle.data(), oracle_gpu, count * sizeof(float), cudaMemcpyDeviceToHost));
                    const bool wire16 = std::getenv("STRATA_MW_META_WIRE16") != nullptr &&
                                        std::atoi(std::getenv("STRATA_MW_META_WIRE16")) != 0;
                    if (wire16) {
                        // fp16 partials round each rank's contribution: same sum order, bounded drift.
                        double num = 0, den = 0;
                        for (size_t i = 0; i < count; ++i) {
                            num += std::fabs((double) actual[i] - oracle[i]);
                            den += std::fabs((double) oracle[i]);
                        }
                        require(std::isfinite(num) && num / std::max(den, 1e-30) < 5e-3,
                                "fp16-wire Meta output drifted beyond 5e-3");
                    } else
                    require(!std::memcmp(actual.data(), oracle.data(), count * sizeof(float)), "Meta output differs bitwise from legacy TP + hit/fallback copy");
                }
            }
        };
        for (int l = 0; l < 48; ++l) {
            for (int nt : {1, 2, 3, 4}) check_case(l, nt, false, false, nt == 1);
            if (l % 4 == 0) check_case(l, 4, true, false, false);
        }
        // Repeated identical jobs exercise staging/list reuse; any nondeterminism
        // is a test failure against the fixed-rank CPU sum and whole-expert oracle.
        for (int i = 0; i < 100; ++i) check_case(8, 4, false, false, false);
        for (int l : {0, 1, 8}) for (int nt : {1, 4, 63, 64}) check_case(l, nt, false, true, false);
    }
    if (bench_meta)
        fixed_benchmark(store, tiers, meta, main_gpu, partial, flags, main_flag, expected, lists,
                         rows, fallback, oracle_gpu, meta_gpu, hits, stream);
    if (test_meta) {
        ck(cudaSetDevice(main_gpu));
        std::vector<float> x(H, 0.1f); cpu::ActQ a; cpu::act_quant_any(x.data(), H, a);
        const int32_t id = store.expert[0];
        require(meta.decode(0, &a, &id, 1, 1, error), error);
        // Double-buffered partials: ONE outstanding consume is safe (the next submission
        // writes the other set); a SECOND outstanding one must be refused.
        require(meta.decode(0, &a, &id, 1, 1, error), error);
        require(!meta.decode(0, &a, &id, 1, 1, error), "Meta allowed a second outstanding submission");
        *main_flag = 0; hits[0] = 0;
        require(meta.record_consume(stream, main_flag, 1, 0, meta_gpu, fallback, 1, H, hits + 1, hits, error), error);
        ck(cudaStreamSynchronize(stream));   // cancellation must release the consumer despite main==0
        std::vector<float> zero(H);
        ck(cudaMemcpy(zero.data(), meta_gpu, H * sizeof(float), cudaMemcpyDeviceToHost));
        for (float v : zero) require(v == 0.f, "aborted Meta consumer did not zero its output");
        require(!meta.decode(0, &a, &id, 1, 1, error), "Meta failure did not latch");
        meta.close();
        require(meta.open(meta_config, assign, Store::copy, &store, error), error);
        const int32_t invalid_mapping = 8;
        require(!meta.prefill(0, &a, &id, 1, 1, fallback, &invalid_mapping, 7, 1, error), "Meta allowed an out-of-range prefill scatter");
        meta.close();
        auto failed_source = [](int64_t, int64_t, uint8_t*, size_t, void*) { return false; };
        require(!meta.open(meta_config, assign, failed_source, nullptr, error) && !meta.valid(), "Meta accepted failed weight loading");
        std::printf("Meta PASS: exact legacy TP parity, hit/fallback rows, consumption tickets, overlap rejection, abort release, scatter bounds, failed-source cleanup; direct=%d heavy=%d\n",
                    meta_config.direct_output, meta_config.heavy_first);
        ck(cudaSetDevice(main_gpu));
        ck(cudaFree(meta_gpu)); ck(cudaFree(oracle_gpu)); ck(cudaFreeHost(fallback)); ck(cudaFreeHost(hits));
    }
    std::printf("PASS: 48-layer loader + exact shard intermediates, %d decode services (%d empty), 100 reuse repetitions, "
                "%d prefill cases, rank-order GPU/CPU bit parity; TP vs whole worst relL1 %.3e\n", services, empty, prefill, worst);
    ck(cudaSetDevice(main_gpu)); ck(cudaStreamDestroy(stream));
    for (auto& t : tiers) t.close();
    for (auto* p : partial) ck(cudaFreeHost(p));
    for (auto* p : flags) ck(cudaFreeHost(p));
    for (auto* p : lists) ck(cudaFreeHost(p));
    ck(cudaFreeHost(main_flag)); ck(cudaFreeHost(expected)); ck(cudaFreeHost(rows));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "maxwell_tp_chain_test: %s\n", e.what());
    return 1;
}
