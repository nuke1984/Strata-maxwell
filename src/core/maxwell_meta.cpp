#include "strata/core/maxwell_meta.hpp"
#include "strata/core/maxwell_experts.hpp"
#include "strata/core/maxwell_tp_plan.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace strata::core {
namespace {
constexpr int H = 2560, CAP = 640;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct RestoreDevice {
    int device = 0;
    RestoreDevice() { check(cudaGetDevice(&device)); }
    ~RestoreDevice() { cudaSetDevice(device); }
};
template<class T> T* mapped(size_t n) {
    T* p = nullptr;
    check(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(p, 0, n * sizeof(T));
    return p;
}
}
struct MaxwellMetaTier::Impl {
    Config config;
    std::array<MaxwellExperts, 4> rank;
    std::array<float*, 4> partial[2]{};   // double-buffered: submission s uses set (s+1)&1
    std::array<uint32_t*, 4> flag{};
    std::array<int32_t*, 2> row_map{};
    float* returned_rows = nullptr;   // reference transport staging, never the caller's read-only fallback
    std::vector<uint8_t> shared_acts; // packed once per window, shared by the four ranks
    bool wire16 = false;
    uint32_t *expected = nullptr, *consumed = nullptr, *cancel = nullptr, *counter = nullptr;
    std::vector<std::pair<int32_t, int32_t>> assignment;
    uint32_t submitted = 0;
    bool ready = false, failed = false;
    std::string failure;
    bool fail(const std::string& why, std::string& error) {
        failed = true; failure = why; error = why;
        if (cancel) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) cancel = 1;
        }
        return false;
    }
    bool reusable(std::string& error) {
        if (failed) { error = failure; return false; }
        // Double-buffered partials: one reduce may still be in flight - the NEXT submission
        // writes the other set. Reusing set p requires the reduce of submitted-1 done, which
        // the in-order consumption ticket already implies at this point.
        if (submitted >= 2 && *(volatile uint32_t*) consumed < submitted - 1)
            return fail("Maxwell Meta: previous decode partials are still in flight", error);
        return true;
    }
    std::array<int, 4> order(int64_t layer) const {
        std::array<int, 4> result{0, 1, 2, 3};
        if (config.heavy_first)
            for (int i = 1; i < 4; ++i) {
                const int key = result[i];
                int j = i;
                while (j && maxwell_tp::slice(key, (int) layer).rows > maxwell_tp::slice(result[j - 1], (int) layer).rows) {
                    result[j] = result[j - 1]; --j;
                }
                result[j] = key;
            }
        return result;
    }
    ~Impl() {
        int previous = 0; cudaGetDevice(&previous);
        if (cancel) { *(volatile uint32_t*) cancel = 1; std::atomic_thread_fence(std::memory_order_seq_cst); }
        // Unblock any recorded consumer before draining/freeing rank resources.
        cudaSetDevice(config.main_device);
        cudaDeviceSynchronize();
        for (auto& r : rank) r.close();
        cudaSetDevice(config.main_device);
        if (counter) cudaFree(counter);
        for (auto& set : partial) for (auto* p : set) if (p) cudaFreeHost(p);
        for (auto* p : flag) if (p) cudaFreeHost(p);
        for (auto* p : row_map) if (p) cudaFreeHost(p);
        if (returned_rows) cudaFreeHost(returned_rows);
        for (auto* p : {expected, consumed, cancel}) if (p) cudaFreeHost(p);
        cudaSetDevice(previous);
    }
};
MaxwellMetaTier::MaxwellMetaTier() = default;
MaxwellMetaTier::~MaxwellMetaTier() = default;
void MaxwellMetaTier::close() { p_.reset(); }
bool MaxwellMetaTier::valid() const { return p_ && p_->ready; }
bool MaxwellMetaTier::owns(int64_t l, int64_t e) const { return valid() && p_->rank[0].owns(l, e); }
int64_t MaxwellMetaTier::resident() const { return valid() ? (int64_t) p_->assignment.size() : 0; }
uint32_t MaxwellMetaTier::sequence() const { return valid() ? p_->submitted : 0; }
const std::vector<std::pair<int32_t, int32_t>>& MaxwellMetaTier::held_pairs() const {
    static const std::vector<std::pair<int32_t, int32_t>> empty;
    return valid() ? p_->assignment : empty;
}
void MaxwellMetaTier::abort() { if (p_) { std::string ignored; p_->fail("Maxwell Meta: aborted", ignored); } }

bool MaxwellMetaTier::open(const Config& config, const std::vector<std::pair<int32_t, int32_t>>& assignment,
                            CopyBlob copy, void* context, std::string& error) {
    close();
    error.clear();
    try {
        RestoreDevice restore;
        const char* mode = std::getenv("STRATA_MW_GPUFLAG");
        if (!mode || (std::atoi(mode) != 1 && std::atoi(mode) != 2))
            throw std::runtime_error("Maxwell Meta requires STRATA_MW_GPUFLAG=1 or 2");
        if (!copy || assignment.empty() || config.layers != 48 || config.experts <= 0)
            throw std::runtime_error("Maxwell Meta: invalid source/assignment/geometry");
        const auto& layout = kernels::cpu::expert_layout();
        if (!layout.native || layout.n_layers != config.layers || layout.n_expert != config.experts)
            throw std::runtime_error("Maxwell Meta: native layout and requested geometry differ");
        std::vector<unsigned char> seen((size_t) config.layers * config.experts, 0);
        for (const auto& [l, e] : assignment) {
            if (l < 0 || l >= config.layers || e < 0 || e >= config.experts || seen[(size_t) l * config.experts + e]++)
                throw std::runtime_error("Maxwell Meta: invalid or duplicate routed expert");
        }
        int previous = 0; check(cudaGetDevice(&previous));
        check(cudaSetDevice(config.main_device));
        p_ = std::make_unique<Impl>();
        p_->config = config; p_->assignment = assignment;
        for (int i = 0; i < 4; ++i) {
            if (config.devices[i] == config.main_device) throw std::runtime_error("Maxwell Meta rank is the primary GPU");
            for (int j = 0; j < i; ++j)
                if (config.devices[i] == config.devices[j]) throw std::runtime_error("Maxwell Meta has duplicate devices");
            cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, config.devices[i]));
            if (prop.major != 5 || prop.minor != 0) throw std::runtime_error("Maxwell Meta needs four sm_50 ranks");
            p_->partial[0][i] = mapped<float>((size_t) CAP * H);
            p_->partial[1][i] = mapped<float>((size_t) CAP * H);
            p_->flag[i] = mapped<uint32_t>(16);
        }
        for (auto& map : p_->row_map) map = mapped<int32_t>(CAP);
        if (!config.direct_output) p_->returned_rows = mapped<float>((size_t) CAP * H);
        p_->expected = mapped<uint32_t>(maxwell_tp::expected_slots);
        p_->consumed = mapped<uint32_t>(16); p_->cancel = mapped<uint32_t>(16);
        check(cudaMalloc(&p_->counter, sizeof(uint32_t))); check(cudaMemset(p_->counter, 0, sizeof(uint32_t)));
        struct Source {
            CopyBlob copy; void* context; std::mutex mutex;
        } source{copy, context, {}};
        auto blob = [](int64_t l, int64_t e, void* context) -> const uint8_t* {
            auto& s = *(Source*) context;
            static thread_local std::vector<uint8_t> owned;
            owned.resize((size_t) kernels::cpu::expert_layout().blob_bytes(l));
            std::lock_guard<std::mutex> lock(s.mutex);
            return s.copy(l, e, owned.data(), owned.size(), s.context) ? owned.data() : nullptr;
        };
        std::array<std::string, 4> errors;
        std::array<unsigned char, 4> ok{};
        std::vector<std::jthread> fills;
        for (int i = 0; i < 4; ++i) fills.emplace_back([&, i] {
            try {
                const auto start = std::chrono::steady_clock::now();
                ok[i] = p_->rank[i].open(config.devices[i], assignment, blob, &source, config.layers,
                                         config.experts, config.reserve_mib, errors[i], i, 4);
                if (ok[i]) std::fprintf(stderr, "strata Meta rank %d CUDA%d: %lld routed shards, %.3f GiB, %.1f s\n",
                    i, config.devices[i], (long long) p_->rank[i].resident(), p_->rank[i].gib(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            } catch (const std::exception& e) { errors[i] = e.what(); }
              catch (...) { errors[i] = "unknown rank loader failure"; }
        });
        for (auto& fill : fills) fill.join();
        for (int i = 0; i < 4; ++i) {
            if (!ok[i]) throw std::runtime_error("Maxwell Meta rank " + std::to_string(i) + ": " + errors[i]);
            if (p_->rank[i].held_pairs() != assignment) throw std::runtime_error("Maxwell Meta resident sets differ");
            p_->rank[i].set_completion_flag(p_->flag[i]);
        }
        if (const char* v = std::getenv("STRATA_MW_META_WIRE16"); v && std::atoi(v) != 0) {
            p_->wire16 = true;
            for (auto& r : p_->rank) r.set_tp_wire16(true);
        }
        p_->shared_acts.reserve((size_t) 64 * kernels::cuda::expert_maxwell_geo::ACTS_STRIDE);
        p_->ready = true;
        check(cudaSetDevice(previous));
        std::fprintf(stderr, "strata Maxwell Meta: ONE routed-expert tier, %zu experts, direct=%d heavy-first=%d wire16=%d\n",
                     assignment.size(), config.direct_output, config.heavy_first, p_->wire16);
        return true;
    } catch (const std::exception& e) {
        error = e.what(); close(); return false;
    }
}
bool MaxwellMetaTier::decode(int64_t l, const kernels::cpu::ActQ* acts, const int32_t* ids,
                              int64_t tokens, int64_t topk, std::string& error) {
    if (!valid()) { error = "Maxwell Meta is not open"; return false; }
    auto& p = *p_;
    if (!p.reusable(error)) return false;
    if (!acts || !ids || l < 0 || l >= p.config.layers || tokens < 1 || tokens > 64 || topk < 1 || topk > 10)
        return p.fail("Maxwell Meta: invalid decode shape", error);
    if (p.submitted == std::numeric_limits<uint32_t>::max()) return p.fail("Maxwell Meta: sequence exhausted; reload tier", error);
    const uint32_t next = p.submitted + 1;
    auto* map = p.row_map[next & 1];
    std::fill_n(map, (size_t) tokens * topk, -1);
    int jobs = 0;
    for (int i = 0; i < tokens * topk; ++i) {
        if (ids[i] < 0 || ids[i] >= p.config.experts) return p.fail("Maxwell Meta: invalid routed ID", error);
        if (owns(l, ids[i])) map[i] = jobs++;
    }
    // Pack the window's activations ONCE: the four ranks consume identical ActQ bytes, so the
    // per-rank begin() bulk-copies this block instead of re-running the per-token loop four times.
    constexpr size_t stride = (size_t) kernels::cuda::expert_maxwell_geo::ACTS_STRIDE;
    p.shared_acts.resize((size_t) tokens * stride);
    for (int64_t t = 0; t < tokens; ++t) {
        const size_t nc4 = (size_t) (H / 32) * 4;
        std::memcpy(p.shared_acts.data() + (size_t) t * stride, acts[t].q, H);
        std::memcpy(p.shared_acts.data() + (size_t) t * stride + H, acts[t].scale, nc4);
        std::memcpy(p.shared_acts.data() + (size_t) t * stride + H + nc4, acts[t].hx, nc4);
    }
    auto* dst_set = p.partial[next & 1].data();
    for (int rank : p.order(l))
        p.rank[rank].set_shared_acts(p.shared_acts.data());
    for (int rank : p.order(l))
        if (!p.rank[rank].begin(l, acts, ids, tokens, topk, nullptr, dst_set[rank], false, error)) return p.fail(error, error);
    for (auto& rank : p.rank) rank.set_shared_acts(nullptr);
    for (auto& rank : p.rank) {
        if (!rank.finish(nullptr, error)) return p.fail(error, error);
        if (rank.seq() != next) return p.fail("Maxwell Meta: rank sequence mismatch", error);
    }
    p.expected[maxwell_tp::expected_slot(l)] = next;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    p.submitted = next;
    return true;
}
bool MaxwellMetaTier::record_consume(void* stream, const uint32_t* main, uint32_t ring, int step,
                                      float* rows, const float* fallback, int64_t n, int64_t width,
                                      const int32_t* hits, const int32_t* count, std::string& error) {
    if (!valid()) { error = "Maxwell Meta is not open"; return false; }
    auto& p = *p_;
    if (!stream || !main || !rows || !fallback || !hits || !count || step < 0 || step >= p.config.layers ||
        n < 1 || n > CAP || width != H || ((uintptr_t) rows & 15u) || ((uintptr_t) fallback & 15u))
        return p.fail("Maxwell Meta: invalid consumer shape/alignment", error);
    const uint32_t* flags[4] = {p.flag[0], p.flag[1], p.flag[2], p.flag[3]};
    auto* exp = p.expected + maxwell_tp::expected_slot(step);
    kernels::maxwell_meta_reduce(main, ring, exp, flags, p.partial[0].data(), p.partial[1].data(),
                                 p.row_map[0], p.row_map[1],
                                 p.config.direct_output ? rows : p.returned_rows, fallback,
                                 n, width, hits, count, p.config.direct_output, p.wire16,
                                 p.cancel, p.counter, p.consumed, stream);
    if (!p.config.direct_output) {
        // Reference transport retained inside Meta for isolated A/B; no rank details escape to the verifier.
        kernels::copy_rows_from_mapped(rows, p.returned_rows, n, width, hits, count, stream);
        kernels::maxwell_meta_complete(exp, p.consumed, stream);
    }
    return true;
}
bool MaxwellMetaTier::prefill(int64_t l, const kernels::cpu::ActQ* acts, const int32_t* ids,
                               int64_t tokens, int64_t topk, float* rows, const int32_t* mapping,
                               int32_t base, int64_t capacity, std::string& error) {
    if (!valid()) { error = "Maxwell Meta is not open"; return false; }
    auto& p = *p_;
    if (!p.reusable(error)) return false;
    if (!acts || !ids || !rows || !mapping || capacity < 1 || l < 0 || l >= p.config.layers || tokens < 1 || tokens > 64 || topk < 1 || topk > 10)
        return p.fail("Maxwell Meta: invalid prefill shape", error);
    for (int i = 0; i < tokens * topk; ++i) {
        if (ids[i] < 0 || ids[i] >= p.config.experts) return p.fail("Maxwell Meta: invalid routed ID", error);
        if (owns(l, ids[i])) {
            if (mapping[i] < base || (int64_t) mapping[i] - base >= capacity)
                return p.fail("Maxwell Meta: invalid prefill scatter", error);
            std::memset(rows + (size_t) (mapping[i] - base) * H, 0, H * sizeof(float));
        }
    }
    for (int rank : p.order(l))
        if (!p.rank[rank].begin(l, acts, ids, tokens, topk, nullptr, nullptr, false, error)) return p.fail(error, error);
    for (auto& rank : p.rank)
        if (!rank.finish_to(rows, mapping, base, H, error, true)) return p.fail(error, error);
    return true;
}
}  // namespace strata::core
