#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
// One routed-expert device from the caller's perspective. Physical ranks,
// sharding, streams, flags, partials, staging and reduction belong to this object.
// Not a backend for attention, shared experts, KV or the rest of the model.
class MaxwellMetaTier {
public:
    struct Config {
        std::array<int, 4> devices{1, 2, 3, 4};
        int main_device = 0, reserve_mib = 150;
        int64_t layers = 48, experts = 512;
        bool direct_output = false;   // separately measured: reduce straight into the consumer's VRAM rows
        bool heavy_first = false;     // separately measured: launch the larger shards first
    };
    using CopyBlob = bool (*)(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes, void* context);
    MaxwellMetaTier();
    ~MaxwellMetaTier();
    MaxwellMetaTier(const MaxwellMetaTier&) = delete;
    MaxwellMetaTier& operator=(const MaxwellMetaTier&) = delete;
    bool open(const Config&, const std::vector<std::pair<int32_t, int32_t>>& assignment,
              CopyBlob, void* context, std::string& error);
    void close();
    bool valid() const;
    bool owns(int64_t layer, int64_t expert) const;
    int64_t resident() const;
    const std::vector<std::pair<int32_t, int32_t>>& held_pairs() const;
    uint32_t sequence() const;

    // One in-flight decode group. A second submission before the GPU has consumed
    // our buffers is rejected, never allowed to overwrite live partials. No hot-path
    // CUDA sync/query is needed: the reduction publishes a host-visible consumption ticket.
    bool decode(int64_t layer, const kernels::cpu::ActQ* acts, const int32_t* ids,
                int64_t tokens, int64_t topk, std::string& error);
    // Called during graph recording. The caller sees one completion/reduction
    // operation, not four rank flags. main_flag protects the parent's CPU/PLE work.
    bool record_consume(void* stream, const uint32_t* main_flag, uint32_t ring, int step,
                        float* device_rows, const float* fallback_rows, int64_t rows, int64_t width,
                        const int32_t* hit_rows, const int32_t* hit_count, std::string& error);
    bool prefill(int64_t layer, const kernels::cpu::ActQ* acts, const int32_t* ids,
                 int64_t tokens, int64_t topk, float* host_rows, const int32_t* row_of_entry,
                 int32_t row_base, int64_t row_capacity, std::string& error);
    void abort();
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace strata::core
