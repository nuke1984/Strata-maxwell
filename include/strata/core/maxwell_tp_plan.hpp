#pragma once

// Pure host-side invariants shared by the routed-expert TP loader, launcher and
// verifier. No CUDA dependency: boundary/error cases are testable without GPUs.
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace strata::core::maxwell_tp {
inline constexpr unsigned expected_slots = 64;
inline constexpr unsigned expected_slot(uint64_t step) { return (unsigned) (step & (expected_slots - 1)); }
struct Slice { int block_base, blocks, row_base, rows; };
inline constexpr Slice slice(int rank, int layer) {
    const int slot = (rank + layer % 4) % 4;
    const int blocks = 2 + (slot < 2);
    const int base = slot * 2 + (slot < 2 ? slot : 2);
    return {base, blocks, base * 64, blocks * 64};
}
inline constexpr uint64_t shard_bytes(Slice s, size_t gu_row) {
    return (uint64_t) s.blocks * (128ull * gu_row + 46080ull);
}
inline constexpr bool iq_slice_valid(int base, int rows, int split, bool wide) {
    if (split != 1 && split != 2 && split != 4) return false;
    const int per_block = (wide && split >= 2 ? 256 : 128) / split;
    return base >= 0 && rows > 0 && base % 32 == 0 && base <= 640 && rows <= 640 - base && rows % per_block == 0;
}
inline std::string parse_devices(std::string_view text, int primary, std::array<int, 4>& out) {
    size_t start = 0;
    int n = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        std::string_view token = text.substr(start, comma == text.npos ? text.size() - start : comma - start);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.remove_prefix(1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.remove_suffix(1);
        int device = -1;
        if (token.empty() || n >= 4) return "needs exactly four distinct M10 devices";
        const auto result = std::from_chars(token.data(), token.data() + token.size(), device);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size() || device < 0 || device == primary)
            return "invalid M10 device (must differ from the primary GPU)";
        for (int i = 0; i < n; ++i) if (out[i] == device) return "duplicate M10 device";
        out[n++] = device;
        if (comma == text.npos) break;
        start = comma + 1;
    }
    return n == 4 ? std::string() : "needs exactly four distinct M10 devices";
}
}  // namespace strata::core::maxwell_tp
