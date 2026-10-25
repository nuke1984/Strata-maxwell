#include "strata/core/maxwell_tp_plan.hpp"
#include <cstdio>
#include <stdexcept>

namespace tp = strata::core::maxwell_tp;
void require(bool ok) { if (!ok) throw std::runtime_error("TP invariant failed"); }
int main() try {
    for (int layer = 0; layer < 48; ++layer) {
        int coverage[640] = {}, blocks = 0;
        uint64_t bytes = 0;
        for (int rank = 0; rank < 4; ++rank) {
            const auto s = tp::slice(rank, layer);
            require(s.rows == 128 || s.rows == 192);
            blocks += s.blocks;
            bytes += tp::shard_bytes(s, 820);
            for (int r = s.row_base; r < s.row_base + s.rows; ++r) ++coverage[r];
            require(tp::iq_slice_valid(s.row_base, s.rows, 4, false));
            require(tp::iq_slice_valid(s.row_base, s.rows, 2, false));
            require(tp::iq_slice_valid(s.row_base, s.rows, 4, true));
            require(tp::iq_slice_valid(s.row_base, s.rows, 1, false) == (s.rows == 128));
            require(tp::iq_slice_valid(s.row_base, s.rows, 2, true) == (s.rows == 128));
        }
        require(blocks == 10 && bytes == 10ull * (128 * 820 + 46080));
        for (int count : coverage) require(count == 1);
    }
    for (int rank = 0; rank < 4; ++rank) {
        int slots[4] = {};
        for (int layer = 0; layer < 48; ++layer) ++slots[(rank + layer) % 4];
        for (int count : slots) require(count == 12);
    }
    for (uint64_t step = 0; step < 10000; ++step) require(tp::expected_slot(step) == step % 64);
    require(!tp::iq_slice_valid(1, 128, 4, false));
    require(!tp::iq_slice_valid(576, 128, 4, false));
    std::array<int, 4> devices{};
    require(tp::parse_devices("1,2,3,4", 0, devices).empty());
    require(tp::parse_devices(" 1, 2,3,4 ", 0, devices).empty());
    for (const auto* bad : {"", "1,2,3", "1,2,3,4,5", "1,2,2,4", "0,1,2,3", "1,x,3,4", "1,2,3,4,", "1,2,3,4junk"})
        require(!tp::parse_devices(bad, 0, devices).empty());
    std::puts("PASS: 48-layer shard coverage, byte sums, rotation, IQ geometries, 10000 ring slots, device validation");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
