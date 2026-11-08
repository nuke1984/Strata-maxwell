// include/strata/kernels/cuda/expert_maxwell_iq4nl.hpp - the IQ4_NL (ggml type 20) down layout the
// Maxwell tier's down kernels read, and the one host conversion that produces it.
//
// WHY THIS FILE EXISTS: the conversion below runs at load() in production (src/core/maxwell_experts.cpp)
// and must also be exercised by the parity tool (src/kernels/cuda/expert_maxwell_iq_parity.cpp).  The
// tool has to prove the code PRODUCTION runs, not a copy of it, so the function lives in a header both
// translation units include.  It is plain host code: no CUDA types, no device calls, no allocation, so
// the engine's C++ compiler and nvcc's host pass build the same bytes-in/bytes-out routine.
//
// THE FORMAT: an IQ4_NL block is 18 bytes - fp16 scale d, then 16 code bytes for 32 weights - and a
// down row is FF = 640 values = 20 blocks = 360 bytes.  Stored raw, a block's codes sit at
// row*360 + b*18 + 2: 2-byte aligned, never 16, which is why the kernel has to read them as eight
// LDG.U16 and never as one vector load (results/38-down-iq4nl-sass-before.txt measures this).
//
// THE PLANAR FORM (same bytes, same total size, no VRAM cost, lossless):
//     codes   at codes_off  + row*320 + b*16 + k   ==  raw[row*360 + b*18 + 2 + k]   (k = 0..15)
//     scales  at scales_off + row*40  + b*2  + j   ==  raw[row*360 + b*18 + j]       (j = 0..1)
// with scales_off = codes_off + rows*320.  So a code block is a naturally aligned int4 (16 B) and the
// scale of the block it belongs to is 40 bytes into a separate plane.  It is a permutation: every input
// byte lands in exactly one output byte and no value is recomputed.  rows*320 + rows*40 == rows*360.
//
// ALIGNMENT: the caller must place codes_off at a 16-byte boundary (the tier's down_off is
// 2*gu_row*FF, and every gate/up row is a whole number of 16-byte blocks, so it is).  The tier asserts
// this at load time rather than trusting it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cuda::expert_maxwell_iq4nl {

inline constexpr int kQK = 32;                                     ///< weights per IQ4_NL block
inline constexpr int kBlocks = 20;                                 ///< FF / kQK, a down row's blocks
inline constexpr size_t kRawBlock = 18;                            ///< fp16 d + 16 code bytes
inline constexpr size_t kRawRow = (size_t) kBlocks * kRawBlock;    ///< 360 B, raw
inline constexpr size_t kCodesRow = (size_t) kBlocks * 16;         ///< 320 B, planar codes
inline constexpr size_t kScalesRow = (size_t) kBlocks * 2;         ///< 40 B, planar fp16 scales
static_assert(kCodesRow + kScalesRow == kRawRow, "the planar form must be the same bytes as the raw one");
static_assert(kCodesRow % 16 == 0, "a codes row must keep every block 16-byte aligned");

/// The planar region's size for `rows` down rows: identical to the raw region's, by construction.
inline constexpr size_t region_bytes(size_t rows) { return rows * (kCodesRow + kScalesRow); }

/// Raw interleaved rows -> planar codes plane followed by the scales plane, inside `dst`.
/// `dst` must hold region_bytes(rows) bytes and `src` rows*kRawRow; they must not overlap.
inline void planarize(const uint8_t* src, uint8_t* dst, int rows) {
    uint8_t* codes_base = dst;
    uint8_t* scales_base = dst + (size_t) rows * kCodesRow;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* srow = src + (size_t) r * kRawRow;
        uint8_t* codes = codes_base + (size_t) r * kCodesRow;
        uint8_t* scales = scales_base + (size_t) r * kScalesRow;
        for (int b = 0; b < kBlocks; ++b) {
            const uint8_t* blk = srow + (size_t) b * kRawBlock;
            scales[(size_t) b * 2 + 0] = blk[0];
            scales[(size_t) b * 2 + 1] = blk[1];
            std::memcpy(codes + (size_t) b * 16, blk + 2, 16);
        }
    }
}

}  // namespace strata::kernels::cuda::expert_maxwell_iq4nl