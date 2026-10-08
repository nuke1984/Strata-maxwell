// src/kernels/cuda/expert_maxwell_iq_parity.cpp - the Maxwell (sm_50) tier's i-quant gate/up kernel
// against the 3060's own production path, plus the act-image bit check.
//
//     build/maxwell_iq_parity <shard1.gguf> [layer ...]    real rows (default layers 0, 1, 8: the IQ2_XS
//                                                          file's three gate/up encodings - IQ2_S, IQ2_XXS,
//                                                          IQ1_M; down is Q2_0 everywhere)
//     build/maxwell_iq_parity --synthetic <gu>/<down> ...   random weights quantized by ggml (e.g. iq2_s/q2_0)
//
// The tool finds BOTH cards it needs and refuses when either is missing: a Maxwell (compute capability 5.0,
// the M10) for the kernel under test, and a Turing-or-newer card (the 3060, sm_86 - this build's
// CMAKE_CUDA_ARCHITECTURES) for the reference.  Run it with every device visible (no CUDA_VISIBLE_DEVICES).
// ggml_cpu_init() first: to_float needs the fp16 tables.
//
// WHAT IS CHECKED, in the order the engine depends on it:
//
//   (0) THE ACT IMAGE, bit for bit: the kernel consumes the tier's staged ActQ and derives each chunk's
//       d = fp16(scale) in shared memory; the reference path consumes the device quantizer's own q8_1
//       blocks (quantize_q8_1_rows, exactly what the 3060's native expert path reads).  ActQ.q is
//       block_q8_1.qs bit for bit, so the derived blocks must equal the quantizer's in every bit - a
//       mismatch is a staging or rounding bug, and everything below is meaningless until it is 0.
//   (1) THE KERNEL vs THE 3060's PRODUCTION DOTS, gated at 1e-5 relative: iq_mmvq (the engine's own
//       i-quant dots, the ones the 3060's native expert path runs in production) on the same rows and
//       the same act blocks, the SwiGLU applied in fp32 on both sides.  This is the number the multi-tier
//       engine actually needs: the M10's rows must be the 3060's rows to the fp32 accumulation order.
//   (2) THE EPILOGUE IMAGE: fasc/fahx against this tool's own quantisation of (1)'s reference (rel
//       <= 1e-5 where no rounding boundary flipped), faq within ONE quantum everywhere, flips counted
//       and gated at 0.1%; fahx is gated per chunk against the chunk's own flips (a flip moves the sum
//       by one, so |d fahx| <= the chunk's flips * its scale + fp32 dust).
//   (3) REPORTED, not gated: the same rows dequantized by ggml's to_float and dotted in fp64 with the
//       raw activation - the act quantization noise, the number the engine accepts for every q8_1 tier.
//       (A second, larger, systematic number this reference shows - the dots' exact-integer epilogues
//       like IQ2_S's truncating `(sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4` are NOT reproduced by a
//       float dequantizer - is why (1), not this, is the gated reference.)

#include "strata/kernels/cuda/expert_maxwell.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace geo = strata::kernels::cuda::expert_maxwell_geo;
namespace k = strata::kernels;

namespace {

constexpr int NT = 3;   // tokens (jobs) per check
constexpr int64_t H = 2560, FF = 640;
constexpr int E = 7;   // the expert whose real rows a shard check reads

// fp32 -> fp16, round to nearest even: an INDEPENDENT implementation of the contract f2h_rn documents
// (expert_maxwell.cu) - the classic construction, not the same code, so a fault in either shows up in
// (0).  Denormals flush to zero, inf/nan pass through.
uint16_t host_f2h(float x) {
    uint32_t b = 0;
    std::memcpy(&b, &x, 4);
    const uint32_t sign = (b >> 16) & 0x8000;
    const int32_t exp = (int32_t)((b >> 23) & 0xff) - 127;
    const uint32_t mant = b & 0x7fffff;
    if (((b >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 1 : 0));   // inf / nan
    if (exp > 15) return (uint16_t)(sign | 0x7c00);                                      // overflow -> inf
    if (exp >= -14) {   // normal fp16
        const uint32_t m = mant >> 13, r = mant & 0x1fff;
        uint32_t h = sign | (uint32_t)((exp + 15) << 10) | m;
        if (r > 0x1000 || (r == 0x1000 && (m & 1))) ++h;   // round half to even
        return (uint16_t)h;
    }
    if (exp < -25) return (uint16_t)sign;   // underflow (incl. zero) -> zero
    const uint32_t mm = mant | 0x800000;     // subnormal fp16: the implicit bit joins the mantissa
    const int shift = -exp - 14;
    const uint32_t m = mm >> (13 + shift), r = mm & ((1u << (13 + shift)) - 1);
    uint16_t h = (uint16_t)(sign | m);
    if (r > (1u << (12 + shift)) || (r == (1u << (12 + shift)) && (m & 1))) ++h;
    return h;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// ggml's block_q8_1 as bytes: {half2 ds; int8_t qs[32]} - only d (ds.x) and the codes matter here.
struct Q81Block { uint16_t d, s; int8_t q[32]; };
static_assert(sizeof(Q81Block) == 36, "block_q8_1 is 36 bytes");

int g_m10 = -1, g_ref = -1;   // the devices the two sides run on

// One blob's gate/up: the kernel on the M10 against the 3060's production dots; returns failed checks.
int check_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label) {
    int failures = 0;
    // ---- the activation: the engine's own ActQ (what the dispatch hands the tier) on the host, and the
    //      device quantizer's q8_1 blocks on the 3060 (what its native path consumes).
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<cpu::ActQ> acts(NT);
    for (int k = 0; k < NT; ++k)   // act_quant_any, as the engine's dispatch: the _q8_1 body is AVX-512
        cpu::act_quant_any(x.data() + (size_t) k * (int) H, (int) H, acts[k]);
    // the host-derived q8_1 blocks: ActQ.q verbatim, d = fp16(scale) with the #606 clamp, s = 0
    std::vector<Q81Block> mine((size_t) NT * (H / 32));
    for (int k = 0; k < NT; ++k)
        for (int c = 0; c < H / 32; ++c) {
            float d = acts[k].scale[c];
            d = std::fabs(d) > 65504.f ? std::copysign(65504.f, d) : d;
            mine[(size_t) k * (H / 32) + c].d = host_f2h(d);
            mine[(size_t) k * (H / 32) + c].s = 0;
            std::memcpy(mine[(size_t) k * (H / 32) + c].q, acts[k].q + c * 32, 32);
        }
    // ---- (0) + (1) on the 3060: its own act blocks, then its own dots (iq_mmvq, the production path)
    std::vector<Q81Block> refblocks((size_t) NT * (H / 32));
    std::vector<float> gdot((size_t) NT * FF), udot((size_t) NT * FF), refff((size_t) NT * FF);
    std::vector<float> h_ff((size_t) NT * FF);   // the kernel's pre-quant SwiGLU output
    {
        cudaSetDevice(g_ref);
        float* dx = nullptr;
        Q81Block* dxq = nullptr;
        float* yg = nullptr, * yu = nullptr;
        uint8_t* wg = nullptr, * wu = nullptr;
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, refblocks.size() * sizeof(Q81Block));
        cudaMalloc(&yg, gdot.size() * 4);
        cudaMalloc(&yu, udot.size() * 4);
        cudaMalloc(&wg, f.up_off);
        cudaMalloc(&wu, f.up_off);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(wg, blob.data(), f.up_off, cudaMemcpyHostToDevice);
        cudaMemcpy(wu, blob.data() + f.up_off, f.up_off, cudaMemcpyHostToDevice);
        k::quantize_q8_1_rows(dx, NT, H, dxq, nullptr);
        k::iq_mmvq(f.gu_type, wg, dxq, yg, (int) H, (int) FF, NT, nullptr);
        k::iq_mmvq(f.gu_type, wu, dxq, yu, (int) H, (int) FF, NT, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the 3060 reference failed: %s\n", label.c_str(), cudaGetErrorString(e));
            return 1;
        }
        cudaMemcpy(refblocks.data(), dxq, refblocks.size() * sizeof(Q81Block), cudaMemcpyDeviceToHost);
        cudaMemcpy(gdot.data(), yg, gdot.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(udot.data(), yu, udot.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(dx); cudaFree(dxq); cudaFree(yg); cudaFree(yu); cudaFree(wg); cudaFree(wu);
        // (0) the act image, bit for bit
        size_t diffs = 0;
        for (size_t i = 0; i < refblocks.size(); ++i) {
            const Q81Block& a = mine[i], & b = refblocks[i];
            if (a.d != b.d || std::memcmp(a.q, b.q, 32) != 0) ++diffs;   // s is 0 in both (quantize zeroes it)
        }
        std::printf("%s: %s gate/up, %d tokens\n", label.c_str(), ggml_type_name((ggml_type) f.gu_type), NT);
        std::printf("          (0) the M10's ActQ-derived q8_1 blocks vs the device quantizer's: "
                    "%zu of %zu differ\n", diffs, refblocks.size());
        if (diffs != 0) {
            std::printf("          *** ACT IMAGE MISMATCH: the M10 kernel and the 3060 path consume "
                        "different activations ***\n");
            ++failures;
        }
        // the SwiGLU in fp32, as the kernel has it
        for (int c = 0; c < NT; ++c)
            for (int r = 0; r < FF; ++r) {
                const float g = gdot[(size_t) c * FF + r], u = udot[(size_t) c * FF + r];
                refff[(size_t) c * FF + r] = (g / (1.f + std::exp(-g))) * u;
            }
    }
    // ---- (1) the kernel under test, on the M10: the same packed ActQ staging begin_chain uploads
    {
        cudaSetDevice(g_m10);
        const int as = (int) geo::ACTS_STRIDE;
        std::vector<uint8_t> packed((size_t) NT * as);
        for (int c = 0; c < NT; ++c) {
            uint8_t* at = packed.data() + (size_t) c * as;
            std::memcpy(at, acts[c].q, H);
            std::memcpy(at + H, acts[c].scale, (size_t) (H / 32) * 4);
            std::memcpy(at + H + (size_t) (H / 32) * 4, acts[c].hx, (size_t) (H / 32) * 4);
        }
        std::vector<int32_t> jobs((size_t) NT * 3);
        for (int c = 0; c < NT; ++c) { jobs[c * 3] = 0; jobs[c * 3 + 1] = c; jobs[c * 3 + 2] = 0; }
        uint8_t* d_blob = nullptr, * d_acts = nullptr;
        int32_t* d_jobs = nullptr;
        int8_t* d_faq = nullptr;
        float* d_fasc = nullptr, * d_fahx = nullptr, * d_ff = nullptr;
        std::vector<int8_t> h_faq((size_t) NT * FF);
        std::vector<float> h_fasc((size_t) NT * (FF / 32)), h_fahx((size_t) NT * (FF / 32));
        cudaMalloc(&d_blob, blob.size());
        cudaMalloc(&d_acts, packed.size());
        cudaMalloc(&d_jobs, jobs.size() * 4);
        cudaMalloc(&d_faq, h_faq.size());
        cudaMalloc(&d_fasc, h_fasc.size() * 4);
        cudaMalloc(&d_fahx, h_fahx.size() * 4);
        cudaMalloc(&d_ff, h_ff.size() * 4);
        cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acts, packed.data(), packed.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::cuda::iq_expert_maxwell_gup_quant_batch_async(
            f.gu_type, d_blob, f.bytes, d_jobs, NT, d_acts, 0, f.up_off, f.gu_row,
            d_faq, d_fasc, d_fahx, d_ff, 0);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the M10 kernel failed: %s\n", label.c_str(), cudaGetErrorString(e));
            return ++failures;
        }
        cudaMemcpy(h_faq.data(), d_faq, h_faq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fasc.data(), d_fasc, h_fasc.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fahx.data(), d_fahx, h_fahx.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ff.data(), d_ff, h_ff.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_blob); cudaFree(d_acts); cudaFree(d_jobs); cudaFree(d_faq);
        cudaFree(d_fasc); cudaFree(d_fahx); cudaFree(d_ff);
        const double r1 = rel(h_ff, refff);
        std::printf("          (1) the M10 kernel vs the 3060's production dots: rel %.2e (gate 1e-5)\n", r1);
        if (r1 > 1e-5) {
            std::printf("          *** MISMATCH vs the 3060's dots (rel %.2e > 1e-5) ***\n", r1);
            ++failures;
        }
        // ---- (2) the epilogue image: this tool's own quantisation of the reference's ff
        size_t flips = 0;
        double fsc = 0;
        for (int c = 0; c < NT; ++c)
            for (int q = 0; q < FF / 32; ++q) {
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(refff[(size_t) c * FF + q * 32 + i]));
                const float sc = amax > 0.f ? amax / 127.f : 0.f;
                const float inv = sc > 0.f ? 1.f / sc : 0.f;
                int sum = 0, cflips = 0;
                for (int i = 0; i < 32; ++i) {
                    const float t = refff[(size_t) c * FF + q * 32 + i] * inv;
                    int qi = (int) (t + (t >= 0.f ? 0.5f : -0.5f));
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    sum += qi;
                    const int got = h_faq[(size_t) c * FF + q * 32 + i];
                    if (got != qi) {
                        ++flips; ++cflips;
                        if (std::abs(got - qi) > 1) {
                            std::printf("          *** faq differs by %d at job %d chunk %d lane %d ***\n",
                                        got - qi, c, q, i);
                            ++failures;
                        }
                    }
                }
                const float hx = sc * (float) sum;
                const float gsc = h_fasc[(size_t) c * (FF / 32) + q];
                fsc = std::fmax(fsc, std::fabs(gsc - sc) / (std::fabs(sc) + 1e-30));
                const float ghx = h_fahx[(size_t) c * (FF / 32) + q];
                // a flip moves the chunk's sum by one, so the sums agree to the chunk's own flips
                if (std::fabs(ghx - hx) > 1.001f * sc * (float) cflips + 1e-5 * (std::fabs(hx) + 1e-30)) {
                    std::printf("          *** fahx mismatch beyond the chunk's %d flips (job %d chunk %d) ***\n",
                                cflips, c, q);
                    ++failures;
                }
                if (std::fabs(gsc - sc) > 1e-5 * (std::fabs(sc) + 1e-30)) {
                    std::printf("          *** fasc mismatch at job %d chunk %d ***\n", c, q);
                    ++failures;
                }
            }
        std::printf("          (2) epilogue image: faq flips %zu of %zu (<= 0.1%%, all within one quantum), "
                    "fasc max rel %.2e (gate 1e-5)\n", flips, (size_t) NT * FF, fsc);
        if (flips > (size_t) NT * FF / 1000) ++failures;
    }
    // ---- (3) REPORTED: the float reference (ggml's to_float, the raw activation, fp64)
    {
        const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
        std::vector<float> G((size_t) FF * H), U((size_t) FF * H);
        for (int64_t r = 0; r < FF; ++r) {
            tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
            tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
        }
        double n = 0, d = 0;
        for (int c = 0; c < NT; ++c)
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) {
                    g += (double) G[(size_t) r * H + i] * x[(size_t) c * H + i];
                    u += (double) U[(size_t) r * H + i] * x[(size_t) c * H + i];
                }
                const float v = (float) g;
                const float ff = (v / (1.f + std::exp(-v))) * (float) u;
                n += std::fabs((double) h_ff[(size_t) c * FF + r] - ff);
                d += std::fabs((double) ff);
            }
        std::printf("          (3) vs the float reference (raw acts, reported): rel %.2e\n", n / (d + 1e-30));
    }
    return failures;
}

int type_of(const std::string& name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t)
        if (name == ggml_type_name((ggml_type) t)) return t;
    return -1;
}

// Random weights (rows of different magnitudes) quantized by ggml into the blob layout, as
// native_expert_parity's synthetic_blob does (the same imatrix handling).
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    return blob;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    // the two cards: a Maxwell for the kernel under test, a Turing-or-newer one for the reference
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::fprintf(stderr, "maxwell_iq_parity: no CUDA device\n");
        return 2;
    }
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        if (p.major == 5 && g_m10 < 0) g_m10 = d;
        if (p.major >= 7 && g_ref < 0) g_ref = d;
    }
    if (g_m10 < 0 || g_ref < 0) {
        std::fprintf(stderr, "maxwell_iq_parity: needs a Maxwell (cc 5.0, the M10) and a cc 7+ card "
                            "(the 3060) visible; found Maxwell=%d reference=%d\n", g_m10, g_ref);
        return 2;
    }
    std::printf("maxwell_iq_parity: M10 is device %d, the reference card is device %d\n", g_m10, g_ref);
    int failures = 0;
    if (argc > 1 && std::strcmp(argv[1], "--synthetic") == 0) {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            // the tier's kernels: an i-quant gate/up of {22, 16, 29} over the S2 (Q2_0) down
            if (gu != 22 && gu != 16 && gu != 29) {
                std::printf("%s: gate/up ggml type %d is not one the Maxwell IQ kernels take "
                            "(22 IQ2_S, 16 IQ2_XXS, 29 IQ1_M)\n", arg.c_str(), gu);
                ++failures;
                continue;
            }
            if (dn != 42) {
                std::printf("%s: down ggml type %d is not the S2 (Q2_0) form the tier converts\n", arg.c_str(), dn);
                ++failures;
                continue;
            }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt(gu, dn, H, FF, f, err)) {
                std::printf("%s: %s\n", arg.c_str(), err.c_str());
                ++failures;
                continue;
            }
            failures += check_blob(f, synthetic_blob(f, i), i, "synthetic " + arg);
        }
    } else if (argc > 1) {
        const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
        std::vector<int> layers;
        for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
        if (layers.empty()) layers = {0, 1, 8};   // IQ2_S, IQ2_XXS, IQ1_M of the IQ2_XS file
        for (int l : layers) {
            const strata::TensorInfo* t[3] = {};
            const uint8_t* data[3] = {};
            const char* roles[3] = {"gate", "up", "down"};
            for (int r = 0; r < 3; ++r) {   // each role from the shard that holds it
                size_t at = 0;
                t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
                if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
            }
            if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
            if (t[2]->type != 42) {
                std::printf("layer %d: down is ggml type %d, not the S2 (Q2_0) form\n", l, (int) t[2]->type);
                ++failures;
                continue;
            }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err) ||
                (f.gu_type != 22 && f.gu_type != 16 && f.gu_type != 29)) {
                std::printf("layer %d: gate/up ggml type %d is not one the Maxwell IQ kernels take\n", l,
                            (int) t[0]->type);
                ++failures;
                continue;
            }
            std::vector<uint8_t> blob(f.bytes);   // expert E's three slices, as the tier's source assembles
            std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off),
                        f.bytes - f.down_off);
            failures += check_blob(f, blob, l, "layer " + std::to_string(l));
        }
    } else {
        std::fprintf(stderr, "usage: maxwell_iq_parity <shard1.gguf> [layer ...]\n"
                             "       maxwell_iq_parity --synthetic <gu>/<down> ...  (ggml type names)\n");
        return 2;
    }
    std::printf("maxwell_iq_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
