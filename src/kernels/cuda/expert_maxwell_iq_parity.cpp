// src/kernels/cuda/expert_maxwell_iq_parity.cpp - the Maxwell (sm_50) tier's i-quant gate/up kernel
// against the 3060's own production path, plus the act-image bit check.
//
//     build/maxwell_iq_parity <shard1.gguf> [layer ...]    real rows (default layers 0, 1, 8: the IQ2_XS
//                                                          file's three gate/up encodings - IQ2_S, IQ2_XXS,
//                                                          IQ1_M; down is Q2_0 everywhere)
//     build/maxwell_iq_parity --synthetic <gu>/<down> ...   random weights quantized by ggml (e.g. iq3_s/q2_0)
//
// --dump <file> may precede either form: it appends, for every type-20 (IQ4_NL) down, the block's output
// as raw IEEE-754 bit patterns plus FNV-1a checksums of the three inputs it was computed from (the codes
// plane AS THE KERNEL READS IT, faq, fasc).  Two binaries comparing their dumps therefore prove both
// "same inputs" and "same bits", which is Iteration 7's contract point 3.
//
// TWO BLOBS, on purpose.  The M10's kernel reads the PLANAR down (expert_maxwell_iq4nl.hpp, the form
// production builds at load()) while the 3060's iq_mmvq and ggml's to_float read the RAW GGUF blocks:
// the reference must not move with the variable under test.  The gate/up rows are identical in both.
//
// The IQ3_S file's gate/up encodings (18 IQ3_XXS, 21 IQ3_S, 23 IQ4_XS) are checked the same way.  Its
// layers 12, 20, 30 (18/42) and 21 (21/42) are real rows this tool reads; layer 47 is 23/20 and the
// down IQ4_NL is Iteration 3, so 23 is checked on synthetic weights only.
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
#include "strata/kernels/cuda/expert_maxwell_iq4nl.hpp"

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
namespace iq4nl = strata::kernels::cuda::expert_maxwell_iq4nl;
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

// f2h_rn's CONTRACT (expert_maxwell.cu) mirrored exactly: denormals FLUSH TO ZERO by design (Maxwell has
// no fp16 arithmetic).  The f16-form comparison must use the same rule, or it would report the deliberate
// flush as a kernel disagreement.  host_f2h above is the full IEEE conversion and stays what the act-image
// check uses (there the device quantizer, not f2h_rn, sets the bits).
uint16_t contract_f2h(float x) {
    uint32_t b = 0;
    std::memcpy(&b, &x, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const uint32_t ex = (b >> 23) & 0xFFu;
    const uint32_t m = b & 0x7FFFFFu;
    if (ex == 0xFFu) return (uint16_t) (sign | 0x7C00u | (m ? 0x200u : 0u));
    const int32_t e = (int32_t) ex - 127 + 15;
    if (e <= 0) return (uint16_t) sign;          // subnormal or zero -> signed zero
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
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

// FNV-1a 64: the dumps below record a checksum of the INPUTS they were computed from.  Bit identity
// of two outputs proves nothing if the two runs did not start from the same bytes, so the comparison
// gates the checksums first and the values second.
uint64_t fnv1a64(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// ---- Iteration 7, contract point 3: THE DOWN OUTPUT, bit for bit.  HEAD (raw 18-byte blocks, the
//      kernel of today) and the branch (planar codes + scale plane, the new kernel) are run on the
//      same weights and the same faq/fasc, and their dumps must differ in ZERO floats - not in 1e-5.
//      The batch form is enough: check (4b) already pins MT and f16 to the batch bits *within* each
//      binary, so batch identity across binaries plus (4b) is the whole surface.
//      Values are the raw IEEE-754 bit patterns, eight per line, in (job, row) order.
//
// The checksums make "same inputs" a gate instead of an assumption.  `codes_stride` is the row stride
// the KERNEL uses, so the codes checksum is layout-dependent by construction (raw 360 interleaved vs
// planar 320 + a separate 40-byte scale plane): it is there to prove the two binaries each consumed a
// well-defined image, not to be equal across them.  faq and fasc MUST be equal across them.
struct DownGeo {
    size_t codes_off = 0, codes_stride = 0;    // the codes plane, as the kernel indexes it
    size_t scales_off = 0, scale_stride = 0;   // 0 = scales interleaved in the block (the raw form)
};

void dump_down_section(std::FILE* fp, const cpu::NativeFmt& f, const std::vector<uint8_t>& blob,
                       const DownGeo& dg, const std::vector<int8_t>& h_faq,
                       const std::vector<float>& h_fasc, const std::vector<float>& got,
                       const std::string& label) {
    std::fprintf(fp, "## %s\n", label.c_str());
    std::fprintf(fp, "# gu %d (%s)  down %d (%s)  H %lld  nt %zu\n", f.gu_type,
                 ggml_type_name((ggml_type) f.gu_type), f.d_type,
                 ggml_type_name((ggml_type) f.d_type), (long long) H, (size_t) (got.size() / (size_t) H));
    std::fprintf(fp, "# fnv1a-64 codes %016llx  (off %zu, stride %zu, %lld rows)\n",
                 (unsigned long long) fnv1a64(blob.data() + dg.codes_off, (size_t) H * dg.codes_stride),
                 dg.codes_off, dg.codes_stride, (long long) H);
    if (dg.scale_stride)
        std::fprintf(fp, "# fnv1a-64 scales %016llx  (off %zu, stride %zu, %lld rows)\n",
                     (unsigned long long) fnv1a64(blob.data() + dg.scales_off, (size_t) H * dg.scale_stride),
                     dg.scales_off, dg.scale_stride, (long long) H);
    else
        std::fprintf(fp, "# fnv1a-64 scales interleaved\n");
    std::fprintf(fp, "# fnv1a-64 faq %016llx\n", (unsigned long long) fnv1a64(h_faq.data(), h_faq.size()));
    std::fprintf(fp, "# fnv1a-64 fasc %016llx\n", (unsigned long long) fnv1a64(h_fasc.data(), h_fasc.size() * 4));
    std::fprintf(fp, "values %zu\n", got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t b = 0;
        std::memcpy(&b, &got[i], 4);
        std::fprintf(fp, "%08x%s", b, ((i % 8) == 7 || i + 1 == got.size()) ? "\n" : " ");
    }
}

// The layout of the down region the tool HANDS the kernel.  ONE place describes it: step 2 of Iteration 7
// flips this from the raw form to the planar one, and nothing else in the tool moves.
DownGeo down_geo_of(const cpu::NativeFmt& f) {
    DownGeo g;
    g.codes_off = f.down_off;
    g.codes_stride = iq4nl::kCodesRow;                                     // 320: planar codes
    g.scales_off = f.down_off + (size_t) H * iq4nl::kCodesRow;             // the plane that follows
    g.scale_stride = iq4nl::kScalesRow;                                    // 40
    return g;
}

// The blob the M10's kernel is given.  TWO blobs exist on purpose, because the two CONSUMERS need two
// different layouts and only one of them is the variable under test:
//   * the 3060's iq_mmvq(type 20) and ggml's to_float read the RAW GGUF blocks - they are the reference,
//     and re-expressing their input would change the thing we measure against;
//   * the M10's kernel reads the PLANAR form.
// `raw` supplies gate/up (identical in both: the i-quant rows are never touched) and the raw down for the
// reference; the returned blob has the planar down in exactly the region production uses.
std::vector<uint8_t> planar_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> blob = raw;
    iq4nl::planarize(raw.data() + f.down_off, blob.data() + f.down_off, (int) H);
    return blob;
}

// Iteration 7, contract point 1: the conversion is a BYTE PERMUTATION and is verified as one.  For every
// row and block, the planar codes equal the raw block's 16 code bytes and the planar scale equals the raw
// block's fp16 d - over all H x 20 pairs of a real expert.  Nothing is inverted and no value is
// recomputed, so this is stronger than a round trip: the check can only fail if a byte went missing.
bool planar_is_a_permutation(const cpu::NativeFmt& f, const std::vector<uint8_t>& raw, std::string& err) {
    const std::vector<uint8_t> planar = planar_blob(f, raw);
    size_t bad_codes = 0, bad_scales = 0;
    const uint8_t* sc = planar.data() + f.down_off + (size_t) H * iq4nl::kCodesRow;
    for (int64_t r = 0; r < H; ++r) {
        const uint8_t* rrow = raw.data() + f.down_off + (size_t) r * f.d_row;
        const uint8_t* prow = planar.data() + f.down_off + (size_t) r * iq4nl::kCodesRow;
        for (int b = 0; b < iq4nl::kBlocks; ++b) {
            const uint8_t* rblk = rrow + (size_t) b * iq4nl::kRawBlock;
            if (std::memcmp(prow + (size_t) b * 16, rblk + 2, 16) != 0) ++bad_codes;
            if (std::memcmp(sc + (size_t) r * iq4nl::kScalesRow + (size_t) b * 2, rblk, 2) != 0) ++bad_scales;
        }
    }
    // and nothing outside the region moved or was left behind
    if (std::memcmp(planar.data(), raw.data(), f.down_off) != 0) err = "the conversion touched the gate/up";
    if (planar.size() != raw.size()) err += " size changed";
    if (bad_codes || bad_scales) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%zu code blocks and %zu scales differ", bad_codes, bad_scales);
        err += buf;
    }
    return err.empty();
}

// The DOWN stage (IQ4_NL, ggml type 20): the M10's iq4nl kernel against the 3060's own production
// dots (iq_mmvq type 20) on the SAME activation image (the faq/fasc the gup above produced), plus
// the float reference.  This is the check the design promises: same faq/fasc, same fp16-rounded
// activation scale, gate 1e-5.
int check_down(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int nt,
               const std::vector<int8_t>& h_faq, const std::vector<float>& h_fasc,
               const std::string& label, std::FILE* dump) {
    int failures = 0;
    // The M10's input: the planar form, produced by THE SAME FUNCTION production calls at load().
    const std::vector<uint8_t> planar = planar_blob(f, blob);
    const DownGeo dg = down_geo_of(f);
    // (P) the permutation itself, before any GPU work depends on it
    {
        std::string perr;
        const bool ok = planar_is_a_permutation(f, blob, perr);
        std::printf("          (P) the planar conversion is a byte permutation: %s\n",
                    ok ? "verified, 2560 x 20 blocks" : perr.c_str());
        if (!ok) {
            std::printf("          *** THE LAYOUT IS WRONG: %s ***\n", perr.c_str());
            ++failures;
        }
    }
    const int nchunk = (int) (FF / 32);          // 20 chunks of 32 values per FF-wide activation
    // the 3060's q8_1 blocks from that image: qs verbatim, d = fp16(scale), s = 0 (unused by type 20)
    std::vector<Q81Block> xq((size_t) nt * nchunk);
    for (int c = 0; c < nt; ++c)
        for (int q = 0; q < nchunk; ++q) {
            float d = h_fasc[(size_t) c * nchunk + q];
            d = std::fabs(d) > 65504.f ? std::copysign(65504.f, d) : d;
            Q81Block& b = xq[(size_t) c * nchunk + q];
            b.d = host_f2h(d);
            b.s = 0;
            std::memcpy(b.q, h_faq.data() + (size_t) c * FF + (size_t) q * 32, 32);
        }
    // ---- the 3060: iq_mmvq, its production i-quant dot, over the raw IQ4_NL down rows
    std::vector<float> refdot((size_t) nt * H);
    {
        cudaSetDevice(g_ref);
        uint8_t* dw = nullptr;
        Q81Block* dxq = nullptr;
        float* dy = nullptr;
        cudaMalloc(&dw, (size_t) H * f.d_row);
        cudaMalloc(&dxq, xq.size() * sizeof(Q81Block));
        cudaMalloc(&dy, refdot.size() * 4);
        cudaMemcpy(dw, blob.data() + f.down_off, (size_t) H * f.d_row, cudaMemcpyHostToDevice);
        cudaMemcpy(dxq, xq.data(), xq.size() * sizeof(Q81Block), cudaMemcpyHostToDevice);
        k::iq_mmvq(f.d_type, dw, dxq, dy, (int) FF, (int) H, nt, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the 3060 down reference failed: %s\n", label.c_str(), cudaGetErrorString(e));
            return 1;
        }
        cudaMemcpy(refdot.data(), dy, refdot.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(dw); cudaFree(dxq); cudaFree(dy);
    }
    // ---- the M10: the tier's own iq4nl kernel, the same jobs the decode chain would launch
    std::vector<float> got((size_t) nt * H);
    {
        cudaSetDevice(g_m10);
        std::vector<int32_t> jobs((size_t) nt * 3);
        for (int c = 0; c < nt; ++c) { jobs[c * 3] = 0; jobs[c * 3 + 1] = c; jobs[c * 3 + 2] = c; }
        uint8_t* d_blob = nullptr;
        int32_t* d_jobs = nullptr;
        int8_t* d_faq = nullptr;
        float* d_fasc = nullptr;
        float* d_out = nullptr;
        cudaMalloc(&d_blob, planar.size());
        cudaMalloc(&d_jobs, jobs.size() * 4);
        cudaMalloc(&d_faq, h_faq.size());
        cudaMalloc(&d_fasc, h_fasc.size() * 4);
        cudaMalloc(&d_out, got.size() * 4);
        cudaMemcpy(d_blob, planar.data(), planar.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_faq, h_faq.data(), h_faq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fasc, h_fasc.data(), h_fasc.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::cuda::iq4nl_expert_maxwell_down_batch_async(
            d_blob, d_jobs, nt, d_faq, d_fasc, d_out, f.bytes, dg.codes_off, dg.scales_off, nullptr, 0, nchunk);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the M10 iq4nl down failed: %s\n", label.c_str(), cudaGetErrorString(e));
            cudaFree(d_blob); cudaFree(d_jobs); cudaFree(d_faq); cudaFree(d_fasc); cudaFree(d_out);
            return ++failures;
        }
        cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_blob); cudaFree(d_jobs); cudaFree(d_faq); cudaFree(d_fasc); cudaFree(d_out);
    }
    const double r = rel(got, refdot);
    std::printf("          (4) down IQ4_NL: the M10 iq4nl kernel vs the 3060's iq_mmvq(type 20): "
                "rel %.2e (gate 1e-5)\n", r);
    if (r > 1e-5) {
        std::printf("          *** DOWN MISMATCH vs the 3060's dots (rel %.2e > 1e-5) ***\n", r);
        ++failures;
    }
    // ---- (4b) THE THREE FORMS: the f16 epilogue and the MT (run-table) launch, both against the
    //          batch result.  Same arithmetic and reduction order by construction, so BIT-identical:
    //          the f16 form only rounds its final row, the MT one hoists the weight block per run.
    {
        cudaSetDevice(g_m10);
        std::vector<int32_t> jobs((size_t) nt * 3);
        for (int c = 0; c < nt; ++c) { jobs[c * 3] = 0; jobs[c * 3 + 1] = c; jobs[c * 3 + 2] = c; }
        // ONE run holding every entry (the widest shape E_MAX = 4 allows, nt = 3 today)
        std::vector<int32_t> runs = {0, nt, 0, 0};
        std::vector<int32_t> entries((size_t) nt * 2);
        for (int c = 0; c < nt; ++c) { entries[c * 2] = c; entries[c * 2 + 1] = c; }
        uint8_t* d_blob = nullptr;
        int32_t* d_jobs = nullptr, * d_runs = nullptr, * d_ent = nullptr;
        int8_t* d_faq = nullptr;
        float* d_fasc = nullptr;
        float* d_out = nullptr;
        uint16_t* d_out16 = nullptr;
        std::vector<float> mt((size_t) nt * H), f16back((size_t) nt * H);
        cudaMalloc(&d_blob, planar.size());
        cudaMalloc(&d_jobs, jobs.size() * 4);
        cudaMalloc(&d_runs, runs.size() * 4);
        cudaMalloc(&d_ent, entries.size() * 4);
        cudaMalloc(&d_faq, h_faq.size());
        cudaMalloc(&d_fasc, h_fasc.size() * 4);
        cudaMalloc(&d_out, mt.size() * 4);
        cudaMalloc(&d_out16, f16back.size() * 2);
        cudaMemcpy(d_blob, planar.data(), planar.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_runs, runs.data(), runs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ent, entries.data(), entries.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_faq, h_faq.data(), h_faq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fasc, h_fasc.data(), h_fasc.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::cuda::iq4nl_expert_maxwell_down_mt_async(
            d_blob, f.bytes, dg.codes_off, dg.scales_off, d_runs, d_ent, 1, d_jobs, d_faq, d_fasc, d_out,
            nullptr, 0, nchunk);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the M10 iq4nl MT down failed: %s\n", label.c_str(), cudaGetErrorString(e));
            ++failures;
        }
        cudaMemcpy(mt.data(), d_out, mt.size() * 4, cudaMemcpyDeviceToHost);
        strata::kernels::cuda::iq4nl_expert_maxwell_down_batch_f16_async(
            d_blob, d_jobs, nt, d_faq, d_fasc, d_out16, f.bytes, dg.codes_off, dg.scales_off, nullptr, 0, nchunk);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::printf("%s: the M10 iq4nl f16 down failed: %s\n", label.c_str(), cudaGetErrorString(e));
            ++failures;
        }
        std::vector<uint16_t> h16(f16back.size());
        cudaMemcpy(h16.data(), d_out16, h16.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_blob); cudaFree(d_jobs); cudaFree(d_runs); cudaFree(d_ent);
        cudaFree(d_faq); cudaFree(d_fasc); cudaFree(d_out); cudaFree(d_out16);
        // MT vs batch: bit for bit
        size_t mtdiff = 0;
        for (size_t i = 0; i < mt.size(); ++i)
            if (std::memcmp(&mt[i], &got[i], 4) != 0) ++mtdiff;
        // f16 vs batch: the batch value rounded by f2h_rn's own contract, bit for bit
        size_t f16diff = 0;
        for (size_t i = 0; i < h16.size(); ++i)
            if (h16[i] != contract_f2h(got[i])) ++f16diff;
        std::printf("          (4b) the three forms: MT vs batch %zu of %zu differ, f16 vs round(batch) "
                    "%zu of %zu differ (both must be 0)\n", mtdiff, mt.size(), f16diff, h16.size());
        if (mtdiff != 0 || f16diff != 0) {
            std::printf("          *** THE FORMS DISAGREE: MT %zu, f16 %zu ***\n", mtdiff, f16diff);
            ++failures;
        }
    }
    // ---- (5) REPORTED: the float reference - ggml's to_float on the raw down, dotted in fp64 with
    //          the SAME dequantised activation image (faq * fasc, no fp16 rounding on this side)
    {
        const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
        std::vector<float> D((size_t) H * FF);
        for (int64_t r2 = 0; r2 < H; ++r2)
            td->to_float(blob.data() + f.down_off + (size_t) r2 * f.d_row, D.data() + (size_t) r2 * FF, FF);
        double n = 0, d = 0;
        for (int c = 0; c < nt; ++c)
            for (int64_t r2 = 0; r2 < H; ++r2) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i)
                    o += (double) D[(size_t) r2 * FF + i] * (double) h_faq[(size_t) c * FF + i] *
                         (double) h_fasc[(size_t) c * nchunk + i / 32];
                n += std::fabs((double) got[(size_t) c * H + r2] - o);
                d += std::fabs(o);
            }
        std::printf("          (5) down vs the float reference (raw acts, reported): rel %.2e\n",
                    n / (d + 1e-30));
    }
    if (dump) dump_down_section(dump, f, planar, dg, h_faq, h_fasc, got, label);
    return failures;
}

// One blob's gate/up: the kernel on the M10 against the 3060's production dots; returns failed checks.
int check_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label,
               std::FILE* dump) {
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
    //          (h_faq/h_fasc outlive the block: the down check consumes the very image the gup made)
    std::vector<int8_t> h_faq((size_t) NT * FF);
    std::vector<float> h_fasc((size_t) NT * (FF / 32)), h_fahx((size_t) NT * (FF / 32));
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
    // ---- (4) + (5): the DOWN stage, when this layer keeps an IQ4_NL (ggml type 20) one.  Feed it the
    //          faq/fasc the gup above produced: the design's chain - the tier's own activation image into
    //          the tier's own down kernel, against the 3060's production dots on the same image.
    if (f.d_type == 20) failures += check_down(f, blob, NT, h_faq, h_fasc, label, dump);
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
    // --dump <file>: the down output of every type-20 layer, bit for bit (Iteration 7, contract point 3).
    // It is CONSUMED here, so the mode dispatch below sees a clean argument list.
    std::FILE* dump = nullptr;
    std::string dump_path;
    {
        std::vector<char*> keep;
        keep.push_back(argv[0]);
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "maxwell_iq_parity: --dump needs a file name\n");
                    return 2;
                }
                dump_path = argv[++i];
                continue;
            }
            keep.push_back(argv[i]);
        }
        argc = (int) keep.size();
        for (int i = 0; i < argc; ++i) argv[i] = keep[(size_t) i];
    }
    if (!dump_path.empty()) {
        dump = std::fopen(dump_path.c_str(), "w");
        if (!dump) {
            std::fprintf(stderr, "maxwell_iq_parity: cannot write %s\n", dump_path.c_str());
            return 2;
        }
        std::fprintf(dump, "# maxwell_iq_parity down dump - raw IEEE-754 bit patterns, eight per line\n");
        std::fprintf(dump, "# the tool is the authority on the LAYOUT: the region checksum above is the\n"
                           "# down bytes AS THE KERNEL READS THEM, so raw and planar differ here by design.\n");
    }
    if (argc > 1 && std::strcmp(argv[1], "--synthetic") == 0) {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            // the tier's kernels: an i-quant gate/up of {22, 16, 29, 18, 21, 23} over the S2 (Q2_0) down
            if (gu != 22 && gu != 16 && gu != 29 && gu != 18 && gu != 21 && gu != 23) {
                std::printf("%s: gate/up ggml type %d is not one the Maxwell IQ kernels take "
                            "(22 IQ2_S, 16 IQ2_XXS, 29 IQ1_M, 18 IQ3_XXS, 21 IQ3_S, 23 IQ4_XS)\n", arg.c_str(), gu);
                ++failures;
                continue;
            }
            if (dn != 42 && dn != 20) {
                std::printf("%s: down ggml type %d is not one the tier takes (42 Q2_0/S2, 20 IQ4_NL)\n",
                            arg.c_str(), dn);
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
            failures += check_blob(f, synthetic_blob(f, i), i, "synthetic " + arg, dump);
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
            if (t[2]->type != 42 && t[2]->type != 20) {
                std::printf("layer %d: down is ggml type %d, neither Q2_0 (42) nor IQ4_NL (20)\n", l,
                            (int) t[2]->type);
                ++failures;
                continue;
            }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err) ||
                (f.gu_type != 22 && f.gu_type != 16 && f.gu_type != 29 &&
                 f.gu_type != 18 && f.gu_type != 21 && f.gu_type != 23)) {
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
            failures += check_blob(f, blob, l, "layer " + std::to_string(l), dump);
        }
    } else {
        std::fprintf(stderr, "usage: maxwell_iq_parity [--dump <file>] <shard1.gguf> [layer ...]\n"
                             "       maxwell_iq_parity [--dump <file>] --synthetic <gu>/<down> ...  (ggml type names)\n");
        if (dump) std::fclose(dump);
        return 2;
    }
    if (dump) std::fclose(dump);
    std::printf("maxwell_iq_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
