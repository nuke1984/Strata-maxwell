// src/core/maxwell_experts.cpp - the Maxwell (sm_50) expert tier.  See the header for the design.
//
// DATA FLOW for one layer:
//   begin(): for each routed entry with kind==-1 owned by this device, dequantise the host ActQ into xq,
//            copy xq to the device, launch gate+up (stage1) for each owned expert, requantise ff on the
//            host, copy ffq, launch down (stage2).  Results are staged in h_out_.
//   finish(): the launches already synchronised in begin(); copy staged rows into `out`.
//
// This is deliberately synchronous (no stream overlap with the CPU pool) for the proof: correctness first,
// overlap later.  The CPU pool and the M10 work sequentially, so the M10 still replaces CPU work 1:1.
#include "strata/core/maxwell_experts.hpp"
#include "strata/core/peer_experts.hpp"

#include "strata/kernels/cuda/expert_maxwell.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-cpu.h"   // ggml_cpu_init (the round-trip gate's to_float tables)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

namespace strata::core {
namespace {

using namespace strata::kernels::cuda::expert_maxwell_geo;
using strata::kernels::cuda::s2_expert_maxwell_gup_async;
using strata::kernels::cuda::s2_expert_maxwell_down_async;
using strata::kernels::cuda::s2_expert_maxwell_gup_batch_async;
using strata::kernels::cuda::s2_expert_maxwell_down_batch_async;
using strata::kernels::cuda::s2_expert_maxwell_down_batch_f16_async;
using strata::kernels::cuda::s2_expert_maxwell_quant_ff_async;
using strata::kernels::cuda::s2_expert_maxwell_gup_quant_batch_async;
using strata::kernels::cuda::iq_expert_maxwell_gup_quant_batch_async;

namespace {
// E1: STRATA_MW_GPUFLAG - 0 off; 1 the GPU-raised tier flags + the graph's multi-wait (host sync kept);
// 2 also drops the decode-path sync (the staging double-buffers).  See WORKLOG 2026-10-12.
int gpuflag_mode() {
    static const int m = [] { const char* e = std::getenv("STRATA_MW_GPUFLAG"); return e ? std::atoi(e) : 0; }();
    return m;
}
}
using strata::kernels::cuda::s2_expert_maxwell_v2;
using strata::kernels::cuda::s2_expert_maxwell_sync;

constexpr int H = 2560;
constexpr int FF = 640;

// RAII device scope, like RemoteExperts::DeviceScope.
struct DeviceScope {
    int prev = -1;
    bool ok = false;
    std::string err;
    explicit DeviceScope(int device) {
        cudaError_t e = cudaGetDevice(&prev);
        if (e != cudaSuccess) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Maxwell DeviceScope: cudaGetDevice failed: %s",
                          cudaGetErrorString(e));
            err = buf;
            return;
        }
        e = cudaSetDevice(device);
        if (e != cudaSuccess) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Maxwell DeviceScope: cudaSetDevice(%d) failed: %s (prev=%d)",
                          device, cudaGetErrorString(e), prev);
            err = buf;
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (ok) cudaSetDevice(prev); }
};

}  // namespace

MaxwellExperts::~MaxwellExperts() { close(); }

bool MaxwellExperts::check(cudaError_t e, const char* what, std::string& err) {
    if (e != cudaSuccess) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): %s: %s", device_, what,
                      cudaGetErrorString(e));
        err = buf;
        return false;
    }
    return true;
}

bool MaxwellExperts::open(int device,
                          const std::vector<std::pair<int32_t, int32_t>>& assign,
                          const uint8_t* (*src_blob)(int64_t layer, int64_t expert, void* ctx), void* ctx,
                          int64_t n_layers, int64_t n_expert, int reserve_mib,
                          std::string& err) {
    close();
    // GUARD: this tier runs the expert encodings its kernels take - the Q2_0 (S2) form, and a native
    // pack's IQ2_S / IQ2_XXS / IQ1_M gate/up over the Q2_0 down (the IQ2_XS file's layers; the down is
    // converted to the S2 planar form at open()).  Anything else (an encoding a future pack may carry)
    // would be read with the wrong geometry - plausible garbage rows, silently, never a fault.  Refuse
    // instead, naming the first offending layer.  42 is ggml's Q2_0 (tools/gguf_reader.py).
    {
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (lay.native) {
            for (size_t L = 0; L < lay.fmt.size(); ++L) {
                const auto& f = lay.fmt[L];
                if (f.d_type != 42 ||
                    (f.gu_type != 42 && f.gu_type != 22 && f.gu_type != 16 && f.gu_type != 29)) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "MaxwellExperts (CUDA%d): layer %zu keeps its experts as gate/up ggml type "
                                  "%d and down type %d; this tier takes Q2_0 (42), and IQ2_S (22) / IQ2_XXS "
                                  "(16) / IQ1_M (29) gate/up over a Q2_0 down",
                                  device, L, f.gu_type, f.d_type);
                    err = buf;
                    return false;
                }
            }
        }
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.err; return false; }

    // how much fits?  A canonical pack's blobs are all BLOB bytes; a native pack's differ per layer
    // (the IQ2_XS file: 1,510,400 / 1,305,600 / 1,177,600 B per expert, by its three gate/up encodings),
    // so the budget is taken in the layer's own blob bytes.
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) return false;
    const size_t reserve_b = (size_t) reserve_mib * 1024 * 1024;
    const size_t usable = (free_b > reserve_b) ? (free_b - reserve_b) : 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto blob_of = [&](int64_t layer) -> uint64_t {
        return lay.native ? lay.blob_bytes(layer) : (uint64_t) BLOB;
    };

    // take experts in assignment order until the device is full
    std::vector<std::pair<int32_t, int32_t>> mine;
    size_t bytes = 0;
    for (const auto& p : assign) {
        if (bytes + blob_of(p.first) > usable) break;
        mine.push_back(p);
        bytes += blob_of(p.first);
    }
    if (mine.empty()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): no experts fit (%.1f GiB free, %.1f GiB reserved)",
                      device, free_b / 1e9, reserve_b / 1e9);
        err = buf;
        return false;
    }

    device_ = device;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    res_.assign((size_t) n_layers * (size_t) n_expert, -1);

    // THE STORE: a native pack keeps each layer's experts in their own contiguous region (the blobs
    // differ in size per layer, so a flat slot*BLOB walk cannot address them); a canonical pack keeps
    // today's flat store.  geo_ carries each region's base and the launch geometry; res_ then holds the
    // slot WITHIN the layer (native) or the flat slot (canonical).
    if (lay.native) {
        std::vector<int64_t> count((size_t) n_layers, 0);
        for (const auto& p : mine) ++count[(size_t) p.first];
        geo_.assign((size_t) n_layers, LayerGeo{});
        size_t off = 0;
        for (int64_t L = 0; L < n_layers; ++L) {
            const auto& f = lay.fmt[(size_t) L];
            LayerGeo& g = geo_[(size_t) L];
            g.gu_type = f.gu_type;
            g.base = off;
            g.blob_bytes = (size_t) f.bytes;
            if (f.gu_type != 42) {   // the i-quant layer: raw GGUF gate/up rows, the down converted
                g.o_gate = 0;
                g.o_up = f.up_off;                       // gate rows first, up rows second (as the source)
                g.gu_row = f.gu_row;
                g.d_codes_off = f.down_off;               // the down region, de-interleaved to S2 planar
                g.d_scales_off = f.down_off + (size_t) H * ROW_D;
            }
            off += (size_t) count[(size_t) L] * (size_t) f.bytes;
        }
    }
    // allocate the blob store on the device
    if (!check(cudaMalloc(&d_blobs_, bytes), "cudaMalloc blobs", err)) { close(); return false; }

    // Convert each blob on copy.  The source blob for a native pack is INTERLEAVED (18 bytes per
    // 64-weight block: fp16 scale + 16 codes, rows at gu_row/d_row strides); the S2 kernels read PLANAR.
    // A Q2_0 layer (native or canonical) is converted to the S2 planar layout as before; an i-quant
    // layer keeps its gate/up rows EXACTLY as the GGUF stores them (the kernel reads the raw blocks -
    // an i-quant row cannot be re-expressed in the S2 form, tools/iq_pack.py) and converts only its
    // Q2_0 down rows to the S2 planar form, inside the blob, at the layer's own offsets.
    std::vector<int32_t> per_layer_slot(lay.native ? (size_t) n_layers : 0, 0);
    std::vector<uint8_t> staging(lay.native ? (size_t) lay.max_blob : BLOB);
    for (size_t i = 0; i < mine.size(); ++i) {
        const int64_t L = mine[i].first;
        const uint8_t* src = src_blob(L, mine[i].second, ctx);
        if (src == nullptr) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): source blob missing for (%d,%d)",
                          device, mine[i].first, mine[i].second);
            err = buf;
            close();
            return false;
        }
        size_t blob_bytes = (size_t) BLOB, dst_off = i * (size_t) BLOB;
        int32_t slot = (int32_t) i;
        if (lay.native) {
            const auto& f = lay.fmt[(size_t) L];
            const LayerGeo& g = geo_[(size_t) L];
            blob_bytes = (size_t) f.bytes;
            slot = per_layer_slot[(size_t) L]++;
            dst_off = g.base + (size_t) slot * blob_bytes;
            if (f.gu_type == 42) {
                // native Q2_0 -> S2 planar: de-interleave each row's 18-byte blocks (as before)
                uint8_t* dst = staging.data();
                // S2 planar offsets (must match expert_maxwell_geo)
                const size_t ROW_GU = 640, ROW_D = 160, SC_GU = 40, SC_D = 10;
                const size_t O_D_CODES = 2ull * 640 * ROW_GU;
                const size_t O_GU_SCALES = O_D_CODES + 2560ull * ROW_D;
                const size_t O_D_SCALES = O_GU_SCALES + 2ull * 640 * SC_GU * 2;
                // gate + up rows
                for (int r = 0; r < 640; ++r) {
                    for (int half = 0; half < 2; ++half) {   // 0 = gate, 1 = up
                        const uint8_t* srow = src + (half == 0 ? 0 : f.up_off) + (size_t) r * f.gu_row;
                        uint8_t* dcode = dst + ((size_t) (2 * r + half)) * ROW_GU;
                        uint8_t* dscale = dst + O_GU_SCALES + ((size_t) (2 * r + half)) * SC_GU * 2;
                        for (int b = 0; b < 40; ++b) {
                            const uint8_t* blk = srow + (size_t) b * 18;
                            dscale[b * 2] = blk[0];
                            dscale[b * 2 + 1] = blk[1];
                            std::memcpy(dcode + b * 16, blk + 2, 16);
                        }
                    }
                }
                // down rows
                for (int r = 0; r < 2560; ++r) {
                    const uint8_t* srow = src + f.down_off + (size_t) r * f.d_row;
                    uint8_t* dcode = dst + O_D_CODES + (size_t) r * ROW_D;
                    uint8_t* dscale = dst + O_D_SCALES + (size_t) r * SC_D * 2;
                    for (int b = 0; b < 10; ++b) {
                        const uint8_t* blk = srow + (size_t) b * 18;
                        dscale[b * 2] = blk[0];
                        dscale[b * 2 + 1] = blk[1];
                        std::memcpy(dcode + b * 16, blk + 2, 16);
                    }
                }
            } else {
                // the i-quant layer: gate/up rows VERBATIM (the raw GGUF blocks the kernel decodes),
                // the Q2_0 down rows de-interleaved into the S2 planar form at the blob's own offsets
                std::memcpy(staging.data(), src, f.down_off);
                for (int r = 0; r < H; ++r) {
                    const uint8_t* srow = src + f.down_off + (size_t) r * f.d_row;
                    uint8_t* dcode = staging.data() + g.d_codes_off + (size_t) r * ROW_D;
                    uint8_t* dscale = staging.data() + g.d_scales_off + (size_t) r * SC_D * 2;
                    for (int b = 0; b < SC_D; ++b) {
                        const uint8_t* blk = srow + (size_t) b * 18;
                        dscale[b * 2] = blk[0];
                        dscale[b * 2 + 1] = blk[1];
                        std::memcpy(dcode + b * 16, blk + 2, 16);
                    }
                }
            }
        } else {
            // already S2 planar (canonical pack): direct copy
            std::memcpy(staging.data(), src, BLOB);
        }
        if (!check(cudaMemcpy((uint8_t*) d_blobs_ + dst_off, staging.data(), blob_bytes,
                              cudaMemcpyHostToDevice),
                   "copy blob", err)) { close(); return false; }
        res_[(size_t) L * (size_t) n_expert + (size_t) mine[i].second] = slot;
        held_.push_back(mine[i]);
    }
    resident_ = (int64_t) mine.size();
    gib_ = (double) bytes / (1024.0 * 1024.0 * 1024.0);

    // activation buffers: int8 codes + fp32 scales + hx per chunk, for H-wide and FF-wide acts.
    // n_tok_max/cap_entries bound the window.  P3.1: 8 tokens was the decode window; the prefill feeds whole
    // prompt chunks, so the window is 64 tokens (640 entries) and callers sub-batch larger chunks.
    n_tok_max_ = 64;
    cap_entries_ = 64 * 10;
    // P3.2: ONE packed ActQ buffer per device, tokens contiguous ([q H][scale][hx] each, ACTS_STRIDE), so a
    // layer's window uploads only n_tok * ACTS_STRIDE bytes - not the whole 64-token buffer (measured: the
    // full 205 KB copy cost 0.073 ms per device per layer on the M10's slow host link).
    acts_bytes_ = (size_t) n_tok_max_ * (size_t) ACTS_STRIDE;
    const size_t ce = (size_t) cap_entries_;
    const size_t jobs_bytes = ce * 3 * sizeof(int32_t);
    // P6.x (measured): ONE contiguous staging region [packed acts][job table] so a layer costs ONE async
    // upload instead of two.  A CUDA call is ~4 us in this VM; 2 calls -> 1 per device per layer.
    up_bytes_ = acts_bytes_ + jobs_bytes;
    // P7 (2026-10-12, MEASURED AND REJECTED as the default): zero-copy staging - the pinned buffer
    // allocated MAPPED, the kernels reading acts+jobs straight from host memory.  It removes the
    // per-layer cudaMemcpyAsync (~4 us of API x 4 devices) but the mapped reads are UNCACHED system
    // memory on sm_50: the kernel stalls on PCIe latency and the engine measured 28.2/34.1/33.8 tok/s
    // vs 40.0/51.0/46.7 (-30%).  OFF by default; STRATA_MW_ZEROCOPY=1 restores it for reference.
    static const bool zerocopy = [] {
        const char* e = std::getenv("STRATA_MW_ZEROCOPY");
        return e != nullptr && std::atoi(e) != 0;
    }();
    zerocopy_ = zerocopy;    if (zerocopy) {
        if (!check(cudaHostAlloc((void**) &h_up_, up_bytes_,
                                 cudaHostAllocMapped | (::strata::core::maxwell_portable() ? cudaHostAllocPortable : 0)),
                   "cudaHostAlloc mapped staging", err)) { close(); return false; }
        if (!check(cudaHostGetDevicePointer((void**) &d_up_, h_up_, 0), "map staging", err)) {
            close(); return false;
        }
        std::fprintf(stderr, "strata maxwell: acts+jobs staging is ZERO-COPY (mapped pinned; measured "
                             "-30%% tok/s on sm_50 - uncached sysmem reads; reference only)\n");
    } else {
        if (!check(cudaMalloc(&d_up_, up_bytes_), "cudaMalloc staging", err)) { close(); return false; }
        if (!check(cudaMallocHost((void**) &h_up_, up_bytes_), "cudaMallocHost staging", err)) { close(); return false; }
        if (gpuflag_mode() >= 2) {   // E1 mode 2: the second staging set (layer l+1 stages while l reads)
            if (!check(cudaMalloc(&d_up2_, up_bytes_), "cudaMalloc staging2", err)) { close(); return false; }
            if (!check(cudaMallocHost((void**) &h_up2_, up_bytes_), "cudaMallocHost staging2", err)) { close(); return false; }
        }
    }
    if (gpuflag_mode() >= 1 && !check(cudaMalloc(&d_ctr_, 4), "cudaMalloc comp ctr", err)) { close(); return false; }
    if (d_ctr_ != nullptr) { cudaMemset(d_ctr_, 0, 4); cudaGetLastError(); }
    d_acts_ = d_up_;                 // the packed acts live at offset 0
    h_acts_ = h_up_;
    d_jobs_ = (int32_t*) (d_up_ + acts_bytes_);   // re-based per layer in begin_chain (right after the acts)
    h_jobs_ = (int32_t*) (h_up_ + acts_bytes_);
    d_aq_ = d_acts_;                 // legacy pointers: the batch kernels index by token now
    d_ascale_ = nullptr;
    d_ahx_ = nullptr;
    if (!check(cudaMalloc(&d_faq_, ce * FF), "cudaMalloc faq", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_fascale_, ce * (FF/32) * sizeof(float)), "cudaMalloc fascale", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_fahx_, ce * (FF/32) * sizeof(float)), "cudaMalloc fahx", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_ff_, ce * FF * sizeof(float)), "cudaMalloc ff", err)) { close(); return false; }
    if (!check(cudaMalloc(&d_out_, ce * H * sizeof(float)), "cudaMalloc out", err)) { close(); return false; }
    if (!check(cudaMallocHost((void**) &h_out_, ce * H * sizeof(float)), "cudaMallocHost out", err)) { close(); return false; }

    // P3.3: per-device streams and events for the overlap DAG
    for (int s = 0; s < 4; ++s) {
        if (!check(cudaStreamCreateWithFlags(&streams_[s], cudaStreamNonBlocking), "stream", err)) { close(); return false; }
        if (!check(cudaEventCreateWithFlags(&ev_gup_[s], cudaEventDisableTiming), "event", err)) { close(); return false; }
        if (!check(cudaEventCreateWithFlags(&ev_down_[s], cudaEventDisableTiming), "event", err)) { close(); return false; }
    }
    if (!check(cudaEventCreateWithFlags(&ev_up_, cudaEventDisableTiming), "event", err)) { close(); return false; }
    if (!check(cudaEventCreateWithFlags(&ev_q_, cudaEventDisableTiming), "event", err)) { close(); return false; }

    owned_idx_.assign((size_t) cap_entries_, -1);
    // P3.4 (rev 3, misurato): the chain runs inline on the caller's thread.  The persistent per-device worker
    // (condvars first, atomic wait/notify second) measured SLOWER: the four devices' cudaDeviceSynchronize
    // calls serialise inside the driver, so parallelising them buys nothing, and the per-layer handoff added
    // ~150 us/layer (29.3 vs 22.2 ms/window).  What pays is shrinking the single finish: the down stage now
    // launches there and scatters straight into the mapped destination (no D2H, no host memcpy).
    //
    // THE ROUND-TRIP GATE (native packs): the conversion above is checked on the real data before this
    // tier answers a token.  The failure mode it exists to catch is the one the first IQ2_XS attempt died
    // of and never isolated: a wrong offset or row stride in the src->blob conversion decodes PLAUSIBLE
    // garbage - never a fault, a model that answers nonsense words instead.  One held expert of each
    // distinct gate/up encoding runs the WHOLE chain on the device (the converted blob, the tier's own
    // packed ActQ, its own launch path) against the raw source blob dequantized by ggml and dotted in
    // fp64.  The tolerance is the act + FF quantisation noise the 3060's own native path carries
    // (~1e-2); a conversion bug is ~1.0 and cannot pass it.
    if (!geo_.empty()) {
        ggml_cpu_init();   // the to_float tables (idempotent; native_fmt already did this)
        int seen[4] = {0, 0, 0, 0};   // by gate/up type: 42, 22, 16, 29
        double worst = 0.0;
        for (const auto& pr : held_) {
            const int64_t L = pr.first, e = pr.second;
            const auto& f = lay.fmt[(size_t) L];
            const int ti = f.gu_type == 42 ? 0 : (f.gu_type == 22 ? 1 : (f.gu_type == 16 ? 2 : 3));
            if (ti < 0 || seen[ti]) continue;
            seen[ti] = 1;
            const uint8_t* src = src_blob(L, e, ctx);
            if (src == nullptr) {
                err = "MaxwellExperts: the round-trip gate lost its source blob";
                close();
                return false;
            }
            const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
            const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
            std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
            for (int64_t r = 0; r < FF; ++r) {
                tg->to_float(src + (size_t) r * f.gu_row, G.data() + (size_t) r * H, H);
                tg->to_float(src + f.up_off + (size_t) r * f.gu_row, U.data() + (size_t) r * H, H);
            }
            for (int64_t r = 0; r < H; ++r)
                td->to_float(src + f.down_off + (size_t) r * f.d_row, D.data() + (size_t) r * FF, FF);
            // the activation: deterministic, the engine's own ActQ (what the dispatch hands the tier)
            std::vector<float> x(H);
            std::mt19937 rng(1234 + (int) L * 31 + (int) e);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);
            strata::kernels::cpu::ActQ a;
            strata::kernels::cpu::act_quant_any(x.data(), (int) H, a);
            // the reference: the float expert (fp64 dots, the SwiGLU in fp32 as the kernels have it)
            std::vector<float> ref(H), h((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) {
                    g += (double) G[(size_t) r * H + i] * x[i];
                    u += (double) U[(size_t) r * H + i] * x[i];
                }
                h[(size_t) r] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i) o += (double) D[(size_t) r * FF + i] * h[i];
                ref[(size_t) r] = (float) o;
            }
            // the device: one job through the CONVERTED blob, the tier's own launch path
            const LayerGeo& g = geo_[(size_t) L];
            const int32_t slot = res_[(size_t) L * (size_t) n_expert_ + (size_t) e];
            const size_t nc4 = (size_t) (H / 32) * 4;
            std::memcpy(h_acts_, a.q, H);
            std::memcpy(h_acts_ + H, a.scale, nc4);
            std::memcpy(h_acts_ + H + nc4, a.hx, nc4);
            h_jobs_ = (int32_t*) (h_up_ + (size_t) 1 * ACTS_STRIDE);   // one token, re-based as begin_chain
            d_jobs_ = (int32_t*) (d_up_ + (size_t) 1 * ACTS_STRIDE);
            h_jobs_[0] = slot; h_jobs_[1] = 0; h_jobs_[2] = 0;
            if (!check(cudaMemcpyAsync(d_up_, h_up_, (size_t) 1 * ACTS_STRIDE + 3 * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, streams_[0]), "round-trip upload", err)) {
                close();
                return false;
            }
            if (g.gu_type == 42) {
                if (s2_expert_maxwell_v2()) {
                    s2_expert_maxwell_gup_quant_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_acts_,
                                                            d_faq_, d_fascale_, d_fahx_, streams_[0]);
                } else {
                    s2_expert_maxwell_gup_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_acts_, d_ff_,
                                                      streams_[0]);
                    s2_expert_maxwell_quant_ff_async(d_ff_, d_faq_, d_fascale_, d_fahx_, FF / 32,
                                                     streams_[0]);
                }
            } else {
                iq_expert_maxwell_gup_quant_batch_async(g.gu_type, d_blobs_ + g.base, g.blob_bytes,
                                                        d_jobs_, 1, d_acts_, g.o_gate, g.o_up, g.gu_row,
                                                        d_faq_, d_fascale_, d_fahx_, nullptr, streams_[0]);
            }
            s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, 1, d_faq_, d_fascale_, d_fahx_,
                                               d_out_, g.blob_bytes, g.d_codes_off, g.d_scales_off,
                                               streams_[0]);
            if (!check(cudaStreamSynchronize(streams_[0]), "round-trip sync", err)) { close(); return false; }
            if (!check(cudaMemcpy(h_out_, d_out_, (size_t) H * sizeof(float), cudaMemcpyDeviceToHost),
                       "round-trip download", err)) { close(); return false; }
            double num = 0, den = 0;
            for (int64_t r = 0; r < H; ++r) {
                num += std::fabs((double) h_out_[r] - ref[(size_t) r]);
                den += std::fabs((double) ref[(size_t) r]);
            }
            const double rv = num / (den + 1e-30);
            worst = std::max(worst, rv);
            std::fprintf(stderr, "MaxwellExperts (CUDA%d): round-trip gate layer %lld expert %lld "
                                 "(%s): rel %.2e vs the float reference\n",
                         device_, (long long) L, (long long) e,
                         ggml_type_name((ggml_type) f.gu_type), rv);
            std::fflush(stderr);
            if (rv > 3e-2) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "MaxwellExperts (CUDA%d): the round-trip gate FAILED on layer %lld "
                              "expert %lld (%s): rel %.2e > 3e-2 - the conversion is wrong, refusing",
                              device_, (long long) L, (long long) e,
                              ggml_type_name((ggml_type) f.gu_type), rv);
                err = buf;
                close();
                return false;
            }
        }
        std::fprintf(stderr, "MaxwellExperts (CUDA%d): round-trip gate PASSED (worst rel %.2e, "
                             "%lld blobs converted)\n", device_, worst, (long long) resident_);
        std::fflush(stderr);
    }
    return true;
}

void MaxwellExperts::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    for (int s = 0; s < 4; ++s) {
        if (streams_[s]) { cudaStreamDestroy(streams_[s]); streams_[s] = nullptr; }
        if (ev_gup_[s]) { cudaEventDestroy(ev_gup_[s]); ev_gup_[s] = nullptr; }
        if (ev_down_[s]) { cudaEventDestroy(ev_down_[s]); ev_down_[s] = nullptr; }
    }
    if (ev_up_) { cudaEventDestroy(ev_up_); ev_up_ = nullptr; }
    if (ev_q_) { cudaEventDestroy(ev_q_); ev_q_ = nullptr; }
    if (d_blobs_) { cudaFree(d_blobs_); d_blobs_ = nullptr; }
    if (d_up_ && !zerocopy_) { cudaFree(d_up_); }   // zero-copy: d_up_ IS h_up_'s mapping
    d_up_ = nullptr;
    if (d_up2_) { cudaFree(d_up2_); d_up2_ = nullptr; }
    if (h_up2_) { cudaFreeHost(h_up2_); h_up2_ = nullptr; }
    if (d_ctr_) { cudaFree(d_ctr_); d_ctr_ = nullptr; }
    if (h_up_) { cudaFreeHost(h_up_); h_up_ = nullptr; }
    d_acts_ = nullptr; h_acts_ = nullptr; d_jobs_ = nullptr; h_jobs_ = nullptr;
    d_aq_ = nullptr; d_ascale_ = nullptr; d_ahx_ = nullptr;
    if (d_faq_) { cudaFree(d_faq_); d_faq_ = nullptr; }
    if (d_fascale_) { cudaFree(d_fascale_); d_fascale_ = nullptr; }
    if (d_fahx_) { cudaFree(d_fahx_); d_fahx_ = nullptr; }
    if (d_ff_) { cudaFree(d_ff_); d_ff_ = nullptr; }
    if (d_out_) { cudaFree(d_out_); d_out_ = nullptr; }
    if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    device_ = -1;
    resident_ = 0;
    gib_ = 0.0;
    res_.clear();
    held_.clear();
}

bool MaxwellExperts::owns(int64_t layer, int64_t expert) const {
    if (device_ < 0 || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] >= 0;
}

int64_t MaxwellExperts::count_owned(int64_t layer, const int32_t* ids, int64_t n_tok, int64_t k) const {
    if (device_ < 0 || layer < 0 || layer >= n_layers_) return 0;
    int64_t c = 0;
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        if (res_[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0) ++c;
    }
    return c;
}

bool MaxwellExperts::begin_chain(std::string& err) {
    const auto bt0 = std::chrono::steady_clock::now();
    const int64_t layer = w_layer_;
    const strata::kernels::cpu::ActQ* acts = w_acts_;
    const int32_t* ids = w_ids_;
    const int64_t n_tok = w_n_tok_, k = w_k_;
    const int32_t* kind = nullptr;   // collect every entry this device owns
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.err; return false; }
    const auto bt1 = std::chrono::steady_clock::now();
    owned_row_.clear();

    // P3.1 BATCHED, three phases per layer (not per expert):
    //   Phase 1: upload all ActQ + launch ALL stage1 (async, one sync at the end)
    //   Phase 2: ONE D2H of all ff, requant all on host, ONE H2D of all ffq
    //   Phase 3: launch ALL stage2 (async, one sync), ONE D2H of all results

    // ---- collect owned entries first (no GPU work yet) ----
    struct Job { int32_t slot; int64_t tok; int32_t row; };
    // thread_local to avoid per-layer allocation (the token path allocates nothing)
    static thread_local std::vector<Job> jobs;
    jobs.clear();
    for (int64_t i = 0; i < n_tok * k; ++i) {
        if (kind && kind[i] != -1 && kind[i] != 3) continue;
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const int32_t slot = res_[(size_t) layer * (size_t) n_expert_ + (size_t) e];
        if (slot < 0) continue;
        jobs.push_back({slot, i / k, (int32_t) i});
    }
    if (jobs.empty()) return true;
    if ((int64_t) jobs.size() > cap_entries_) {
        err = "MaxwellExperts: more owned entries than cap_entries";
        return false;
    }
    // P3.1 stats: histogram of jobs/layer to validate batching effectiveness
    {
        static thread_local long hist[16] = {};
        static thread_local long n_calls = 0;
        if (jobs.size() < 16) ++hist[jobs.size()];
        if (++n_calls % 20000 == 0) {
            std::fprintf(stderr, "Maxwell STAT jobs/layer histogram (n=%ld):", n_calls);
            for (int i = 0; i < 16; ++i) if (hist[i]) std::fprintf(stderr, " [%d]=%ld", i, hist[i]);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
    }

    // ---- ONE packed upload (acts + job table, contiguous) then ONE launch per stage: the whole layer's
    //      jobs are processed by the two batched kernels (gup -> quant -> down).  No events, no per-job
    //      launches; finish() is the single wait per layer. ----
    {
        // re-base the job table right after the packed acts so ONE async memcpy covers both (P6.x).
        // E1 mode 2: alternate the two staging sets (layer l+1 stages while layer l's kernels still
        // read set 0) - the selection is the SERVICE parity, matching the launch sequence.
        uint8_t* hsel = h_up_;
        uint8_t* dsel = d_up_;
        if (gpuflag_mode() >= 2 && d_up2_ != nullptr && (comp_seq_ & 1u)) { hsel = h_up2_; dsel = d_up2_; }
        h_jobs_ = (int32_t*) (hsel + (size_t) n_tok * ACTS_STRIDE);
        d_jobs_ = (int32_t*) (dsel + (size_t) n_tok * ACTS_STRIDE);
        h_acts_ = hsel;
        d_acts_ = dsel;
        const size_t nc4 = (size_t) (H / 32) * 4;
        for (int64_t t = 0; t < n_tok; ++t) {
            const auto& A = acts[t];
            uint8_t* at = h_acts_ + (size_t) t * ACTS_STRIDE;
            std::memcpy(at, A.q, H);
            std::memcpy(at + H, A.scale, nc4);
            std::memcpy(at + H + nc4, A.hx, nc4);
        }
    }
    // ---- ONE job table + ONE launch per stage: the whole layer's jobs are processed by two batched
    //      kernels (gup -> quant -> down), so per-job launch overhead disappears and the SMs fill. ----
    for (size_t j = 0; j < jobs.size(); ++j) {
        h_jobs_[j * 3] = jobs[j].slot;
        h_jobs_[j * 3 + 1] = (int32_t) jobs[j].tok;
        h_jobs_[j * 3 + 2] = jobs[j].row;   // destination row: the token path's entry index.  finish_chain
                                            // rewrites it (and re-uploads) for the prefill's staging order.
    }
    // ONE async upload for the acts + the job table (contiguous staging); finish() is the only wait.
    // Zero-copy: the mapped staging needs no upload at all - the kernel reads it over PCIe.
    if (!zerocopy_ &&
        !check(cudaMemcpyAsync(d_acts_, h_acts_, (size_t) n_tok * ACTS_STRIDE + jobs.size() * 3 * sizeof(int32_t),
                               cudaMemcpyHostToDevice, streams_[0]), "copy acts+jobs", err)) return false;
    // v2 (default): gate+up and the FF quantisation in ONE launch - the fused kernel's block covers exactly
    // one 32-value chunk of its job, so the separate quant launch (~4 us of API per device per layer in this
    // VM, plus the kernel off the chain) disappears.  v1 keeps the two-launch pair.  A native pack's
    // i-quant layers (IQ2_S/IQ2_XXS/IQ1_M) take the IQ kernel - it is fused by construction, and the
    // STRATA_S2_V2 arm is the S2 form's geometry A/B, not a choice the i-quant path offers.
    const LayerGeo& g = geo_of(layer);
    if (g.gu_type == 42) {
        if (s2_expert_maxwell_v2()) {
            s2_expert_maxwell_gup_quant_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_acts_,
                                                    d_faq_, d_fascale_, d_fahx_, streams_[0]);
        } else {
            s2_expert_maxwell_gup_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_acts_, d_ff_,
                                              streams_[0]);
            s2_expert_maxwell_quant_ff_async(d_ff_, d_faq_, d_fascale_, d_fahx_,
                                             (int) (jobs.size() * (size_t) (FF / 32)), streams_[0]);
        }
    } else {
        iq_expert_maxwell_gup_quant_batch_async(g.gu_type, d_blobs_ + g.base, g.blob_bytes, d_jobs_,
                                                (int) jobs.size(), d_acts_, g.o_gate, g.o_up, g.gu_row,
                                                d_faq_, d_fascale_, d_fahx_, nullptr, streams_[0]);
    }
    // The TOKEN path hands begin() the mapped destination: its down stage launches HERE and scatters each row
    // straight into `w_dst_` (the job table already carries every row's destination), so it runs while the
    // caller finishes the layer - no D2H copy and no host scatter.  The prefill's destination is only known at
    // finish_to, so ITS down launches there, into the staging block.
    down_launched_ = false;
    if (w_dst_mapped_) {
        // E1: in GPU-flag mode the decode down launch carries the completion flag - the LAST block
        // raises it once this tier's rows are visible, and (mode 2) the host never syncs this layer.
        // fp16 rows take the no-tail variant, so the flag stays off there (the sync path is kept).
        const bool e1 = comp_flag_ != nullptr && !w_rows_f16_ && gpuflag_mode() >= 1;
        const uint32_t ring = e1 ? (comp_seq_ + 1) : 0;
        if (w_rows_f16_) {
            // fp16 rows (opt-in): H halves packed into the fp32 row's own slot (the first H/2 floats), so
            // the per-group destination offset stays whatever `w_dst_` already carries.  Half the bytes
            // cross the uplink; the verifier's row copy converts them back (its kind16 flags mark them).
            s2_expert_maxwell_down_batch_f16_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_faq_,
                                                   d_fascale_, d_fahx_, (uint16_t*) w_dst_, g.blob_bytes,
                                                   g.d_codes_off, g.d_scales_off, streams_[0]);
        } else {
            s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, (int) jobs.size(), d_faq_,
                                               d_fascale_, d_fahx_, w_dst_, g.blob_bytes, g.d_codes_off,
                                               g.d_scales_off, streams_[0],
                                               e1 ? comp_flag_ : nullptr, ring, e1 ? d_ctr_ : nullptr);
        }
        if (e1) comp_seq_ = ring;
        down_launched_ = true;
    }
    for (size_t j = 0; j < jobs.size(); ++j) {
        owned_row_.push_back(jobs[j].row);
        computed_.fetch_add(1, std::memory_order_relaxed);
    }
    if (std::getenv("STRATA_MAXWELL_TIMING") != nullptr) {
        const auto bt2 = std::chrono::steady_clock::now();
        static thread_local double a_set = 0, a_all = 0;
        static thread_local long n = 0;
        a_set += std::chrono::duration<double, std::milli>(bt1 - bt0).count();
        a_all += std::chrono::duration<double, std::milli>(bt2 - bt0).count();
        if (++n % 2000 == 0)
            std::fprintf(stderr, "Maxwell TIMING begin  dev=%d n=%ld setdev=%.1f us total=%.1f us\n",
                         device_, n, 1000.0 * a_set / (double) n, 1000.0 * a_all / (double) n);
    }
    return true;
}

bool MaxwellExperts::begin(int64_t layer, const strata::kernels::cpu::ActQ* acts, const int32_t* ids,
                           int64_t n_tok, int64_t k, const int32_t* /*kind*/, float* dst_mapped,
                           bool rows_f16, std::string& err) {
    (void) err;
    if (device_ < 0) return true;
    if (n_tok > n_tok_max_ || n_tok * k > cap_entries_) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "MaxwellExperts (CUDA%d): window too large (%lld tok x %lld)",
                      device_, (long long) n_tok, (long long) k);
        err = buf;
        return false;
    }
    w_layer_ = layer; w_acts_ = acts; w_ids_ = ids; w_n_tok_ = n_tok; w_k_ = k;
    w_dst_ = dst_mapped;
    w_dst_mapped_ = dst_mapped != nullptr;
    w_rows_f16_ = rows_f16 && dst_mapped != nullptr;   // fp16 applies to the token path only
    w_row_of_ = nullptr;
    w_row_base_ = 0;
    w_n_embd_ = H;
    return begin_chain(err);
}

bool MaxwellExperts::finish(float* out, std::string& err) {
    (void) out;   // the destination was handed to begin(); only the wait and the scatter-less finish remain
    return finish_chain(err);
}

bool MaxwellExperts::finish_to(float* dst_rows, const int32_t* row_of_entry, int32_t row_base, int64_t n_embd,
                               std::string& err) {
    if (device_ < 0) return true;
    w_dst_ = dst_rows; w_row_of_ = row_of_entry; w_row_base_ = row_base; w_n_embd_ = n_embd;
    w_dst_mapped_ = false;
    return finish_chain(err);   // the prefill's down stage launches here, where its destination is known
}

bool MaxwellExperts::finish_chain(std::string& err) {
    // E1: a decode service this tier sat out still advances the completion sequence - the verifier's
    // expected value counts every service, so the flag must rise here (a mapped write, one fence).
    if (owned_row_.empty()) {
        if (comp_flag_ != nullptr && w_dst_mapped_ && gpuflag_mode() >= 1) {
            __atomic_store_n(comp_flag_, comp_seq_ + 1, __ATOMIC_SEQ_CST);
            comp_seq_ += 1;
        }
        return true;
    }
    const auto ft0 = std::chrono::steady_clock::now();
    // THIS device, explicitly: without the scope the sync below lands on whatever device the caller had
    // current (the primary GPU, whose graph is waiting for this very host step: a deadlock).
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.err; return false; }
    const auto ft1 = std::chrono::steady_clock::now();
    // The down stage: launched at begin() for the token path (see there).  For the prefill it launches NOW,
    // into the staging block, and the host copies the rows out below.  One wait per device per layer.
    if (!down_launched_) {
        for (size_t j = 0; j < owned_row_.size(); ++j) h_jobs_[j * 3 + 2] = (int32_t) j;
        if (!zerocopy_ &&
            !check(cudaMemcpyAsync(d_jobs_, h_jobs_, owned_row_.size() * 3 * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams_[0]), "copy jobs dst", err)) return false;
        const LayerGeo& g = geo_of(w_layer_);
        s2_expert_maxwell_down_batch_async(d_blobs_ + g.base, d_jobs_, (int) owned_row_.size(), d_faq_,
                                           d_fascale_, d_fahx_, d_out_, g.blob_bytes, g.d_codes_off,
                                           g.d_scales_off, streams_[0]);
        down_launched_ = true;
    }
    // E1 mode 2: the decode path does not wait - the down kernel's last block raises the tier flag,
    // the 3060's graph multi-waits on it, and the NEXT service's staging uses the other buffer set.
    // The prefill (non-mapped destination) still syncs: its rows are read on the host.
    if (w_dst_mapped_ && comp_flag_ != nullptr && !w_rows_f16_ && gpuflag_mode() >= 2) {
        w_dst_mapped_ = false;      // the service is complete; the scatter was in-kernel
        owned_row_.clear();
        return true;
    }
    if (!check(cudaStreamSynchronize(streams_[0]), "finish sync", err)) return false;
    const auto ft2 = std::chrono::steady_clock::now();
    if (const char* e = std::getenv("STRATA_MAXWELL_TIMING")) {
        static thread_local double a_set = 0, a_sync = 0;
        static thread_local long n = 0;
        a_set += std::chrono::duration<double, std::milli>(ft1 - ft0).count();
        a_sync += std::chrono::duration<double, std::milli>(ft2 - ft1).count();
        if (++n % 2000 == 0)
            std::fprintf(stderr, "Maxwell TIMING finish dev=%d n=%ld setdev=%.1f us sync=%.1f us\n",
                         device_, n, 1000.0 * a_set / (double) n, 1000.0 * a_sync / (double) n);
    }
    if (w_dst_mapped_) { owned_row_.clear(); return true; }
    if (!check(cudaMemcpy(h_out_, d_out_, owned_row_.size() * (size_t) w_n_embd_ * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy results batch", err)) return false;
    for (size_t s = 0; s < owned_row_.size(); ++s) {
        const int32_t i = owned_row_[s];
        const int32_t row = w_row_of_ ? w_row_of_[i] - w_row_base_ : i;
        std::memcpy(w_dst_ + (size_t) row * (size_t) w_n_embd_, h_out_ + s * (size_t) w_n_embd_,
                    (size_t) w_n_embd_ * sizeof(float));
    }
    return true;
}

}  // namespace strata::core
