# WORKLOG — Strata + Maxwell M10 (measured-facts log)

**Repo**: `/home/toor/Strata`. This file records **only measured facts and results**: numbers,
active modifications, verification methods, closed explorations. Campaign details live in
`bench/results/<date>-<name>/`; the earlier narrative versions of this file are in git history.
No speculation: anything without a measurement behind it does not appear here.

---

## 1. Current state (2026-10-21)

**Branch state**: `maxwell-0.2.2` = the full Maxwell layer merged into upstream **v0.2.2**
(engine 0.1.47), merge commit `e734bd38` (7 conflicts, all documented in the commit message).
Branch `maxwell-0.1.45` and tag `maxwell-0.1.45-split-production-OK` frozen as the rollback
anchor; `engine/strata.bak-maxwell-0.1.45` is the seconds-level binary rollback.
**Production decision on the 0.2.2 port: pending explicit go** (all gates green, below).

**v0.2.2 port gates** (production config `strata-iq2xs-embed-vram-split.json`, `--spec-split`,
binary `164fa8a1`; details in `bench/results/2026-10-21-022-port/`):

| Gate | Result |
|---|---|
| Build (strata + parity + bench targets) | clean; upstream `nv_defaults` (sm_86 table) in |
| `maxwell_iq_parity` | **0 failures** (fasc 3.09e-07, rel 8.38e-03 — identical values to 0.1.45) |
| `iq_gup_bench` layer 0 | **294.6 µs/launch** = 21.4 GB/s — identical to 0.1.45 |
| Maxwell round-trip gate at open | PASSED (worst rel 1.15e-02, 5,675 blobs) |
| Load time | **231 s** (0.1.45: 424–554 s — the new page-cache file tier reads less) |
| 3×3 mean tok/s (temp 0) | **53.1** (story 47.5 · fib 59.2 · code 52.7) vs 0.1.45's 52.44 = +1.3% (noise) |
| TTFT (1,546-tok prompt) | 22.18 s (0.1.45: 22.5 s) |
| Fixed answers / context recall+negative / 8× concurrent / prefix reuse | 3/3 · 5/5 · 1 output of 8 · 57/62 cached |
| A/B creative run-to-run | diverges — the **same pre-existing base property** as 0.1.45, not new |
| 30-min soak (4 threads, temp 0.7) | **559 requests, 0 errors, 73,818 tokens** (0.1.45: 276 req, 0 err) |

**Upstream behavior changes carried by the port** (base properties, not defects): the chat
template defaults to **thinking** (answer correct with `max_tokens ≥ 96`, or disable via
`"chat_template_kwargs": {"enable_thinking": false}` — verified 0 reasoning chars, direct
answer; `serve/server.py` now exposes `reasoning_content`); the Linux file tier defaults to
page-cached experts (`STRATA_COLD_OVERLAP=4`) with gated IO prefetch.

**Production serving**: port 8088, config `strata-iq2xs-embed-vram-split.json`
(= standard production + `STRATA_EMBED_VRAM=1` + **`--spec-split`**), currently running the
ported 0.2.2 binary `164fa8a1` for the gates; rollback = `engine/strata.bak-maxwell-0.1.45`
or restart on `strata-iq2xs-embed-vram.json`.

| Measurement (3×3 + TTFT protocol, temp 0) | Value |
|---|---|
| Mean tok/s, split enabled (0.1.45 base, 2026-10-21 morning) | **52.44** (story 45.1–48.6 · fib 54.5–58.7 · code 52.4–53.3) |
| Mean tok/s, split enabled (0.2.2 port, 2026-10-21 night) | **53.1** (story 47.5 · fib 59.2 · code 52.7) |
| Mean tok/s, 0.2.2 + restructured gup (2026-10-22 evening, binary `ba92a4cb`) | **55.07** (story 50.3 · fib 60.3 · code 54.7) |
| Sustained tok/s, 30-min soak on 0.1.45 (276 requests, 0 errors) | **48.7** mean, 56.1 peak |
| Same-day 0.1.45 baseline without split | 43.06 → **+21.8%** |
| TTFT, 1,546-token prompt | 22.5 s (0.1.45) / 22.18 s (0.2.2) |
| Utilization under sustained load | 3060 63–92%, M10 29–50% |
| Soak decay (0.1.45) | none (first quarter 48.3 vs last quarter 48.8 tok/s) |

Load time: 424–554 s depending on page-cache state (dominated by ~35 GB of blobs from NVMe).
One model at a time: the GPUs hold a single system.

## 2. The system and its numbers

**Model**: Qwen3.8-Flash-Next IQ2_XS — 48 layers, 24,576 routed experts (512/layer, top-10 +
1 shared), H 2560, FF 640. **Tiers**: 1,874 experts in the 3060 cache (2.53 GiB) + 22,702 whole
experts on 4× Tesla M10 (5,676/5,676/5,675/5,675, ~7.6 GiB each, zero overlap). Context 256k,
KV k8v4, spec 4 (MTP threshold 0.5).

**The per-layer decode chain** (MWPROF, arm-against-arm):

| Config | ring (3060 graph + tier wait) | pool (host dispatch) | post | window |
|---|---:|---:|---:|---|
| baseline (no split) | 415 µs | 688 µs (plan 6 + actq 122 + jobs 14 + run 542) | 14 µs | 54.3 ms @ 2.06 tok |
| split + persist | 407 µs | 54 µs | 9 µs | **41.1 ms @ 2.20 tok** |

**The 0.2.2 anatomy (2026-10-22, instrumented ~3.7%, `2026-10-22-token-passage-anatomy`)**: the
ring collapsed to 59 µs/L (GPU-reach wait) — the port's kernels absorbed it. The window is now
**81% M10-tier host protocol**: finish (down launch + sequential card syncs + scatter) 32.56 ms
= 64%, begin (pack/staging/H2D/gup) 8.59 ms = 17%, draft 3.47, GPU-reach 2.84. The cards'
kernels overlap (device wall = max ~435 µs/L; gup ~240 uniform = the instruction floor, down
65–185 µs/L by visit order) while the host burns 857 µs/L serially ≈ 2× the device wall; duty:
3060 83–86% near-saturated, M10s 38–58% (2× headroom). 2-slot serving amortizes the protocol
(each request 52–59 tok/s, aggregate >100). M10 kernel work is deprioritized by this data until
the host protocol cost falls.

The split window pipelines the two token groups (group B's mixer does not depend on group A's
experts) and fills the M10 wait with legal 3060 work.

**M10 kernels (GM107, sm_50)** — measured facts that drive the design:
- sm_50 has **no DP4A** (cc ≥ 6.1), **no fp16 arithmetic** (cc ≥ 5.3), no tensor cores. `dp4a`
  is emulated as 4× `vmad.s32` = **one instruction per product** (FMA parity); hypothetical
  native DP4A hardware would be worth ≤14% of the gup loop (SASS: ~347 instructions/iteration,
  64 of them VMAD, ~213 logic/shift for i-quant decode).
- gup IQ: instruction-bound, 5.6–7.8× above the DRAM floor (practical floor **83 GB/s*);
  ~21 GB/s effective. Down (S2 planar): 90 µs/layer on the first-served card, 240–550 on the
  following ones.
- **The poison**: every host driver call during the down window stretches the cards served after
  it (~+55 µs/card). Proof: `STRATA_MW_REV=1` flips the stagger; GENTLE_SYNC keeps it (it is the
  calls, not the spin).
- **Mapped reads are uncached** on sm_50: a kernel reading acts/jobs from mapped memory inside
  the dot loop loses 30% (P7, `STRATA_MW_ZEROCOPY` kept as reference); the persist v0 that
  pulled staging over mapped memory cost ~2.2 ms/layer. Mapped **writes** work fine (the down
  kernel has always used them).
- Concurrent syncs and driver calls serialize inside this VM's driver: one CUDA call ≈ 4 µs;
  a host worker thread for the tier measured **slower** than inline (P3.4).

**Bus/PCIe** (guest measurements; PLX PEX 8747: 16 real lanes to the CPU, 4×8 to the M10s; the
3060 has a dedicated root port): H2D 3.14 · D2H 3.35 · mapped write 3.35 · mapped read 3.13 GB/s
aggregated over 4×M10; 3060: 11.4/13.2. **Gen1 is final**: at Gen2 bandwidth doubles (6.70/6.12)
but the bus becomes unstable (correctable errors) and the end-to-end gain is ~1%. The guest's
link-state readout is stale: trust measured bandwidth, not the register.

**Operational rules**: persistence mode `nvidia-smi -pm 1` is ON — **re-enable it after every
reboot**; never measure under a profiler (CUPTI speeds up M10 decode by ~17%); the
LAYERPROF+DECODE_TIMING instrumentation costs ~3.6% (arm-against-arm comparisons only); the
wrapper restarts the engine on its own (START_RETRIES) — "engine stopped unexpectedly" means a
reload is in progress.

## 3. Active modifications (measured, in tree)

1. **`--spec-split` (production)**: the two-group verify-window pipeline. **+21.8%**
   (52.44 vs 43.06, same day/protocol), TTFT unchanged, full quality gate passed (§4).
   The old "~7% slower" verdict (comment in `generate.cpp`, pre-kernel-campaigns) is **stale**
   and should be corrected next time that file is touched.
2. **IQ gup kernel campaigns (2026-10-12, in every binary since)**: codebook grids staged in
   shared memory → **+8.4% end-to-end** (42.4→46.0); SPLIT reduction via warp shuffles,
   bit-identical → kernel 300→275 µs (engine flat); **SPLIT=4 is the measured optimum**
   (2: −23…−31%; T256: parity); `iq_gup_bench` harness (~40 s per iteration vs ~8 min of
   engine reload).
3. **`STRATA_EMBED_VRAM=1`** (active in production): the embedding table (322.07 MiB IQ4_XS) in
   3060 VRAM; rows bit-identical across all 248,320×2,560 (`native_embed_placement_test`);
   tok/s +0.21% = noise. Kept on request.
4. **`STRATA_HOST_WAIT_US=10`**: +0.7%, free, never negative. Kept.
5. **`STRATA_MW_PERSIST=1` (opt-in, requires the E1 flags)**: one resident kernel per M10;
   dispatch = packed H2D `[hdr|acts|jobs]` + `cuStreamWriteValue32` on a **device** doorbell
   (two calls per service per card, issued only while the M10s are idle); staging via DMA; halt
   = host store to a mapped word (a stream-write can never reach a kernel that owns its own
   stream); grid = the kernel's own occupancy (20 blocks / 5 SMs — query the CURRENT device,
   not ordinal 0). **In-order protocol**: per-launch relative doorbell, the kernel serves
   strictly `seen+1`, empty services ride the doorbell too (with the split window the host
   legitimately runs two services ahead: serving "the newest" skips one in between and its flag
   never rises — a hang we lived and fixed). Results: alone 42.08 tok/s (−2.3%; TTFT 39.1 s,
   measured pre-fix and never repeated alone afterwards); **compound with split 52.52 tok/s,
   TTFT 22.6 s** (parity with split alone); kernel parity 0 failures. **Under review**: L1/RO
   cache invalidation between services inside one kernel (empirically clean across all benches;
   a formal analysis is owed before any promotion).
6. **E1 / `STRATA_MW_GPUFLAG` (1 = GPU-raised tier flags + multi-wait, 2 = also no host sync)**:
   de-poisons the downs (uniform 123–135 µs) but alone it lands at parity (26.9 vs 27.0
   ms/token): the gain re-absorbs into the ring. It is the machinery persist and the tier use.
7. **Verification infrastructure (at every `open()`)**: the round-trip gate (one expert per
   encoding per card, full chain vs fp64: rel ~1.2e-2 = quantization noise; a wrong offset gives
   ~1.0 and cannot pass); `maxwell_iq_parity` (0 failures; **the right reference is the 3060
   production path, not an fp64 dequantizer** — its exact integer epilogues are not reproducible
   in float); `expert_maxwell_parity`.
8. **PLE direct guard** (`generate.cpp`): on this machine (7.7 GiB RAM) the 0.1.42 mmap probe
   thrashes the page cache (synchronous NVMe faults, 92–121 ms windows, −45%); with a Maxwell
   tier configured it stays direct. Closing commit: `b3c03f34` ("ple: with a Maxwell tier the
   probe keeps direct - mmap measured catastrophic on a RAM-thrashed rig").
9. **`STRATA_SPEC_DEPTH=1`**: enables the MTP acceptance diagnostic report (does not change
   the speculation depth).

## 4. Verification and method

- **Measurement protocol**: burn-in + 3 prompts × 3 runs, temp 0, 512 tokens, tok/s from the
  engine log. For sweeps: alternating blocks A-B-B-A / B-A-A-B, CI95 (Student-t) over blocks,
  promotion threshold ≥ +0.5% with CI above zero.
- **Gate suite**: fixed answers (17×23=391, 9999, Paris, 23), 391×8 concurrent identical,
  1,509–1,546-token prefill with no prefix reuse, round-trip ×4 at every load.
- **Base nondeterminism (measured 2026-10-21)**: at temperature 0, creative prompts diverge
  run-to-run **even without split** (two different stories on the same prompt; fib 463/506/512)
  — a property of the k8v4 base + 3060-cache rows vs M10 rows, documented in
  `bench/results/2026-09-27-cache-parity`. The split **adds no divergence** (A/B: baseline
  diverges identically; under split fib was identical in the sample). Factual prompts are
  stable on both.
- **Split quality gate (all green, 2026-10-21)**: factual 4/4; 8× concurrent → 1 output;
  6,225-token context → correct first-word recall + correct absent-word negative; prefix reuse
  6,220/6,225 cached; 30-min soak 276 requests / 0 errors / 85,710 tokens, zero decay.
- **Spec sweep**: 2/3/4/5/6 = 41.6/44.0/**45.0**/42.0/41.1 → 4 is optimal.
- **Batching** (`--batch N`, no-split era): aggregate 39.4→58.2 tok/s (N=1→4), plateau 58–59 at
  N=6–8, per-request 7.4 at N=8. Batch × split **not tested**.

## 5. Parked explorations (measured, closed)

| What | Measured outcome | Status |
|---|---|---|
| Tensor-split TP (`--maxwell-split tensor`) | 34.1–34.6 tok/s = −19% vs whole-expert; the 4 M10s in TP are truly parallel (simultaneous spans ~640 µs/layer/card) but the wire/assembly does not pay back | opt-in, parked (`2026-10-19-routed-tp-audit`) |
| Routed Meta (optimized autonomous TP) | 35.44–35.51 tok/s; tier +23.6% on the fixed bench but the window stayed flat (the bottleneck is the 3060 ring) | opt-in, parked (`2026-10-20-maxwell-meta`) |
| fp16 rows (`STRATA_MW_ROWS_F16`) | below noise, changes outputs | off |
| Zero-copy staging (`STRATA_MW_ZEROCOPY`) | −30% (uncached mapped reads) | off, reference only |
| Down via local staging + D2H (`STRATA_MW_LOCAL_DOWN`) | worse (469→771 µs/layer) | off |
| gup micro-diet (qh hoist + unroll) | noise (294.8→289.2 µs; XXS/IQ1_M mixed signs); the kernel is latency/issue-bound at 25–50% occupancy anyway | reverted (2026-10-20) |
| Shared expert on the M10s (FP16 prototype) | loses on isolated compute already (22 vs 60–86 µs/layer) | not integrated (`2026-10-19-maxwell-shared-screen`) |
| FP32 experts on the M10s | weights = 446 GB (14× VRAM); even if resident, the bandwidth floor alone would be ~1.2 ms/layer (2–3× slower than today's instruction-bound path) | closed by arithmetic |
| E1m2 alone | parity (26.9 vs 27.0 ms/token): the gain moves into the ring | opt-in (the base of persist) |
| gup multi-row-per-warp (`STRATA_MW_IQ_MR`, upstream 0.2.2's Gyro trick adapted) | parity-perfect (0 failures at MR 1/2/4/5) but **−17/−45/−29%** at MR 2/4/5: the block count is the latency hiding (120→30 blocks starves 5 SMs) and the SASS's 36% per-code unpack never amortises | closed (`2026-10-22-m10-mr-gup`); the knob stays, default 1 |
| GENTLE_SYNC (sleep-poll finish sync) | **−9.6%** (48.90 vs 54.07): the 30 µs wake quantum × 4 cards × 48 layers ≈ 5.8 ms/window swamps any spin-poison relief; the old "spin poison" verdict does not hold on 0.2.2 | closed (`2026-10-22-m10-mr-gup`) |

## 6. Chronology (one line per session; details in `bench/results/`)

- **2026-10-07/08**: bus/PLX measurements, Gen1 final; `--host-core last` (+~1%); IQ2_XS ported
  to the tier — all 24,576 experts on GPU; the round-trip gate and parity tools were born here.
- **2026-10-11**: the 0.1.42 port; −45% regression found (the PLE→mmap probe on 7.7 GiB RAM)
  and closed (direct guard in code); 23/23 logpos across binaries.
- **2026-10-12**: gup kernel campaigns (shared grids +8.4%; shuffle −36% kernel); SPLIT=4;
  `iq_gup_bench`; ABBA methodology; spec sweep (4 optimal); HOST_WAIT_US=10.
- **2026-10-13/14**: per-layer budget T1 (ring 420 / pool 688); S0 (cards balanced, gup
  instruction-bound, down staggered); the poison demonstrated (REV); E1 built (parity);
  batch plateau 58–59; the 256k base (42–43 tok/s).
- **2026-10-19**: embedding in VRAM (parity); shared-expert screening (negative); the full
  tensor-split campaign (−19%); routed-TP audit.
- **2026-10-20**: autonomous Meta (35.5, parked); **the documented-architecture audit**
  (1,117 µs/layer chain reconciled at 99%, duty 3060 35% / M10 21–25%) in
  `2026-10-20-arch-audit`; gup micro-diet measured dead; **the persistent tier built in three
  iterations** (mapped-pull 17.4 → DMA+doorbell 42.08; the SM-query deadlock and the halt
  deadlock lived and fixed) — `2026-10-20-maxwell-persist`.
- **2026-10-21**: **spec-split wins: +21.8%** (the historical verdict was stale); compound with
  persist: in-order bug found/fixed, 52.52; **the full production gate** (A/B, battery, soak) —
  promoted and serving — `2026-10-21-split-persist`, `2026-10-21-split-production-gate`.
- **2026-10-21 (night)**: the v0.2.2 port — merge `e734bd38` on `maxwell-0.2.2` (7 conflicts:
  union everywhere; pinned.cu takes upstream's `pin_ranges` API since our guard's precondition
  vanished; pool.cpp reconciles their constructor NUMA-plan with our lazy `start_workers()`).
  All gates green: parity 0 failures, kernel 294.6 µs identical, 3×3 mean 53.1 (0.1.45: 52.44),
  load 231 s, recall 5/5, 8× identical, soak 559 req/0 err/73,818 tok. Thinking-default template and the page-cache
  file tier are the base's new behavior — `2026-10-21-022-port`. Production switched by user go.
- **2026-10-22**: production GO on 0.2.2 (tag `maxwell-0.2.2-split-production-OK` @ `7e03f289`,
  pushed to the fork). The token-passage anatomy: **the lock moved** — the ring collapsed to
  59 µs/L and the window is now 81% M10-tier host protocol (finish 64% + begin 17%); the host
  runs 857 µs/L serial vs the cards' overlapped 435 µs/L device wall; M10 kernel work
  deprioritized by the data; decision menu A–E in the campaign dir — `2026-10-22-token-passage-anatomy`.
  Same day, Option A closed: the E1+persist no-wait protocol family re-tested with the ring gone
  (54.46 vs 54.07 = +0.7%, nothing) — the wait is real chain latency, not host waste; window
  invariant when waits move host→graph. Instrumenting that combo deadlocks (unfused wait +
  doorbell + split; bug note in the campaign). Spec-depth stays 4 (user call). The remaining
  levers: M10 serve-chain kernel work (E returns), 3060 per-layer graph work, multi-slot
  amortization.

- **2026-10-22 (evening)**: the M10 gup kernel campaign - SASS first (the hot loop is 11% VMAD,
  36% per-code unpack: MR's amortisable pool ~10%), then STRATA_MW_IQ_MR built (bit-exact at
  1/2/4/5) and measured: MR 2/4/5 = -17/-45/-29% - **closed** (block count is the latency
  hiding). Free byproduct: the restructured body's MR=1 codegen = **kernel 295 to 270 us (-8%)**,
  engine **55.07 vs 54.07 (+1.85%)**, deployed `ba92a4cb`. GENTLE_SYNC closed at -9.6%.
  Bench bimodality (cold ~350 / warm ~270-295) documented - warm-only comparisons -
  `2026-10-22-m10-mr-gup`.
- **2026-10-22 (night)**: the 3060 x-ray (STRATA_VERIFY_PROFILE, first use) - the graph's
  31.6 ms/window decomposed: **waitCPU 13.2 (42%, the M10 chain unhidden)**, hyper-connection
  projections ~5.1 (16%, six ~19 us launches/layer = latency, pair-fusion candidate ~+5%),
  dense stack ~11, waits ~2.5. **No waste found** - the useless-work hypothesis closed. The
  one new attackable item: the HC stack - `2026-10-22-3060-xray`. 2-slot demo closed 0.80x
  (slots double the per-token M10 jobs; nothing amortizes on this tier).

---

**Conventions**: `engine/strata` = `build/strata`, backups as `engine/strata.bak-<name>-<date>`;
local configs with credentials are gitignored; no commits/pushes by agents; comparisons always
arm-against-arm, same day/same binary whenever possible.
