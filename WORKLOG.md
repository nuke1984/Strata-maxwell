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
- **2026-10-22 (night 2)**: the PLE-in-RAM A/B on the 39 GB VM - **null result, closed**:
  identical diverse 9,873-token prompt = 75.1 tok/s prefill in BOTH `--ple-io ram` (28.8 GB
  mlocked) and `direct` - the prompt path is compute-bound on the M10 tier's 64-token prefill
  windows, not storage (the QD-256 batched reader + 1M-row cache keep SSD ahead of compute).
  Production stays direct (faster loads, RAM free). The VM disk's random reads also improved
  19 to 49 MB/s with the host change. New exposure: the ~75 tok/s prefill ceiling is the
  tier's prefill path - `2026-10-22-ple-ram-ab`.
  Follow-up: the full 3x3 head-to-head A1 (PLE RAM) vs A2 (PLE direct): 59.59 vs 58.87
  tok/s = parity inside the noise band; TTFT 36.87 vs 36.77 s. Confirmed under the full suite.
- **2026-10-22 (night 3)**: the head-to-head - M10 vs RAM, both arms from scratch, one
  variable (where the 22,702 non-cached experts compute): **A (M10s) decode 61.30 vs B (CPU
  pool + page cache) 12.12 tok/s = the M10s are worth 5.1x on decode**; B wins prefill 3.5x
  (259.9 vs 73.8 tok/s; TTFT 10.6 vs 37.0 s) - the tier's 64-token prefill windows are the
  documented cost, the concrete target if prompt speed ever matters. A also ran 61.3 vs its
  historical 53-55 family (the 40 GB RAM lifted the file tier; same-day observation) -
  `2026-10-22-m10-vs-ram`.

- **2026-10-23**: the M10 prefill anatomy (STRATA_MW_PREFILLPROF, env-gated): per layer, the
  **serve is 94% of the prefill block and 86% of the whole prompt time** (d2h 577 + quant 796
  + serve 59,669 + h2d 2,210 us/L; begin 0.18 + finish 15.02 ms per 64-token sub-batch - the
  sync absorbs real kernel walls). The kernels are instruction-throughput-bound (~4.7x off
  the byte floor, same physics as decode). Same day, the fix campaign: **#2 wider windows
  (NTM=256) NULL** (76.3 vs 75.8 tok/s - round trips cost ~nothing), **job-sort probe NULL**
  (not locality-bound), **#4 CPU-share hybrid NEGATIVE 3x in both forms** (27.5 tok/s: the
  pool drafts the host thread; 26.4 with --no-host-worker: the pool's page-cache reads
  saturate the host memory path - d2h alone 0.58 -> 2.9 ms/layer). Two crash bugs fixed
  (rodata const_cast, pool MAXT overflow); a latent tier-expert double-mark closed. Remaining:
  **#1 multi-token jobs** (+25-40% expected with NTM=256), #3 overlap bounded ~7% —
  `2026-10-23-prefill-anatomy`.

---

**Conventions**: `engine/strata` = `build/strata`, backups as `engine/strata.bak-<name>-<date>`;
local configs with credentials are gitignored; no commits/pushes by agents; comparisons always
arm-against-arm, same day/same binary whenever possible.

## 2026-10-26 - prefill campaign day 2: 75.8 -> 137.4 tok/s (+81%), the wall map after
- MT down (S2 runs): 83.4 -> 96.6 (+15.8%), serve 53.7 -> 45.1 ms/L.  REG 55, 8.3KB smem, zero spills.
- S0 split exposed the rank staircase (down 4.2/10.9/16.8/24.3 by card): prefill.cpp's finish loop
  ran each card's FULL finish before the next card's down launched.  TWOPHASE (launch all downs,
  rank-ordered completes, add order kept): 96.6 -> 126.2 (+31%).  Bit-identical block by construction.
- Geometry A/B on the MT gup (one deploy, env-switched): 256x4 KEEPER 137.4 (+9%); 256x8 131.8-133.0,
  256x4g (acts-in-global, full 64 warps/SM) 132.2, E=2 128.8-129.4 - all closed negative.
- GENTLE_SYNC retested twice on the new stack (95.8, 136.2): NULL both times - the spin is not the
  residual bottleneck.  PFPIPE (upload pipeline, double-buffered staging+faq, event completes):
  136.5-137.0 = NULL - the uploads are 155us/window (the 2.2ms h2d figure was the per-LAYER
  aggregate, a misread); nothing to hide.  Kept opt-in, documented null.
- Correctness at every step: fixed 2.7k answer, 10k topics, Marguerite recall 2/2; parity 0 failures.
- The wall map at 137.4 (51ms window): M10 serve ~32ms (gup 11.5 instruction-bound + down 4.1 with a
  cross-card stretch to 12.6 + completes ~2ms/card) + ~19ms of untouched 3060-side prefill.
  Next: measure/attack the 3060 share; the completion-channel serialization; the gup smem packing.

## 2026-10-26 evening - AD2H +11% (152.5 tok/s), the completes mapped, the no-overlap finding
- STRATA_PREFILL_TIMING (existed, unused by this campaign): the 10k's GPU timeline 63.7s = host grouping
  84.3%; the 3060's own compute just 15.7% - and it NEVER overlaps the M10 chain (the chunk loop is
  strictly sequential per layer).  The wall is the sum of the two engines.
- STRATA_MAXWELL_TIMING: finish sync 3.0-3.4 ms/card vs begin 0.17 - the completes carry the tail.
- STRATA_MW_AD2H: the completes' D2H pre-enqueued behind each down at launch (copy-engine overlap),
  rank-ordered adds kept: 137.4 -> 152.5-152.6 tok/s (+11%).  Recall 2/2, parity 0.
- STRATA_MW_REV flip test: the down staircase follows the COMPLETION ORDER (reversed completes ->
  reversed staircase) - order-driven, and with AD2H partly an attribution artifact (the S0 b_ record
  is enqueued by the rank-ordered host loop).  Budget at 152.5 (per 256-tok chunk-layer, 47 ms):
  gup 8 + down 4.4-8 + completes ~18 + 3060 5.8 + transfers 3.6 + grouping ~4.
- GENTLE_SYNC on the event syncs: NULL (148.5) - third and final GENTLE null; the spin hypothesis is
  dead on every path it could live.
- Campaign: 75.8 -> 152.5 (+101%).  The honest path to 250: (a) the chunk-wavefront overlap (caps at
  the M10 chain alone: ~200 tok/s), then (b) completes parallelization + (c) the gup packing for the
  last ~20%.

## 2026-10-26 night - DRED quarantined: five forensic cycles, the fault survives all benign isolations
- STRATA_MW_DRED (device-side rank-ordered reduce of the four ranks' partials into Dm, bit-exact add
  order, est. ~165): crashes with an illegal memory access on MULTI-WINDOW prompts only (the single-
  window 13-token request answers correctly).  Isolated through five forms - direct peer copy,
  cross-device stream waits, host event syncs + foreign-pinned H2D, portable pinned, pure-CPU staged
  local-pinned H2D - the crash persists; remaining deltas: mid-layer device-0 allocations, the
  early-return completes, local H2Ds.  compute-sanitizer is blocked (cuBLAS init fails under memcheck).
- DRED committed DEFAULT OFF (do not enable; needs a standalone repro harness).  The winning stack is
  unchanged and production runs the default path: MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 = 152.5.
- Also this session: meta-tier reuse ruled out (MaxwellMetaTier requires a 48-layer model, not ours).

## 2026-10-27 early - DRED: the crash cracked open (the completes' timing gates it), still quarantined
- The standalone harness (tools/mw_dred_repro.cu) passes the arm's exact op sequence fault-free - the
  fault is an engine interaction, not the ops.
- The engine mode map (README above): mode 4 (no arm, standard completes) passes; mode 5 (arm +
  standard completes) passes CORRECTLY; mode 1/3 (arm + early-return) crash; mode 6 (arm + waits only,
  no scatter) runs 154-155 tok/s but answers WRONG - a latency race.  The completes' ~6 ms of host
  time gates both the fault and the correctness; the arm's own syncs provably cover the staging/D2H
  ordering, so the racing consumer sits elsewhere (next: audit Dm's tail consumers for a non-m.cs
  stream; probe with cudaStreamSynchronize(m.cs) in the arm).
- Found on the way: PFPIPE's staging alternation never ran (d_up2_ is E1-gated) - the null stands but
  only for the upload form.
- No working DRED form beats 152.5 (mode 5 = 146.8; mode 6 = 155 but wrong).  DRED remains DEFAULT
  OFF; production restored and verified; parity 0.

## 2026-10-27 - the barrier named: the completes' host time is a load-bearing pipeline dependency
- mode 6 + m.cs-only sync: still wrong answers at 152.5 (the sync ate the speed too).  mode 7 (+ a
  NULL-stream readback barrier): CORRECT, zero checksum mismatches - the reduce's data is right
  whenever a device-wide barrier separates it from the consumer.
- THE FINDING: a Dm-tail consumer runs off m.cs (combine marks cs; the MMQ/peer paths use pp->s/s_cp/
  s_out) and the prefill's correctness has depended on the completes' ~4-6 ms/window latency ALL
  ALONG - including the shipped 152.5 stack (latency-hidden, not ordered).  Any future speedup (the
  wavefront) must first make that ordering explicit.
- DRED closed: every correct form re-imports a per-window barrier costing what the completes cost;
  it cannot beat the incumbent on this stack.  Quarantined (default off) with the complete mode map.
- Campaign final: 152.5 shipped (+101% over 75.8); the 240+ path is the wavefront + the cross-stream
  ordering fix, recorded as the named next work.  Production default-path restored and verified;
  parity 0.

## 2026-10-27 final - mode 7 full-gated (correct, 150.3-152.0) and the CEILING VERDICT
- mode 7 passes all gates (fixed answers, 10k topics, recall 2/2) at 150.3-152.0: DRED closed three
  ways (146.8 dead-adds / 155-wrong / 150-152 barrier-priced).  The incumbent 152.5 stands.
- THE VERDICT: the M10 rig's prefill ceiling within this architecture is 152-155 tok/s; the CPU
  rig's 259.9 is unreachable without a ground-up pipeline redesign.  The measured case (three
  legs): (1) the kernels are at their measured optimum (gup instruction-bound at 27x DRAM slack -
  six levers closed; down 4.4 ms); (2) correctness measurably REQUIRES ~4-6 ms of per-window
  pipeline latency (five experimental forms) - with it the completes cost 13-18 ms and the unit is
  47.4 ms = 152.5; (3) the two engines never overlap (measured) and overlapping them requires
  removing exactly that barrier - the consumer could not be isolated in-engine (all identified
  readers are m.cs-ordered; only a NULL-stream barrier works - a driver/vGPU-level visibility
  dependency).  Full argument + every number: bench/results/2026-10-26-prefill-campaign/README.md.

## 2026-10-24 - the decode campaign opens: baseline 61.7-66.5, the anatomy, three levers closed
- Baseline (default path): 61.7-66.5 tok/s flat over ctx 0-8k; window 47.0 ms @ T=3.3,
  2.97 tokens/window, accepted 198-204 of 210-249.
- Anatomy (three profilers, cross-checked): verify 44.3 = graph-real 18.7 + waitCPU 12.8 (the
  lockstep: the graph spins on the host's completion flag ~265 us/L) + GPU-reach 3 + host staging
  ~6 + slop; draft 3.3 serial.  M10 gup 198-219 us/L is the INSTRUCTION floor (~113M instr/card/L
  vs 640G/s issue; DRAM slack 4.7-7x) - same physics as prefill, not recoverable on sm_50.
  One serve per layer carries both split groups' jobs.
- Perfect-overlap floor ~= 27-30 ms ~= 100-110 tok/s: the target IS the perfect-pipelining
  boundary.
- CLOSED (measured): GPUFLAG=2 parity + a hang mode at spec_min_p=0.1; spec_min_p plateau
  (0.65 parity x20 trials, 0.35/0.2 negative - accepted is constant ~200, the MTP prefix is
  ~3 tokens); expert-cache 2300 fails to boot (the MTP head needs 138 MiB, 99 free).
- NEXT: speculative drafting - the draft's 3.3 ms hidden in the waitCPU shadow.
- Full numbers: bench/results/2026-10-24-decode-campaign/README.md

## 2026-10-24 (session 2) - the decode lever table: 62.5 -> 71.2 tok/s (+14%), the space closed
- Closed negative (measured): GPUFLAG=1 -3%; GPUFLAG=2 parity + hang at smp=0.1; spec_min_p plateau
  (0.4-0.65 parity at depth 6 too); expert-cache 2300 does not boot (MTP head 138 MiB vs 99 free);
  HOST_WAIT_US parity; --batch 2 aggregate 44 (each ~22) - the 0.1.x "aggregate >100" is gone on
  0.2.2; --lookup-chain 4 prose -40%; --mtp-q4 crashes at load (both cache sizes); speculative
  drafting closed by economics (a missed guess = a 44 ms redo to save 3.3 ms; needs a second
  verifier stage this rig lacks).
- WINNERS: --spec 6 (+5.3% - T-flat windows make deeper verified drafts free; greedy outputs
  IDENTICAL, every row target-verified; NOTE this supersedes the old "depth 4" call, which dated
  from the CPU-pool era); + STRATA_MW_MT=1 (+6.9% more - T=6 windows cross the MT n_tok>=4 gate);
  + STRATA_MW_MT_T=256x4 (+0.7%).  Depth 8 with MT: 70.0 - the optimum is 6 (acceptance-capped).
- Final: 71.2 tok/s on the standard list protocol; full gates green (arithmetic, 10k topics,
  needle 2/2, parity 0).  Prefill side effect: ~101 tok/s with the MT envs (from 75).
- The remaining 40% to 100: the waitCPU lockstep (12.8 ms) + draft (3.3) + reach (3.3) - every
  incremental lever is in the table above; the escape is the decode wavefront (a second verifier
  resource), unbuildable on this rig as configured.
- Full table: bench/results/2026-10-24-decode-campaign/README.md

## 2026-10-24 (session 3) - q4 forensics + the decode ceiling verdict: ~71 tok/s, 100 needs a second verifier
- CORRECTION: the winning stack's window is ~61 ms @ 4.35 tokens/window (not the ~43 ms session 2
  mis-stated); rate 70.3-71.9 across sessions on the standard protocol.
- The mtp-q4 crash root-caused: the Q4_0 image (153.07 MB) is BIGGER than this model's draft head
  (144.57 MB, ~4.25 bits/value - denser than Q4_0); the in-place rewrite cannot work.  Hardening
  left in tree: OnDevice guards + an honest error message.  --mtp-q4 proj (loads after the fix):
  66.2 - NEGATIVE (noisier drafts: accepted 230/278 vs 234/258).  spec_min_p re-closed on the MT
  stack (0.5 optimal).
- THE CEILING VERDICT: the decode ceiling within every architecture this tree offers is ~71 tok/s.
  The 61 ms window = graph-real ~20 (floors) + waitCPU 14.3 (every protocol removal measured) +
  draft 5.2 (data-serial; q4 negative) + staging ~7 + reach 2.7.  100 needs the window <= 43.5 =
  both walls gone - only the decode wavefront (speculative windows on a SECOND verifier) can, and
  this rig cannot host one (the M10s cannot run the dense model; a drafter on an M10 runs ~10x
  slower).  Aggregate (--batch 2) measured 44 - no side door.  Full gates + parity 0 on the final
  binary.  Every number: bench/results/2026-10-24-decode-campaign/README.md.

## 2026-10-24 (production switch, user go) - the full fast stack is LIVE
- Production now serves: config strata-spec6.json (the production config + --spec 6) with env
  STRATA_MW_MT=1 STRATA_MW_MT_T=256x4 STRATA_MW_NTM=256 STRATA_MW_TWOPHASE=1 STRATA_MW_AD2H=1.
- Launch: setsid env STRATA_MW_MT=1 STRATA_MW_MT_T=256x4 STRATA_MW_NTM=256 STRATA_MW_TWOPHASE=1
  STRATA_MW_AD2H=1 nohup .venv/bin/python serve/server.py --engine strata
  --config strata-spec6.json --port 8088 &
- Live-verified (2026-10-24): prefill 151.9/152.1 (2.7k), 150.1 (10k diverse); decode 72.0-72.6
  (300-tok lists); gates green live (2+2, 17x23=391, 9999^2, Paris, 10k topics, needle 2/2).
- NEVER add STRATA_MW_DRED (quarantined; see bench/results/2026-10-26-prefill-campaign).  The
  latent cross-stream ordering note applies to this stack as documented (safe as deployed).

## 2026-10-25 - prefill-250 campaign opens: research phases done, the incremental space re-closed with fresh measurements
- The user's research plan executed: nvidia-docs mined (tuning guide + GM107 whitepaper + PTX ISA;
  distillation in bench/results/2026-10-25-prefill-250), web research distilled (llama.cpp MMQ family,
  minfer SASS study, vLLM/TRT-LLM MoE routing, Sarathi chunked prefill; reality-checked against our
  kernel - ours already stage-once/reuse at run granularity).  Docs and web AGREE: no missed kernel
  silver bullet on sm_50; every >=200 path is pipeline.
- NTM=512 measured NULL (152.1/152.2 vs 152.4; 10k 150.4 vs 150.6): no fixed per-window cost to
  amortize - the completes' syncs absorb token-scaled work.  Window ladder closed on the modern stack.
- MT_E=8 REJECTED: -8% speed (140.5-140.9) AND wrong multi-window answers - root cause found and
  hardened: the runs table is a SHARED gup/down contract, the down's E_MAX was hardcoded 4 and
  dropped 8-wide runs.  Down now follows STRATA_MW_MT_E; with the fix E=8 is correct and measures
  128.9 (-15%).  E-ladder closed at E=4.  The gates caught it (2.7k/10k protocols; short gates alone
  missed it - multi-window is where wide runs live).
- NEXT (the measured walls, in order): (1) parallel completes + EXPLICIT cross-stream ordering (also
  retires the latent correctness debt), (2) the chunk wavefront (est. 245-265, the only >=240 path).
- Production restored on the fast stack and verified (2+2, recall OK).

## 2026-10-25 (session 2) - the dies are PARALLEL; PFPIPE never ran; the SBPIPE arm designed
- tools/mw_parallel_probe.cu (new): four M10 contexts, DRAM-bound kernel - all four wall 4.5 ms
  ~= one die 4.4 ms.  NO vGPU timeslicing; the completes' "staircase" is transfer/completion
  artifact.  The wavefront overlap math stands (chain 44 s of 65.6 s at 152.5).
- The ordering audit re-read: block H2D + combine are m.cs-ordered; no unstreamed launches in
  the prefill path; dequant_s2's NULL-stream kernel is parity-only.  Mode 7's zero mismatches
  already proved Dm data right - the race is a consumer read; in-engine hunt stays open.
- The prompt loop is CHUNK-MAJOR - cross-chunk overlap is illegal (data dependency); the legal
  overlap is INSIDE each layer's sub-batch loop (independent expert rounds, disjoint Dm slots).
- DISCOVERY: PFPIPE never ran (its staging alternation was dead code - d_up2_ allocated only
  under gpuflag>=2); its null is VOID.  Next arm: STRATA_MW_SBPIPE - rotate staging (fix the
  allocation hole), begin(s+1) before completes(s), M10 serve overlaps the completes' host
  work; the schedule adds the barrier distance structurally.  Measured one change at a time.
- Production restored (fast stack) and verified (answer 4).

## 2026-10-25 (session 3) - PFPIPE armed: NEGATIVE (-1%) and FAULTING; the barrier's third confirmation
- Fixed the PFPIPE allocation hole (d_up2_ under the knob) + two never-run-path bugs (ev_pf_down_
  never recorded; the completes' blocking D2H discarding AD2H).  The arm then RAN for the first
  time (banner x4): 150.6/150.7 (2.7k), 148.8 (10k) vs 152.0/152.2 same-binary baseline - NEGATIVE -
  and after passing the big-prompt gates the engine FAULTED ("pf down sync: illegal memory access",
  CUDA2) on a later small request.  PFPIPE = DO-NOT-ENABLE (hardening kept; armed form measured).
- WHY: the completes' wall is device-wait time (the M10 rounds serialize per-card stream), not
  host-CPU time - overlapping host waits buys nothing; compressing the completes' host latency is
  the DRED-modes-1/3 fault family.  THIRD independent barrier confirmation.
- Floor arithmetic (measured): perfect 3060 overlap still floors at serve 16-19 + completes 13-18
  + join 4-6 = 33-43 ms/unit = ~200-227 tok/s ceiling; 250 needs the serve+completes wall broken -
  the wall three experimental families price as correctness-load-bearing/faulting on this stack.
- Remaining arms before a verdict: interleave-grouping (~4 ms bound), fp16 partials (~2 ms).
- Production restored (fast stack), verified (answer 4).

## 2026-10-25 (session 4, campaign close) - vmad 1.07x; interleave empty by dependency; THE VERDICT
- tools/vmad_ffma_probe.cu: vmad 326.8 vs FFMA 350.7 G MAC/s on the M10 (1.07x) - the web's 3-4x
  assumed the scalar fallback; our vmad form is ~at the ceiling.  The kernel lever inventory is
  fully measured/bounded.  The interleave-grouping arm closed STRUCTURALLY: every filler depends
  on this layer's completes (completes -> H2D -> combine -> prelude) - the 3060's completes-idle
  has no legal work.
- THE CEILING VERDICT (4 legs, all measured): kernels closed (1); the completes' compression is
  correctness-load-bearing across FOUR forms - DRED 1/3 crash, mode 6 wrong, mode 7 priced,
  PFPIPE crash - consumer below the engine (2); the dies parallel, every legal in-architecture
  overlap measured negative/faulting/empty, the layer-major wavefront floors at 33-43 ms/unit =
  200-240 at zero join cost (3); 250 = 28.9 ms/unit sits BELOW the floor (4).  Not reachable on
  this rig/stack.  Shippable: 152.5 (re-verified 152.8/153.0 warm, 150.3 10k, gates + parity 0).
- Micro-arms documented bounded <=2% each (fp16 legs, smem packing, scatter-parallel).  Production
  serving the fast stack, verified.

## 2026-10-25 (ordering-debt session 1) - the premise audit turns the story: the DRED arm never engages (mw_tp=0); production is row-split
- The user set the goal: make the speed/safety dependency explicit.  Session 1 ran a 14-form
  discriminator matrix on the mode-6 family (deterministic: mode 6 wrong x3, mode 7 correct x3;
  sleeps/readbacks/host-reads/reads-only/writes-only/split-passes ALL wrong; only the real dead
  scatter correct) - then the execution banners revealed WHY: **mw_tp=0**.  Neither config sets
  --maxwell-split (default expert/ROW; tensor also needs GPUFLAG) -> the DRED arm NEVER RUNS on
  this rig -> "mode 6" = completes early-return before the scatter -> mw_block never written ->
  the always-running block H2D uploads stale garbage.  A mundane unwritten-buffer bug, not a race.
- Two verification bugs fixed on the way (recorded as rules): the engine log TRUNCATES per launch
  (grep full+immediately, never tail-by-stale-count); small prompts never reach the sub-batch path.
- CONSEQUENCES: the historical barrier forensics STAND for the TENSOR-split regime they ran under
  (their crashes prove the arm engaged); the correct-by-latency narrative belongs to that dormant
  configuration.  PRODUCTION (row split): scatter->H2D->combine is plain program+stream order - no
  latency borrowing visible.  The one real compressed-latency fault in production's regime is the
  PFPIPE arm (armed this campaign, crashed in row split).
- NEXT: the decoupling proof on production (compress the completes to their floor; gates at max
  speed = construction proof); PFPIPE disposition; tensor-regime documentation.
