# 2026-10-25 - the prefill-250 campaign: research plan (user), docs distillation, re-check map

Target: 152.5 -> 250 tok/s prefill on the M10 rig (RTX 3060 + 4x Tesla M10 GM107, Qwen3.8-Flash-Next
IQ2_XS, 48 layers, 512 experts/layer top-10).  Standing discipline: one arm at a time, gates on every
winner (fixed answers, 10k topics, needle 2/2, parity 0), numbers measured only, winners opt-in,
NEVER STRATA_MW_DRED, push to nuke only.

## Phase 1 - the nvidia-docs distillation (/home/toor/nvidia-docs)

Sources read: Maxwell_Tuning_Guide (v13.4), Maxwell_GM107_Whitepaper (extracted text), CUDA Best
Practices (section map).  The M10 die IS GM107-class: 5 SMMs (4 partitions x 32 cores, 4 warp
schedulers, no shared functional units), 64 KB DEDICATED shared/SMM (48 KB/block cap, <=32 KB
recommended for 2+ blocks/SM), 2 MB L2, ~86 GB/s DRAM (750 Ti figures; M10 similar class), FP32
~1.3 TFLOPS/die.  Extracted levers, cross-referenced against our kernels (src/kernels/cuda/expert_maxwell.cu):

| doc point | our code | verdict |
|---|---|---|
| Global loads cached L2-only unless __ldg | weights stream once/run (no reuse), codebook already smem-staged, acts smem/L2 (GACT arm) | nothing missed |
| 64 KB dedicated smem/SMM; <=32 KB/block for residency | 256x4 gup stages sact (E_MAX x ACTS_B x 40B) + grid + sx ~ 16-20 KB -> ~4 blocks/SM (32 warps); the SESSION-2-NAMED 34B packing (sact 5 q8_1-blocks instead of padded 8?) was NEVER MEASURED | OPEN ARM (code) |
| dual-issue math+memory; reduced dependent latencies | hoist (LDG/LDS+int) interleaved with apply (vmad chains) | structure already exploits |
| signed int preferred; power-of-2 structs | indices are int, strides power-of-2 | ok |
| native smem 32-bit atomics | no smem atomics in hot kernels | n/a |
| dp4a is sm_61+; sm_50 needs emulation | dp4a_s32 = 4x vmad.s32.s32.s32 PTX (the llama.cpp m10-final patch trick) - already the best primitive in the docs' arsenal | ok |
| 2 MB L2 (big for its class) | GACT acts-in-global arm measured -4 (132.2): L2 latency + recompute loses to smem staging | closed by measurement |

Docs conclusion: NO missed kernel-level silver bullet - the official guides' tricks are in the tree,
and every geometry/amortization lever was measured.  The kernel named-lever (smem packing) stays open.
The instruction-bound gup (8 ms, 27x DRAM slack, SASS ~36% unpack) is the price of dequant on sm_50.

## Phase 3 - the re-check map (closed arms vs the new look)

The unit budget at 152.5 (47.4 ms per 256-tok chunk-layer): gup 8 + down 4.4-8 + completes ~18
(sync 3.0-3.4 ms/card + D2H 0.55 + rank scatter; the 4-6 ms load-bearing barrier lives here) +
3060 compute 5.8 + transfers 3.6 + grouping ~4.  For 250: unit <= 28.9 ms - must remove ~18.5 ms.

| arm | status | the re-check verdict |
|---|---|---|
| NTM window ladder (64/128/256/512) | "closed" 2026-10-23: 256 vs 64 NULL (76.3 vs 75.8) | STALE: that null predates MT kernels, TWOPHASE, AD2H.  The fixed per-window costs (4 syncs, barrier, grouping, block zero/H2D) are ~4x larger a share now; 512 is an allowed value (maxwell_experts.cpp:456) NEVER measured on the modern stack.  RE-OPEN (env-only, first arm). |
| gup smem packing (sact 34B/entry-block -> 19.2 KB, 5 blocks/SM) | NAMED in session 2 "next kernel lever" | NEVER MEASURED.  OPEN (code arm). |
| parallel completes (overlap event-waits, rank-ordered adds kept) | NAMED session 2 as next lever (b) | NEVER MEASURED as an arm (TWOPHASE parallelized the down LAUNCHES; the completes still sync rank-serial).  OPEN (code arm). |
| host grouping ~4 ms/unit (84.3% of the 3060-side GPU timeline) | measured, untouched | OPEN: overlap with the M10 chain or move act-quant (host loop prefill.cpp:4167) to GPU. |
| completes -> device reduce (DRED family) | closed 3 ways (146.8 dead / 155 wrong / 150-152 barrier-priced) | STAYS CLOSED - never re-enable STRATA_MW_DRED.  The barrier finding binds ANY completes rewrite: correctness needs ~4-6 ms equivalent device-wide ordering until the racing consumer is found and explicitly ordered. |
| the wavefront (chunk c+1 3060-side overlaps chunk c M10 chain) | est. 245-265, never built | THE >=240 PATH (lateral phase): needs (a) the explicit cross-stream ordering fix first, (b) double-buffered chunk state.  Prerequisite work: identify the Dm-tail consumer that only the NULL-stream barrier orders. |
| CPU/RAM assist (#4 CPU-share hybrid) | NEGATIVE 3x (27.5/26.4 tok/s) | STAYS CLOSED - and the user's plan explicitly says GPU-main, no RAM prefill. |

Working arithmetic for 250: parallel completes + NTM=512 + smem packing could plausibly take the unit
47.4 -> ~36-38 ms (~200-210 tok/s) WITHOUT the wavefront; the wavefront then needs the chain at
<= ~36 s per 10k (chain-limited ~278 at perfect overlap) - both estimates, to be measured arm-by-arm.

## Phase 2 - the web research distillation (agent report, reality-checked)

Sources: llama.cpp ggml-cuda (MMQ/mmq-load-tiles/vecdotq/common.cuh, the pascal-older configs),
minfer's SASS-level MMQ analysis, vLLM fused-MoE + moe_align_block_size, TensorRT-LLM expert
parallelism, Sarathi/Sarathi-Serve chunked prefill, the CUDA 12.8 arithmetic-throughput table.

Reality-check against our kernel (the report assumed per-token LUT dequant; ours already hoists
once per (row,k) and applies across the run's E tokens, llama.cpp-style):

| report lever | our verdict |
|---|---|
| stage-once/reuse across a token tile (MMQ's core) | ours IS that, at run granularity; widening the reuse (E=8) MEASURED DEAD this session (below) - the reuse factor is bounded by tokens/expert (~5 at NTM=256) and the smem/occupancy price of wider forms exceeds the hoist saving |
| FFMA + byte-to-float instead of integer dots | the report assumed the scalar emulation fallback; our dp4a_s32 is 4x vmad PTX (single-issue video ops, the m10-final.patch trick) - near-parity with the FFMA trick's 2x128-rate ops; low expected value, parked |
| sorted-expert dense tiling (vLLM/TRT-LLM) | equivalent to our stable-sort + run table; already in |
| occupancy 2 blocks/SMM + bank padding (minfer) | measured BOTH directions: GACT full-64-warps -4 (2026-10-26), E=8 1-block/SM -15 (today) - the 256x4/E=4 point IS the optimum |
| chunked-prefill wavefront (Sarathi) | CONFIRMS our >=240 path: chunk c+1's dense (3060) work overlaps chunk c's expert (M10) work; requires the explicit ordering fix first (our barrier finding) |
| CUDA graphs | our launch overhead is ~0.2 ms/card/layer begin + small; low priority |
| dead ends (cp.async, tensor cores, LOP3, half2 math, ITS) | confirmed sm_50-absent; nothing missed |

Research conclusion (docs + web agree): the incremental/kernel space is closed WITH MEASUREMENTS
on the modern stack; every remaining >=200 path is PIPELINE: (1) parallel completes with the
cross-stream ordering made EXPLICIT (retires the latent debt AND compresses the ~18 ms), then
(2) the chunk wavefront (est. 245-265).

## Session 1 arms (2026-10-25, measured same-day pairs, binary f044b252 + E-arms build)

Protocol: 2.7k x3 + 10k x1 + full gates; engine-log tok/s; production down during arms, restored+verified after.

| arm | 2.7k | 10k | gates | verdict |
|---|---:|---:|---|---|
| baseline (incumbent stack, same day) | 152.4/152.4 | 150.6 | green | the reference |
| NTM=512 (all else incumbent) | 152.1/152.2 | 150.4 | green | **NULL** - window amortization is dead: the completes' syncs absorb token-scaled work; there is no fixed per-window cost to dilute.  The 64/128/256/512 ladder is now closed ON THE MODERN STACK (the 2026-10-23 256-vs-64 null was pre-MT/TWOPHASE/AD2H) |
| MT_E=8 (runs widened, smem 256x4) | 140.5/140.9 | 140.8 | **WRONG** (2.7k/10k answer classes corrupted; short-prompt gates passed) | **REJECTED twice over**: speed -8%, and the corruption exposed a SHARED-CONTRACT bug: the runs table feeds BOTH the gup and the down; the down's E_MAX was hardcoded 4, so 8-wide runs were silently dropped.  HARDENED: the down now instantiates the env's E_MAX (2/4/8); with the fix E=8 answers correctly and measures 128.9 - the arm closes at **-15%**, and the E-ladder (2/4/8) is closed at its measured optimum E=4 |

The E=8 event also re-validated the gate suite: the 2.7k/10k protocols caught a corruption that
the short fixed-answer + needle gates alone missed (multi-window prompts are where wide runs live).

## Session 2 (2026-10-25 evening): the dies are PARALLEL - the wall map redrawn

### The four-die parallelism probe (tools/mw_parallel_probe.cu, new)
4 M10 contexts, DRAM-bound kernel ~4.4 ms/die, launched back-to-back then synced:
- one die alone: 4.43 ms | all four wall: 4.53-4.69 ms | sync waits: 4.4/0.1/0.0/0.1
- reverse-order syncs: the wait MOVES to the last-synced die (0.0/0.0/0.05/4.47) - pure
  wait-order artifact, not execution order.
VERDICT: the vGPU does NOT timeslice the dies - they execute CONCURRENTLY.  The 4x3.4 ms
"staircase" in the completes is transfer/completion artifacts, not device serialization.
Consequences: (a) the M10 chain's device work is already the parallel minimum - the wavefront
overlap math stands (chain ~44 s of the 65.6 s per 10k at 152.5); (b) the completes' wall is
host-wait + shared-PCIe transfer time, not four serial device executions.

### The ordering audit (the wavefront prerequisite) - what the code re-read says
- The block H2D and the combine are BOTH on m.cs (ordered); the completes' host scatter is
  host-sequential.  No unstreamed launches in prefill.cpp; dequant_s2's NULL-stream kernel is
  parity-test-only (dead lead, noted as library hygiene).
- Mode 7's zero-mismatch checksums already proved the Dm DATA is always right - the corruption
  is a consumer read.  "Only a NULL-stream barrier works" still points at a legacy-NULL-stream
  consumer, but it is not in the prefill file; the 5-session in-engine hunt stays unconcluded.
- KEY STRUCTURAL FACT for the wavefront: the prompt loop is CHUNK-MAJOR (chunks of C tokens,
  all 48 layers each - prefill.cpp:2753) - chunks are strictly sequential (data dependency).
  The legal overlap is INSIDE each layer's sub-batch loop (the NTM windows, prefill.cpp:4186):
  sub-batch experts are independent (disjoint token rows of the same layer input), and the
  per-sub-batch Dm tail writes are disjoint by construction.

### THE DISCOVERY: PFPIPE (the wavefront-lite arm) NEVER RAN
Session 4's own conclusion: "PFPIPE's staging alternation never ran in ANY test: d_up2_/h_up2_
are allocated only under gpuflag>=2, so pf_on was always false (the PFPIPE null stands but
measured only the streams_[1] upload form)".  The one arm that pipelines the sub-batch loop
(begin s+1 under completes s, rotated staging) measured NOTHING - its null is VOID.

### The next arm, designed: STRATA_MW_SBPIPE (the sub-batch pipeline)
Per layer, for sub-batches s: [downs(s) TWOPHASE] -> [begin(s+1) on the ROTATED staging set]
-> [completes(s): waits+scatter+H2D+combine-slot].  The M10 serve(s+1) overlaps completes(s)
host work (~18 ms of waits/scatter/transfer per round at 152.5).  Pipelining also ADDS distance
between Dm writes and downstream reads (the barrier's requirement) instead of removing it -
the latent ordering debt is structurally satisfied by the schedule, not borrowed from latency.
Prerequisites: fix the PFPIPE allocation hole (allocate set 2 when the knob is on), keep the
Dm tail slots per-sub-batch disjoint (they are), gates on every step.  Est. per-layer saving at
8 sub-batches: up to ~7x the overlappable host share - to be measured, not assumed.

## Session 3 (2026-10-25 night): PFPIPE armed for the first time - NEGATIVE and FAULTING (the barrier's third confirmation)

The arm: the PFPIPE allocation hole fixed (d_up2_/h_up2_ now allocated under the knob - it was
gpuflag>=2-only, which is why pf_on was always false), plus two never-run-path bugs hardened:
ev_pf_down_ is now recorded under PFPIPE (it was DRED-only - the completes would have synced a
never-recorded event and scattered stale partials), and the PFPIPE completes now use the AD2H
pre-arm (the original form did a blocking D2H, discarding AD2H's overlap).

| run (same binary, same day) | 2.7k | 10k | gates |
|---|---:|---:|---|
| baseline (incumbent stack) | 152.0/152.2 | - | green |
| +STRATA_MW_PFPIPE=1 (ARMED, banner x4) | 150.6/150.7 | 148.8 | green on the big prompts - then the engine FAULTED on a later small request: "MaxwellExperts (CUDA2): pf down sync: an illegal memory access" |

VERDICT: the sub-batch pipeline (the only legal overlap inside the chunk-major architecture) is
measured NEGATIVE (-1%) and UNSTABLE.  The overlap thesis failed because the completes' wall is
DEVICE-WAIT time, not host-CPU time: the M10 round s+1 sits behind round s on the same per-card
stream, so overlapping the host's waits buys nothing - and compressing the completes' host latency
trips the SAME illegal-access family as DRED modes 1/3.  This is the THIRD independent confirmation
of the barrier (DRED modes 1/3: crash; mode 6: wrong answers; PFPIPE armed: crash).

PFPIPE joins DRED as DO-NOT-ENABLE (the hardening stays - a broken knob that never ran was worse -
but its armed form is measured negative and faulting).  The winning stack is unchanged.

### The arithmetic after the third confirmation (measured floors only)

The unit floor with PERFECT 3060-side overlap (the layer-major wavefront, whose M10 rounds stay
exactly today's serial form): M10 serve 16-19 ms + completes ~13-18 ms (barrier-priced - three
confirmations) + combine/H2D ~4-6 ms = ~33-43 ms per 256-tok unit = a ceiling of ~200-227 tok/s
EVEN IF the 3060+host work hid perfectly.  250 (unit <= 28.9 ms) requires compressing the
serve+completes wall itself - precisely what DRED (5 sessions), mode 6/7 economics, and now PFPIPE
each measured to be correctness-load-bearing or faulting on this vGPU stack.
Remaining honest arms before any verdict: (a) interleave the next grouping into the completes'
idle waits (bounded by the ~4 ms grouping), (b) fp16 partials (halve the D2H/H2D bytes, ~2 ms).
Both are sub-10% arms; neither reaches 250 alone or stacked.
