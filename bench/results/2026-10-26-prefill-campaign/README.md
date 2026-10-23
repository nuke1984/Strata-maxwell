# 2026-10-26 - the prefill campaign: 75.8 -> 137.4 tok/s (+81%), the wall map after

Protocol: 10k diverse prompt + 2.7k fixed + Marguerite recall, engine-log tok/s, NTM=256 windows,
all arms quality-gated (fixed answers, topic recall, 2/2 needle).  CPU rig reference: 259.9.

## The ladder (each step alone on the previous winner)

| step | tok/s | serve ms/L | note |
|---|---:|---:|---|
| baseline (per-entry kernels) | 75.8-76.3 | 59.7 | the 2026-10-23 starting point |
| + MT gup E_MAX=4 | 83.4 | 53.7 | d543f001 |
| + MT down (S2 runs, REG 55, 8.3KB) | 96.6 | 45.1 | the down's code rows read once per run |
| + STRATA_MW_TWOPHASE | 126.2-126.3 | ~34 | the finishes' rank-order loop serialized the DOWNS |
| + STRATA_MW_MT_T=256x4 | **137.4** | ~32 | 8 warps/block, 64 rows/block, arithmetic identical |
| 256x8 | 131.8-133.0 | | closed (SPLIT=8 reassociates floats, loses to 256x4) |
| 256x4g acts-in-global | 132.2 | | closed - full 64 warps/SM loses to L2 latency + recompute |
| MT_E=2 | 128.8-129.4 | | closed - amortization beats occupancy |
| STRATA_MW_PFPIPE | 136.5-137.0 | | NULL, closed - uploads are 155us/window (the 2.2ms figure was the LAYER aggregate); nothing to hide |
| GENTLE_SYNC (retest x2) | 95.8 / 136.2 | | NULL twice, closed for good |

## The wall map at 137.4 (why 250 is not one lever away)

- Layer window ~51ms; the M10 serve share ~32ms (S0: gup 11.2-11.8 uniform on all four cards,
  down 4.1 stretched to 12.6 on the later-served cards - a cross-card serialization that is NOT
  the sync spin (GENTLE_SYNC null) and NOT the uploads (PFPIPE null); suspect the VM's single
  completion channel), completes' D2H+scatter ~2ms/card.
- The OTHER ~19ms/window is the 3060-side prefill (attention, shared experts, the 29MB block
  upload, host act-quant) - untouched by this campaign; must be measured and attacked next.
- The gup kernel itself is instruction-bound (SASS ~36% unpack): 11.5ms at ~27x DRAM slack.
  Next kernel lever: the 5-block smem packing (sact 34B/entry-block instead of 40B -> 19.2KB).

## Knobs (all default off; default paths byte-identical, parity 0 failures throughout)

STRATA_MW_MT, STRATA_MW_MT_T (128x4 default | 256x4 KEEPER | 256x8 | 256x4g), STRATA_MW_MT_E
(4 | 2), STRATA_MW_TWOPHASE, STRATA_MW_PFPIPE, STRATA_MW_NTM, STRATA_MW_LAYERPROF.

## Session 2 (2026-10-26 evening): AD2H +11% (152.5), the completes mapped, the overlap finding

- STRATA_PREFILL_TIMING (the 3060-side phase profiler, pre-existing): 10k prompt = GPU timeline 63.7s of
  which host grouping 84.3%; the 3060's OWN compute (attention+gdn+qsa+gemms+everything) = 15.7% (~10s).
  The wall is the SUM: the 3060's attention and the M10 chain never overlap (strict per-chunk layer loop).
- STRATA_MAXWELL_TIMING: finish sync 3.0-3.4 ms/card, begin 0.17 ms/card -> the completes (not begins)
  carry the host tail; the down staircase's ~2 ms steps matched the per-card complete cost.
- STRATA_MW_AD2H (the completes' D2H pre-enqueued behind each down, rank-ordered adds kept):
  137.4 -> 152.5-152.6 tok/s (+11%), recall 2/2.  THE DAY'S WIN.
- STRATA_MW_REV (flip test, measurement only, reversed add order): the down staircase FLIPPED with the
  completion order (CUDA4 4.7 fastest / CUDA1 8.9 slowest, reversed from normal) -> the stretch is
  ORDER-driven, not device-driven; later cards' [m_,b_] spans absorb the earlier completes' host time.
  With AD2H the span's b_ record is enqueued by the rank-ordered loop - part of the "stretch" is an
  attribution artifact; the wall-clock budget closes as: gup 8 + down 4.4-8 + completes ~18 + 3060
  5.8 + transfers 3.6 + grouping ~4 = ~47 ms/unit (152 tok/s).
- GENTLE_SYNC on the AD2H event syncs (30us polls): 148.5 = NULL (third and final GENTLE null).
- The per-unit budget at 152.5: the M10 chain ~16-19 ms wall + the 3060 ~5.8 ms + host ~20 ms; the two
  engines NEVER overlap.  Named next levers: (a) the chunk-wavefront pipeline (chunk c+1's attention
  layers overlap chunk c's expert layers - the 3060 side is only 10s, so perfect overlap caps at the
  M10 chain ~44s = 200 tok/s for the current chain), (b) parallel completes (the event-waits overlap,
  the adds stay rank-ordered), (c) the gup smem packing (5 blocks/SM).

## Session 3 (2026-10-26 night): STRATA_MW_DRED - quarantined WIP (broken on this stack), full forensics

The idea (sound, unproven here): move the completes' rank-ordered adds to the 3060 - each rank's
AD2H partials H2D to the primary GPU (dred_part[4]) and mw_dred_reduce writes Dm's tail directly
(the exact fp add order of the host adds; one block/row kernel in elementwise.cu).  Would remove
the 4 serial host adds (~2.4 ms) + the block memset (~1.6) + the block H2D (~2.2) per layer-chunk,
costing one 16 MB partials H2D on m.cs and a ~0.05 ms kernel - est. 152.5 -> ~165.

THE CRASH: an illegal memory access on MULTI-WINDOW prompts (the 13-token single-window request
passes and answers correctly; the 2.7k faults at window 1's begin).  Five isolation cycles:
1. the direct cross-device peer copy (d_out_ M10 -> d_part 3060, D2D): crash.
2. cross-device stream waits (m.cs waiting M10 events) + plain H2D (DRED=2 bisect, no reduce/map): crash.
3. host cudaEventSynchronize per rank (the proven completes pattern) + H2D from the M10's pinned h_out_: crash.
4. h_out_/h_out2_ re-alloc'd PORTABLE (cudaHostAllocPortable): crash at the same point.
5. pure-CPU staging into prefill-owned pinned buffers (no GPU ever touches foreign pinned): STILL crash.
Remaining deltas vs the 152.5 stack at the crash point: mid-layer cudaMalloc/cudaHostAlloc on device 0,
the early-return completes, and the local-pinned H2Ds.  compute-sanitizer cannot run the engine
(cuBLAS fails to init under memcheck on this stack) - the repro needs a standalone harness.  DRED stays
in the tree, DEFAULT OFF, and must not be enabled until that harness localises the fault.

The winning stack is unchanged: MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 = 152.5 tok/s.

## Session 4 (2026-10-27 early): the DRED crack cracked open - the mode map

The standalone harness (tools/mw_dred_repro.cu - 4 M10 contexts, the arm's exact op sequence, dummy
kernels) PASSES 11 windows fault-free: the op sequence is clean; the fault is an engine interaction.
Engine bisect modes (each a run of the real 2.7k):

| mode | arm | completes | result |
|---|---|---|---|
| 1 | full (waits+stage+H2D+map+reduce->Dm) | early return | CRASH (illegal access, multi-window only) |
| 3 | reduce->scratch, map VALIDATED in-range | early return | CRASH |
| 4 | none | standard (152.5 path) | PASS, correct (146.8 tok/s class) |
| 5 | full | standard (+dead scatter) | PASS, CORRECT answers - but 146.8 tok/s (dead adds cost) |
| 6 | full | ad2h-waits only, no scatter | NO CRASH, 154.4-155.4 - but WRONG answers (a latency race: correct only when the completes' ~6 ms of scatter time is present) |
| harness | full, dummy kernels | - | PASS |

Conclusions (measured): (a) the arm's every CUDA op is individually clean; (b) the COMPLETES' waits/
scatter timing gates BOTH the crash (mode 1/3) and the correctness (mode 6) - remove ~6 ms of host
time from the completes and the chain both faults (early return) and mis-orders (fast form); with it,
everything passes. The arm's own event syncs provably cover the staging/upload/D2H ordering, so the
racing pair is elsewhere - most likely a consumer of Dm's tail that is NOT m.cs-ordered and was
latency-hidden by the completes' host time (audit + a cudaStreamSynchronize(m.cs) probe in the arm
are the named next steps). (c) PFPIPE's staging alternation never ran in ANY test: d_up2_/h_up2_ are
allocated only under gpuflag>=2, so pf_on was always false (the PFPIPE null measurement stands but
measured only the streams_[1] upload form).

Economics: no working DRED form beats the incumbent (mode 5 = 146.8 < 152.5; mode 6 = 155 but wrong).
DRED stays quarantined DEFAULT OFF.  The winning stack is unchanged: 152.5 tok/s.

## Session 5 (2026-10-27): the barrier named - the completes' host time is load-bearing; DRED closed

The decisive probes (same binary, env-switched):
- mode 6 + cudaStreamSynchronize(m.cs) after the reduce: STILL WRONG (152.5 tok/s) - an m.cs-only
  barrier neither fixes correctness nor keeps the speed (the sync ate mode 6's +2).
- mode 7 (= mode 6 + a blocking NULL-stream readback memcpy): CORRECT - the NULL-stream barrier
  serializes the racing consumer; zero checksum mismatches (the reduce's Dm data is RIGHT whenever
  a full barrier stands between it and the consumer).

THE FINDING: the completes' ~4-6 ms/window of host time (or any equivalent device-wide barrier) is a
DE FACTO pipeline dependency - a Dm-tail consumer runs off m.cs (the combine marks cs, but the MMQ/peer
machinery uses pp->s / s_cp / s_out), and the pipeline's correctness has depended on that latency all
along.  The standard 152.5 stack carries the SAME latent exposure (its block H2D is m.cs-ordered too);
it is latency-hidden, not properly ordered.  Every correct DRED form re-imports a per-window barrier
costing what the completes cost: the economics are closed, DRED can never beat the incumbent on this
stack as architected.

## The campaign's final measured position

- Shipped: 152.5 tok/s (baseline 75.8, +101%): MT gup (E=4) + MT down + TWOPHASE + AD2H + MT_T=256x4.
- Ceiling of THIS orchestration: 155 (mode 6, barrier-less) - not shippable (wrong answers).
- The per-unit budget (47.4 ms per 256-tok chunk-layer): gup 8 + down 4.4-8 + completes ~18 (the
  load-bearing barrier) + 3060 compute 5.8 + transfers 3.6 + grouping ~4.
- The only >= 240 path: the chunk-wavefront overlap (est. 245-265 with the micro-levers) - which (a)
  requires eliminating exactly the latency this pipeline secretly depends on (the cross-stream
  ordering bug must be found and fixed first, else any speedup corrupts the KV), and (b) is major
  surgery (double-buffered chunk state) beyond this campaign's completed scope.
