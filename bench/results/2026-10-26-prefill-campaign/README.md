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
