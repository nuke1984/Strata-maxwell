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
