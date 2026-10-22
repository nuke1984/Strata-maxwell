# 2026-10-22 - the M10 gup multi-row campaign (STRATA_MW_IQ_MR) + the restructure win

Question: does multi-row-per-warp (upstream 0.2.2's Gyro trick) pay on OUR IQ gup kernel?

## SASS first (the design input)

The MR=1 hot loop (694-instruction unrolled body, TY=22 IQ2_S): 64 VMAD.S8.S8 (11%) vs
~253 unpack ops (36% - per-WEIGHT-code, does not amortize across rows), 91 address ops,
17 LDS, 14 LDG. MR's amortizable pool (act reads + act-side index math) ~10% - and the
register budget was nearly empty (REG 40-47), so MR was cheap to try.

## MR: implemented, parity-perfect, decisively SLOWER - closed

STRATA_MW_IQ_MR in tree (1 default, 2/4/5 opt-in; slices not divisible by PER_BLOCK*MR
fall back to 1; the persist kernel stays MR=1). maxwell_iq_parity: 0 failures at MR 1/2/4/5.
Interleaved sweep (3 rounds x 4 arms, idle GPUs, after warm-up, layer 0, us/launch):

| MR | rounds | verdict |
|---|---|---|
| 1 | 270 / 274 | reference |
| 2 | 318-319 | -17% |
| 4 | 394 | -45% (REG 62; blocks 120 -> 30 starve 5 SMs) |
| 5 | 350-351 | -29% (REG 64) |

Verdict: the block count IS the latency hiding; the weight path (the real cost) never
amortizes. The 4-rows/warp idea is CLOSED on this kernel/hardware family with numbers.

## The free win: the restructured body - PROMOTED (it is the default path)

The MR restructure (one hoisted act pointer per k: `sa`, the r-unrolled dot loop) changed
the MR=1 codegen for the better. Binary A/B, same hour, warm runs:

| binary | warm us/launch | notes |
|---|---|---|
| shipped (pre-MR) | 294.9 | 365.6 / 294.9 / 361.9 - bimodal cold/warm |
| restructured MR=1 | 270.3 / 274.8 | stable warm (348 cold) |

Kernel -8% (295 -> 270), REG 46, zero spills, bit-exact (parity 0 failures, identical
values). Engine (3x3, same config, new binary ba92a4cb): **55.07 tok/s vs 54.07 same-day
baseline = +1.85%** (story 50.3 / fib 60.3 / code 54.7, TTFT 21.86) - inside the predicted
1.5-2.5% band for gup's share of the chain, above the +/-0.5% noise band. Fixed answers
3/3-family, recall spot-check 2/2.

## Also closed this session: STRATA_MW_GENTLE_SYNC

Sleep-poll stream sync (30 us quantum): 48.90 vs 54.07 = **-9.6%** - the wake latency
(30 us x 4 cards x 48 layers ~ 5.8 ms/window) swamps any spin-poison relief. The old
"spin poison" verdict does not hold on this base; the driver spin stays. (First arm ran
against the wrong server - engagement now verified via /proc/<pid>/environ + banner.)

## Run-to-run note

The gup bench is BIMODAL today (cold ~350-365, warm ~270-295 us) - single runs are not
comparable; all numbers above are warm-state or interleaved.
