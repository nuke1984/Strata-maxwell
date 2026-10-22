# 2026-10-22 - the 3060's half of the assembly line (STRATA_VERIFY_PROFILE x-ray)

Question: what does the MAIN GPU actually do inside a decode window - and is any of it waste?

Method: STRATA_VERIFY_PROFILE + STRATA_DECODE_TIMING + STRATA_MW_LAYERPROF (NOT GPUFLAG - that
combo deadlocks), burn + fib 384 + story 384. The profiler stamps every graph section per
window (gpu_stamp; NOTE: profiling disarms some fusions - br/fuse_ab/sh_fork - so the DENSE
numbers are slightly pessimistic vs production; the WAITS are exact). Cross-checked against
the decode-timing line and MWPROF dispatch spans of the same windows.

## The window, three ways (fib 57-window avg / story 197-window)

- GPU stages total: **31.6-32.4 ms/window** (the graph's own timeline)
- decode timing: 43.8-51.0 ms/window wall = verify [GPU-reach + host (CPU+actq)] + draft
- MWPROF: dispatch span 426 us/L direct

## The graph's 31.6 ms, decomposed (fib, ms/window)

| section | ms | share | note |
|---|---:|---:|---|
| **waitCPU** (graph waits for the host/M10 chain) | **13.24** | **42%** | the M10 serial chain, unhidden portion |
| **hyper-connection projections** (hc0/hc1 norm/down/up, GDN+QSA) | **~5.1** | **16%** | ~6 tiny kernels/layer; ~19 us each = launch-latency dominated |
| q8+qkv/q-idx gemv | 1.81 | 6% | act quant + QKV, already fused |
| z + rec + conv + ab (mixer) | 2.11 | 7% | |
| out-proj | 1.55 | 5% | |
| head | 1.13 | 4% | |
| shared+quant | 1.16 | 4% | |
| router+plan+bell | 0.90 | 3% | includes the tier doorbell |
| QSA attention stack (k/v+rope, append, scores+topk, attention, gate) | ~2.2 | 7% | |
| copy+combine (the M10 rows' assembly) | 0.79 | 2% | |
| waitA+waitB+PCIe+VRAM-hits | ~2.5 | 8% | other flag/cache waits |
| (gap) | 0.02 | 0% | |

## Verdicts

1. **No waste found.** No useless computation, no dead sections, no runaway kernel - the
   "extra unuseful work" hypothesis from the anatomy brief is CLOSED: the 3060's own ~18.4 ms
   is real model work.
2. **The M10 chain remains the wall**: 13.2 ms of pure graph-side waiting (42% of the graph's
   time) - consistent with everything measured this week (protocol closed, kernel floor near).
3. **The one new attackable item: the hyper-connection stack, ~5.1 ms/window.** Six tiny
   launches per layer x 48 layers, each ~19 us = latency, not compute. Pair-fusion (norm+down,
   up+up) could recover an estimated 2-3 ms/window (~+5% end-to-end) - a real but modest CUDA
   campaign on verify_kernels.cu. The HC read already has a staged path (STRATA_HC_SPLIT).
4. actq/begin 7.4 ms host-side: bounded ceiling, bulk-staging only.

## Updated assembly line (per window, ~44 ms instrumented / ~41 clean)

42% real 3060 work - 30% M10 chain (unhidden) - 17% host dispatch - 6% draft - 5% other waits.

## Files
- xray.log: the full instrumented log (GPU stage reports + decode timing + MWPROF lines)
