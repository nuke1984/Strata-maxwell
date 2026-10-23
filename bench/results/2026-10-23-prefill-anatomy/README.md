# 2026-10-23 - the M10 prefill anatomy: WHERE the 75 tok/s falls

Question: the M10 rig reads prompts at ~75 tok/s while the no-M10 rig does 260 - where exactly
does the M10 prefill spend its time?

Method: STRATA_MW_PREFILLPROF (new, env-gated, inert by default) times the four phases of the
Maxwell prefill block in prefill.cpp per layer per chunk: d2h (acts D2H + the pipeline-syncing
cudaStreamSynchronize), quant (the host act_quant_any loop), serve (the MW_SUB=64 sub-batch
round trips: begin = staging/H2D/gup launch, finish_to = down launch + stream sync + D2H +
scatter), h2d (the rows' upload). Two prompts (1,842 and 8,836 tokens), 1,920 layer-calls,
avg chunk T = 251, ~3.9 sub-batches/layer. Identical prefill rates with and without the
instrumentation (74.2-75.8 tok/s - the timers are free).

## The decomposition (per LAYER, us; cumulative over 1,920 layers, stable across checkpoints)

| phase | us/layer | share of the block |
|---|---:|---:|
| acts D2H + sync | 577 | 0.9% |
| host act-quant | 796 | 1.3% |
| **M10 serve (the sub-batch round trips)** | **59,669** | **94.0%** |
| rows H2D | 2,210 | 3.5% |

Inside the serve, per 64-token sub-batch: **begin 0.18 ms (1.2%) + finish 15.02 ms (98.8%)**.
The begin is trivial; the finish's stream sync absorbs the gup+down kernel walls.

**The serve is 86% of the whole prompt time** (1,920 x 59.7 ms = 114 s of the 10k prompt's
~117 s prefill). The M10 prefill serve IS the prefill.

## Why the M10s lose to a 4-core CPU (the B rig's 260 tok/s)

1. **The kernels are instruction-bound, and prefill multiplies their work ~25x**: each 64-token
   sub-batch asks each card for ~160 (expert, token) JOBS; the byte floor is ~67 MB/card at
   21 GB/s ~ 3.2 ms, measured 15 ms = ~4.7x off floor - the SAME slack as decode (S0's 5-7x).
   [inference, anchored to the S0 slack measurements + byte math; not directly evented]
2. **No expert-dedup across tokens**: every (expert, token) pair is its own job - an expert
   activated by several tokens of the sub-batch has its weights read once PER TOKEN. The CPU
   pool's kernels are multi-token (one weight load -> all tokens' dots; the engine's own AVX2
   banner), the M10 path is token-per-job.
3. **Serial with the 3060**: the block opens with a cudaStreamSynchronize - the tier's round
   trips sit squarely ON the chunk's critical path; B's CPU pool runs UNDER the GPU's work.
4. The 64-token cap is the tier's staging geometry (n_tok_max), not a knob.

## The fix candidates (ranked, NOT implemented)

| # | direction | ceiling | cost |
|---|---|---|---|
| 1 | multi-token jobs (dedup experts across the sub-batch's tokens; the kernel reads each expert's weights once, all its tokens' dots against them) | up to ~2-3x serve (the dedup ratio) on top of amortization | real kernel work (job form change) |
| 2 | wider sub-batches (MW_SUB 64 -> 256/512; bigger staging) | fewer round trips, more jobs/launch amortization | buffer resize + re-gate |
| 3 | overlap the tier block with the 3060 chunk compute (pipeline the sync) | hides the ~1.4 ms/layer of non-serve + start latency | architecture |
| 4 | CPU-share hybrid for prefill only (let the pool take part of the prefill misses beside the M10s) | bounded by the 4 vCPUs | config+plumbing |

Data: strata-pp4.out (the MWPROF prefill lines + engine prefill rates).

## The fix campaign (each fix tested alone, same prompts, same session)

| fix | arm | 10k prefill | verdict |
|---|---|---:|---|
| (baseline) | - | 75.8 tok/s (serve 59.7 ms/L, 15.0 ms/sub-batch) | reference |
| #2 wider sub-batches | STRATA_MW_NTM=256 | **76.3 tok/s** (serve 59.1 ms/L: begin 0.70 + finish 58.4 per 256-tok batch) | **NULL** - round trips cost ~nothing; per-token serve identical (237 vs 238 us/tok). The serve is pure kernel throughput. |
| (probe) job sort by expert | STRATA_MW_PSORT=1 | **76.1 tok/s** (finish 14.9) | **NULL** - not L2-locality-bound either; the kernel is instruction-issue-bound |
| #4 CPU-share hybrid | STRATA_MW_PSHARE=50 | **27.5 tok/s** (serve 82.6 ms/L, d2h 2.9 ms) | **NEGATIVE 3x** - the pool drafts the host thread (the tier's driver) |
| #4b + --no-host-worker | PSHARE=50 + no-host-worker | **26.4 tok/s** (serve 89.3, d2h 2.9) | **NEGATIVE** - the pool's page-cache reads saturate the host memory path the tier's staging needs (the D2H phase alone went 0.58 -> 2.9 ms/layer) |

Two crash bugs found and fixed on the way (recorded in the commits): a const_cast write to a
rodata flag, and the pool's MAXT-fixed job arrays overflowing on popular experts (c>MAXT now
stays on the tier). The tier-expert guard in the CPU-share candidate list also closes a latent
double-mark in our fork (tier-owned experts were CPU-share candidates).

**Remaining fix: #1 multi-token jobs (one weight-unpack serving M tokens' dots)** - the only
lever that cuts the instruction stream itself; expected +25-40% combined with NTM=256 (the
multiplicity at 64-token windows is only ~1.75x, at 256 ~5x). #3 overlap is bounded ~7% (the
non-serve phases are 3.7 ms of a ~61 ms layer).
