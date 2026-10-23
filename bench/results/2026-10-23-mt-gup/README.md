# 2026-10-23 - the multi-token gup kernel (STRATA_MW_MT) - first working checkpoint

The kernel: expert_iq_stage1_mt_batch (per format: IqMt<TY>::hoist/apply) - a RUN (one expert,
E<=E_MAX sorted-adjacent entries) per block; each (row,k) weight element hoisted ONCE (codes load +
codebook + sign work into 8 int payloads) and applied to every entry's act; the dp4a order and the
float tail verbatim the per-entry vecdots -> bit-exact per (expert, token) BY CONSTRUCTION, verified
by fixed answers + recall (the first cut had the faq indexing bug: the down indexes faq by the job's
SORTED TABLE POSITION, not its pre-sort entry id - one line, found by the recall gate).

## Measured (the 10k diverse prompt, NTM=256 windows, PREFILLPROF)

| arm | prefill tok/s | serve ms/layer | REG | SHARED | blocks/SM |
|---|---:|---:|---:|---:|---:|
| baseline (per-entry) | 75.8-76.3 | 59.7 | 47 | 11.2 KB | 4-5 |
| MT E_MAX=8 | 79.0 | 57.1 | 72 | 32.2 KB | 1 (!) |
| MT E_MAX=4 | **83.4** | **53.7** | 56 | 20.2 KB | 2 |

Correct at both E: fixed answers, recall OK. The amortization is real but LATENCY-capped: the
acts-in-smem staging costs occupancy (1-2 blocks/SM vs 4-5), and the codebook-lookup stalls lose
their hiding warps - E=8's bigger amortization is eaten by 4-warps-in-flight; E=4 trades back.

## The honest state of "beat the CPU's 260"

- The gup side: -10% serve so far; the DOWN kernel still runs per-entry (the same multiplicity
  waste on its share, ~1/3 of the serve) - the next chunk of work (the same run/helix treatment:
  the down's weights per expert read once, all entries' faq rows against them).
- Occupancy levers untried: THREADS=256/SPLIT=8 (8 warps/block at the same smem), acts-in-global
  (zero act smem, L2-shared across blocks) vs smem staging, E_MAX=2.
- The 260 target likely needs down-MT + occupancy work together; current honest band: 100-130.

All behind STRATA_MW_MT (default OFF - the standard path byte-identical; parity 0 failures).
