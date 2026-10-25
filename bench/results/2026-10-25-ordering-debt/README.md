# 2026-10-25 (evening) - the ordering-debt campaign opens: the premise audit TURNS THE STORY

Goal (user): make the dependency explicit so speed and safety stop being coupled - convert
"correct by latency" into "correct by construction".

## Session 1: the mode-6 family re-measured, and the discovery that changes the narrative

### The discriminator campaign (all same-day, same-binary, engine-log verdicts + recall gates)
Instruments: STRATA_MW_ORDER_PROBE (prefill.cpp, mode-6-gated, quarantined family) + op6 probes
(maxwell_experts.cpp finish_chain, mode-6 early-return path).  Every run: warmup + 2.7k x2 + recall.

| form | verdict |
|---|---|
| DRED mode 6 (control) x3 | WRONG (deterministic) |
| DRED mode 7 (positive control) x3 | CORRECT (deterministic) |
| mode 6 + sleep 6 ms (prefill side / at scatter pos) | WRONG |
| mode 6 + sleep 1 ms | WRONG |
| mode 6 + 4 B NULL-stream Dm readback (after the m.cs sync) | WRONG |
| mode 6 + 4 B NULL-stream unrelated readback | WRONG |
| mode 6 + 12 KB Dm readback raw | WRONG |
| mode 6 + 12 KB unrelated readback | WRONG |
| mode 6 + dead h_out_ reads (prefill side) | WRONG |
| mode 6 + reads-only at scatter position | WRONG |
| mode 6 + writes-only at scatter position (constants) | WRONG |
| mode 6 + split read/write passes | WRONG |
| mode 6 + THE REAL DEAD SCATTER at position | **CORRECT** |

### THE DISCOVERY: the DRED arm never engages on this rig - mw_tp = 0
Execution banners (after fixing two verification bugs: the engine log TRUNCATES per launch - grep
immediately, full log; and small prompts never reach the sub-batch path) printed:
  DRED-ARM conjuncts: mw_dred=1 m4=6 **mw_tp=0** count=4

Neither strata-spec6.json nor strata-iq2xs-embed-vram-split.json sets --maxwell-split (default
"expert" = ROW split; tensor additionally needs STRATA_MW_GPUFLAG).  Therefore in EVERY run this
session (and in the whole 250 campaign, and in today's production):
- the DRED staging/reduce NEVER runs (requires mw_tp);
- dred_owns is always false, so the block H2D ALWAYS runs;
- in row split the block memset is skipped (each row is written exactly once by its owner card -
correct for row split);
- "mode 6" = the completes early-return BEFORE the scatter -> mw_block NEVER WRITTEN -> the H2D
  uploads stale garbage -> WRONG.  "mode 7" and the dead-scatter probe = the scatter runs ->
  mw_block correct -> the H2D uploads correct data -> CORRECT.

EVERY probe result in the table above reduces to "the block must be written before it is uploaded"
- a mundane unwritten-buffer bug, not a race, not latency, not a NULL-stream anything.

### What this does - and does not - retract
- The HISTORICAL forensics (2026-10-26/27: DRED modes 1/3 crashes, mode 6 wrong-at-155, mode 7
  barrier-priced) ran under TENSOR split - the arm's crashes prove it engaged there.  Those
  measurements STAND for the tensor-split regime.
- The "load-bearing 4-6 ms / correct-by-latency" narrative belongs to TENSOR split, which NO
  current config runs (dormant code path, DRED-adjacent).
- The PRODUCTION regime (row split): completes = per-card event waits + per-card DISJOINT-row
  memcpy scatter (host) -> block H2D (m.cs) -> combine (m.cs) -> downstream (m.cs).  Plain
  program order + stream order.  No latency borrowing is visible in this path.
- The one REAL compressed-latency fault in the production regime is the PFPIPE arm (this campaign's
  armed run crashed with an illegal access in row split) - never-run code that was armed with
  three latent fixes; its own set-rotation code may still carry lifetime/parity bugs.

### Measurement-methodology rules learned (recorded so they stick)
1. The engine log TRUNCATES on every launch: grep the FULL log IMMEDIATELY after the run; never
   tail-by-stale-line-count.
2. Prompts below the sub-batch path (the CPU-share threshold) never reach the tier loop - always
   run the 2.7k before judging instrumentation.
3. Every probe needs an execution banner before its verdict is trusted.

## Next (the actual goal work)
1. THE DECOUPLING PROOF on the production regime: compress the completes' host time to its floor
   (all waits issued together, scatters joined, writes-before-H2D program order kept) - gates at
   max speed = proof that production is correct by construction (and likely a measured speed win).
2. PFPIPE disposition in row split: fix its lifetime bugs or hard-disable the knob - no faulting
   knob stays reachable.
3. The tensor-split regime's ordering debt: document as belonging to that (unused) configuration;
   fix only if the user runs it.
