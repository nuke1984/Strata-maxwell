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

## Session 2 (2026-10-25, before the goal was stopped): the DECOUPLING PROVEN + FASTCOMPLETE (+2.8%), PFPIPE hard-disabled, and the hardware incident

### STRATA_MW_FASTCOMPLETE - the decoupling proof and a winner (opt-in, default off)
Four host threads, one per card: [wait own AD2H event -> scatter own rows]; all join before the
block H2D.  Row-split safety: the cards' row sets are DISJOINT (each entry owned by exactly one
card, memcpy not accumulate), so the scatters are independent and the writes-before-upload program
order is kept BY CONSTRUCTION (the join).  This compresses the completes' host wall to ~max(per
card) - if the gates stay green at compressed completes time, production correctness NEVER
depended on the completes' latency: the speed/safety decoupling, proven.

Same-binary pair (this binary, same day, full gates on both):
| run | 2.7k | 10k | gates |
|---|---|---:|---|---|
| incumbent stack | 152.2/152.0 | 150.5 | green |
| +STRATA_MW_FASTCOMPLETE=1 | **156.2/156.7** | **154.5** | green (all: answers, topics, needle, parity) |

VERDICT: decoupling PROVEN (correct at compressed completes time), and a measured +2.8% winner.
Outputs bit-identical by construction (disjoint-row memcpy, same bytes, same H2D).

### PFPIPE: the parity bug found, the fault persists, the knob HARD-DISABLED
Found + fixed a real bug in the never-run code (the faq member swap fired only on par==1,
degenerating the alternation to A,B,B,A... so from window 2 each gup overwrote the image the
previous window's in-flight down read).  With the fix it STILL faults ("pf down sync: illegal
memory access", this time CUDA3) and corrupts small-prompt answers (2+2 -> "2", 17x23 -> "105").
Four bugs fixed across two campaigns and it still crashes, while measuring 150-151 vs FASTCOMPLETE's
156: pf_pipe_env() now returns false UNCONDITIONALLY - the arm cannot run even with the env set.
The sub-batch pipelining idea is closed: the completes' wall is device-wait time; compressing the
begins buys nothing the concurrent completes don't buy safely.

### THE HARDWARE INCIDENT (the journal forensics, 2026-10-25 12:15)
The last PFPIPE re-test's Xid 31 (illegal read, die PCI 04:00 = CUDA3) was followed within a
second by 8x Xid 158 (NV_UFLUSH_FB_FLUSH timeouts) - the die WEDGED mid-fault-recovery; RT
throttling; the kernel log stops 12:15:30; the machine died hard ~12:18 and the M10 did not come
back on reboot.  NO PCIe/AER bus errors anywhere in the journal; the earlier Oct-23 Xid-31 burst
(the DRED-era forensics, ~40 faults) all recovered cleanly - the 158-escalation was the rare
bad-luck timing (fault landing mid-FB-flush), not a new kind of stress.  A full host power-off
cleared it: all four dies back, enumerated clean, zero Xids since.

### Post-recovery verification (this build, 2026-10-25 ~13:00)
Persistence mode re-applied; parity 0 failures; production (incumbent fast stack - NO
FASTCOMPLETE) relaunched and verified: 2.7k 152.8/152.8, 10k 150.6, all gates green, zero Xids.
FASTCOMPLETE is documented, opt-in, awaiting the user's go for production.
