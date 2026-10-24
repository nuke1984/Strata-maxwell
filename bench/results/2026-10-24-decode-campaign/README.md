# The decode campaign (2026-10-24): baseline anatomy and the closed levers

Target: 100 tok/s generation (from ~62).  Protocol: 300-token generations, temperature 0,
engine-log tok/s, quality gates (fixed answers, needle recall, list sanity) after every arm.

## Baseline (default production path, 2026-10-24)
- 61.7-66.5 tok/s across ctx 0-8k (flat - KV is not the wall); window 47.0 ms clean,
  T 3.3, 2.94-2.97 tokens/window, drafts accepted 198-204 of 210-249.

## The anatomy (STRATA_DECODE_TIMING + STRATA_MW_LAYERPROF + STRATA_VERIFY_PROFILE, ~3.6-6% overhead)
- window 47-48.6 ms = verify 44.3-45.3 (GPU-reach 2.8-3.5 + per-layer host 38.4-39.1
  [plan 0.25, actq 5.3, jobs 0.8, run 31.7] + stage 0.9) + commit 0.03 + DRAFT 3.3.
- Per card (MWPROF S0, 1440 layers): gup 198-219 us/L (jobs/L 4.1, 4.8 MB/L, DRAM ideal 57 us,
  slack 4.7-7x - INSTRUCTION-BOUND: ~113M instr/card/layer at the 640G/s issue floor ~= 176 us),
  down 60-180 us/L (visit-order stagger), sync-wait 224/71/35/22 us/L by finish order.
- Host dispatch span 475 us/L (plan 3.7 + actq 91 + jobs 9.3 + run 371 - mostly standing at the
  sequential card syncs waiting for real device work).
- GPU side (VERIFY_PROFILE): REAL graph work ~18.7 ms/window; **waitCPU 12.8 ms (9.73 GDN + 3.10
  QSA ~= 265 us/L uniform)** - the graph spinning on the mapped completion flag while the host
  stages and the M10 chain runs: THE LOCKSTEP.  hc0/hc1 hyper-connection reads ~5 ms, head 1.16,
  waitA/B small, one serve per layer carries both split groups' jobs (4.1 jobs/card ~= 33 visits
  = T x top-10).

## The perfect-overlap floor
48 x max(graph-real 0.39 ms, M10 chain 0.38-0.49 ms) + draft 3.3 + handshake slop ~= 27-30 ms
~= 100-110 tok/s.  The 100 target sits AT the perfect-pipelining boundary: every recoverable
layer of overlap matters, and the gup instruction floor is not recoverable on this silicon.

## Closed levers (each measured, gates run)
1. STRATA_MW_GPUFLAG=2 (E1 no-wait finish): PARITY 62.3/62.5 vs 61.7-63.2 baseline; gates pass
   (recall OK, 17x23=391).  PLUS a hang mode at spec_min_p=0.1 (a 4-token window stuck with flags
   A/B/M raised; issue-#29 self-stop caught it).  The 2026-10-22 verdict ("the wait is real chain
   latency, not host waste") re-confirmed at T=3.3 post-MT-kernels.
2. spec_min_p (per-request tune, 20 interleaved trials): 0.65 = PARITY (4.89 vs 4.90 s median);
   0.35 = -2%; 0.2 = -4%.  Accepted drafts stay 198-202 at every threshold - the MTP chain's
   confident prefix is ~3 tokens; tokens/window ~= 3 is a MODEL property.  Production 0.5 stands.
3. --expert-cache 2300 (from 1800): FAILS TO BOOT - the MTP draft head needs 138 MiB, 99 free
   ("strata serve: mtp: the draft head does not fit").  Cache capped by VRAM; max ~2100-2200 with
   zero KV margin at long contexts; expected +2% - closed as not worth the risk.

## The named next surgery
Hide the serial draft (3.3 ms, measured 7% of the window) inside the waitCPU shadow (12.8 ms of
graph idle): speculative drafting - draft K+1's chain during K's verify on the assumed full
acceptance, validate against the verdict, re-draft on miss.  The draft runs on the 3060 whose
graph idles 12.8 ms/window spinning on the tier flag, so the overlap target is genuinely idle
hardware, not contention.

## Session 2 (2026-10-24, later): the lever table - closed negative x8, the winners x3 (62.5 -> 71.2, +14%)

| Arm | Result (300-tok list protocol) | Verdict |
|---|---|---|
| STRATA_MW_GPUFLAG=1 (device flags, host sync kept) | 58.1-60.1 (-3%) | closed negative: the multi-wait nodes cost more than the de-poisoned downs save |
| STRATA_MW_GPUFLAG=2 (no-wait) | parity (62.3/62.5) + a hang at spec_min_p=0.1 | closed (see session 1) |
| spec_min_p 0.4-0.65 x depth-6 (per-request, n=2 each) | all parity (min 4.48-4.62 s) | 0.5 stands |
| --expert-cache 2300 | fails to boot (MTP head needs 138 MiB, 99 free) | closed: VRAM-capped |
| STRATA_HOST_WAIT_US=10 | 61.8-64.4 (parity) | closed |
| --batch 2 (ctx 64k; 256k does not fit the head) | each request ~22, aggregate ~44 (-30%) | closed negative: the 0.1.x "aggregate >100" does not hold on 0.2.2's serving stack |
| --lookup-chain 4 | lists neutral, PROSE 39.8 (-40%) | closed negative |
| --mtp-q4 all | crashes at load ("writing the draft head"), at cache 1800 AND 1700 | closed: broken on this stack |
| speculative draft overlap (analysis) | a missed guess costs a 44 ms verify redo to save 3.3 ms; only pays on a second idle verifier stage, which this rig lacks | closed by economics |
| **--spec 6** (from 4) | 65.4-66.2 (**+5.3%**), accepted 230 of 270-283 | **WINNER** (greedy outputs identical: every row still target-verified) |
| **+ STRATA_MW_MT=1** | 67.0-70.6 | **WINNER** (+6.9% more): T=6 decode windows cross the MT kernels' n_tok>=4 gate - the run-table gup beats the batch kernel at T=6 |
| **+ STRATA_MW_MT_T=256x4** | 70.3-71.3 | **WINNER** (+0.7%): the prefill geometry holds on decode |
| --spec 8 + MT + 256x4 | 69.4-70.5 | the optimum stays at 6 (acceptance-model-capped) |

The winning stack's gates (all pass): 2+2=4, 17x23=391, 9999^2=99980001, Paris, 10k topics correct,
needle recall 2/2, maxwell_iq_parity 0 failures.  Side effect: with the MT envs, prefill also rose
75 -> ~101 tok/s (the full 152.5 prefill stack needs its own AD2H/TWOPHASE envs on top).

**Final position: 71.2 tok/s (baseline 62.5, +14%)** via --spec 6 (config arg) + STRATA_MW_MT=1 +
STRATA_MW_MT_T=256x4 (env knobs, default off).  The window is now ~43 ms; the remaining 40% to 100
is the waitCPU lockstep (12.8 ms) + the serial draft (3.3) + reach (3.3) - the incremental lever
space is measured-closed (the table above); the next step would be the decode wavefront, which
needs a second verifier resource this rig does not have.

## Session 3 (2026-10-24, evening): the q4 forensics, the corrected arithmetic, the ceiling verdict

CORRECTION to session 2's closing note: the winning stack's window is **~61 ms @ 4.35 tokens/window**
(300 tok / 69-70 windows / 4.21-4.26 s), not ~43 ms - the T=6 window does more verify work than the
T=3.3 one; the rate gain came from tokens/window 2.97 -> 4.35 outrunning the window 47 -> 61 ms.

Fresh anatomy on the winning stack (instrumented ~7%): verify ~55 = graph-real ~20 + **waitCPU 14.26**
(10.89 GDN + 3.37 QSA, ~300 us/L uniform) + host staging ~7 + GPU-reach 2.65 + stage/slop; **draft 5.16**
(deeper chains draft 5 steps).  T avg 4.74 of 6, VRAM hits 2.23/L of ~47 visits.

### The mtp-q4 forensics (the load crash root-caused)
- The real error (after making the message honest): "writing the draft head: invalid argument
  (dev 0, dst 153070560 B, head 144566640 B)" - **the Q4_0 image is BIGGER than the head**: this
  model's draft head is ~4.25 bits/value (1360 B/row), denser than Q4_0 (1440 B/row).  The in-place
  rewrite cannot work; the feature assumes a Q8_0 head.  Upstream hardening left in the tree:
  OnDevice guards in make_q4_dense/make_q4_head (the bind can hold another device current) + the
  error now carries the sizes.  q4-head: physically inapplicable to this model.
- --mtp-q4 proj (Q8_0 -> Q4_0 draft projections; loads fine after the fix): **66.2 tok/s - NEGATIVE**
  vs 71.2 (at cache 1700; the mechanism is visible: noisier 4-bit drafts - accepted 230/278 vs
  234/258).  Closed.
- spec_min_p re-swept on the MT stack (0.35-0.5, interleaved): 0.5 optimal (4.287 s median; all
  lower thresholds worse).  Closed (twice now, once per stack).

## THE CEILING VERDICT (measured, 2026-10-24)

The M10 rig's decode ceiling within every architecture this tree offers is **~71 tok/s**
(single request, greedy, standard list protocol; 70.3-71.9 across sessions).  The case:

1. EVERY incremental lever is measured-closed (16 arms across sessions 1-3: the tables above).
   The winners: --spec 6 + STRATA_MW_MT=1 + STRATA_MW_MT_T=256x4 (62.5 -> 71.2).
2. The window at the optimum (~61 ms @ 4.35 tok/window) decomposes as: graph-real ~20 ms (each
   phase at or under its bandwidth/instruction floor - hc reads ~5 ms ARE the weight bytes), 
   waitCPU 14.3 ms (the per-layer lockstep: the graph spins on the host's tier flag while
   [staging + H2D + gup(instruction floor) + down] runs - every protocol removal measured:
   E1 mode 1 -3%, mode 2 parity + a hang, persist parity, dispatch threads negative in P3.4),
   draft 5.2 ms (data-serial: the chain needs the verify's own residual rows; q4 closed above;
   overlap needs a speculative window whose miss costs a full-window redo >= the 5.2 saved),
   host staging ~7 + reach 2.7 + slop.
3. 100 tok/s needs the window at <= 43.5 ms - i.e. removing ~ALL of waitCPU AND the draft: the two
   walls whose every in-architecture attack is closed with numbers.  The remaining mechanism is
   the decode wavefront (speculative windows + snapshot undo on a second verifier resource), which
   this rig cannot host: the M10s cannot run the dense model/KV, and a drafter moved to an M10
   would compute its chain ~10x slower (sm_50, no tensor cores).  tokens/window is model-capped
   (depth ladder 4/6/8 closed; threshold closed twice; acceptance ~4.35 at the optimum).
4. Aggregate throughput is not a side door: --batch 2 measured 44 aggregate (worse than one slot).

Like prefill's 155-vs-250, the 100 target sits exactly at the perfect-overlap boundary of this
architecture; the measured, shippable optimum is 71.  Full gates green on the final binary
(arithmetic, 10k topics, needle 2/2, maxwell_iq_parity 0 failures; the binary carries only the
inert q4 hardening, default paths unchanged).
