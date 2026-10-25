# WORKLOG — Strata on the M10 rig: the point-in-time truth of production

## Read this first — what this file is

This file reflects **only what is measured and currently working** on the production rig. It is the
single place to check "what is the rig running, and what do we know for sure". Three rules:

1. **Everything here is reproduced and verified** — every number was measured, most of them more
   than once. Nothing here is a projection or a hope.
2. **Unknowns are kept OPEN, not declared dead.** Where we could not tell a bad-luck combination
   of factors from something that never works, the item sits in "Open ways" with what IS known.
   Nothing in this file should be read as "impossible" unless a measurement says so.
3. The full experiment history — every arm, every number, every forensics session — lives in
   `bench/results/<date>-<name>/`. The prefill and decode campaigns were **closed 2026-10-25**;
   the historical WORKLOG text is archived at `bench/results/2026-10-25-worklog-full-history.md`.

Rules for future entries: measured facts only, one change at a time, same-binary arm pairs,
full quality gates on anything that ships.

---

## Production now (2026-10-25, tag `maxwell-0.2.6-port-production-OK`; the 0.2.2 line is preserved as
## branch `maxwell-0.2.2` and tag `maxwell-0.2.2-opti-production-OK`)

**Launch recipe** (verified live):
```sh
cd /home/toor/Strata
setsid env STRATA_MW_MT=1 STRATA_MW_MT_T=256x4 STRATA_MW_NTM=256 \
  STRATA_MW_TWOPHASE=1 STRATA_MW_AD2H=1 STRATA_MW_FASTCOMPLETE=1 \
  nohup .venv/bin/python serve/server.py --engine strata \
  --config strata-spec6.json --port 8088 &
```
- Config: `strata-spec6.json` (the production config + `--spec 6`). Restart needs ~230-340 s load.
- Health: `curl -H "Authorization: Bearer <key>" http://127.0.0.1:8088/health` → `"loaded": true`.
- After any reboot: `nvidia-smi -pm 1`. Never measure under a CUPTI profiler (~17% skew).

**Verified live performance** (engine-log tok/s, standard protocol) — on the 0.2.6 port:
| path | 0.2.6 (production) | 0.2.2 line (reference) |
|---|---|---|
| prefill, 2.7k prompt | **153.6 / 154.6** | 157.5 / 157.9 |
| prefill, 10k diverse | **154.8** | 155.3 |
| generation (decode) | **69.2 / 70.2** | 72.0 – 72.6 |

The ~2-3% deltas are measured and OPEN (the port entry below names the candidates).  The engine
reports `engine=0.1.51` on this build.

**Quality gates green on this exact stack**: fixed answers (2+2=4, 17×23=391, 9999²=99980001),
10k diverse-prompt topics correct, needle recall 2/2, `maxwell_iq_parity` **0 failures**.

---

## What works for sure (the shipped ladder, each step measured alone)

### Prefill: 75.8 → 157.9 tok/s (+108%)
| change | effect | measured |
|---|---|---|
| MT gup (multi-token weight-hoist runs, E=4) | one dequant per (row,k) serves up to 4 tokens | 75.8 → 83.4 |
| MT down (S2 runs) | the down reads code rows once per run | → 96.6 |
| `STRATA_MW_TWOPHASE=1` | all downs launch first, completes stay ordered | → 126.2 |
| `STRATA_MW_MT_T=256x4` | gup geometry: 256 threads, 64 rows/block, arithmetic identical | → 137.4 |
| `STRATA_MW_AD2H=1` | each card's D2H pre-rides behind its down | → 152.5 |
| `STRATA_MW_FASTCOMPLETE=1` | the four completes run concurrently (disjoint rows, joined before upload — correctness by construction) | → 156.2–156.7 (live 157.5/157.9) |
| `STRATA_MW_NTM=256` | prefill window size (64/128/256/512 measured equal; 256 shipped) | layout choice |

### Generation: 62.5 → ~72 tok/s (+15%)
| change | effect | measured |
|---|---|---|
| `--spec 6` (config) | deeper speculative window; greedy outputs still target-verified | 62.5 → 65.8 |
| `STRATA_MW_MT=1` | T=6 decode windows cross the MT n_tok≥4 gate | → 70.6 |
| `STRATA_MW_MT_T=256x4` | same geometry win as prefill | → 71.2 (live 72.0–72.6) |

### Why the kernels are what they are (measured, so we don't re-derive)
sm_50 (GM107) has no dp4a/fp16-math/tensor cores: our dequant-dot runs on 4×`vmad` PTX, measured
within **1.07×** of the best alternative primitive (326.8 vs 350.7 G MAC/s). The occupancy ×
amortization space was swept in both directions (256x8, GACT, E=2, E=8, NTM 64↔512): the shipped
geometry is the measured optimum. The gup is instruction-bound at 27× DRAM headroom — that is the
silicon's price for 2-bit dequant, not a tuning gap.

---

## Hard rules — quarantines and cautions

- **NEVER enable `STRATA_MW_DRED`** (any mode). Quarantined since its campaign; the dangerous arm
  requires a `--maxwell-split tensor` configuration this rig does not run, but the knob stays off.
- **`STRATA_MW_PFPIPE` is hard-disabled in code** (`pf_pipe_env()` returns false): its armed form
  faults (illegal memory access) and corrupts answers, through four fixed bugs. It cannot arm even
  with the env set. Do not re-enable without re-deriving its lifetime/parity from scratch.
- **A faulting CUDA op can wedge an M10 die** (2026-10-25 incident: Xid 31 → 8× Xid 158, machine
  died hard; ~40 earlier Xid 31s had all recovered cleanly, so the escalation is possible but not
  the rule). Recovery that worked: full host power-off. Treat new faulting experiments on the M10s
  accordingly — reproduce risky ops standalone first.
- Production config JSONs are gitignored (credentials). Push only to the `nuke` remote.

---

## Measurement rules (learned the hard way)

1. The engine log **truncates on every launch** — grep the full log immediately after a run; never
   tail by a stale line count.
2. Small prompts never reach the prefill sub-batch path — always run the 2.7k/10k protocols before
   judging anything about the tier loop.
3. Every probe/instrument needs an execution banner before its verdict is trusted (the `mw_tp=0`
   discovery: days of conclusions from an arm that never engaged).
4. Short gates (fixed answers + needle) can miss multi-window corruption — the 2.7k/10k protocols
   are the correctness bar.
5. Server ops: kill by PID, wait for VRAM < 500 MiB on GPU0 AND port free before relaunch.

---

## Open ways (unknown — deliberately left open, not closed)

- **f16 on the wire** (halve the completes' D2H bytes): bounded ~+1.2%, untried. Quality risk is
  real (a lower-precision arm previously measurably noisier outputs — gates decide).
- **The tensor-split regime's fault family**: `--maxwell-split tensor` produced real crashes
  (illegal accesses) in its era. No current config runs it; the mechanism was never explained.
  Open: whether it can be made stable at all.
- **Cross-device peer copies / cross-device stream waits**: crashed historically (tensor-split era).
  Untested in the current row-split regime (the safe two-hop form IS verified clean, 250 rounds,
  `tools/xdev_twohop_probe.cu`). Open: whether the fast forms work here.
- **The layer-major wavefront** (overlap the 3060-side work under the M10 serve): bounded 200-240
  tok/s by measured floors, never built, and its partial forms are the family that faults. Open:
  whether a full, careful build is stable — high effort, bounded payoff.
- **Decode beyond ~72**: measured at its floor for this rig's serial verification loop (waitCPU
  ~14.3 ms/window; every removal form measured). Going materially higher needs a second verifier
  resource the M10s cannot host. Open only in the sense of different hardware.
- **The CPU rig reaches ~260 prefill on plain RAM** — for the record, not a mystery: it pays no
  cross-device pipeline and streams near memory bandwidth; the M10s pay the dequant instruction
  price plus the joins. Different machine class, kept as the reference number.

---


## 2026-10-25 (the 0.2.6 port) - the Maxwell layer on Strata 0.2.6, verified live
- Branch maxwell-0.2.6 cut from upstream v0.2.6; our tier grafted with the confirmed drops: the
  DRED arm, the PFPIPE machinery, the ORDER_PROBE instrumentation, and the losing kernel
  geometries (256x8/GACT/E=2/E=8) are NOT in 0.2.6.  The shipped tier + six knobs + hardening +
  tools + tests + records carried.  Upstream's 0.2.3-0.2.6 changes (DFlash2 taps, the dflash2-kern
  warm window, the 0.2.5 prompt-path restructuring, serve changes) coexist cleanly.
- VERIFIED LIVE on the rig (same launch recipe, same config): parity 0 failures; ALL gates green
  (2.7k answers, 10k topics, needle 2/2, arithmetic, decode lists).
- Measured vs the 0.2.2-based stack (same day, same protocol): 2.7k 153.6/154.6 (was 157.5/157.9),
  10k 154.8 (was 155.3 - par), decode 69.2/70.2 (was 72.0-72.6).  The deltas (~2-3% on 2.7k and
  decode) are measured and OPEN: candidates are 0.2.5's decode launch-batching and cache-sizing
  changes interacting with the tier (expert_slots 1874 vs 1800; the INFO line's spec=8 is computed
  identically in both versions - not the cause).  A follow-up tuning pass can chase them; the port
  itself is functionally verified.
- LAUNCH HYGIENE (bit us once): drain ALL FIVE GPUs before relaunch (a GPU0-only check left the
  M10s holding parity's memory; the tier's 8 GB/die open then failed silently at "policy is").

## 2026-10-25 (the head-to-head) - strata-maxwell-M10 vs strata-standard CPU/RAM, same host, equal expert split
- Stock 0.2.6 built clean (/home/toor/Strata-stock026 worktree); A/B with IDENTICAL generic args
  and EQUAL 3060 expert caches (1874 slots both, verified).  Full gates green on every run.
- THE NUMBERS (engine-log tok/s): A (M10) prefill 153-154, decode 69.7-70.4.  B (CPU/RAM,
  stock 0.2.6): prefill 42-54 partially warm -> **219-300 (2.7k) / 397.8 (10k) fully warm**;
  decode 26-42.  The historical 259.9 reproduces and is exceeded: it was this host's warm
  page-cache path (39 GB of experts cached).
- VERDICT: the M10 stack wins decode ~2x and is warmth-independent; the warm CPU/RAM path wins
  prefill ~2x and is cache-state-dependent.  Production stays on A (stable decode + stable
  prefill).  Open: the B warmth/flags residual, and a never-tried mixed mode.
- bench/results/2026-10-25-m10-vs-standard/README.md carries the full schema and every number.
