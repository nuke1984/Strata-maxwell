# 2026-10-25 — strata-maxwell-M10 vs strata-standard-CPU/RAM: the head-to-head (same host, same model, equal expert split)

Protocol: both sides on THIS host (RTX 3060 + 4×M10, 4-core CPU, 39 GB RAM), same model/pack/PLE/
MTP/256k/kv-k8v4/mmap args, full quality gates on every run (all green, both sides).  Standard
protocol: fresh 2.7k prompts ×3 + the 10k diverse (always 0-reused), 300-token decode lists ×2.

## The schema (as run)

| | A: strata-maxwell (our port, tag maxwell-0.2.6-port-production-OK) | B: strata-standard (stock upstream v0.2.6, built clean in /home/toor/Strata-stock026) |
|---|---|---|
| binary | our port (0.1.51 base + Maxwell layer) | stock v0.2.6 (no Maxwell code exists in it) |
| 3060 | dense spine + **1874 expert-cache slots (2.53 GiB)** | dense spine + **1874 slots (B-main; pinned identically)** / 2522 auto (B-stock) |
| remaining routed experts | **M10 tier: 5676 blobs per die, 7.61–7.63 GiB on each of 4 dies** (the profiled set) | **CPU + RAM** (mmap file tier, page cache, 3-worker pool) |
| PLE | direct (default — never enters RAM), same shard | direct (default), same shard |
| other args | spec 6, spec-min-p 0.5, spec-split, prefill auto | B-main: identical; B-stock: spec 4, no spec-split, auto cache |
| env | MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 FASTCOMPLETE=1 | none (the knobs do not exist in stock) |

The A-main/B-main pair differs ONLY in: the binary, the two --maxwell flags, and the env knobs.
The 3060 expert count is EQUAL (1874, verified in both logs).

## The numbers

| run | prefill 2.7k (fresh) | prefill 10k | decode 300-tok | cache state |
|---|---:|---:|---:|---|
| **A: M10 stack** | 153.0 / 153.9 | **154.3** | **69.7 / 70.4** | VRAM-resident (warmth-independent) |
| B-main, first pass | 42.4 / 46.2 | 53.8 | 26.3 / 39.6 | page cache partially warm |
| **B-main, warm** | **219.4 / 287.8 / 300.4** | **397.8** | (same as above) | page cache fully warm (39 GB cached) |
| B-stock (reference) | 217.5 / 243.4 | 160.7 | 26.0 / 41.8 | warm-ish |

Every number: engine-log tok/s.  All gates green on both sides (fixed answers, 10k topics,
needle 2/2; B-stock's decode accepts fewer drafts at spec 4 — expected, not a fault).

## What the numbers say

1. **DECODE: the M10 stack wins ~2×** (70 vs 26–42).  Every token's experts must compute in the
   verify window; the M10s' kernels beat this host's 4-core pool decisively there.
2. **PREFILL: the warm CPU/RAM path wins ~2×** (up to 397.8 on the 10k vs the M10's 154.3) — and
   it is **page-cache-state-dependent**: the same config measured 42→53 (partial warmth) and
   219→398 (fully warm).  The M10 stack's 154 is warmth-independent (VRAM-resident experts).
3. **The historical 259.9 (2026-10-22, `2026-10-22-m10-vs-ram`) reproduces and is exceeded** —
   it was this host's CPU/RAM path on a warm cache; today's fully-warm 10k hit 397.8.  The
   earlier campaign framing ("259.9 unreachable by the M10 stack") stands: the M10 prefill
   ceiling (~154) is far below what this host's RAM does warm.
4. The decode-side reality inverts the prefill story: at 26–42 tok/s the CPU/RAM rig is not a
   practical serving config for generation-heavy use; at 70 the M10 stack is.

## Open (measured, unexplained — left open)

- B-main's FIRST pass (42–54) vs B-stock same-day (160–243): the delta is warmth + possibly the
  pinned cache/reserve or spec-split; not isolated this run (the warm B-main rerun converged to
  B-stock's class, so warmth is the dominant term — the residual is unexplored).
- The 0.2.5/0.2.6 chunked-prompt path's interaction with the CPU tier at various cache sizes.
- Whether a mixed mode (M10 for decode-critical experts + RAM for prefill breadth) could hold
  ~70 decode AND ~300 prefill — never tried; the two paths were never run simultaneously.

**Serving choice stays A (the M10 stack)** for production: its decode advantage is stable and
its prefill does not depend on a 39 GB page cache staying warm.
