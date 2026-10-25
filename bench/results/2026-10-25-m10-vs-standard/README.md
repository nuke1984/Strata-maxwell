# 2026-10-25 — THE test: strata-maxwell (experts in M10 VRAM) vs strata-standard (experts in CPU RAM)

One head-to-head, this host (RTX 3060 + 4×M10, 4-core AVX2 CPU, 39 GB RAM), same model/pack/PLE
shard/MTP/256k/kv-k8v4/profile. Both sides deterministic: every routed expert lives in real
memory (VRAM or page-locked RAM), no page cache, no mmap-on-demand, no file reads after load.
This entry supersedes every earlier B run in this directory (those measured the mmap file tier,
a mis-specified config; see the note at the end).

## The schema (as run, verified from load logs)

| | **A — strata-maxwell** | **B — strata-standard** |
|---|---|---|
| binary | our port, tag `maxwell-0.2.6-port-production-OK` | stock upstream v0.2.6 (clean build, /home/toor/Strata-stock026) |
| recipe | production: spec 6, spec-min-p 0.5, spec-split, prefill auto + 6 env knobs (MT=1 MT_T=256x4 NTM=256 TWOPHASE=1 AD2H=1 FASTCOMPLETE=1) | stock: spec 4 (default), prefill auto, no knobs (none exist) |
| 3060 | dense spine + **1874 expert-cache slots (2.53 GiB)** — pinned `--expert-cache 1800 --vram-reserve-mib 976`, fit-path, log-verified | **identical: 1874 slots (2.53 GiB)** — same pins, log-verified |
| remaining routed experts | **22,702 in M10 VRAM**: 5676+5676+5675+5675 experts, 7.61–7.63 GiB per die, resident at load | **22,702 in CPU RAM**: the resident complement (30.49 GiB, page-locked) + 1304 of 1584 lend-slot duplicates = 32.24 GiB total. Log: `resident RAM mode: 32.24 GiB of experts in RAM (page-locked), 1874 in the GPU cache; adaptive swaps ... (no file reads)` |
| PLE | **`--ple-io direct` explicit** — pinned; 0 auto-switch lines in log | identical |
| expert totals | 1874 + 22,702 = 24,576 routed | 1874 + 22,702 = 24,576 routed — EQUAL |

Code-verified config note: the bare `--resident-cpu-experts` form hard-codes an 8 GiB RAM
headroom (generate.cpp:544, no override) — 30.49 + 8 GiB exceeds this host's 39 GB, so it
refuses to start (measured: hard exit, `CPU expert residency:` error — the no-silent-fallback
behavior worked as designed). B therefore uses the engine's own setup form `--resident-experts`
(same resident machinery, copy page-locked, 4 GiB headroom: 30.49+4 < 36.28 available). Its
soft-fallback path was hard-gated: the success line above is from THIS launch (0 fallback
lines after this launch's engine-start line). The 280 un-kept lend slots are read from file
only when a prompt lends them (a prefill-path detail; decode never lends).

## The numbers (engine-log tok/s, this test's launches only)

**Prefill** — fresh prompts, 0-reused verified in every line:

| | 2.7k trial 0 (1842 tok) | 2.7k trial 1 (1728 tok) | 2.7k trial 2 (1728 tok) | 10k diverse (8836 tok) |
|---|---:|---:|---:|---:|
| **A — M10 VRAM** | 148.5 | 144.5 | 154.0 | 154.3 |
| **B — CPU RAM** | 99.9 | **430.3** | **434.9** | **275.3** |

**Decode** — exact request, both sides identical:
`POST /v1/chat/completions`, `temperature 0` (greedy), `max_tokens 300`, `stream false`,
`enable_thinking false`, prompt: *"List 100 distinct animals, one per line, numbered 1. to 100.
Do not stop early."* (39 tokens as tokenized; response = numbered list, cut by the 300-token
cap at ~item 53-55 — the model does not stop early).

| side | spec | trial | 300 tok in | **decode tok/s** | drafts accepted |
|---|---|---|---:|---:|---|
| **A — M10 VRAM** | 6 + min-p 0.5 + split | 0 | 4318 ms | **69.5** | 234/258 |
| **A — M10 VRAM** | 6 + min-p 0.5 + split | 1 | 4279 ms | **70.1** | 234/258 |
| **B — CPU RAM** | 4 (stock default) | 0 | 8355 ms | **35.9** | 202/300 |
| **B — CPU RAM** | 4 (stock default) | 1 | 5719 ms | **52.5** | 207/285 |

Quality gates: ALL GREEN on both sides (2+2=4, 17×23=391, 9999²=99980001, 10k topics correct,
needle recall 2/2, decode list shape intact).

## What the numbers say

1. **Prefill: the RAM path wins decisively** — 430-435 vs ~154 on fresh 2.7k (2.8×), 275 vs 154
   on the 10k (1.8×). The CPU's wide prefill windows beat the M10 tier's 64-token windows.
2. **Decode: the M10 stack wins ~1.8× over B's best** (70 vs 52.5). Verify-window expert compute
   on four M10 dies beats the 4-core AVX2 pool even with experts page-locked in RAM.
3. **Both sides are now deterministic** — no page-cache state anywhere. B's remaining
   trial-0→1 spread (99.9→430 prefill, 35.9→52.5 decode) is engine first-touch warmup and
   VRAM-cache adaptation (adaptive swaps into the 1874 slots), not file-cache warmth: the
   experts were already all in RAM. A shows the same shape compressed (148→154, flat decode).
4. Production choice stays **A** (the M10 stack): decode is the interactive metric and A's
   prefill does not depend on 32 GiB of page-locked RAM leaving no headroom (B runs at
   ~3 GiB MemAvailable; A leaves the host 39 GB free for everything else).

## Superseded

The earlier entries in this directory (B-main / B-stock mmap runs, 42→398 "warmth gradient")
measured `--mmap-experts` alone — the file tier, NOT the specified RAM-resident mode. They
stand as measurements of that other config; the warmth story explains them and does not apply
to this test.
