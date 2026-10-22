# 2026-10-22 - the token-passage anatomy (0.2.2 port)

Question: after the v0.2.2 port, which passage constrains decode - and is there useless work?

Method: instrumented serving (STRATA_MW_LAYERPROF + STRATA_DECODE_TIMING, measured cost ~3.7%:
fib 57.1 tok/s instrumented vs 59.3 clean same-day), story+fib 384-token requests, MWPROF per-card
counters (1440-layer averages), nvidia-smi duty sampling under a 2-thread steady load, a 2-slot
concurrency probe. nsys capture attempted 4x (idle capture window / orphaned port / session
interrupts / hung finalize) - abandoned; the one item marked INFERENCE below is not nsys-verified.

## The window budget (fib, 132 windows, 2.91 tok/window, 50.98 ms instrumented)

| passage | ms/window | share |
|---|---:|---:|
| M10 tier FINISH (down launch + per-card sync waits + row scatter) | 32.56 | 64% |
| M10 tier BEGIN (act pack + staging + H2D + gup launch) | 8.59 | 17% |
| draft (MTP) | 3.47 | 7% |
| GPU-reach wait (the old "ring") | 2.84 | 6% |
| plan + jobs + stage + commit | 1.30 | 2.5% |

**81% of the window is the M10 tier's host-side protocol.** The ring COLLAPSED on the port:
GPU-reach 2.84 ms/window = 59 us/layer vs 407 us/layer on 0.1.45-split. The lock moved from the
3060 ring to the host-side tier protocol - as predicted, a previously-hidden wall.

Story (193 windows): 44.61 ms @ 1.99 tok/window - acceptance-driven, +46% protocol cost per token.

## Per-card walls (us/layer, MWPROF, 1440-layer averages, 3 reports consistent)

| card | gup | down | kernel total | host sync-wait |
|---|---:|---:|---:|---:|
| CUDA1 | 243 | 65 | 292-312 | 242-260 |
| CUDA2 | 240 | 130 | 344-377 | 64-67 |
| CUDA3 | 240 | 165 | 394-410 | 37-40 |
| CUDA4 | 248 | 185 | 426-439 | 27-34 |

- gup is the uniform instruction floor (~240 us/L, jobs/L 4.1-4.7 balanced across cards)
- down scales with finish visit order (65 -> 185): D2H contention by visit order, known property
- the sync-wait concentrates on the FIRST-visited card: the finish loop absorbs the cards' walls
  SEQUENTIALLY; by the time card 1's flag rises, cards 2-4 are done
- the four cards overlap (tier device wall = max ~435 us/L, not the 1520 sum); the host burns
  begin 179 + finish 678 = 857 us/L serially = **~2x the device wall**

## Host hidden blocking (INFERENCE - nsys discriminator failed environmentally)

Direct section spans (MWPROF layer, 30,720-layer history): plan 3.3 + actq 92 + jobs 7.7 +
run 331 = **434 us/L**. The verify residual accounting (decode-timing) says **~885 us/L** of host
time. The ~440 us/L difference is the host blocked INSIDE its own sections - staging-buffer parity
waits / H2D-D2H pacing are the candidates, not attributed (no nsys ground truth).

## Duty under steady 2-thread load

3060: 83-86% util @ 1920-1927 MHz (full boost), mem-bus 57-63% - near-saturated.
M10s: 38-58% util - ~2x headroom. The 3060 keeps up only because the host throttles everything.

## Concurrency probe (2 slots, fib, engine-log decode rates)

Each request: 52-59 tok/s -> aggregate >100 tok/s. The per-window protocol amortizes across
slots (upstream --parallel machinery works on this config).

## Other measured facts

- load: hot page cache 8 s (was 231 s same morning; 424-554 s on 0.1.45 cold)
- after a restart the file tier re-read 40 GB from storage; the first request waited 178 s in
  decode fetches - restarts are expensive unless the page cache holds
- acceptance (drafts accepted/tested): 66-77% typical; one 4.6k-token generation at 52%
- under instrumentation only 490 MiB VRAM free (#2065 warning, threshold 512) - the profiling
  buffers' VRAM cost; clean production has more headroom
- CPU pool: 12 tasks/phase automatic, 4 participating threads, 0 CPU entries served - inert

## The decision menu (measured ceilings, NOTHING implemented)

| # | direction | ceiling | cost |
|---|---|---|---|
| A | host-protocol attack (finish 32.56 ms: pipeline depth, on-device down-chaining, batched D2H, fused multi-wait under split) | window -> ~30 ms = ~70+ tok/s if waits overlap | kernel+protocol work |
| B | spec-depth re-sweep on 0.2.2 | lifts the story worst case (1.99 tok/window, +46% protocol/token) | config-only, cheapest |
| C | multi-slot serving for concurrent clients | measured working: 2 x ~56 = 112 aggregate | config/ops only |
| D | begin-path bulk staging | 8.59 ms ceiling | moderate |
| E | M10 kernel work (4-rows/warp etc.) | DEPRIORITIZED BY DATA: device wall 435 us/L hides under host 885 us/L; kernel gains cannot surface until the protocol cost falls. Retroactively explains Meta's flat window (+23.6% tier gain, 2026-10-20) | - |

## Files
- instr.log: the instrumented run's log (MWPROF + decode timing lines)
- duty.log: nvidia-smi duty samples under steady load

## Follow-up (same day): Option A — the host-protocol attack, CLOSED by measurement

The anatomy's A direction assumed the finish-path host syncs were waste. The tree already
contains the complete no-wait protocol family (E1 mode 2: GPU-raised tier flags, the graph's
fused wait-copy, no-sync decode finish, dual staging sets; STRATA_MW_GPUFLAG=2 — which also
auto-arms the persistent doorbell tier). It was shelved on 0.1.44 "the gain moves into the ring".
With the ring now gone (59 us/L), the definitive re-test:

| arm (same binary, env-only, same hour) | 3x3 mean | TTFT |
|---|---:|---:|
| standard protocol (fresh baseline) | **54.07** (story 49.4 / fib 58.8 / code 54.0) | 21.96 s |
| E1 mode 2 + persist (STRATA_MW_GPUFLAG=2) | **54.46** (+0.7% = noise) | 21.85 s |

**Verdict: the protocol family pays NOTHING even with the ring gone.** The window is invariant
when the waits move from the host sync into the graph's multi-wait — the cost is real
device/chain latency, not host waste: the serial per-layer chain [3060 graph work + the M10
serve (gup ~240 + down 65-185, cards overlapped, wall = max ~435 us/L)] sums to ~41 ms = the
window. A protocol can re-position waits; it cannot remove latency. Option A is closed.

**Bug note**: instrumenting the compound (STRATA_MW_LAYERPROF with GPUFLAG=2) DEADLOCKS the
first request — profiling disarms the fused wait-copy (prof_on_), and the unfused wait path +
the persist doorbell + the split window miss a wakeup (hang: "verify window: waiting for the
GPU (flags A/B/M raised) 3", caught by #29, waits released by #267). Never profile that combo;
the non-instrumented compound is correct (full 3x3 ran clean).

**Menu after A**: the remaining levers are (i) shorten the M10 serve chain = KERNEL work —
option E returns as the main code-level lever (gup's 240 us/L instruction floor is ~55% of the
tier wall; the 4-rows/warp amortization idea now has a path to surface IF the down's visit-order
contention is also cut), (ii) shorten the 3060 per-layer graph work, (iii) amortize the chain
over more tokens: multi-slot serving (measured working, 2 x ~56 aggregate 112), (iv) fewer M10
layers per window (static expert table — deliberately left as is).
