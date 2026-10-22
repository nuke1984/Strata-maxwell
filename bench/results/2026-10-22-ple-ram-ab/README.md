# 2026-10-22 - the PLE-in-RAM A/B (39 GB VM) - NULL RESULT, closed

Trigger: the VM's RAM was raised 7.7 -> 39 GB; --ple-io ram became possible (28.8 GB table
mlock'd in 102.7 s; load 336 s vs 230 s direct). The engine's own probe ALSO improved with
the VM change: direct-mode random 4 KiB reads 19-20 MB/s (historical, all boots) -> 49 MB/s
(11,885 rows/s) today.

## The A/B (identical diverse 9,873-token prompt - spread n-grams, real row-cache misses;
## repetitive 1,885-token prompt shown for the row-cache-covered case)

| arm | prompt | read time | rate | answer |
|---|---|---:|---:|---|
| --ple-io ram | diverse 9,873 tok | 131.45 s | **75.1 tok/s** | "Lighthouse keepers of Brittany, Silk Road caravans" |
| --ple-io direct | diverse 9,873 tok | 131.48 s | **75.1 tok/s** | identical, byte for byte |
| --ple-io ram | repetitive 1,885 tok | 25.6 s | 73.6 tok/s | - |

**Verdict: PLE storage mode is invisible at this rig's prefill rate.** The prompt path is
compute-bound at ~75 tok/s in both modes - the M10 tier's prefill architecture (64-token
sub-batch windows through the tier staging), NOT storage: the batched reader (QD 256) +
the 1M-row cache keep even direct-mode SSD reads ahead of the compute, and the PLE block
overlaps the chunk pipeline by design. The upstream 572-1,290 tok/s prefill numbers are
single-GPU machines without the tier.

Production stays `--ple-io direct` (same speed, faster loads, 28.8 GB free). The RAM is
not wasted: warm-restart page-cache headroom + the disk probe improved 2.5x with the VM
change (49 vs 19-20 MB/s random) - but the PLE hypothesis specifically is CLOSED.

**Newly exposed**: the prefill ceiling (~75 tok/s) is the TIER's prefill path - the next
candidate if TTFT ever matters enough to attack (a real campaign: wider tier prefill
windows / batched staging).
