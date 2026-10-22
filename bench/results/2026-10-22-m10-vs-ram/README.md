# 2026-10-22 - M10 vs RAM: the head-to-head (both arms measured from scratch, same session)

The question: on a 40 GB machine, what do the M10s actually buy vs computing the same experts
from RAM? One variable - where the 22,702 non-cached experts live and compute. Everything else
identical (same binary, same profile, same --expert-cache 1800 -> the SAME 3060 expert set,
same spec 4 / k8v4 / 256k / embed-VRAM).

| arm | A - the M10 rig | B - the RAM rig |
|---|---|---|
| 3060 | dense + 1,874 experts | dense + the same 1,874 experts |
| other 22,702 experts | M10 VRAM, GPU-computed (the tier) | RAM page cache, CPU pool (3 workers + host, 4 vCPU) |
| PLE | locked in RAM (--ple-io ram, 102.7 s) | SSD (--ple-io direct, 22 MB/s probe today) |
| flags | --maxwell-devices 1,2,3,4 --spec-split | none (standard paths) |

## The numbers (fresh 3x3 + TTFT each, temp 0, engine-log decode rates)

| metric | A | B | ratio |
|---|---:|---:|---:|
| decode, 3x3 overall | **61.30 tok/s** (story 63.6 / fib 54.9 / code 65.3) | **12.12 tok/s** (steady 14-16; cold runs 3-11) | **A = 5.1x B** |
| prompt reading | 73.8 tok/s | **259.9 tok/s** | B = 3.5x A |
| TTFT, same 2,726-tok prompt | 36.98 s | **10.60 s** | B = 3.5x faster |
| load | 338 s (PLE lock 103 s + M10 fill 46 s) | 41 s | |
| fixed answers | 4 ok | 4 ok | |

B internals: the 3060 cache covered 12.7% of decode lookups (952/7,516); the rest computed on
4 vCPUs from the page cache (1.25 TB cumulative in-RAM reads this run). B ran with its PLE on
the SSD and STILL won prefill 3.5x - the tier's 64-token prefill windows are A's prefill cost,
not storage.

## Verdicts

1. **The M10s are worth 5.1x on decode.** The whole Maxwell tier, validated by the cleanest
   possible comparison: 61.3 vs 12.1 tok/s, same model, same 3060 expert set, same hour.
   A 4-vCPU pool cannot compute 24k experts competitively; 30 GB of device-resident experts
   on the M10s can.
2. **The tier costs 3.5x on prompt reading** (73.8 vs 259.9 tok/s): the standard prefill path
   (CPU share + chunk pipeline, no 64-token tier windows) reads prompts far faster. TTFT
   37 s vs 10.6 s. The tier's prefill windowing is now THE documented cost of the M10 rig -
   the concrete target if prompt speed ever matters (widen the windows / batch the staging).
3. A itself measured 61.3 today vs the historical 53-55 family - same-day observation, not a
   controlled claim: the 40 GB RAM also lifted A (the file tier's reads now cache; no thrash).

Config files: A = strata-iq2xs-embed-vram-split.json (+ --ple-io ram during the arm, since
restored to default); B = strata-iq2xs-embed-vram-B.json (kept for reference).
Data: bench-armA-m10.json, bench-armB-ram.json.
