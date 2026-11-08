# IQ3_S on the Maxwell tier — a fork of Strata-maxwell

**Craicek** · branch `iq3` · November 2026

This is the guide to what this fork adds, what it does not add, what was measured, and how to
rebuild and re-measure it.

Upstream Strata-maxwell keeps the routed experts of a big MoE model in the VRAM of **four** Tesla
M10 dies (one card). This fork takes that to **eight dies (two cards)** and makes a **Qwen3.8-Flash-Next
IQ3_S** pack — the 3-bit format, not the 2-bit one the upstream tier was built for — run on them at
useful speed. The engine is the upstream one; the new work is in the Maxwell kernels, the expert
dispatcher, the parity tool and a campaign of measurements.

The campaign's working notes are in **[LOG.md](LOG.md)** (Italian, one entry per iteration) and the
raw numbers in **[results/](results/)**. This file is the English summary that a newcomer should read
first. The brief the campaign started from is preserved verbatim in **[BRIEF.md](BRIEF.md)**.

---

## 1. What the fork adds, and what it does not

### Ours

| Area | What this fork adds | Files |
|---|---|---|
| Device count | Up to **8 Maxwell dies** on the whole-expert path (was 4). The E1 completion-flag path and tensor/Meta stay 4-way by construction and refuse more. | `include/strata/core/expert_source.hpp`, `include/strata/core/maxwell_experts.hpp`, `src/core/expert_source.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp` |
| Gate/up formats | **IQ3_XXS (18), IQ3_S (21), IQ4_XS (23)** dot products on the M10, transcribed from the 3060's IQ kernels and verified against them. | `src/kernels/cuda/expert_maxwell.cu`, `include/strata/kernels/cuda/expert_maxwell.hpp` |
| Down | A **native IQ4_NL down** for the Maxwell tier: a new stage-2 kernel, selected per layer by `d_type`, with the blob's down region **repacked planar** at load (one shared conversion) instead of copied verbatim. | `src/kernels/cuda/expert_maxwell.cu`, `include/strata/kernels/cuda/expert_maxwell_iq4nl.hpp`, `src/core/maxwell_experts.cpp` |
| Dispatcher | Per-layer kernel selection on `(gu_type, d_type)`, including the IQ4_NL down; explicit refusals for the persistent kernel and for E1/tensor with `d_type 20` rather than a silent wrong-layout read. | `src/core/maxwell_experts.cpp`, `src/program/generate.cpp` |
| Parity tool | `maxwell_iq_parity` extended to the new formats (synthetic and shard modes), plus a **bit-for-bit dump** of the down output used to prove the planar repack is lossless. A timing harness, `iq_gup_bench`, measures the production gate/up kernel on real GGUF rows at decode-like job counts. | `src/kernels/cuda/expert_maxwell_iq_parity.cpp`, `src/kernels/cuda/iq_gup_bench.cpp` |
| Bench tooling | `iq3_bench.py` (fixed bench, long context, soak), `iq3_power.py` (power and throttle sampling), `iq3_profile.py` (per-layer span), each with tests. | `tools/iq3_*.py`, `tools/test_iq3_*.py` |
| Documentation | The campaign: brief, log, and every measurement as a file under `results/`. | `docs/iq3/` |

### Upstream (not ours)

| Work | Where it comes from |
|---|---|
| The Strata engine itself: model loading, expert arena, KV cache, server, CLI, the CUDA backend | **Niko1221**, [Strata](https://github.com/Niko1221/Strata) |
| The Maxwell tier: the whole-expert split, the S2/Q2_0 down kernel, the IQ2 gate/up kernels, the round-trip quantization gate, the safety guards, the 4-die implementation | **nuke1984**, [Strata-maxwell](https://github.com/nuke1984/Strata-maxwell) |
| The IQ dot products (`vec_dot_iq*_q8_1`) that the Maxwell kernels are transcribed from, the `block_iq*` types and their reference dequantization | **Strata** and **ggml / llama.cpp** ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)) |
| The model | **Qwen team**, Qwen3.8-Flash-Next |
| The quantization format and the pack | **ISTA-DASLab**, GSQ-RCO |

The transcription is mechanical and deliberate: the new Maxwell kernels are the 3060's own dot
products, restated for the Maxwell register file, and the parity tool compares them against the
3060's production path to 1e-5 relative. This fork did not invent the arithmetic; it moved it.

### A claim, stated carefully

To our knowledge, the upstream projects do not support **IQ3_S** on the Maxwell tier. Checked
against the upstream code on 2026-11-08: at the baseline commit `6175fa0`, `expert_maxwell.cu`
instantiates exactly three i-quant gate/up types — `16` (IQ2_XXS), `22` (IQ2_S), `29` (IQ1_M) — and
`maxwell_experts.cpp` accepts only a `d_type 42` (Q2_0) down, over gate/up `42` (S2) or those three.
An IQ3_S pack uses gate/up `18`/`21`/`22`/`23` over a down that is `20` (IQ4_NL) on 39 of its 48
layers, so it is refused at load, by design. If you know of an upstream implementation, this
paragraph is wrong and should be corrected.

---

## 2. Hardware

One HP Z440. Everything below was read from the machine on 2026-11-08; the raw evidence, including
the exact commands and their output, is **[results/66-pcie-slots-and-links.txt](results/66-pcie-slots-and-links.txt)**.

### Cards, slots and bus addresses

| Physical slot | Card | Slot wiring (SMBIOS / SltCap) | Root port | Bridge | PCI bus of the dies | `nvidia-smi` index | CUDA index |
|---|---|---|---|---|---|---|---|
| **Slot 2** | RTX 3060 12 GB | x16 PCIe 3 (`SLOT 2`, bus `07:00.0`) | `00:02.0` | — | `07:00.0` | 4 | **CUDA0** |
| **Slot 3** | NVMe SSD (Samsung SM951/PM951) | x4 (SMBIOS: `x4 PCI Express 2 x8`) | `00:1c.4` | — | `12:00.0` | — | — |
| **Slot 4** | Tesla M10 #1 | **x8** PCIe 3 (`SLOT 4`, bus `01:00.0`) | `00:01.0` | PLX 8747 `01:00.0` → `02:08/09/10/11.0` | `03:00.0`, `04:00.0`, `05:00.0`, `06:00.0` | 0–3 | **CUDA1–CUDA4** |
| **Slot 5** | Tesla M10 #2 | x16 PCIe 3 (`SLOT 5`, bus `08:00.0`) | `00:03.0` | PLX 8747 `08:00.0` → `09:08/09/10/11.0` | `0A:00.0`, `0B:00.0`, `0C:00.0`, `0D:00.0` | 5–8 | **CUDA5–CUDA8** |
| Slot 1 | free | x1 PCIe 2 x4 | — | — | no device (SMBIOS shows the slot's address `0f:00.0`) | — | — |
| Slot 6 | free | 32-bit PCI | — | — | no device (SMBIOS shows the slot's address `11:00.0`) | — | — |

The slot numbers are **measured**, not declared: `dmidecode -t slot` names the designation and the
bus address of the device in it, and `lspci -vv -s 00:0x.0` independently reports `Slot #2`, `Slot #4`
and `Slot #5` in the root ports' `SltCap`. The two readings agree, which is why this table can name
the slot for each bus group instead of leaving the correspondence open.

The M10 is not one GPU: each card is a **PLX PEX 8747 48-lane, 5-port PCIe switch** with **four
GM107 dies** behind it, each die with its own 8 GB. Two cards therefore give **eight dies**, which
is what the engine calls eight Maxwell devices.

### Device order

```
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=4,0,1,2,3,5,6,7,8
```

This puts the 3060 at CUDA0 and the eight dies at CUDA1..CUDA8, slot 4 first and then slot 5, bus
ascending inside each card. The engine is launched with `--maxwell-devices 1,2,3,4,5,6,7,8`.

### PCIe links: what was measured, and why it matters

At idle every link sits at 2.5 GT/s — that is NVIDIA's link power management, not a cap. Under load
the picture separates:

| Link | Idle | Under load | Verdict |
|---|---|---|---|
| 3060 `07:00.0` | 2.5 GT/s x16 | **8 GT/s x16** | Gen3, as expected |
| die `03–06:00.0` (slot 4) | 2.5 GT/s x8 | **8 GT/s x8** | die↔switch is Gen3 x8 |
| die `0A–0D:00.0` (slot 5) | 2.5 GT/s x8 | **8 GT/s x8** | die↔switch is Gen3 x8 |
| **uplink** `01:00.0` (slot 4 → PLX) | 2.5 GT/s x8 | **2.5 GT/s x8** | **never trains up** |
| **uplink** `08:00.0` (slot 5 → PLX) | 2.5 GT/s x16 | **2.5 GT/s x16** | **never trains up** |

So the dies do negotiate Gen3 with their own switch, and the switch's uplink to the host stays at
**2.5 GT/s**. The bandwidth measurement confirms which of the two matters:

| Device | Host→device, best of 4 × 256 MiB | Matches |
|---|---|---|
| CUDA0, RTX 3060 | **12.05 GB/s** | Gen3 x16 |
| CUDA1–4, M10 in slot 4 | **1.58 GB/s** | Gen1 **x8** |
| CUDA5–8, M10 in slot 5 | **3.14 GB/s** | Gen1 **x16** |

That is the operator's BIOS setting showing up as a number: the M10 slots are **forced to Gen1**,
and the measurement says exactly what that costs — 1.58 GB/s per die in the x8 slot, 3.14 GB/s in
the x16 one, against 12.05 for the 3060. Slot 4 is also physically an x8 slot, so its card could
never do better than x8 even at Gen3.

Two consequences worth knowing before reading any number below:

- **The engine's own probe line is the 3060, not the M10.** `strata generate: PCIe probe: 11.3 GB/s`
  is printed for the device the probe ran on; the separate per-device measurement above gives the
  same link 12.05 GB/s. Do not read 11.3 as a Maxwell figure. (The brief does; this is one of the
  premises the LOG supersedes.)
- **The uplink is shared by four dies.** An expert row read by a die crosses the Gen1 uplink once,
  and all four dies on that card share it. The production log's own summary is worth quoting exactly:
  `decode expert cache hit rate: 100.0% (34263 hits / 34263 lookups); 346377 more read by the GPU
  over PCIe or from another GPU (91.0% of all 380640 routed)`. That 91% is not a measure of uplink
  traffic specifically — the line does not separate a copy from the host from a copy from another
  die — so no bottleneck share is claimed here. The link numbers above are what was measured.

### Why Gen1: what the campaign observed

The Gen1 forcing is the operator's, and it predates this work — the BIOS is not touched by this fork.
The reason given for it is an electromagnetic-compatibility problem that had shown up as PCIe errors.
What the campaign can add is the *observation* of what the link layer does afterwards, and that
observation does **not** confirm the remedy: correctable errors continue on the second card's bridge
while the slots are already at Gen1 (series below). Gen1 is a stated platform constraint here, not a
result of this fork.

The relevant counter is AER (`/sys/bus/pci/devices/*/aer_dev_correctable`, read by `/root/aer.sh`).
It resets when the machine reboots, so it is read as a per-boot series. Across the campaign, from
2026-11-05 to 2026-11-08:

- **Every** event is a correctable `RxErr` (receiver error, physical layer) on the PLX bridge
  **`0000:09:11.0`** — the downstream port of the **second** M10 (slot 5, child `0D:00.0`).
- **No GPU ever reports anything**: all nine devices at zero, in every window.
- Series, per boot and read at each window's edges: **0** on 2026-11-05 (already at Gen1; the forcing
  predates this campaign) → **2** by the Iteration 0a baseline → **3** before and after the Iteration 3 window →
  **6 → 8** across the Iteration 5 window → **11** at the Iteration 6 close → **14** at the start of the
  Iteration 7 window, **16** at its end over ~16.9 h of that boot (11 events in the first 10 h, 5 in the
  last 6.9 h — steady, not a burst). The machine rebooted on 2026-11-08 before the Iteration 8 window,
  which then closed at **4 before and 4 after** a 10-minute, 84-run soak — no event in that window. The
  documentation session that produced this file found **6** at its first reading and **7** three
  minutes later, then stayed at 7 for six readings 20 s apart. The two events between the I8 close and
  that first reading were not observed as they happened.

None is fatal, non-fatal or uncorrectable; none coincides with an OOM, a CUDA error or a failed
allocation; the counter does not track load (it rose during a light documentation session and stayed
still through the 10-minute soak). It is recorded, not explained. The distinction that matters for
the tier is **bridge, not GPU** — and the bridge in question is the one in front of the second card,
which is also the card whose Gen1 link is twice as wide as the first card's.

### Power

| Card | Limit now | Default | Minimum | Maximum |
|---|---|---|---|---|
| RTX 3060 | **120 W** | 170 W | 100 W | 180 W |
| Tesla M10, each of 8 dies | **30 W** | 53 W | 26.5 W | 53 W |

The M10 limit is set low on purpose and raising it buys nothing — see the known limits in §6.

The 3060 stays at **120 W** because it was measured not to be on the critical path. In an A/B/A
experiment on IQ2_XS (`results/39-` … `results/44-`), the limit was raised 120 → 140 → 120 W:
power draw went 119.0 → 135.1 → 118.9 W (median), SM clock 1777 → 1875 → 1777 MHz,
`SwPowerCap` 82.8% → 64.8% → 81.6% — and decode went **59.4 → 59.2 → 58.8 tok/s**, i.e. B minus the
mean of A1 and A2 is **+0.1 tok/s against a ±0.6 bar**. The mechanism works and the speed does not
move: the 3060's compute is overlapped with the M10 tier's, so a faster 3060 waits longer. The limit
went back to 120 W and stayed there.

---

## 3. Software

| Layer | Version / setting |
|---|---|
| Hypervisor | **Proxmox VE 9** (declared; not readable from inside the container), host kernel **6.17.13-21-pve** (pinned; `-pve` kernel measured) |
| Host kernel command line | `... nomodeset pci=realloc` — read from `/proc/cmdline`; `pci=realloc` is what lets the BARs of two 4-die PLX switches fit (without it the second switch does not enumerate) |
| Container | **LXC 101**, Debian 12 (bookworm), **privileged** (`/dev/nvidia*` passed through, `cap_sys_rawio`, `cap_sys_module`, `cap_mac_admin`), 24 GiB RAM, 8 threads, 220 GB rootfs on ZFS |
| NVIDIA driver | **580.196.07** — the last branch with Maxwell (sm_50) support |
| CUDA | **12.9** (`V12.9.86`) — CUDA 13 dropped sm_50 |
| Build | CMake **4.4.4** from `/opt/cmake-venv`; the system cmake 3.25.1 does not understand the CUDA20 dialect nvcc 12.9 emits |
| Engine base | upstream **Strata v0.2.6**, engine **0.1.51**, branch `maxwell-0.2.6` (tag `maxwell-0.2.6-port-production-OK`) |
| Server | the upstream Python server (`serve/server.py`) in a venv, run as a systemd unit on port 8080 |
| Model | Qwen3.8-Flash-Next, **IQ3_S** GSQ-RCO pack (ISTA-DASLab), `--max-context 128000`, `--kv k8v4`, MTP spec 6 |

Production engine is the binary from commit **`e1de86e8`**, md5 `0f5effb3d86b2251cd80439079a50575`,
kept at `/root/Strata-maxwell/engine-iq3-e1de86e8/strata` with `libstrata_kq.so` beside it. The tag
**`aidrogen-prod-2026-11-08`** points at that same commit. Everything committed after `e1de86e8` up to
the branch tip is documentation only — verified with `git diff --name-only e1de86e8..HEAD`, which
lists nothing but `docs/iq3/`.

`libstrata_kq.so` is **opt-in**: `mmq_kq_loader.cpp` opens it only with `STRATA_MMQ_KQ=1`. Production
does not arm it, so the absence of "K-quant prompt kernels" lines in the log is correct, not a
failed load.

---

## 4. Results: IQ2_XS vs IQ3_S

All numbers below come from files in [`results/`](results/); each row names its source. Nothing here
is quoted from memory or recomputed.

### The bench protocol (declared before the run)

- **Prompts:** `city` (describe Florence in ~150 words) and `code`, **3 runs each**, 6 runs per engine.
- **Thinking ON** (no `--no-thinking`), `--max-tokens 600`, `temperature 0`.
- **`STRATA_MW_LAYERPROF` off** — the profiler costs 4.7% on this bench, so it would be a second variable.
- Expert cache: **988** for IQ3_S, **1800** for IQ2_XS (the two models' best-fit values, established separately).
- **Same binary** (`e1de86e8`) for both engines, same pack-loading path, same machine, same window.
- **Criterion:** mean of city+code over the 6 runs **≥ 55 tok/s**. Calibrated on IQ2_XS with thinking
  on, because `--no-thinking` closes an answer in ~240 tokens and measures a transient.

### Speed

| | city (3 runs) | code (3 runs) | **mean, 6 runs** | median | vs criterion 55 |
|---|---|---|---|---|---|
| **IQ3_S** (planar IQ4_NL down) | 51.30 (sd 0.90) | 57.30 (sd 1.14) | **54.30** | 54.10 | **FAIL**, by 0.70 tok/s = 1.3% |
| **IQ2_XS** (upstream reference) | 56.60 (sd 3.24) | 63.97 (sd 1.69) | **60.28** | 60.50 | PASS |
| ratio | 0.91× | 0.90× | **0.90×** | | |

Source: [`results/57-bench-i7-summary.txt`](results/57-bench-i7-summary.txt).

The FAIL is reported as a FAIL because the declared criterion says so, and it is also reported as
what it is: **inside the noise of this bench**. Pooling the per-run standard deviations of both
engines gives a combined sd near 2.0 tok/s, i.e. a standard error near **0.80** on a 6-run mean;
the gap is 0.70. The bench supports the *ordering* and the *size of the gain*, not a 1% verdict —
resolving 0.7 tok/s at 95% confidence would need about 2 × 121 runs. The operator promoted IQ3_S to
production with that stated, and the two optimisations that could close the gap (gate/up at 134 µs/L,
shared-memory LUT) are deferred and non-blocking.

### What the fork changed in the token budget

The planar down is the single biggest win of the campaign, measured A/B in one window with the old
engine rebuilt from `83580f58` and otherwise identical configs:

| down-20 | µs/layer | µs/MB | GB/s |
|---|---|---|---|
| before (raw 18 B `block_iq4_nl`) | **271** | 66.8 | 15.0 |
| after (planar repack) | **118** | 28.8 | 34.8 |
| change | **−56.5%** | **−57.0% (2.32×)** | |

The unit matters: 271 → 118 is µs **per layer**, and the ×2.32 is the same change measured in µs
**per MB** (66.8 → 28.8). They differ because the down-20 group's MB/layer moved ~2% between arms.
The honest headline is the −57% per MB, the stage's own speed; −56.5% per layer is what the token
sees.

The untouched Q2_0 down (`d_type 42`) reads **101 µs/L in both arms**, identical to the microsecond,
which is what makes the A/B clean. The earlier Iteration 6 figure of 296 µs/L is *not* used as the
denominator: in this window even the untouched kernel reads 13.7% faster, so 296 → 271 is a window
effect and is not claimed as a result.

Against IQ2_XS the distance closed from **−25.3%** (Iteration 6, same profiler on both arms) to
**−9.9%** (this window, same protocol on both). That is the campaign result.

Where the token goes now (profile arm, thinking off, so the shares describe structure, not this
bench's exact budget): kernel span of 48 layers 12,010 µs = 62.9%, of which gate/up 6,432 µs
(33.7%) and down-20 4,602 µs (24.1%, was 49.3% in Iteration 6). The down is no longer the dominant
stage; gate/up is, and it did not change.

Source: [`results/57-bench-i7-summary.txt`](results/57-bench-i7-summary.txt), LOG iteration 7 step 4.

### Long context and memory

| Workload | Prompt | Prefill | Decode | Needle | VRAM free on the 3060 |
|---|---|---|---|---|---|
| IQ3_S, 64k control | **62,703 tok** | 180.1 tok/s | 50.4 tok/s | **FOUND** (`meadow-copper-504`) | 1073 MiB |
| IQ3_S, production 128k | **119,275 tok** | 178.4 tok/s | 44.7 tok/s | **FOUND** | 1035 MiB |
| IQ3_S, 32k (Iteration 5, 1749-slot cache) | 32,135 tok | 183.0 tok/s | 58.2 tok/s | **FOUND** | 93 MiB |

Sources: [`results/60-final-i7-restore-verified.txt`](results/60-final-i7-restore-verified.txt),
[`results/62-longctx-i8-128k.json`](results/62-longctx-i8-128k.json),
[`results/09-window-i5.txt`](results/09-window-i5.txt).

The 128k test is the production configuration (988-slot cache, 8 tiers, `--max-context 128000`): one
needle at the middle of a 119,275-token haystack, found, with **zero** OOM, CUDA errors or failed
allocations, and ~1 GB free on the 3060 throughout.

At load and at steady state, production holds:

| | IQ3_S production | IQ2_XS (for comparison) |
|---|---|---|
| 3060 expert cache | **1267 slots**, 2.45 GiB, pre-filled in 4.4 s (600 MB/s) | 1874 slots, 2.53 GiB, 16.7 s (163 MB/s) |
| Maxwell tier | **8 tiers**, 2914×5 + 2913×3 experts, **5.54–5.56 GiB** per die | 8 tiers, 2838×6 + 2837×2, 3.80–3.82 GiB per die |
| Round-trip gate | **8/8 PASSED**, worst rel 1.18e-02 … 1.29e-02, 23,309 blobs converted | 8/8 PASSED, worst rel 1.12e-02 … 1.26e-02 (identical to the 0.1.51 baseline) |
| VRAM free on the 3060 | 1332 MiB at load → **1033 MiB** steady | 2242 MiB at load → 1943 MiB steady |
| Load time | 251 s to `/health` (SSD) | 640 s (pack on HDD) |

Sources for these rows: [`results/61-final-i8-production-verified.md`](results/61-final-i8-production-verified.md)
and [`results/00-baseline.txt`](results/00-baseline.txt) / [`results/06-window-iq2-xs.txt`](results/06-window-iq2-xs.txt)
for the IQ2_XS side. The blob count is the sum of the production log's eight `gate ... PASSED (... N blobs
converted)` lines, which is 23,309 and matches the log's own `0 of 23309`; the archived report prints
23 315 in its table, a transcription typo corrected here.

The 988-slot cache is the knob that buys that margin: at 1749 slots (IQ3_S's own auto value, measured
in Iteration 5, [`results/09-window-i5.txt`](results/09-window-i5.txt)) the 3060 drops to **93 MiB**
free during generation, which is why production runs 988.

### Production soak and probe

| | Value |
|---|---|
| Soak, 10 min, city prompt, 300 tokens, thinking on | **84 runs**, mean **43.51** tok/s, median 43.45, sd 1.29, min/max 39.7/46.7 |
| Drift, first → last run | **−0.8 tok/s** (inside the sd; no progressive decay) |
| VRAM during soak | **1033 MiB constant**, no drift |
| AER in the soak window | bridge `09:11.0` at 4 before and after → **no event in window**; all GPUs 0 |
| City probe, 600 tokens, first request after load | **50.1 tok/s**, drafts accepted 364/554 (65.7%) |

Sources: [`results/61-final-i8-production-verified.md`](results/61-final-i8-production-verified.md),
[`results/64-soak-i8-10min.json`](results/64-soak-i8-10min.json),
[`results/63-bench-i8-city-probe.json`](results/63-bench-i8-city-probe.json).

43.5 tok/s is a **production exercise figure** (300 tokens, one prompt) and must not be compared with
the 54.30 of the fixed bench (600 tokens, two prompts, 3 runs each). They measure different things.

---

## 5. Reproducing it

### 5.1 Build

The build is the upstream one plus the flags the tier needs. On a machine with CUDA 12.9 and the
driver above:

```sh
cd /root/Strata-maxwell                      # a checkout of this fork
git fetch https://github.com/BitMakerMan/Strata-maxwell.git iq3
git checkout -B iq3 FETCH_HEAD

export PATH=/opt/cmake-venv/bin:$PATH        # cmake 4.x; 3.25.1 cannot compile the CUDA20 dialect
cmake -S . -B build-iq3 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TQ_OVERLAY=ON -DGGML_STATIC=ON
cmake --build build-iq3 -j 8 --target strata
mkdir -p engine-iq3 && cp build-iq3/strata build-iq3/libstrata_kq.so engine-iq3/
```

`STRATA_EXPERIMENTAL_SM60` is what allows compute capability 5.x/6.x builds; on this fork it is On.
`-DCMAKE_CUDA_ARCHITECTURES=86` targets the 3060 — it is **not** how the Maxwell code gets built:
`CMakeLists.txt` puts `src/kernels/cuda/expert_maxwell.cu` in its own static library
(`strata_maxwell`, line ~600) with `CUDA_ARCHITECTURES 50`, because the engine's own kernels target
sm_75+. That separation is upstream's and is why the Maxwell tier can live in the same binary as the
3060 path.
`CMAKE_CUDA_RUNTIME_LIBRARY` is only relevant to a Windows/MSVC build and is not set here.

`maxwell_iq_parity` and `iq_gup_bench` sit in the CUDA block of `CMakeLists.txt` **outside** the
`STRATA_BUILD_TESTS` guard, so they build in a plain configuration:

```sh
cmake --build build-iq3 -j 8 --target maxwell_iq_parity iq_gup_bench
```

Both were rebuilt and linked on this checkout during the documentation pass.

### 5.2 Configuration

The server takes a JSON config. This is the **production** one with the key replaced — adapt the
paths, never commit the file:

```json
{
  "model_name": "qwen3.8-flash-next-iq3_s",
  "exe": "/root/Strata-maxwell/engine-iq3-e1de86e8/strata",
  "cwd": "/root/Strata-maxwell",
  "host": "0.0.0.0",
  "port": 8080,
  "cuda": 12,
  "lib_dirs": ["/usr/local/cuda-12.9/bin", "/usr/local/cuda-12.9/lib64"],
  "log": "/root/Strata-maxwell/strata-iq3_s-prod.log",
  "tokenizer": "/root/Strata-data/packs/iq3_s/tokenizer",
  "gpu": 0, "gpus_asked": true,
  "api_key": "LA_TUA_API_KEY",
  "args": [
    "--pack", "/root/Strata-data/packs/iq3_s",
    "--native", "/root/Strata-data/models/IQ3_S/....gguf",
    "--ple-gguf", "/root/Strata-data/models/IQ3_S/....gguf",
    "--expert-profile", "/root/Strata-maxwell/data/expert-profile.bin",
    "--expert-cache", "988",
    "--prefill", "auto", "--spec", "6", "--spec-min-p", "0.5",
    "--mtp", "/root/Strata-data/mtp/rt",
    "--max-context", "128000", "--kv", "k8v4", "--mmap-experts",
    "--maxwell-devices", "1,2,3,4,5,6,7,8",
    "--maxwell-reserve-mib", "250", "--spec-split"
  ],
  "env": {
    "STRATA_EMBED_VRAM": "1",
    "STRATA_MW_MT": "1",
    "STRATA_MW_MT_T": "256x4",
    "STRATA_MW_NTM": "256",
    "STRATA_MW_TWOPHASE": "1",
    "STRATA_MW_AD2H": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "4,0,1,2,3,5,6,7,8"
  }
}
```

`api_key` must be a real secret at run time. Keep the file mode `0600` and out of git — the campaign
greps every archived result for `api_key`/`Bearer` and committed none.

Defaults worth knowing, because they are what makes the tier do its work at **prefill** as well as
decode: `STRATA_MW_MT=1`, `STRATA_MW_MT_T=256x4`, `STRATA_MW_NTM=256`, `STRATA_MW_TWOPHASE=1`,
`STRATA_MW_AD2H=1`. With these, the log says
`strata prefill: 8 Maxwell tier(s) compute their experts' rows of every prompt chunk (the arena is
not read for them)` — the M10s prefill, not just decode.

Serve it:

```sh
/root/Strata-maxwell/.venv/bin/python serve/server.py --engine strata \
  --config /root/Strata-maxwell/strata-iq3_s-prod.json --port 8080
```

The order of `CUDA_VISIBLE_DEVICES` is load-bearing: `4,0,1,2,3,5,6,7,8` makes the 3060 CUDA0 and the
dies CUDA1..8, which is what `--maxwell-devices 1,2,3,4,5,6,7,8` expects. Swap it and the engine
will try to hold expert rows on the 3060 as if it were a Maxwell die.

### 5.3 Switching models and the reserve

The server's systemd unit points at one config. `/root/strata-switch.sh` rewrites `--config`, backs up
the unit, restarts, and waits for `/health`:

```sh
/root/strata-switch.sh iq3        # IQ3_S production   (engine e1de86e8, cache 988, 8 tiers, 128k)
/root/strata-switch.sh iq2        # IQ2_XS on the same new engine, cache 1800 (verified non-regressing)
/root/strata-switch.sh iq2-old    # IQ2_XS on engine-mw8, the pre-Iteration-8 production, untouched
/root/strata-switch.sh            # print which config is active, change nothing
```

Return time is the **load** (~250 s from SSD, ~640 s from HDD), not the switch. `engine-mw8` stays on
disk as the reserve.

### 5.4 Load checks

Two lines tell you the tier came up correctly, and both must be present:

```sh
grep -c "round-trip gate PASSED" /path/to/engine.log      # expect 8
grep -m1 "tier(s)" /path/to/engine.log                    # expect "8 Maxwell tier(s)"
curl -sf http://127.0.0.1:8080/health
```

Expected, measured at the production load of 2026-11-08:

```
strata generate: expert cache 1267 slots, 2.45 GiB of VRAM; policy is ...
strata generate: pre-filled 1267 of 1267 slots from the profile in 4.4 s (600 MB/s); slot 0 verified
MaxwellExperts (CUDA1): round-trip gate PASSED (worst rel 1.28e-02, ...)
... eight of these, one per die ...
strata generate: Maxwell GPU 1: 2914 experts, 5.54 GiB (filled in 23.2 s)
strata prefill: 8 Maxwell tier(s) compute their experts' rows of every prompt chunk (the arena is not read for them)
strata serve: 1332 MiB of VRAM free with everything loaded
```

`--expert-cache N` is a request, not a promise: the engine resolves it against the profile and the
free VRAM. **988 in the config produced 1267 real slots here**, and the number to compare against is
the one on the `expert cache` line, not the one in the config.

Then a smoke request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer LA_TUA_API_KEY" \
  -d '{"model":"strata","max_tokens":64,"temperature":0,
       "messages":[{"role":"user","content":"Say hello in five words."}]}'
```

If the tier silently did not come up, the usual causes are: the `CUDA_VISIBLE_DEVICES` order, a
`--maxwell-devices` list that includes device 0, or fewer than 8 GPUs visible in the container.
Note that `server.py` loads the engine in its own process: the engine's load lines go to the `log`
file named in the config, not to the server's stdout.

### 5.5 Re-measuring

The campaign's tools are in `tools/` and are the ones that produced `results/`:

```sh
tools/iq3_bench.py --label iq3_s --runs 3 --labels city,code \
    --engine-log /root/Strata-maxwell/strata-iq3_s-prod.log \
    --api-key-json /root/Strata-maxwell/strata-iq3_s-prod.json
tools/iq3_bench.py --label iq3_s --long-context 32k,64k \
    --api-key-json /root/Strata-maxwell/strata-iq3_s-prod.json
tools/iq3_profile.py --log /root/Strata-maxwell/strata-iq3_s-prod.log   # needs STRATA_MW_LAYERPROF=1 in the log
tools/iq3_power.py --label iq3_s --minutes 10 --out power.csv
```

Read the protocol each one prints before trusting a number. Three details that cost time if you
learn them the hard way:

- The bench runs **thinking ON by default**; `--no-thinking` turns it off, and that is a different
  measurement, not comparable with the 55 tok/s criterion.
- `--long-context` always plants a hidden code word; there is no separate needle flag.
- `iq3_profile.py` does not run the engine, it reads a log the engine wrote with
  `STRATA_MW_LAYERPROF=1`, which costs ~4.7%. Do not leave it on across a bench and compare the two.

---

## 6. Known limits

**The M10's application clocks are not settable.** The driver answers `Setting applications clocks
is not supported`. The dies run at **~1032 MHz base** against an NVML maximum of 1202 MHz, and it is
a base clock, not a reached ceiling: draw is 21–23 W of a 30 W limit and `SW Power Capping` stays
idle, so **raising the 30 W limit cannot buy clock**. Every projection that assumes 1202 MHz is
wrong; the tier's compute ceiling is what is measured, not what NVML advertises.

**Determinism is not bit-for-bit, and never was.** Temperature-0 answers were already not
bit-identical before this work (LOG, Iteration 0a: the baseline itself is not bit-reproducible), and
the campaign did not investigate why. Quality is therefore judged by **kernel parity** (the dot
products against the 3060's path, 1e-5 relative) and by the **round-trip gate** at load, not by
comparing answers between runs.

**Tensor split and E1 stay 4-way.** `--maxwell-split tensor` accepts exactly four devices, and the
E1 completion-flag path refuses more than four (`STRATA_MW_GPUFLAG` / `STRATA_MW_PERSIST`), because
the verifier allocates four flags and the assembly kernel is four-way. Whole-expert split, which is
what production uses, handles eight. An IQ4_NL (`d_type 20`) down is refused in both paths rather
than read with the wrong layout.

**AER on the second M10's bridge is observed, not explained.** See §2: correctable `RxErr` only, on
`0000:09:11.0`, GPUs always clean, no correlation with load or errors. Followed, not cured.

**The second card's uplink is Gen1 x16 = 3.14 GB/s** (host→device, measured), against 12.05 GB/s
for the 3060's Gen3 x16. Every expert row a die does not hold locally has to arrive through that
link, shared by the card's four dies. The forcing is a platform choice (electromagnetic
compatibility), not a software limit, and no engine change can recover it.

**The 55 tok/s criterion is a FAIL, formally.** 54.30 measured, 0.70 short, SE ~0.80 — inside the
noise but not met. It is written as FAIL here and everywhere in the campaign.

**`STRATA_MW_LAYERPROF=1` costs ~4.7%** on the bench. It was on during some early windows and off
for every number quoted in §4; do not mix arms.

---

## 7. Credits

This fork is a thin layer on a lot of other people's work.

- **[Strata](https://github.com/Niko1221/Strata)** — Niko1221 and the Strata contributors. The
  engine, the server, the CUDA backend, the whole architecture. MIT.
- **[Strata-maxwell](https://github.com/nuke1984/Strata-maxwell)** — nuke1984. The Maxwell tier this
  fork extends: the whole-expert split, the S2/Q2_0 down kernel, the IQ2 gate/up kernels, the
  round-trip gate and the safety guards. Without that foundation there is nothing here to extend.
- **[Qwen](https://github.com/QwenLM)** — the Qwen team, for Qwen3.8-Flash-Next.
- **[ISTA-DASLab](https://github.com/ISTA-DASLab)** — for the GSQ-RCO quantization and the IQ3_S pack
  this work runs.
- **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** — for the `block_iq*` types, their
  reference dequantization, and the dot-product layouts the 3060 kernels and therefore the Maxwell
  transcriptions follow.

The mistakes in this branch are Craicek's, not theirs.

## 8. License

The code stays **MIT**, as upstream. `LICENSE` and the original copyright notices are untouched. The
changes this fork makes are enumerated in the Ours/Upstream table in §1 and detailed, with
measurements, in [LOG.md](LOG.md).

---

*Craicek — `https://github.com/BitMakerMan`*