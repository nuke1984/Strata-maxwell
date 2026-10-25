# Strata-Maxwell

An optimization of **[Strata](https://github.com/Niko1221/Strata)** for a fixed rig: **NVIDIA
RTX 3060 12 GB + Tesla M10**, serving the **Qwen3.8-Flash-Next** MoE (24,576 routed experts,
~33.7 GB quantized).

This is a **proof of work**: it checks whether the Tesla M10 — a cheap, everywhere-available
datacenter cast-off — can usefully offload MoE routed experts. The goal is M10-based expert
offload for medium-big MoE models.

## Why the M10

In October 2026 the Tesla M10 sells for about **$100-120 on the backmarket** and carries
**32 GB of GDDR5** — memory cheaper per GB than DDR4 system RAM, on a single board. If it can
hold a model's routed experts, medium-big MoE models become affordable to serve.

## The architecture (fixed)

**Hardware (one rig, nothing varies):**

- **RTX 3060 (Ampere)** — the dense spine: attention, dense layers, the KV cache (k8v4), the
  MTP draft head, PLE — plus a **1874-slot routed-expert cache (2.53 GiB)**.
- **Tesla M10** — four Maxwell GK210 dies, ~7.6 GiB usable each = **30.5 GB expert tier**,
  holding the **other 22,702 routed experts**, resident in VRAM at load.
- **Host** — 4-core AVX2 CPU, 39 GB RAM. In production it holds no experts at all.

**Software:**

- Base: upstream **Strata v0.2.6** (engine 0.1.51), branch `maxwell-0.2.6`, tag
  `maxwell-0.2.6-port-production-OK`.
- The **Maxwell tier** (new): loads the profile-ranked routed experts into the M10 dies at
  start, verifies every blob with a round-trip quantization gate, and serves all 3060-cache
  misses from M10 VRAM — no file reads, no page cache, fully deterministic.
- Six environment knobs tune the tier: `STRATA_MW_MT`, `STRATA_MW_MT_T`, `STRATA_MW_NTM`,
  `STRATA_MW_TWOPHASE`, `STRATA_MW_AD2H`, `STRATA_MW_FASTCOMPLETE`.
- MTP speculative decoding (spec 6); PLE n-gram lookups over direct I/O.

How to build and run it: **[docs/MAXWELL.md](docs/MAXWELL.md)**.

```text
                       prompt
                          |
                          v
          +---------------------------------+
          |  RTX 3060 (sm_86)               |
          |  dense spine, KV k8v4, MTP      |
          |  expert cache: 1874 experts     |
          +---------------------------------+
                          |
      cache misses: the other 22,702 experts
             |                         |
             v                         v
   +-------------------+     +-------------------+
   | setup A           |     | setup B           |
   | Tesla M10         |     | CPU RAM           |
   | 4x GK210, 30.5 GB |     | 32.2 GB locked    |
   | resident tier     |     | resident copy     |
   +-------------------+     +-------------------+
```

**The M10 tier was written with GLM 5.3 (Z.ai).**

## The test: M10 VRAM vs CPU RAM

**The test's specific purpose: serve the model at its full 256K context.** At that context the
KV cache consumes most of the 3060's 12 GB, leaving only ~2.5 GiB for experts — so **22,702 of
the 24,576 routed experts (92%) must be offloaded**, and their main load falls on the M10
(setup A) or on host RAM (setup B). This is the particular test that measures the effective
cost of offloading experts, and whether the M10's 32 GB can carry most of a big MoE's experts.

Same host, same model, **equal expert split** — the one comparison that counts, both sides
fully deterministic (every expert in real memory, no page cache, no file reads after load).

| | **Setup A — M10** | **Setup B — CPU RAM** |
|---|---|---|
| context | **256K** (the model's full length) | **256K** (the model's full length) |
| KV cache | k8v4 | k8v4 |
| binary | this repo | stock upstream v0.2.6 |
| recipe | production (spec 6, min-p 0.5, spec-split + knobs) | stock (spec 4 default) |
| 3060 expert cache | 1874 experts | 1874 experts (identical, log-verified) |
| the other 22,702 experts | **M10 VRAM**, resident at load | **page-locked host RAM**, resident at load |
| PLE | direct (pinned explicit) | direct (pinned explicit) |

Protocol: three fresh 2.7k prompts + one 10k diverse prompt (all 0-reused, verified in the
engine logs), two 300-token greedy decodes, full quality gates on both sides — all green.

| tok/s (engine-log) | **Setup A — M10** | **Setup B — CPU RAM** |
|---|---:|---:|
| prefill 2.7k, fresh ×3 | 148.5 / 144.5 / 154.0 | 99.9 / 430.3 / 434.9 |
| prefill 10k diverse | 154.3 | 275.3 |
| decode, 300 tokens ×2 | **69.5 / 70.1** | 35.9 / 52.5 |

**Verdict:** with the experts in M10 VRAM, decode runs at ~1.8× the best CPU-RAM number, fully
deterministic, and leaves all 39 GB of host RAM free. Host RAM wins prefill (~2.8× on fresh
2.7k prompts; the CPU's wide prefill windows against the M10 tier's 64-token windows). For
interactive serving the M10 tier is the production choice on this rig.

The implication goes past this one rig: good results with the experts offloaded to an M10
**open the way to very cheap serving hardware** — a $100 cast-off card carries the experts —
and **free the host's CPU and RAM for other compute**.

## Limits & open questions

- Measured on exactly **one rig and one model** (Qwen3.8-Flash-Next, IQ2_XS pack). No other
  model, host GPU, or M10 count has numbers behind it.
- The M10 tier prefills in 64-token windows: host RAM wins prompt speed ~2.8× (measured);
  decode is where the M10 pays (~1.8×). Prompt-heavy, decode-light workloads would pick B.
- **Never tried**: a mixed tier — M10 for the decode-hot experts + RAM for prefill breadth.
- A full load takes ~4-6 minutes; the tier must see the M10's dies as CUDA devices with
  enough free VRAM (verified at load — the engine refuses to serve half-loaded).

## Use it freely

The code is MIT licensed, same as Strata (see [LICENSE](LICENSE)). Fork it, run your own
experiments — other cards, other MoE models, other expert splits, other context lengths — and
treat `bench/results/` as the template for recording what you measure (the protocol script is
[`bench/prefarm.py`](bench/prefarm.py), the index is
[`bench/results/README.md`](bench/results/README.md)). Contributions and reproductions are
welcome.

Full records: [`WORKLOG.md`](WORKLOG.md) and [`bench/results/`](bench/results/) (the
head-to-head with every number and its load-log evidence:
[`bench/results/2026-10-25-m10-vs-standard/README.md`](bench/results/2026-10-25-m10-vs-standard/README.md)).
