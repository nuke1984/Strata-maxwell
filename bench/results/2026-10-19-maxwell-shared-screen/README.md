# Maxwell shared-expert TP screening (X1)

This is a standalone kernel/transport experiment, **not a serving optimization**.
The production binary, configuration, expert placement, and spec 4 remain untouched.

## Hardware and model

- Linux KVM guest; RTX 3060 12 GB plus four Tesla M10 dies (sm_50, 8 GB each).
- M10s share a Gen1 x16 uplink, measured previously at approximately 3.3 GB/s.
  No usable P2P. No PCIe settings changed in this experiment.
- NVIDIA driver 580.178.04. Persistence mode reported disabled; each measurement
  burns for two seconds before timing. No CUPTI profiler.
- Strata 0.1.45, source parent `9b5edbfb`, Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS.
- Production remained loaded but was not sent inference requests during the
  microbenchmarks. Its existing allocations remained in VRAM.

## What is measured

`maxwell_shared_bench` reads real shared-expert tensors from the GGUF. The native
3060 comparison calls `shared_expert_multi`, including its quantizers and scalar
gate, through a captured CUDA graph. It is measured in isolation, not as exposed
time on the serving critical path: serving already overlaps it on another stream.

The prototype expands quantized weights to FP16, output-shards gate/up into 160
channels per card, and input-shards down correspondingly. Four tokens share each
weight load. Inputs/accumulators/partials are F32. Each card runs two kernels.
Read-only loads use the device cache, never mapped host activation reads.

- **TP compute:** separate event-timed batches on each card, reporting the slowest
  card. Not a measurement or proof of four-way concurrent execution.
- **TP host-staged:** launch four captured H2D/compute/D2H graphs, then wait on
  their streams and sum pinned partials in fixed rank order on the CPU. This
  includes host/driver synchronization penalties. It does **not** include final
  delivery to the 3060 or application of the scalar gate.
- Neither path overlaps the prototype with routed M10 expert kernels. GPU-raised
  completion flags and compressed-weight M10 kernels are **not** implemented here.
- One layer's weights are reused. FP16 shard weights occupy 2.34 MiB/card, larger
  than an M10's L2; no full 48-layer residency/startup test was performed. Full
  FP16 weights would require 112.5 MiB/card, excluding scratch and runtime overhead.

## Accuracy checks

- Exhaustive GPU half-to-float conversion check against `f16_bits.hpp`: 65,536 bit
  patterns, including subnormals and signed zero. NaNs compare by class, not payload.
- Full FP16-weight/F32-input TP chain versus an independent CPU fp64 calculation:
  required relative L2 error below 2e-5, including a finite-output check.
- The prototype omits production's Q8_1 activation quantization. Its difference
  from production is reported separately and is **not** a parity gate. No generated
  token correctness or full-model parity claim is made for the prototype.

## Results and decision

Three fresh process repetitions, 500 iterations per measurement. Median timings
in microseconds/layer; individual observations are in [timings.json](timings.json).

| Layer | Tokens | Native 3060 graph | TP compute, slowest card | TP host-staged |
|---:|---:|---:|---:|---:|
| 0 | 1 | 22.25 | 60.16 | 99.00 |
| 0 | 2 | 27.96 | 67.23 | 121.07 |
| 0 | 4 | 37.58 | 86.18 | 188.04 |
| 1 | 4 | 40.04 | 86.22 | 188.22 |
| 8 | 4 | 38.06 | 86.17 | 188.72 |

All 15 math checks pass (relative L2 2.34e-7 to 2.66e-7). Half conversion checks
pass. Difference from native Q8_1 output is 1.03–1.20%, not bitwise parity.

**Do not integrate this FP16 prototype:** its compute-only path is already slower
than the existing native shared expert, before transport, scalar gating, or
competition with routed experts on the M10s. This does **not** prove that every
quantized dense-TP kernel or overlap schedule would lose. Those remain unmeasured;
the benchmark is a reproducible screening tool, not a hardware lower-bound proof.

Earlier versions using software half decoding and uncached activation loads were
slower; the table uses PTX half-to-float conversion and read-only cached loads.

After screening: `/health` reports loaded/ok at 256000 context; 391, 9999, and eight
concurrent 391 requests all pass. The serving binary remains SHA256
`171c4035ab503220fd08463205192edb9b70ee52740d9ce975d5deb649636384`.
No restart, full-context ingestion stress test, or new load-time round-trip gate
was performed. Only the CUDA screening target was built; HIP/SYCL were not tested.

## Reproduction

From the repository root:

```bash
cmake --build build --target maxwell_shared_bench -j4
python3 bench/maxwell_shared_screen.py /path/to/model-00001-of-00002.gguf \
  --repeats 3 --iters 500 --output /tmp/opencode/maxwell-shared-screen.json
```

The CUDA-only target is `EXCLUDE_FROM_ALL` and is not linked into the engine.
No `engine/strata` copy or server restart is needed. It requires exactly four
visible sm_50 dies and one sm_70+ main GPU; do not run it during live inference.

## Stable serving anchor

Same-day short prompts, temperature 0, 512-token cap, three repeats each. This
is an anchor only: the prototype was never integrated or A/B-tested in serving.

| Prompt | Per-run tok/s | Mean tok/s |
|---|---|---:|
| tell me a story about AI | 38.0, 39.7, 39.3 | 39.0 |
| explain Fibonacci | 44.8, 45.3, 44.9 | 45.0 |
| write an example code in python, medium difficulty | 45.2, 45.4, 41.9 | 44.17 |

Overall mean: **42.72 tok/s**. Generation length varies (including one code request
that reached the cap entirely in reasoning); this table is not a quality evaluation.
