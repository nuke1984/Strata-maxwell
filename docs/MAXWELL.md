# The Maxwell tier — build & run

The Maxwell tier turns a **Tesla M10** (four GK210 dies, 32 GB GDDR5) into a resident expert
store for MoE models: at load it copies the profile-ranked routed experts the host GPU's
cache cannot hold into the M10's VRAM, verifies every blob with a round-trip quantization
gate, and serves every later cache miss from VRAM — no file reads after load, no page cache.

## Build (Linux, CUDA 12.x)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target strata -j4
```

This is the exact configure line of the shipped engine (the host GPU is sm_86). The Maxwell
expert kernel (`src/kernels/cuda/expert_maxwell.cu`) builds in its own library for **sm_50**
— the M10's dies — beside the engine's sm_75+ kernels; nothing extra to pass.

## Hardware notes (learned the hard way, all measured on this rig)

- A full load takes **~4-6 minutes** (pack scan + tier fill). Wait for the health endpoint.
- **Drain every GPU below ~500 MiB before a relaunch**; after a host reboot set persistence
  mode first: `nvidia-smi -pm 1`.
- Small prompts (a few hundred tokens) never reach the chunked prefill path — do not judge
  prefill speed by them.
- **Never measure under a profiler**: CUPTI skewed decode ~17% here.
- This rig's PCIe measures 11.3 GB/s host→device.

## Run

The server takes a JSON config (the real one is gitignored; this is the shape):

```json
{
  "model_name": "qwen3.8-flash-next-iq2_xs",
  "api_key": "YOUR-API-KEY",
  "exe": "engine/strata",
  "cwd": "/path/to/Strata",
  "tokenizer": "/path/to/packs/iq2_xs/tokenizer",
  "lib_dirs": ["/usr/local/cuda/bin", "/usr/local/cuda/lib64"],
  "log": "strata-maxwell.log",
  "port": 8088,
  "host": "127.0.0.1",
  "args": [
    "--pack", "/path/to/packs/iq2_xs",
    "--native", "/path/to/models/iq2_xs/Qwen3.8-Flash-Next-...-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/models/iq2_xs/Qwen3.8-Flash-Next-...-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "1800",
    "--prefill", "auto",
    "--spec", "6", "--spec-min-p", "0.5",
    "--mtp", "/path/to/mtp/rt",
    "--max-context", "256000", "--kv", "k8v4",
    "--maxwell-devices", "1,2,3,4", "--maxwell-reserve-mib", "150",
    "--mmap-experts", "--host-core", "last", "--spec-split"
  ]
}
```

Production launch, with the six tier knobs:

```sh
setsid env STRATA_MW_MT=1 STRATA_MW_MT_T=256x4 STRATA_MW_NTM=256 STRATA_MW_TWOPHASE=1 \
  STRATA_MW_AD2H=1 STRATA_MW_FASTCOMPLETE=1 \
  nohup python serve/server.py --engine strata --config strata-maxwell.json --port 8088 &
```

The six knobs (the production set, selected by the campaigns recorded in
[`WORKLOG.md`](../WORKLOG.md)):

| knob | value | what it arms |
|---|---|---|
| `STRATA_MW_MT` | 1 | the tier's multi-token job batching |
| `STRATA_MW_MT_T` | 256x4 | its tile shape |
| `STRATA_MW_NTM` | 256 | the multi-token batch depth |
| `STRATA_MW_TWOPHASE` | 1 | the two-phase kernel completion |
| `STRATA_MW_AD2H` | 1 | the asynchronous device→host completion path |
| `STRATA_MW_FASTCOMPLETE` | 1 | early completion (measured +2.8% decode) |

**Verify at load** (grep the engine log): `expert cache N slots` (the host GPU's share),
`Maxwell GPU k: N experts, X GiB` (per die), and the round-trip gate's `PASSED` line. If
those lines are missing, the tier did not engage — do not measure.

## Benchmark

[`bench/prefarm.py`](../bench/prefarm.py) is the exact protocol behind the records in
[`bench/results/`](../bench/results/): warmup, three fresh 2.7k prompts, a 10k-token diverse
prompt, needle recall, fixed answers, and two 300-token greedy decodes. Point it at a small
config with the server's URL key:

```sh
cp bench/bench-config.example.json bench/bench-config.json   # edit the key
python3 bench/prefarm.py full bench/bench-config.json
```

Numbers are read from the engine log (`prompt N tokens = ... read in X ms (Y tok/s)`) — the
script prints the gate results; the tok/s come from the log lines themselves.
