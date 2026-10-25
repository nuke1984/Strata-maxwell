# bench/results — index

**This fork's campaigns** (the Maxwell line, measured on this rig: RTX 3060 + Tesla M10):

- [`2026-10-25-m10-vs-standard/`](2026-10-25-m10-vs-standard/) — **the head-to-head**: M10 VRAM
  vs CPU RAM at the model's full 256K context, equal expert split (1874 cache + 22,702
  offloaded experts on both sides), both sides fully resident. The README's numbers come from
  here. (The mmap-tier entries inside it are superseded, noted in place.)
- [`2026-10-22-token-passage-anatomy/`](2026-10-22-token-passage-anatomy/) — how a token
  passes the pipeline on the 3060+M10 rig (the 0.2.2 line).
- [`2026-10-22-3060-xray/`](2026-10-22-3060-xray/) — the 3060's VRAM layout at 256K context:
  what fits, what must offload.
- [`2026-10-20-0.2.1-bench/`](2026-10-20-0.2.1-bench/) — the first `strata bench run` on the
  released 0.2.1.
- [`2026-10-20-0.2.1-idle-gpu/`](2026-10-20-0.2.1-idle-gpu/) — what the idle-GPU switches do
  on NVIDIA (the M10's idle dies).

The older 0.2.2-line campaigns (ordering-debt, PLE RAM-vs-direct, the first M10-vs-RAM
head-to-head) live on the `maxwell-0.2.2` branch of this repository.

**Everything else** in this directory is upstream Strata's community benchmark archive,
inherited unchanged with the fork — records other people sent upstream, not measurements
from this rig. `COMMUNITY.md` there is upstream's too.

The benchmark protocol itself is [`../prefarm.py`](../prefarm.py); how to run it is in
[`docs/MAXWELL.md`](../../docs/MAXWELL.md#benchmark).
