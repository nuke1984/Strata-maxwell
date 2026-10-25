
## 2026-10-25 (the 0.2.6 port) - the Maxwell layer on Strata 0.2.6, verified live
- Branch maxwell-0.2.6 cut from upstream v0.2.6; our tier grafted with the confirmed drops: the
  DRED arm, the PFPIPE machinery, the ORDER_PROBE instrumentation, and the losing kernel
  geometries (256x8/GACT/E=2/E=8) are NOT in 0.2.6.  The shipped tier + six knobs + hardening +
  tools + tests + records carried.  Upstream's 0.2.3-0.2.6 changes (DFlash2 taps, the dflash2-kern
  warm window, the 0.2.5 prompt-path restructuring, serve changes) coexist cleanly.
- VERIFIED LIVE on the rig (same launch recipe, same config): parity 0 failures; ALL gates green
  (2.7k answers, 10k topics, needle 2/2, arithmetic, decode lists).
- Measured vs the 0.2.2-based stack (same day, same protocol): 2.7k 153.6/154.6 (was 157.5/157.9),
  10k 154.8 (was 155.3 - par), decode 69.2/70.2 (was 72.0-72.6).  The deltas (~2-3% on 2.7k and
  decode) are measured and OPEN: candidates are 0.2.5's decode launch-batching and cache-sizing
  changes interacting with the tier (expert_slots 1874 vs 1800; the INFO line's spec=8 is computed
  identically in both versions - not the cause).  A follow-up tuning pass can chase them; the port
  itself is functionally verified.
- LAUNCH HYGIENE (bit us once): drain ALL FIVE GPUs before relaunch (a GPU0-only check left the
  M10s holding parity's memory; the tier's 8 GB/die open then failed silently at "policy is").
