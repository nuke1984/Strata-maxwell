# 2026-10-21 - the v0.2.2 port (upstream engine 0.1.47)

Merge of the full Maxwell layer (maxwell-0.1.45 @ 221b5172) into upstream v0.2.2 (tag v0.2.2, 011c9e60).
7 conflicts resolved (see merge commit e734bd38). No kernel or numerics change to the tier.

## Gates (all on the production config strata-iq2xs-embed-vram-split.json, --spec-split)

- build: clean (strata, maxwell_iq_parity, iq_gup_bench); upstream nv_defaults (sm_86 table) in the build
- maxwell_iq_parity: 0 failures (fasc 3.09e-07, rel 8.38e-03 - identical values to the 0.1.45 base)
- iq_gup_bench layer 0: 294.6 us/launch, 21.4 GB/s - identical to 0.1.45 (294.6)
- load: 231 s (0.1.45 family: 424-554 s; the new page-cache file tier reads less)
- Maxwell round-trip gate at open: PASSED (worst rel 1.15e-02, 5675 blobs)
- fixed answers: 3/3 (2+2, Paris, 17 prime)
- 3x3 (temp 0, 512 tok): story 47.5 / fib 59.2 / code 52.7 = mean 53.1 tok/s; TTFT 22.18 s (1,546-tok prompt)
  [0.1.45 same-week split family: 52.44 peak mean, TTFT 22.5 s -> parity within noise, +1.3%]
- quality: context recall+negative 5/5 at ~9k ctx (no-think); 8x concurrent -> 1 distinct output of 8;
  prefix reuse 57/62 cached; A/B creative diverges run-to-run = the same pre-existing base property as 0.1.45
- soak 30 min x 4 threads, temp 0.7, mixed 128-384 tok: **559 requests, 0 errors, 73,818 tokens** (~41 tok/s sustained output incl. gaps; GPUs at the 0.1.45 duty: 3060 89%, M10 54-59%; 100% expert cache hit)

## Behavior changes carried by the upstream base (not port defects)

- the chat template defaults to THINKING: short max_tokens can be eaten by reasoning; the answer is correct
  with room (mt>=96) or via "chat_template_kwargs": {"enable_thinking": false} (verified: 0 reasoning chars,
  direct answer). serve/server.py now exposes reasoning_content.
- new file tier engaged: page-cached experts from the mapping (STRATA_COLD_OVERLAP=4), gated IO prefetch -
  load faster, decode parity.

## Data files

- bench-port-022-split.json: the 3x3+TTFT run
- soak022.py: the soak driver (30 min, 4 threads)
