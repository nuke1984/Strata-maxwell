#!/usr/bin/env python3
"""Fixed-routing ABBA screening of the Meta tier against legacy routed TP.

Stop serving before running. This measures tier microseconds, not model tok/s.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/maxwell_tp_chain_test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    pattern = re.compile(r"META_BENCH (\d+) ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+)")
    for name, direct, heavy in [("reference", 0, 0), ("direct", 1, 0), ("heavy", 0, 1), ("both", 1, 1)]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_MW_", "STRATA_S2_"))}
        env.update(STRATA_MW_GPUFLAG="2", STRATA_MW_META_DIRECT=str(direct), STRATA_MW_META_HEAVY_FIRST=str(heavy))
        run = subprocess.run([str(args.binary.resolve()), str(args.gguf.resolve()), str(args.pack.resolve()), "--bench-meta"],
                             env=env, capture_output=True, text=True, timeout=180)
        blocks = [list(map(float, match.groups()[1:])) for match in pattern.finditer(run.stdout)]
        row = dict(case=name, direct=direct, heavy=heavy, exit=run.returncode,
                   stdout=run.stdout, stderr=run.stderr, abba_us_per_layer=blocks)
        results.append(row)
        if run.returncode == 0 and len(blocks) == 7:
            gains = [100 * ((a0 + a1) / (b0 + b1) - 1) for a0, b0, b1, a1 in blocks]
            mean = statistics.mean(gains)
            half_ci = 2.446912 * statistics.stdev(gains) / math.sqrt(7)  # t95, df=6
            row.update(mean_gain_percent=mean, ci95_percent=[mean - half_ci, mean + half_ci],
                       legacy_us=statistics.mean((v[0] + v[3]) / 2 for v in blocks),
                       meta_us=statistics.mean((v[1] + v[2]) / 2 for v in blocks))
            print(name, {k: row[k] for k in ("legacy_us", "meta_us", "mean_gain_percent", "ci95_percent")}, flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        if run.returncode or len(blocks) != 7:
            raise RuntimeError(f"{name} failed: {run.stdout}\n{run.stderr}")


if __name__ == "__main__":
    main()
