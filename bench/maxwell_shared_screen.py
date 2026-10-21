#!/usr/bin/env python3
"""Run the standalone X1 shared-expert screening target; never restart serving."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="first model GGUF shard")
    parser.add_argument("--binary", type=Path, default=Path("build/maxwell_shared_bench"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.iters < 1:
        parser.error("repeats and iters must be positive")
    pattern = re.compile(
        r"RESULT main-native-graph ([0-9.]+) us/L; "
        r"TP-compute-max ([0-9.]+) us/L; TP-host-staged ([0-9.]+) us/L"
    )
    results = []
    for repeat in range(args.repeats):
        for layer, tokens in [(0, 1), (0, 2), (0, 4), (1, 4), (8, 4)]:
            command = [str(args.binary.resolve()), str(args.gguf.resolve()),
                       str(layer), str(tokens), str(args.iters)]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"repeat={repeat + 1}\n{run.stdout}", flush=True)
            match = pattern.search(run.stdout)
            if not match:
                raise RuntimeError("benchmark did not report timing results")
            results.append({
                "repeat": repeat + 1, "layer": layer, "tokens": tokens,
                "native_graph_us": float(match[1]), "tp_compute_max_us": float(match[2]),
                "tp_host_staged_us": float(match[3]), "stdout": run.stdout,
            })
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
