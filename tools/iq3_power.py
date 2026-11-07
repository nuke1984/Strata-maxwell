"""tools/iq3_power.py - what the cards are doing while the campaign's bench runs.

Iteration 6 asks whether the M10 tier is power-capped: the soak did 37.5 tok/s where the bench did 44 on the same
engine, and a sustained-cap explanation is the first hypothesis. The operator's sampler is

    nvidia-smi --query-gpu=index,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,\\
        clocks_throttle_reasons.active --format=csv -lms 1000

which loops forever; this tool runs it for a chosen window, timestamps every poll, writes the CSV, and prints the
summary the report needs (power mean/median/max, SM clock, temperature, throttle reasons seen, total draw against
the sum of the limits).

    python tools/iq3_power.py --label iq3_s-city --minutes 3 --out docs/iq3/results/16-power-iq3_s-city.csv
    python tools/iq3_power.py --parse docs/iq3/results/16-power-iq3_s-city.csv     # summarize an existing run

It changes nothing: read-only queries, no clocks, no limits, no state. Setting power limits is the operator's alone
(30 W on the M10s here, against a 53 W default: not this tool's business).
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

QUERY = ("index,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,"
         "clocks_throttle_reasons.active")
FIELDS = QUERY.split(",")

# the driver's bit names for clocks_throttle_reasons.active
THROTTLE = {
    0x1: "GpuIdle", 0x2: "SwPowerCap", 0x4: "HwSlowdown", 0x8: "SyncBoost",
    0x10: "SwThermalSlowdown", 0x20: "HwThermalSlowdown", 0x40: "HwPowerBrakeSlowdown",
    0x80: "DisplayClockSetting",
}


def num(text: str) -> float | None:
    """'8.08 W' -> 8.08, '135 MHz' -> 135.0, '0x0...1' -> None (handled elsewhere)."""
    part = text.strip().split()[0] if text.strip() else ""
    try:
        return float(part)
    except ValueError:
        return None


def parse_row(line: str) -> dict | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(FIELDS) or not parts[0].lstrip("-").isdigit():
        return None
    row = {
        "gpu": int(parts[0]),
        "draw_w": num(parts[1]),
        "limit_w": num(parts[2]),
        "sm_mhz": num(parts[3]),
        "mem_mhz": num(parts[4]),
        "temp_c": num(parts[5]),
        "throttle": parts[6],
    }
    return row


def throttle_names(mask: str) -> list[str]:
    try:
        bits = int(mask, 16)
    except ValueError:
        return [mask]
    if bits == 0:
        return []
    return [name for bit, name in THROTTLE.items() if bits & bit]


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "no samples"
    by_gpu: dict[int, list[dict]] = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)
    out = []
    total_draw = 0.0
    total_limit = 0.0
    for gpu in sorted(by_gpu):
        rs = by_gpu[gpu]
        draw = [r["draw_w"] for r in rs if r["draw_w"] is not None]
        lim = [r["limit_w"] for r in rs if r["limit_w"] is not None]
        sm = [r["sm_mhz"] for r in rs if r["sm_mhz"] is not None]
        tmp = [r["temp_c"] for r in rs if r["temp_c"] is not None]
        masks: dict[str, int] = {}
        for r in rs:
            masks[r["throttle"]] = masks.get(r["throttle"], 0) + 1
        mean_draw = statistics.fmean(draw) if draw else 0.0
        total_draw += mean_draw
        if lim:
            total_limit += statistics.fmean(lim)
        reasons = []
        for mask, count in sorted(masks.items(), key=lambda kv: -kv[1]):
            names = throttle_names(mask)
            label = "+".join(names) if names else "none"
            reasons.append(f"{label} {100.0 * count / len(rs):.0f}%")
        out.append(
            f"GPU {gpu}: draw mean {mean_draw:6.1f} max {max(draw) if draw else 0:6.1f} W, "
            f"limit {statistics.fmean(lim) if lim else 0:6.1f} W, "
            f"SM {statistics.fmean(sm) if sm else 0:6.0f} MHz (max {max(sm) if sm else 0:.0f}), "
            f"temp mean {statistics.fmean(tmp) if tmp else 0:4.1f} max {max(tmp) if tmp else 0:4.1f} C, "
            f"{len(rs)} samples; throttled: {', '.join(reasons)}")
    out.append(f"total draw (mean per GPU, summed): {total_draw:.1f} W; "
               f"sum of the limits: {total_limit:.1f} W")
    return "\n".join(out)


def sample(label: str, minutes: float, out_path: str, interval_ms: int) -> int:
    if shutil.which("nvidia-smi") is None:
        print("no nvidia-smi on this machine: nothing to sample", file=sys.stderr)
        return 2
    # -lms takes its value as a SEPARATE argv ("-lms", "1000"): as one string the driver answers
    # "Option -lms 1000 is not recognized", exits, and the sampler would report a silent 0 samples.
    cmd = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader", "-lms", str(interval_ms)]
    print(f"# power sampling, {label}, {minutes} min, one poll every {interval_ms} ms")
    print("  " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    rows: list[dict] = []
    lines: list[str] = [f"# {label}, query {QUERY}, interval {interval_ms} ms"]
    t0 = time.time()
    deadline = t0 + minutes * 60
    try:
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            r = parse_row(line)
            if r is None:
                continue
            elapsed = time.time() - t0
            r["t"] = round(elapsed, 1)
            rows.append(r)
            lines.append(f"{elapsed:8.1f}," + line.strip())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    err_text = ""
    if proc.stderr is not None:
        try:
            err_text = proc.stderr.read().strip()
        except (OSError, ValueError):
            err_text = ""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    if not rows:
        # A sampler that collected nothing is a broken measurement, not an idle machine: say why and
        # fail, so a bench never goes into the report with a power column that was never sampled.
        print("*** the sampler collected NO samples: the measurement did not happen", file=sys.stderr)
        if err_text:
            print("--- nvidia-smi said ---", file=sys.stderr)
            print(err_text, file=sys.stderr)
        return 2
    print(summarize(rows))
    print(f"\n{len(rows)} samples over {time.time() - t0:.0f} s, CSV at {out_path}")
    return 0


def parse_file(path: str) -> int:
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # A raw nvidia-smi line parses as-is; sample()'s own CSV prepends the elapsed seconds, which
        # makes the first field non-integer.  Try the line unchanged FIRST: stripping the first field
        # from a raw line would silently drop the GPU index (and 8.08 would look like a GPU number).
        r = parse_row(line)
        if r is None and "," in line:
            r = parse_row(line.split(",", 1)[1])
        if r is not None:
            rows.append(r)
    print(f"# {path}: {len(rows)} samples")
    print(summarize(rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", default="power")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--out", default="power.csv")
    ap.add_argument("--parse", default="", help="summarize an existing CSV instead of sampling")
    a = ap.parse_args()
    if a.parse:
        return parse_file(a.parse)
    return sample(a.label, a.minutes, a.out, a.interval_ms)


if __name__ == "__main__":
    sys.exit(main())