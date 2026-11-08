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

# The 3060 asks a second question: WHICH throttle reason.  Its `active` mask says HwSlowdown, but that is
# one bit for several mechanisms, so the causes are sampled both ways - the aggregate mask AND the
# individual reason fields, plus the two inputs a thermal answer needs (memory temperature, fan speed).
# Kept as a separate query so the default CSV format the earlier windows wrote stays parseable.
DEEP_QUERY = ("index,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
              "fan.speed,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_slowdown,"
              "clocks_throttle_reasons.hw_power_brake_slowdown,"
              "clocks_throttle_reasons.hw_thermal_slowdown,"
              "clocks_throttle_reasons.sw_power_cap,"
              "clocks_throttle_reasons.sw_thermal_slowdown")
DEEP_FIELDS = DEEP_QUERY.split(",")
# the individual reasons, in the order DEEP_QUERY asks them (field index in the row)
DEEP_REASONS = [(9, "HwSlowdown"), (10, "HwPowerBrake"), (11, "HwThermal"), (13, "SwThermal"),
                (12, "SwPowerCap")]

# The driver's bit names for clocks_throttle_reasons.active (NVML's frozen nvmlClocksThrottleReason
# enum).  This table was WRONG until iteration 7 and every report before it inherited the error: bits
# from 0x2 up were shifted by one, so an M10's ApplicationsClocksSetting read as "SwPowerCap" and the
# 3060's real SwPowerCap read as "HwSlowdown".  The mapping below is pinned by observation, not by
# memory: at the same instant the 3060 reports mask 0x4 with sw_power_cap Active, and an M10 reports
# 0x2 with applications_clocks_setting Active (see test_iq3_power.BitMap).
THROTTLE = {
    0x001: "GpuIdle", 0x002: "ApplicationsClocksSetting", 0x004: "SwPowerCap", 0x008: "HwSlowdown",
    0x010: "SyncBoost", 0x020: "SwThermalSlowdown", 0x040: "HwThermalSlowdown",
    0x080: "HwPowerBrakeSlowdown", 0x100: "DisplayClockSetting",
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


def parse_deep_row(line: str) -> dict | None:
    """A DEEP_QUERY row: the same fields plus memory temp, fan and each throttle reason on its own."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(DEEP_FIELDS) or not parts[0].lstrip("-").isdigit():
        return None
    row = {
        "gpu": int(parts[0]),
        "draw_w": num(parts[1]),
        "limit_w": num(parts[2]),
        "sm_mhz": num(parts[3]),
        "mem_mhz": num(parts[4]),
        "temp_c": num(parts[5]),
        "temp_mem_c": num(parts[6]),
        "fan_pct": num(parts[7]),
        "throttle": parts[8],
        "reasons": {name: parts[i] for i, name in DEEP_REASONS},
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


def summarize_deep(rows: list[dict], active_only: bool = False) -> str:
    """Which throttle REASON, per card: shares of the samples where the card is not GpuIdle.

    `active_only` is the reading that matters for a cause: a card at 135 MHz is not thermally throttled,
    it is asleep, and counting its idle samples in the denominator dilutes every reason.
    """
    if not rows:
        return "no samples"
    by_gpu: dict[int, list[dict]] = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)
    out = []
    for gpu in sorted(by_gpu):
        rs = by_gpu[gpu]
        sel = [r for r in rs if not (int(r["throttle"], 16) & 0x1)] if active_only else rs
        if not sel:
            out.append(f"GPU {gpu}: no {'active ' if active_only else ''}samples")
            continue
        n = len(sel)
        head = (f"GPU {gpu}: {n} samples" + (" (GpuIdle excluded)" if active_only else ""))
        out.append(head)
        for _, name in DEEP_REASONS:
            cnt = sum(1 for r in sel if r["reasons"].get(name) == "Active")
            out.append(f"    {name:<14} {100.0 * cnt / n:5.0f}%" if cnt else
                       f"    {name:<14} {0.0:5.0f}%")
        # the inputs a thermal answer needs: they must be present to be read
        for key, label, unit in (("temp_c", "temp gpu", "C"), ("temp_mem_c", "temp mem", "C"),
                                 ("fan_pct", "fan", "%")):
            vals = [r[key] for r in sel if r.get(key) is not None]
            if vals:
                out.append(f"    {label:<14} mean {statistics.fmean(vals):5.1f} max {max(vals):5.1f} {unit}"
                           + (f"  ({n - len(vals)} N/A)" if len(vals) < n else ""))
            else:
                out.append(f"    {label:<14} not exposed by this driver (N/A on every sample)")
        masks: dict[str, int] = {}
        for r in sel:
            masks[r["throttle"]] = masks.get(r["throttle"], 0) + 1
        agg = ", ".join(f"{'+'.join(throttle_names(m)) or 'none'} {100.0 * c / n:.0f}%"
                        for m, c in sorted(masks.items(), key=lambda kv: -kv[1]))
        out.append(f"    aggregate mask: {agg}")
    return "\n".join(out)


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


def sample(label: str, minutes: float, out_path: str, interval_ms: int, deep: bool = False) -> int:
    if shutil.which("nvidia-smi") is None:
        print("no nvidia-smi on this machine: nothing to sample", file=sys.stderr)
        return 2
    query = DEEP_QUERY if deep else QUERY
    parse = parse_deep_row if deep else parse_row
    # -lms takes its value as a SEPARATE argv ("-lms", "1000"): as one string the driver answers
    # "Option -lms 1000 is not recognized", exits, and the sampler would report a silent 0 samples.
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader", "-lms", str(interval_ms)]
    print(f"# power sampling, {label}, {minutes} min, one poll every {interval_ms} ms"
          + (", DEEP (per-reason throttle bits)" if deep else ""))
    print("  " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    rows: list[dict] = []
    lines: list[str] = [f"# {label}, query {query}, interval {interval_ms} ms"]
    t0 = time.time()
    deadline = t0 + minutes * 60
    try:
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            r = parse(line)
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
    print(summarize(rows) if not deep else summarize_deep(rows))
    if deep:
        print("\n--- the same, GpuIdle excluded (the reading that answers 'which cause') ---")
        print(summarize_deep(rows, active_only=True))
    print(f"\n{len(rows)} samples over {time.time() - t0:.0f} s, CSV at {out_path}")
    return 0


def parse_file(path: str) -> int:
    rows = []
    deep_rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # A raw nvidia-smi line parses as-is; sample()'s own CSV prepends the elapsed seconds, which
        # makes the first field non-integer.  Try the line unchanged FIRST: stripping the first field
        # from a raw line would silently drop the GPU index (and 8.08 would look like a GPU number).
        for attempt in (line, line.split(",", 1)[1] if "," in line else None):
            if attempt is None:
                continue
            rd = parse_deep_row(attempt)
            if rd is not None:
                deep_rows.append(rd)
                break
            r = parse_row(attempt)
            if r is not None:
                rows.append(r)
                break
    print(f"# {path}: {len(rows)} plain samples, {len(deep_rows)} deep samples")
    if rows:
        print(summarize(rows))
    if deep_rows:
        print()
        print(summarize_deep(deep_rows))
        print("\n--- the same, GpuIdle excluded ---")
        print(summarize_deep(deep_rows, active_only=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", default="power")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--out", default="power.csv")
    ap.add_argument("--parse", default="", help="summarize an existing CSV instead of sampling")
    ap.add_argument("--deep", action="store_true",
                    help="also sample the individual throttle reasons, memory temperature and fan speed")
    a = ap.parse_args()
    if a.parse:
        return parse_file(a.parse)
    return sample(a.label, a.minutes, a.out, a.interval_ms, a.deep)


if __name__ == "__main__":
    sys.exit(main())