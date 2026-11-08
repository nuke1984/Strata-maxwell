"""tools/iq3_profile.py - read the engine's own per-layer timings out of a log, split by down type.

Iteration 6 asks where the IQ3_S's ~20% goes: the IQ4_NL down (d_type 20), the IQ3 gate/up, or outside
the M10s.  With STRATA_MW_LAYERPROF=1 the engine prints its phase averages and, since Iteration 6, also
splits the card's own span by the layer's DOWN type.  This tool reads such a log and puts the numbers
side by side; it does not touch the engine.

    python tools/iq3_profile.py --log /root/Strata-maxwell/strata-iq3_s-profile.log
    python tools/iq3_profile.py --dump                # print the layer map this tool uses

The lines it knows (see src/core/maxwell_experts.cpp, src/core/expert_source.cpp, src/core/verify.cpp):

    MWPROF m10 CUDAk layers N: jobs/L .. distinct/L .. MB/L .. ideal .. kernel .. [gup .. + down ..], sync-wait ..
    MWPROF m10type CUDAk down20|down42 layers N: MB/L .., kernel .. us/L [gup .. + down ..], sync-wait ..
    MWPROF layer N: plan .. actq .. jobs .. run .. us/layer (dispatch span ..)
    MWPROF vlayer N: ring .. pool .. post .. us/layer

The `m10type` lines exist only for a pack that mixes down types, and only when both buckets have spans,
so an IQ2_XS log keeps the old output.  An IQ3_S log gets one down20 and one down42 line per card.

Measured map (from /root/Strata-data/packs/iq3_s/native_experts.txt, 48 layers, 512 experts each):
39 layers carry down 20 (IQ4_NL) and 9 carry down 42 (Q2_0).
"""
from __future__ import annotations

import argparse
import re

# IQ3_S, verified against the pack's native_experts.txt: 39 down-20 + 9 down-42 = 48 layers.
DOWN20 = sorted([0, 2, 3, 4, 5, 6, 7, 9, 11, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29,
                 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])
DOWN42 = sorted([1, 8, 10, 12, 13, 20, 21, 30, 36])
N_LAYERS = 48

M10 = re.compile(
    r"MWPROF m10 CUDA(\d+) layers ([0-9.]+): jobs/L ([0-9.]+) distinct/L ([0-9.]+), ([0-9.]+) MB/L, "
    r"ideal ([0-9.]+) us/L, kernel ([0-9.]+) us/L \(slack ([0-9.]+)x\) \[gup ([0-9.]+) \+ down ([0-9.]+)\], "
    r"sync-wait ([0-9.]+) us/L")
M10TYPE = re.compile(
    r"MWPROF m10type CUDA(\d+) (down20|down42) layers ([0-9.]+): ([0-9.]+) MB/L, kernel ([0-9.]+) us/L "
    r"\[gup ([0-9.]+) \+ down ([0-9.]+)\], sync-wait ([0-9.]+) us/L")
LAYER = re.compile(
    r"MWPROF layer (\d+): plan ([0-9.]+) actq ([0-9.]+) jobs ([0-9.]+) run ([0-9.]+) us/layer "
    r"\(dispatch span ([0-9.]+)\)")
VLAYER = re.compile(r"MWPROF vlayer (\d+): ring ([0-9.]+) pool ([0-9.]+) post ([0-9.]+) us/layer")


def down_of(layer: int) -> int:
    """The down type of one layer; everything not in DOWN20 is Q2_0 (42)."""
    return 20 if layer in DOWN20 else 42


def read_log(path: str) -> dict:
    m10, m10type, layers, vlayers = [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "MWPROF" not in line:
                continue
            if (m := M10.search(line)):
                m10.append({"gpu": int(m.group(1)), "layers": float(m.group(2)),
                            "jobs": float(m.group(3)), "distinct": float(m.group(4)),
                            "mb": float(m.group(5)), "ideal": float(m.group(6)),
                            "kernel": float(m.group(7)), "slack": float(m.group(8)),
                            "gup": float(m.group(9)), "down": float(m.group(10)),
                            "sync": float(m.group(11))})
            elif (m := M10TYPE.search(line)):
                m10type.append({"gpu": int(m.group(1)), "tag": m.group(2), "layers": float(m.group(3)),
                                "mb": float(m.group(4)), "kernel": float(m.group(5)),
                                "gup": float(m.group(6)), "down": float(m.group(7)),
                                "sync": float(m.group(8))})
            elif (m := LAYER.search(line)):
                layers.append({"n": int(m.group(1)), "plan": float(m.group(2)), "actq": float(m.group(3)),
                               "jobs": float(m.group(4)), "run": float(m.group(5)),
                               "span": float(m.group(6))})
            elif (m := VLAYER.search(line)):
                vlayers.append({"n": int(m.group(1)), "ring": float(m.group(2)),
                                "pool": float(m.group(3)), "post": float(m.group(4))})
    return {"m10": m10, "m10type": m10type, "layer": layers, "vlayer": vlayers}


def pack_share() -> str:
    n20, n42 = len(DOWN20), len(DOWN42)
    return (f"# pack: {n20 + n42} expert layers, {n20} with down-20 (IQ4_NL) and {n42} with down-42 (Q2_0)\n"
            f"# down-20 layers: {DOWN20}\n# down-42 layers: {DOWN42}")


def aggregate_by_card(rows: list[dict]) -> list[dict]:
    """One row per card, weighted over the windows it printed (the engine prints every 1440 layers)."""
    by_gpu: dict[int, list[dict]] = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)
    out = []
    for gpu in sorted(by_gpu):
        rs = by_gpu[gpu]
        n = sum(r["layers"] for r in rs)
        if n <= 0:
            continue
        w = lambda key: sum(r[key] * r["layers"] for r in rs) / n  # noqa: E731
        out.append({"gpu": gpu, "windows": len(rs), "layers": n,
                    "jobs": w("jobs"), "distinct": w("distinct"), "mb": w("mb"),
                    "ideal": w("ideal"), "kernel": w("kernel"), "slack": w("slack"),
                    "gup": w("gup"), "down": w("down"), "sync": w("sync")})
    return out


def aggregate_by_type(rows: list[dict]) -> list[dict]:
    """One row per (card, down type), weighted over that card's windows."""
    by_key: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["gpu"], r["tag"]), []).append(r)
    out = []
    for (gpu, tag), rs in sorted(by_key.items()):
        n = sum(r["layers"] for r in rs)
        if n <= 0:
            continue
        w = lambda key: sum(r[key] * r["layers"] for r in rs) / n  # noqa: E731
        out.append({"gpu": gpu, "tag": tag, "windows": len(rs), "layers": n,
                    "mb": w("mb"), "kernel": w("kernel"), "gup": w("gup"),
                    "down": w("down"), "sync": w("sync")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", default="", help="engine log (with STRATA_MW_LAYERPROF=1)")
    ap.add_argument("--dump", action="store_true", help="print the layer map and stop")
    a = ap.parse_args()

    print(pack_share())
    if a.dump:
        for layer in range(N_LAYERS):
            print(f"layer {layer:2d} down {down_of(layer)}")
        return 0
    if not a.log:
        ap.error("--log is required unless --dump is given")

    data = read_log(a.log)
    m10, m10type, layers, vlayers = data["m10"], data["m10type"], data["layer"], data["vlayer"]
    print(f"\n# {a.log}: {len(m10)} m10 lines, {len(m10type)} m10type lines, "
          f"{len(layers)} layer lines, {len(vlayers)} vlayer lines")

    if m10:
        cards = aggregate_by_card(m10)
        print("\n## per card (MWPROF m10), weighted over the windows each card printed")
        print(f"{'card':>5} {'windows':>8} {'layers':>7} {'jobs/L':>7} {'dist/L':>7} {'MB/L':>7} {'ideal':>7} "
              f"{'kernel':>7} {'slack':>6} {'gup':>7} {'down':>7} {'sync':>7}")
        for r in cards:
            print(f"{r['gpu']:>5} {r['windows']:>8} {r['layers']:>7.0f} {r['jobs']:>7.2f} {r['distinct']:>7.2f} "
                  f"{r['mb']:>7.3f} {r['ideal']:>7.0f} {r['kernel']:>7.0f} {r['slack']:>6.2f} "
                  f"{r['gup']:>7.0f} {r['down']:>7.0f} {r['sync']:>7.0f}")
        avg = lambda key: sum(r[key] for r in cards) / len(cards)  # noqa: E731
        krn, gup, down, sync = avg("kernel"), avg("gup"), avg("down"), avg("sync")
        print(f"\nmean over the cards: kernel {krn:.0f} us/L = gup {gup:.0f} + down {down:.0f}; "
              f"sync-wait {sync:.0f} us/L")
        if krn > 0:
            print(f"  gup is {100 * gup / krn:.1f}% of the kernel span, down {100 * down / krn:.1f}%, "
                  f"ideal (83 GB/s floor) {avg('ideal'):.0f} us/L -> slack {avg('slack'):.2f}x")
        print(f"  the host waits {sync:.0f} us/L beyond the card's own span")

    if m10type:
        cards = aggregate_by_type(m10type)
        print("\n## per card, split by down type (MWPROF m10type), weighted over the windows")
        print(f"{'card':>5} {'down':>7} {'windows':>8} {'layers':>7} {'MB/L':>7} {'kernel':>7} {'gup':>7} "
              f"{'down us':>7} {'sync':>7}")
        for r in cards:
            print(f"{r['gpu']:>5} {r['tag']:>7} {r['windows']:>8} {r['layers']:>7.0f} {r['mb']:>7.3f} "
                  f"{r['kernel']:>7.0f} {r['gup']:>7.0f} {r['down']:>7.0f} {r['sync']:>7.0f}")
        # The comparison that answers the question: the same span, per layer, on the two down types.
        for tag in ("down20", "down42"):
            rs = [r for r in cards if r["tag"] == tag]
            if not rs:
                continue
            n = sum(r["layers"] for r in rs)
            krn = sum(r["kernel"] * r["layers"] for r in rs) / n
            gup = sum(r["gup"] * r["layers"] for r in rs) / n
            dn = sum(r["down"] * r["layers"] for r in rs) / n
            mb = sum(r["mb"] * r["layers"] for r in rs) / n
            sy = sum(r["sync"] * r["layers"] for r in rs) / n
            print(f"  {tag}: weighted over {n:.0f} layer-spans and {len(rs)} cards -> "
                  f"{mb:.3f} MB/L, kernel {krn:.0f} us/L (gup {gup:.0f} + down {dn:.0f}), "
                  f"sync-wait {sy:.0f} us/L")
        both = {t: [r for r in cards if r["tag"] == t] for t in ("down20", "down42")}
        if both["down20"] and both["down42"]:
            def wavg(tag, key):
                rs = both[tag]
                n = sum(r["layers"] for r in rs)
                return sum(r[key] * r["layers"] for r in rs) / n
            dn20, dn42 = wavg("down20", "down"), wavg("down42", "down")
            k20, k42 = wavg("down20", "kernel"), wavg("down42", "kernel")
            print(f"\n  down stage: down20 {dn20:.0f} us/L vs down42 {dn42:.0f} us/L "
                  f"({dn20 - dn42:+.0f}); whole span: {k20:.0f} vs {k42:.0f} us/L ({k20 - k42:+.0f})")
            print("  (per-layer, per-card; the two groups differ in expert COUNT per layer, so read MB/L too)")

    if layers:
        print("\n## dispatch phases (MWPROF layer)")
        print(f"{'windows':>8} {'plan':>7} {'actq':>7} {'jobs':>7} {'run':>7} {'span':>7}  us/layer")
        for r in layers:
            print(f"{r['n']:>8} {r['plan']:>7.1f} {r['actq']:>7.1f} {r['jobs']:>7.1f} {r['run']:>7.1f} "
                  f"{r['span']:>7.1f}")
        last = layers[-1]
        share = {k: 100.0 * last[k] / last["span"] for k in ("plan", "actq", "jobs", "run") if last["span"] > 0}
        print("\n  share of the dispatch span: " +
              ", ".join(f"{k} {v:.1f}%" for k, v in share.items()))

    if vlayers:
        print("\n## verifier phases (MWPROF vlayer)")
        print(f"{'windows':>8} {'ring':>7} {'pool':>7} {'post':>7}")
        for r in vlayers:
            print(f"{r['n']:>8} {r['ring']:>7.1f} {r['pool']:>7.1f} {r['post']:>7.1f}")

    if not (m10 or m10type or layers or vlayers):
        print("\nno MWPROF lines: was STRATA_MW_LAYERPROF set for this run?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())