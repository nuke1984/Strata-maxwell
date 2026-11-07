"""tools/iq3_profile.py's log parsing and layer map, offline.

    python -m unittest tools.test_iq3_profile

The lines below are copies of the engine's printf formats (src/core/maxwell_experts.cpp and
src/core/expert_source.cpp); if a format changes, the test fails rather than the report silently
showing nothing.  The layer map is checked against the pack's census (39 down-20 + 9 down-42).
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import iq3_profile as PR  # noqa: E402

M10 = ("MWPROF m10 CUDA1 layers 1440: jobs/L 12.34 distinct/L 9.87, 3.456 MB/L, ideal 42 us/L, "
       "kernel 55 us/L (slack 1.32x) [gup 33 + down 22], sync-wait 7 us/L")
M10TYPE20 = ("MWPROF m10type CUDA1 down20 layers 1170: 3.900 MB/L, kernel 58 us/L "
             "[gup 34 + down 24], sync-wait 8 us/L")
M10TYPE42 = ("MWPROF m10type CUDA1 down42 layers 270: 1.700 MB/L, kernel 44 us/L "
             "[gup 28 + down 16], sync-wait 4 us/L")
LAYER = "MWPROF layer 1920: plan 10.5 actq 22.3 jobs 4.8 run 60.1 us/layer (dispatch span 97.7)"
VLAYER = "MWPROF vlayer 1920: ring 3.2 pool 1.4 post 8.9 us/layer"


class Aggregate(unittest.TestCase):
    def test_windows_collapse_to_one_row_per_card(self):
        """The engine prints a window every 1440 layers: 496 raw rows must become 8 cards, not 496 lines."""
        rows = [{"gpu": 1, "layers": 1440.0, "jobs": 3.0, "distinct": 2.0, "mb": 4.0, "ideal": 50.0,
                 "kernel": 300.0, "slack": 6.0, "gup": 150.0, "down": 150.0, "sync": 100.0},
                {"gpu": 1, "layers": 1440.0, "jobs": 5.0, "distinct": 4.0, "mb": 6.0, "ideal": 70.0,
                 "kernel": 500.0, "slack": 8.0, "gup": 250.0, "down": 250.0, "sync": 200.0},
                {"gpu": 2, "layers": 1440.0, "jobs": 1.0, "distinct": 1.0, "mb": 2.0, "ideal": 20.0,
                 "kernel": 100.0, "slack": 5.0, "gup": 50.0, "down": 50.0, "sync": 10.0}]
        cards = PR.aggregate_by_card(rows)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["gpu"], 1)
        self.assertEqual(cards[0]["windows"], 2)
        self.assertAlmostEqual(cards[0]["kernel"], 400.0)       # mean of 300 and 500
        self.assertAlmostEqual(cards[0]["mb"], 5.0)
        self.assertAlmostEqual(cards[1]["kernel"], 100.0)

    def test_by_type_keeps_the_two_buckets_apart(self):
        rows = [{"gpu": 1, "tag": "down20", "layers": 1166.0, "mb": 4.8, "kernel": 344.0,
                 "gup": 159.0, "down": 185.0, "sync": 286.0},
                {"gpu": 1, "tag": "down42", "layers": 274.0, "mb": 3.8, "kernel": 226.0,
                 "gup": 167.0, "down": 59.0, "sync": 164.0}]
        out = PR.aggregate_by_type(rows)
        self.assertEqual([r["tag"] for r in out], ["down20", "down42"])
        self.assertAlmostEqual(out[0]["down"], 185.0)
        self.assertAlmostEqual(out[1]["down"], 59.0)


class Map(unittest.TestCase):
    def test_counts_match_the_pack(self):
        self.assertEqual(len(PR.DOWN20), 39)
        self.assertEqual(len(PR.DOWN42), 9)
        self.assertEqual(len(PR.DOWN20) + len(PR.DOWN42), PR.N_LAYERS)
        self.assertEqual(sorted(PR.DOWN20 + PR.DOWN42), list(range(PR.N_LAYERS)))

    def test_known_layers(self):
        self.assertEqual(PR.down_of(0), 20)
        self.assertEqual(PR.down_of(1), 42)
        self.assertEqual(PR.down_of(47), 20)

    def test_dump_prints_every_layer(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sys.argv = ["iq3_profile.py", "--dump"]
            PR.main()
        lines = [l for l in buf.getvalue().splitlines() if l.startswith("layer ")]
        self.assertEqual(len(lines), 48)
        self.assertIn("layer  1 down 42", lines)


class Parse(unittest.TestCase):
    def write(self, text: str) -> str:
        fh = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
        fh.write(text)
        fh.close()
        self.addCleanup(Path(fh.name).unlink, missing_ok=True)
        return fh.name

    def test_four_line_families(self):
        path = self.write("\n".join(["noise", M10, M10TYPE20, M10TYPE42, LAYER, VLAYER]) + "\n")
        d = PR.read_log(path)
        self.assertEqual(len(d["m10"]), 1)
        self.assertEqual(len(d["m10type"]), 2)
        self.assertEqual(len(d["layer"]), 1)
        self.assertEqual(len(d["vlayer"]), 1)

    def test_m10_fields(self):
        d = PR.read_log(self.write(M10 + "\n"))["m10"][0]
        self.assertEqual(d["gpu"], 1)
        self.assertEqual(d["layers"], 1440.0)
        self.assertAlmostEqual(d["mb"], 3.456)
        self.assertAlmostEqual(d["kernel"], 55.0)
        self.assertAlmostEqual(d["gup"], 33.0)
        self.assertAlmostEqual(d["down"], 22.0)
        self.assertAlmostEqual(d["sync"], 7.0)

    def test_m10type_fields(self):
        rs = PR.read_log(self.write(M10TYPE20 + "\n" + M10TYPE42 + "\n"))["m10type"]
        self.assertEqual([r["tag"] for r in rs], ["down20", "down42"])
        self.assertAlmostEqual(rs[0]["down"], 24.0)
        self.assertAlmostEqual(rs[1]["mb"], 1.7)

    def test_report_runs_and_weights_the_two_types(self):
        path = self.write("\n".join([M10, M10TYPE20, M10TYPE42, LAYER, VLAYER]) + "\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            sys.argv = ["iq3_profile.py", "--log", path]
            rc = PR.main()
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("down stage: down20 24 us/L vs down42 16 us/L (+8)", out)
        self.assertIn("gup is 60.0% of the kernel span", out)

    def test_empty_log_says_why(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sys.argv = ["iq3_profile.py", "--log", self.write("nothing here\n")]
            rc = PR.main()
        self.assertEqual(rc, 1)
        self.assertIn("STRATA_MW_LAYERPROF", buf.getvalue())


if __name__ == "__main__":
    unittest.main()