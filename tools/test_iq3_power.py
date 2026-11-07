"""tools/iq3_power.py's CSV handling, offline: real nvidia-smi lines stand in for a running host.

    python -m unittest tools.test_iq3_power

What can be checked without the cards: that the driver's W/MHz suffixes are read, that the raw and the
elapsed-prefixed CSV forms both keep the GPU index, that the throttle mask is decoded by bit, and that
the summary reports a share for a sustained caps.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import iq3_power as P  # noqa: E402

RAW_IDLE = "0, 8.08 W, 30.00 W, 135 MHz, 405 MHz, 28, 0x0000000000000001"
RAW_LOAD = "3, 29.51 W, 30.00 W, 1100 MHz, 810 MHz, 61, 0x0000000000000002"
RAW_MIXED = "4, 118.20 W, 120.00 W, 1900 MHz, 7500 MHz, 70, 0x0000000000000000"


class Parse(unittest.TestCase):
    def test_raw_line_keeps_the_gpu_index(self):
        r = P.parse_row(RAW_IDLE)
        self.assertEqual(r["gpu"], 0)
        self.assertAlmostEqual(r["draw_w"], 8.08)
        self.assertAlmostEqual(r["limit_w"], 30.0)
        self.assertAlmostEqual(r["sm_mhz"], 135.0)
        self.assertAlmostEqual(r["temp_c"], 28.0)
        self.assertEqual(r["throttle"], "0x0000000000000001")

    def test_elapsed_prefixed_line_is_not_a_raw_line(self):
        self.assertIsNone(P.parse_row("   3.0," + RAW_IDLE))

    def test_file_with_both_forms(self):
        text = ("# label, query, interval\n"
                "     0.0,0, 8.08 W, 30.00 W, 135 MHz, 405 MHz, 28, 0x0000000000000001\n"
                "     1.0,3, 29.51 W, 30.00 W, 1100 MHz, 810 MHz, 61, 0x0000000000000002\n"
                + RAW_MIXED + "\n")
        fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        fh.write(text)
        fh.close()
        self.addCleanup(Path(fh.name).unlink, missing_ok=True)
        rows = []
        for line in Path(fh.name).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            r = P.parse_row(line)
            if r is None:
                r = P.parse_row(line.split(",", 1)[1])
            if r is not None:
                rows.append(r)
        self.assertEqual([r["gpu"] for r in rows], [0, 3, 4])
        self.assertAlmostEqual(rows[2]["draw_w"], 118.2)


class Throttle(unittest.TestCase):
    def test_idle(self):
        self.assertEqual(P.throttle_names("0x0000000000000001"), ["GpuIdle"])

    def test_sw_power_cap(self):
        self.assertEqual(P.throttle_names("0x0000000000000002"), ["SwPowerCap"])

    def test_two_bits(self):
        self.assertEqual(P.throttle_names("0x0000000000000003"), ["GpuIdle", "SwPowerCap"])

    def test_none(self):
        self.assertEqual(P.throttle_names("0x0000000000000000"), [])


class Sampler(unittest.TestCase):
    def test_lms_is_two_argv(self):
        """-lms takes a separate argv: as one string nvidia-smi exits and the sampler reads nothing."""
        cmd = ["nvidia-smi", "--query-gpu=x", "--format=csv,noheader", "-lms", "1000"]
        self.assertEqual(cmd[-2:], ["-lms", "1000"])
        self.assertNotIn("-lms 1000", cmd)
        src = (ROOT / "tools" / "iq3_power.py").read_text(encoding="utf-8")
        self.assertIn('"-lms", str(interval_ms)', src)
        self.assertNotIn('f"-lms {interval_ms}"', src)

    def test_empty_run_is_not_a_measurement(self):
        """A sampler with 0 rows must fail loudly, never summarize as if the cards were idle."""
        src = (ROOT / "tools" / "iq3_power.py").read_text(encoding="utf-8")
        self.assertIn("the sampler collected NO samples", src)


class Summary(unittest.TestCase):
    def rows(self, mask: str, n: int, draw: float = 29.5) -> list[dict]:
        return [{"gpu": 0, "draw_w": draw, "limit_w": 30.0, "sm_mhz": 1100.0, "mem_mhz": 810.0,
                 "temp_c": 60.0, "throttle": mask} for _ in range(n)]

    def test_sustained_cap_is_a_share(self):
        out = P.summarize(self.rows("0x0000000000000002", 3) + self.rows("0x0000000000000001", 1))
        self.assertIn("SwPowerCap 75%", out)
        self.assertIn("GpuIdle 25%", out)
        self.assertIn("limit   30.0 W", out)

    def test_total_against_the_limits(self):
        out = P.summarize(self.rows("0x0", 2, draw=20.0))
        self.assertIn("total draw", out)
        self.assertIn("20.0 W", out)

    def test_empty(self):
        self.assertEqual(P.summarize([]), "no samples")


if __name__ == "__main__":
    unittest.main()