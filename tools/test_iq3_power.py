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
RAW_LOAD = "3, 29.51 W, 30.00 W, 1100 MHz, 810 MHz, 61, 0x0000000000000004"
RAW_MIXED = "4, 118.20 W, 120.00 W, 1900 MHz, 7500 MHz, 70, 0x0000000000000000"
# what an M10 actually reports under load: applications clocks, NOT a power cap
RAW_M10_LOAD = "1, 22.20 W, 30.00 W, 1032 MHz, 2600 MHz, 36, 0x0000000000000002"


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


class BitMap(unittest.TestCase):
    """The bit values, pinned to what the driver was OBSERVED to report.

    Iteration 7 found the old table shifted by one: it read an M10's ApplicationsClocksSetting as
    SwPowerCap and the 3060's SwPowerCap as HwSlowdown.  These three assertions come from
    simultaneous readings of the aggregate mask and the driver's own named fields on aidrogen.
    """

    def test_3060_slowdown_is_bit_0x4_and_it_is_sw_power_cap(self):
        # mask 0x4 while clocks_event_reasons.sw_power_cap = Active, on the RTX 3060 under load
        self.assertEqual(P.throttle_names("0x0000000000000004"), ["SwPowerCap"])

    def test_m10_bit_0x2_is_applications_clocks_not_power_cap(self):
        # mask 0x2 while applications_clocks_setting = Active, on a Tesla M10 under load
        self.assertEqual(P.throttle_names("0x0000000000000002"), ["ApplicationsClocksSetting"])

    def test_idle_is_still_bit_1(self):
        self.assertEqual(P.throttle_names("0x0000000000000001"), ["GpuIdle"])

    def test_every_bit_is_distinct_and_in_ascending_order(self):
        bits = sorted(P.THROTTLE)
        self.assertEqual(bits, [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100])
        self.assertEqual(len(set(P.THROTTLE.values())), len(P.THROTTLE))


class Throttle(unittest.TestCase):
    def test_idle(self):
        self.assertEqual(P.throttle_names("0x0000000000000001"), ["GpuIdle"])

    def test_m10_load_is_an_application_clock_not_a_cap(self):
        self.assertEqual(P.throttle_names(RAW_M10_LOAD.split(",")[-1].strip()),
                         ["ApplicationsClocksSetting"])

    def test_two_bits(self):
        self.assertEqual(P.throttle_names("0x0000000000000003"), ["GpuIdle", "ApplicationsClocksSetting"])

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


class Deep(unittest.TestCase):
    # a real 3060 row under load: HwSlowdown on its own bit, no thermal bit, memory temp N/A
    DEEP = ("4, 112.20 W, 120.00 W, 1810 MHz, 7500 MHz, 58, N/A, 0 %, 0x0000000000000004, "
            "Active, Not Active, Not Active, Not Active, Not Active")

    def test_separate_reasons_are_read(self):
        r = P.parse_deep_row(self.DEEP)
        self.assertEqual(r["gpu"], 4)
        self.assertAlmostEqual(r["draw_w"], 112.2)
        self.assertIsNone(r["temp_mem_c"])
        self.assertAlmostEqual(r["fan_pct"], 0.0)
        self.assertEqual(r["reasons"]["HwSlowdown"], "Active")
        self.assertEqual(r["reasons"]["HwThermal"], "Not Active")
        self.assertEqual(r["reasons"]["HwPowerBrake"], "Not Active")

    def test_deep_row_is_not_a_plain_row(self):
        """The two formats must not be confused: a deep row has 14 fields, a plain one 7."""
        self.assertIsNone(P.parse_row(self.DEEP))
        self.assertIsNone(P.parse_deep_row("4, 112.20 W, 120.00 W, 1810 MHz, 7500 MHz, 58, "
                                           "0x0000000000000004"))

    def test_summary_names_the_cause_and_says_when_memory_temp_is_missing(self):
        rows = [P.parse_deep_row(self.DEEP) for _ in range(4)]
        out = P.summarize_deep(rows)
        self.assertIn("HwSlowdown", out)
        self.assertIn("100%", out)
        self.assertIn("not exposed by this driver", out)   # temperature.memory is N/A here
        self.assertIn("fan", out)

    def test_active_only_drops_the_idle_samples(self):
        idle = ("4, 18.60 W, 120.00 W, 210 MHz, 405 MHz, 54, N/A, 0 %, 0x0000000000000001, "
                "Not Active, Not Active, Not Active, Not Active, Not Active")
        rows = [P.parse_deep_row(self.DEEP) for _ in range(3)] + [P.parse_deep_row(idle)]
        active = P.summarize_deep(rows, active_only=True)
        every = P.summarize_deep(rows)
        self.assertIn("(GpuIdle excluded)", active)
        self.assertIn("HwSlowdown       100%", active)
        self.assertIn("HwSlowdown        75%", every)


class Summary(unittest.TestCase):
    def rows(self, mask: str, n: int, draw: float = 29.5) -> list[dict]:
        return [{"gpu": 0, "draw_w": draw, "limit_w": 30.0, "sm_mhz": 1100.0, "mem_mhz": 810.0,
                 "temp_c": 60.0, "throttle": mask} for _ in range(n)]

    def test_sustained_cap_is_a_share(self):
        out = P.summarize(self.rows("0x0000000000000004", 3) + self.rows("0x0000000000000001", 1))
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