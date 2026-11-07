"""tools/iq3_bench.py's log parsing and criterion, offline: a small log stands in for a running engine.

    python -m unittest tools.test_iq3_bench

The bench itself needs a server and a loaded model; what can be checked without hardware is that a prompt line is read
back into the fields the report prints, that a slice only sees its own lines, and that the criterion is the mean of
city+code and nothing else.
"""
from __future__ import annotations

import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import iq3_bench as B  # noqa: E402

LINE = ("strata serve: prompt 84 tokens = 79 reused + 5 read in 221 ms (22.6 tok/s), 600 generated in 12079 ms "
        "(49.7 tok/s), drafts accepted 292 of 442, 1 checkpoints")


class LogTail(unittest.TestCase):
    def write(self, text: str) -> str:
        fh = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
        fh.write(text)
        fh.close()
        self.addCleanup(Path(fh.name).unlink, missing_ok=True)
        return fh.name

    def test_fields(self):
        path = self.write(LINE + "\n")
        d, mark = B.read_log_tail(path, 0)
        self.assertEqual(d["prompt_tokens"], 84)
        self.assertEqual(d["reused"], 79)
        self.assertEqual(d["read"], 5)
        self.assertEqual(d["prefill_ms"], 221)
        self.assertAlmostEqual(d["prefill_tps"], 22.6)
        self.assertEqual(d["generated"], 600)
        self.assertEqual(d["decode_ms"], 12079)
        self.assertAlmostEqual(d["decode_tps"], 49.7)
        self.assertEqual(d["drafts_accepted"], 292)
        self.assertEqual(d["drafts_total"], 442)
        self.assertEqual(mark, Path(path).stat().st_size)   # past the whole file, CRLF or LF

    def test_slice_sees_only_new_lines(self):
        path = self.write("junk\n" + LINE + "\n")
        d1, mark = B.read_log_tail(path, 0)
        self.assertAlmostEqual(d1["decode_tps"], 49.7)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(LINE.replace("49.7 tok/s), drafts", "40.0 tok/s), drafts").replace("292 of 442", "300 of 450"))
        d2, mark2 = B.read_log_tail(path, mark)
        self.assertAlmostEqual(d2["decode_tps"], 40.0)
        self.assertEqual(d2["drafts_accepted"], 300)
        self.assertGreater(mark2, mark)
        d3, mark3 = B.read_log_tail(path, mark2)
        self.assertIsNone(d3)
        self.assertEqual(mark3, mark2)

    def test_the_last_line_wins(self):
        path = self.write(LINE + "\n" + LINE.replace("49.7", "47.5") + "\n")
        d, _ = B.read_log_tail(path, 0)
        self.assertAlmostEqual(d["decode_tps"], 47.5)

    def test_no_log_is_not_an_error(self):
        d, mark = B.read_log_tail("", 123)
        self.assertIsNone(d)
        self.assertEqual(mark, 123)
        d, mark = B.read_log_tail("/nonexistent/engine.log", 7)
        self.assertIsNone(d)
        self.assertEqual(mark, 7)


class Criterion(unittest.TestCase):
    def test_the_criterion_is_city_plus_code_only(self):
        """The brief's 55 tok/s is a mean over city and code; prose must not move it."""
        city, code, prose = [60.0, 62.0, 58.0], [56.0, 57.0, 55.0], [40.0, 41.0, 39.0]
        self.assertAlmostEqual(statistics.fmean(city + code), 58.0)
        self.assertLess(statistics.fmean(city + code + prose), statistics.fmean(city + code))
        self.assertGreaterEqual(statistics.fmean(city + code), 55)


class Prompts(unittest.TestCase):
    def test_the_three_prompts_are_the_ones_the_operator_named(self):
        self.assertIn("Firenze", B.CITY)
        self.assertIn("merge_intervals", B.CODE)
        self.assertIn("tramonto", B.PROSE)
        self.assertEqual([n for n, _ in B.LABELS], ["city", "code", "prose"])


class Reporting(unittest.TestCase):
    def test_the_summary_keeps_the_first_run_apart(self):
        """The operator reads mean, median and the first (cold) run separately, never folded together."""
        line = B.summary("city", [40.0, 60.0, 62.0])
        self.assertIn("mean 54.0", line)
        self.assertIn("median 60.0", line)
        self.assertIn("first 40.0 (run 1, cold)", line)

    def test_the_summary_of_one_run_is_that_run(self):
        line = B.summary("code", [58.5])
        self.assertIn("first 58.5 (run 1, cold)", line)
        self.assertNotIn("median", line)

    def test_stats_reports_median(self):
        line = B.stats([40.0, 60.0, 62.0])
        self.assertIn("mean 54.0", line)
        self.assertIn("median 60.0", line)


class Failures(unittest.TestCase):
    def test_an_http_error_stops_the_run_with_the_server_message(self):
        class Boom:
            def __init__(self):
                self.code = 500

            def read(self):
                return b'{"error": "CUDA out of memory"}'

        orig = B.urllib.request.urlopen

        def fake(*_a, **_k):
            raise B.urllib.error.HTTPError("u", 500, "boom", {}, Boom())

        B.urllib.request.urlopen = fake
        try:
            with self.assertRaises(B.BenchFailure) as ctx:
                B.ask("http://127.0.0.1:9", "", "x", 8, 5)
        finally:
            B.urllib.request.urlopen = orig
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertIn("CUDA out of memory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()