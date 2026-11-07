"""tools/iq3_bench.py - the fixed benchmark of the IQ3 campaign, through a running Strata server.

Three prompts (city: the Firenze prompt of /root/det_test.py; code: a medium Python function; prose: the "200 ways to
describe a sunset" prompt of the Iteration 0 baseline), temperature 0, 600 tokens, 3 runs each. Each run is matched to
its own engine-log line, so the tok/s and the accepted drafts come from the engine and not from the client's clock.

    python tools/iq3_bench.py --engine-log /root/Strata-maxwell/strata-iq3_s-i3.log --label iq3_s \
        --api-key-json /root/Strata-maxwell/strata-iq3_s-i3.json --texts-out docs/iq3/results/10-quality-iq3_s.md

The texts of the city and code prompts are written to --texts-out (first run of each: temperature 0), for the operator
to compare IQ2_XS against IQ3_S. Every category is reported as mean, median and the first run kept apart (the run right
after a load is the cold one), plus the accepted drafts summed over the runs.

Two more modes, same server and same log parsing:

    --long-context 32k,64k   one haystack prompt per length, a code word hidden in it, short answer; reports the
                             prompt's real token count, prefill and decode tok/s, and the free VRAM after.
    --soak-minutes 10        back-to-back city prompts until the time is up; reports the run count, mean, median,
                             first and last run and the drift.

Any HTTP or network error stops the run with a "*** STOP" line carrying the server's message and the tail of the engine
log; the JSON written so far stays on disk. OOM is called out by name. The exit status is 2.

The api key never appears on the command line: give --api-key-json FILE (field api_key) or --api-key-env NAME, and the
key is read there. A missing key is not fatal: the tool then sends no Authorization header.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CITY = "Descrivi la città di Firenze in circa 150 parole: storia, arte e cosa visitare."
CODE = ("Scrivi una funzione Python di circa 40 righe chiamata merge_intervals che, data una lista di intervalli "
        "[inizio, fine], unisca quelli sovrapposti o adiacenti e restituisca la lista ordinata. Includi il docstring, "
        "i type hint e due casi di test.")
PROSE = ("Elenca in modo prolisso e dettagliato 200 modi diversi di descrivere un tramonto, uno per riga, "
         "senza ripeterti.")
LABELS = [("city", CITY), ("code", CODE), ("prose", PROSE)]

# one engine line: "prompt 84 tokens = 0 reused + 84 read in 905 ms (92.8 tok/s), 600 generated in 12643 ms
# (47.5 tok/s), drafts accepted 276 of 416, 1 checkpoints"
PROMPT_LINE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\), drafts accepted (\d+) of (\d+)")

SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"),
           ("tools", "*.py")]
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
         "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]
CHARS_PER_TOKEN = 3.2
OOM_HINTS = ("out of memory", "cuda error", "failed to allocate", "oom")


class BenchFailure(RuntimeError):
    """The server refused or broke a run: stop the bench, keep what was measured."""


def read_key(args) -> str:
    if args.api_key_env:
        return os.environ.get(args.api_key_env, "")
    if args.api_key_json:
        with open(args.api_key_json, encoding="utf-8") as fh:
            return str(json.load(fh).get(args.api_key_field) or "")
    if args.api_key_file:
        return Path(args.api_key_file).read_text(encoding="utf-8").strip()
    return ""


def read_log_tail(path: str, mark: int) -> tuple[dict | None, int]:
    """Everything the engine wrote after `mark`, and the new mark. The last prompt line of the slice is this run's."""
    if not path:
        return None, mark
    try:
        with open(path, "rb") as fh:
            fh.seek(mark)
            chunk = fh.read()
            end = fh.tell()
    except OSError:
        return None, mark
    text = chunk.decode("utf-8", "replace")
    last = None
    for m in PROMPT_LINE.finditer(text):
        last = {"prompt_tokens": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
                "prefill_ms": int(m.group(4)), "prefill_tps": float(m.group(5)),
                "generated": int(m.group(6)), "decode_ms": int(m.group(7)), "decode_tps": float(m.group(8)),
                "drafts_accepted": int(m.group(9)), "drafts_total": int(m.group(10))}
    return last, end


def ask(url: str, key: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict, float]:
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise BenchFailure(f"HTTP {exc.code} from the server: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchFailure(f"could not reach the server: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchFailure(f"the server's reply is not JSON: {exc}") from exc
    return out, time.time() - t0


def answer_text(out: dict) -> str:
    msg = out["choices"][0]["message"]
    think = (msg.get("reasoning_content") or "").strip()
    body = (msg.get("content") or "").strip()
    if think and body:
        return think + "\n\n---\n\n" + body
    return think or body


def free_vram(gpu: int) -> int | None:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == gpu:
            return int(parts[1].split()[0])
    return None


def fmt_row(label: str, i: int, r: dict, secs: float, vram: int | None) -> str:
    d = r["engine"]
    if d is None:
        return f"{label:<5} run {i}: client {r['generated']/secs:6.1f} tok/s (no engine line matched)"
    return (f"{label:<5} run {i}: {d['decode_tps']:6.1f} tok/s engine ({d['generated']} tok in {d['decode_ms']} ms), "
            f"prompt {d['prompt_tokens']} tok at {d['prefill_tps']:.1f} tok/s, "
            f"drafts {d['drafts_accepted']}/{d['drafts_total']}"
            + (f", 3060 {vram} MiB free" if vram is not None else ""))


def stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return (f"mean {statistics.fmean(values):.1f}, median {statistics.median(values):.1f}, "
            f"stdev {statistics.stdev(values):.1f}, min {min(values):.1f}, max {max(values):.1f}")


def summary(label: str, values: list[float]) -> str:
    """Mean, median and the first run kept apart: the first run after a load is the cold one."""
    if not values:
        return f"{label:<5}: n/a"
    first = f"{values[0]:.1f} (run 1, cold)"
    if len(values) == 1:
        return f"{label:<5}: first {first}"
    return (f"{label:<5}: mean {statistics.fmean(values):.1f}, median {statistics.median(values):.1f}, "
            f"first {first}, stdev {statistics.stdev(values):.1f}, min {min(values):.1f}, "
            f"max {max(values):.1f}")


def save_rows(path: str, rows: list[dict]) -> None:
    if path:
        Path(path).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def tail_text(path: str, nbytes: int = 4000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - nbytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def build_haystack(n_chars: int) -> str:
    files = []
    for d, pat in SOURCES:
        files += sorted((ROOT / d).rglob(pat))
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"\n\n=== {f.relative_to(ROOT).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= n_chars:
                break
        if not files:
            raise SystemExit("no text files found to build the haystack from")
    return "".join(parts)[:n_chars]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--engine-log", default="", help="the engine's own log; without it only client time is reported")
    ap.add_argument("--label", default="engine", help="a name for this engine in the report")
    ap.add_argument("--api-key-json", default="", help="config JSON holding the api key")
    ap.add_argument("--api-key-field", default="api_key")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--labels", default="city,code,prose", help="which fixed prompts to run")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--vram-gpu", type=int, default=4, help="nvidia-smi index to report free VRAM for (the 3060 here)")
    ap.add_argument("--texts-out", default="", help="write the city and code texts here (markdown)")
    ap.add_argument("--long-context", default="", help="e.g. 32k,64k: haystack prompts instead of the fixed bench")
    ap.add_argument("--soak-minutes", type=float, default=0, help="run city prompts back to back for this long")
    ap.add_argument("--json", default="", help="write every row here as JSON")
    a = ap.parse_args()

    key = read_key(a)
    mark = os.path.getsize(a.engine_log) if a.engine_log and os.path.exists(a.engine_log) else 0
    rows: list[dict] = []

    def run_one(label: str, prompt: str, max_tokens: int) -> dict:
        nonlocal mark
        try:
            out, secs = ask(a.url, key, prompt, max_tokens, a.timeout)
        except BenchFailure as exc:
            low = str(exc).lower()
            kind = "OOM" if any(h in low for h in OOM_HINTS) else "ERROR"
            tail = tail_text(a.engine_log)
            raise BenchFailure(f"{kind} during {label}: {exc}\n--- engine log tail ---\n{tail}") from exc
        d, mark = read_log_tail(a.engine_log, mark)
        usage = out.get("usage") or {}
        row = {"label": label, "engine": d, "seconds": round(secs, 2),
               "generated": int(usage.get("completion_tokens") or 0),
               "client_tps": round((usage.get("completion_tokens") or 0) / secs, 1) if secs > 0 else 0.0,
               "text": answer_text(out)}
        rows.append(row)
        save_rows(a.json, rows)   # whatever was measured survives a stop
        return row

    if a.long_context:
        print(f"# long context, {a.label}, {a.url}")
        rnd = random.Random(7)
        for spec in a.long_context.split(","):
            tokens = int(float(spec.lower().rstrip("k")) * 1024) if spec.lower().endswith("k") else int(spec)
            text = build_haystack(int(tokens * CHARS_PER_TOKEN * 0.98))
            word = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"
            cut = text.rfind("\n", 0, int(len(text) * 0.5)) + 1 or len(text) // 2
            needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
            prompt = (text[:cut] + needle + text[cut:] +
                      "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
            row = run_one("ctx-" + spec, prompt, 64)
            row["target"] = spec
            row["needle"] = word
            row["found"] = word in row["text"]
            save_rows(a.json, rows)
            d = row["engine"]
            vram = free_vram(a.vram_gpu)
            if d is None:
                print(f"{spec:>5}: request ok, no engine line matched (client {row['client_tps']:.1f} tok/s)")
            else:
                print(f"{spec:>5}: prompt {d['prompt_tokens']:,} tok, prefill {d['prefill_tps']:.1f} tok/s "
                      f"({d['prefill_ms']} ms), decode {d['decode_tps']:.1f} tok/s, "
                      f"needle {'FOUND' if row['found'] else 'MISSED'}"
                      + (f", 3060 {vram} MiB free" if vram is not None else ""), flush=True)
        missed = [r for r in rows if not r.get("found")]
        return 0 if rows and not missed else 1

    if a.soak_minutes > 0:
        print(f"# soak, {a.label}: city prompt back to back for {a.soak_minutes} min")
        deadline = time.time() + a.soak_minutes * 60
        speeds: list[float] = []
        i = 0
        while time.time() < deadline:
            i += 1
            row = run_one("soak", CITY, 300)
            d = row["engine"]
            tps = d["decode_tps"] if d else row["client_tps"]
            speeds.append(tps)
            vram = free_vram(a.vram_gpu)
            print(f"run {i:>3}: {tps:6.1f} tok/s, {row['generated']} tok" +
                  (f", 3060 {vram} MiB free" if vram is not None else ""),
                  flush=True)
        print()
        print(f"{i} runs, {stats(speeds)} tok/s")
        if speeds:
            print(f"first {speeds[0]:.1f} (run 1, cold), last {speeds[-1]:.1f} "
                  f"(drift {speeds[-1] - speeds[0]:+.1f} tok/s)")
        return 0

    print(f"# fixed bench, {a.label}, {a.url}, temperature 0, max_tokens {a.max_tokens}, {a.runs} runs each")
    wanted = [x.strip() for x in a.labels.split(",") if x.strip()]
    chosen = [(n, p) for n, p in LABELS if n in wanted]
    if not chosen:
        raise SystemExit(f"--labels {a.labels!r} names no fixed prompt (city, code, prose)")
    by_label: dict[str, list[float]] = {}
    drafts: dict[str, list[tuple[int, int]]] = {}
    texts: dict[str, str] = {}
    for label, prompt in chosen:
        by_label[label] = []
        drafts[label] = []
        for i in range(1, a.runs + 1):
            row = run_one(label, prompt, a.max_tokens)
            d = row["engine"]
            tps = d["decode_tps"] if d else row["client_tps"]
            by_label[label].append(tps)
            if d:
                drafts[label].append((d["drafts_accepted"], d["drafts_total"]))
            vram = free_vram(a.vram_gpu)
            print(fmt_row(label, i, row, row["seconds"], vram), flush=True)
            if i == 1:
                texts[label] = row["text"]
    print()
    for label, _ in chosen:
        print(summary(label, by_label[label]) + " tok/s")
        if drafts[label]:
            ok = sum(x for x, _ in drafts[label])
            tot = sum(y for _, y in drafts[label])
            print(f"      drafts accepted {ok}/{tot} ({100.0 * ok / tot:.1f}%) over {len(drafts[label])} runs")
    city, code = by_label.get("city", []), by_label.get("code", [])
    if city and code:
        both = city + code
        crit = statistics.fmean(both)
        med = statistics.median(both)
        print(f"\ncriterion (mean of city+code, {len(both)} runs): {crit:.1f} tok/s, median {med:.1f} -> "
              f"{'PASS' if crit >= 55 else 'FAIL'} against 55")
    if a.texts_out and texts:
        out = [f"# {a.label} - answers at temperature 0, first run\n"]
        for label in (n for n, _ in chosen):
            if label in texts:
                out.append(f"\n## {label}\n\n```\n{texts[label]}\n```\n")
        Path(a.texts_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.texts_out).write_text("".join(out), encoding="utf-8")
        print(f"texts written to {a.texts_out}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BenchFailure as exc:
        print(f"\n*** STOP: {exc}", flush=True)
        sys.exit(2)