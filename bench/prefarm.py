import json, urllib.request, random, sys, os

# The benchmark protocol behind the records in bench/results/ (see docs/MAXWELL.md).
# Usage: python3 prefarm.py [full|rate|10k] [config.json]
#   config.json (default: $STRATA_BENCH_CFG or bench/bench-config.json) needs
#   {"model_name": ..., "api_key": ...} and optional "port" (default 8088).
CFG_PATH = (sys.argv[2] if len(sys.argv) > 2 else
            os.environ.get("STRATA_BENCH_CFG") or
            os.path.join(os.path.dirname(__file__), "bench-config.json"))
CFG = json.load(open(CFG_PATH))
PORT = CFG.get("port", 8088)

def ask(q, mt=48):
    body = json.dumps({"model": CFG["model_name"], "temperature": 0, "max_tokens": mt, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": q}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, method="POST")
    req.add_header("Authorization", "Bearer " + CFG["api_key"])
    req.add_header("Content-Type", "application/json")
    r = json.loads(urllib.request.urlopen(req, body, timeout=1800).read())
    return r["choices"][0]["message"]["content"]

mode = sys.argv[1] if len(sys.argv) > 1 else "full"

ask("hi", 8)  # warmup

P = ("The history of computing is long. " * 260) + " In one word, what is this about?"
if mode in ("full", "rate"):
    for t in range(3):
        a = ask(P, 24)
        print("2.7k trial %d: %s" % (t, a[:16]))

if mode in ("full", "10k"):
    random.seed(7)
    topics = ["lighthouse keepers", "the Silk Road", "anglerfish", "the Bronze Age collapse", "azulejo tiles",
              "the semaphore", "glacier retreat", "the Voynich manuscript", "Ethiopian coffee", "the Antikythera mechanism",
              "arctic terns", "the Kalahari", "Bauhaus chairs", "penicillin", "Japanese joinery", "Baltic amber",
              "Iceland volcanoes", "the Enigma machine", "Andean terraces", "the Dunhuang caves"]
    parts = ["Note %d on %s: values %d.%02d and %d/%d recorded in %s, season %d." %
             (i, t, random.randint(1, 999), random.randint(0, 99), random.randint(1, 89), random.randint(90, 400),
              random.choice(["spring", "autumn", "winter", "summer"]), 1900 + random.randint(0, 120))
             for i, t in enumerate(topics * 12)]
    a = ask(" ".join(parts) + " Name two topics above, one line.", 64)
    print("10k topics:", a[:40])

if mode == "full":
    ctx = "Notes: the keeper is Marguerite; the code is ZX-7041-QF. " + " ".join("filler %d." % i for i in range(300))
    a1 = ask(ctx + " Who is the keeper?", 48)
    a2 = ask(ctx + " What is the code?", 48)
    print("needle1:", "OK" if "marguerite" in a1.lower() else "FAIL",
          "| needle2:", "OK" if "zx-7041-qf" in a2.lower() else "FAIL")
    print("2+2:", ask("2+2? number only", 16)[:4])
    print("17x23:", ask("17 times 23? number only", 24)[:8])
    print("9999sq:", ask("What is 9999 times 9999? number only", 32)[:12])
    Q = "List 100 distinct animals, one per line, numbered 1. to 100. Do not stop early."
    for t in range(2):
        a = ask(Q, 300)
        print("decode trial %d: %d lines" % (t, len(a.splitlines())))
