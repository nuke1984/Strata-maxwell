#!/usr/bin/env python3
import json, time, random, urllib.request, threading
CFG = json.load(open("/tmp/opencode/spec-sweep.json"))
stop = time.time() + 1800
stats = {"req": 0, "err": 0, "tok": 0, "lock": threading.Lock()}
PROMPTS = ["explain Fibonacci briefly", "write a haiku about rain", "what is the capital of Japan?",
           "write a python function that reverses a list", "summarize the water cycle",
           "name three primary colors", "explain gravity simply", "write a limerick about a cat",
           "what is 12 * 17?", "describe a sunset over the sea"]
def worker(wid):
    rnd = random.Random(wid)
    while time.time() < stop:
        q = rnd.choice(PROMPTS)
        body = json.dumps({"model": CFG["model_name"], "temperature": 0.7, "max_tokens": rnd.choice([128, 256, 384]),
                           "stream": False, "messages": [{"role": "user", "content": q}]}).encode()
        req = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions", method="POST")
        req.add_header("Authorization", "Bearer " + CFG["api_key"]); req.add_header("Content-Type", "application/json")
        try:
            r = json.loads(urllib.request.urlopen(req, body, timeout=600).read())
            n = r["usage"]["completion_tokens"]
            with stats["lock"]:
                stats["req"] += 1; stats["tok"] += n
        except Exception as e:
            with stats["lock"]:
                stats["req"] += 1; stats["err"] += 1
            print("ERR: %r" % e, flush=True)
        time.sleep(rnd.uniform(0.5, 3.0))
ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("SOAK DONE: %d requests, %d errors, %d tokens" % (stats["req"], stats["err"], stats["tok"]))
