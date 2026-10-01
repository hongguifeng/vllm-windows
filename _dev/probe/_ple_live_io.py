"""Sample disk read throughput while vLLM generates, to see if PLE rows are read."""
import json
import subprocess
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8111/v1/chat/completions"
MODEL = "qwen3.8-flash-next-full"
PROMPTS = [
    "List every integer from 3000 down to 2001, one per line.",
]

results = {}


def ask(i, prompt, done):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        ch = data.get("choices", [{}])[0]
        results[i] = {
            "finish": ch.get("finish_reason"),
            "completion": data.get("usage", {}).get("completion_tokens"),
            "wall": time.time() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        results[i] = {"error": repr(exc), "wall": time.time() - t0}
    done.append(i)


def counters():
    out = subprocess.run(
        ["typeperf",
         "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec",
         "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec",
         "\\Memory\\Available MBytes", "-sc", "40", "-si", "2"],
        capture_output=True, text=True, shell=False)
    rows = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.strip().strip('"').split('","')
        if len(parts) >= 4:
            try:
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    return rows


done = []
threads = [threading.Thread(target=ask, args=(i, p, done))
           for i, p in enumerate(PROMPTS)]
for t in threads:
    t.start()
samples = counters()
for t in threads:
    t.join()

read_mb = [r[0] * 8 / 2**20 for r in samples]
avail = [r[2] for r in samples]
print(f"samples={len(samples)} read MB/s: min={min(read_mb):.1f} "
      f"max={max(read_mb):.1f} mean={sum(read_mb)/len(read_mb):.1f}")
print(f"total bytes read during window: "
      f"{sum(r[0] for r in samples) * 2 / 2**30:.2f} GiB (2s intervals)")
print(f"available RAM MiB: first={avail[0]:.0f} min={min(avail):.0f}")
for i in sorted(results):
    print(f"req {i}: {results[i]}")
