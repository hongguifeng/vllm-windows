"""Sample GPU clocks/power while a long decode runs on the recipe server."""

import json
import subprocess
import threading
import time
import urllib.request

PROMPT = "\u8bf7\u4ece1\u6570\u52301000\uff0c\u6bcf\u4e2a\u6570\u5b57\u540e\u52a0\u4e00\u4e2a\u9017\u53f7\uff0c\u4e0d\u8981\u505c\u4e0b\u6765\u3002"


def request() -> None:
    body = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1200,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=600).read()


t = threading.Thread(target=request, daemon=True)
t.start()
time.sleep(2)
for _ in range(24):
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,clocks.sm,power.draw,temperature.gpu,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(time.strftime("%H:%M:%S"), out.replace("\n", " | "), flush=True)
    time.sleep(0.5)
t.join(timeout=5)
