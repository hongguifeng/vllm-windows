# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fetch a Hugging Face file in short-lived range requests.

The single long transfer stalls on this host (through the local TUN), while
short HTTPS requests through the same stack complete, so this pulls fixed-size
ranges and appends them, resuming from whatever is already on disk.

  python _dev/probe/_range_download.py --url <resolve url> --out <path>
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk-mb", type=int, default=32)
ap.add_argument("--retries", type=int, default=8)
args = ap.parse_args()

chunk = args.chunk_mb * 1024 * 1024


def total_size() -> int:
    req = urllib.request.Request(args.url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return int(resp.headers["Content-Length"])


def fetch(start: int, stop: int) -> bytes:
    req = urllib.request.Request(
        args.url, headers={"Range": f"bytes={start}-{stop - 1}"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        if resp.status not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status}")
        return resp.read()


size = total_size()
have = os.path.getsize(args.out) if os.path.exists(args.out) else 0
if have > size:
    have = 0
    os.remove(args.out)
print(f"target {size / 2**20:.1f} MiB, have {have / 2**20:.1f} MiB", flush=True)

last_report = 0.0
with open(args.out, "ab") as fh:
    while have < size:
        stop = min(have + chunk, size)
        for attempt in range(1, args.retries + 1):
            try:
                data = fetch(have, stop)
                break
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                print(f"  retry {attempt} at {have / 2**20:.1f} MiB: {exc}", flush=True)
                time.sleep(min(2 * attempt, 20))
        else:
            print("giving up", flush=True)
            sys.exit(1)
        if len(data) != stop - have:
            print(f"short read: {len(data)} != {stop - have}", flush=True)
            sys.exit(1)
        fh.write(data)
        fh.flush()
        have = stop
        now = time.time()
        if now - last_report > 15 or have == size:
            last_report = now
            print(f"  {have / 2**20:.1f} / {size / 2**20:.1f} MiB", flush=True)
print("DONE", flush=True)
