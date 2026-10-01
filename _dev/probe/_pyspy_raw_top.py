"""Tally py-spy raw flamegraph samples by leaf frame and by any-frame."""
import collections
import sys

path = (sys.argv[1] if len(sys.argv) > 1 else
        r"D:\code\vllm-windows\_dev\out\pyspy_raw.txt")
leaf = collections.Counter()
anywhere = collections.Counter()
threads = collections.Counter()
total = 0

with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        frames = line.split(";")
        count = 1
        # The raw format appends the sample count to the last frame with a space.
        if frames:
            last = frames[-1].rsplit(" ", 1)
            if len(last) == 2 and last[1].isdigit():
                frames[-1] = last[0]
                count = int(last[1])
        total += count
        if frames:
            leaf[frames[-1]] += count
            for f in frames:
                anywhere[f] += count

print(f"total samples: {total}")
print("\ntop leaf frames (what the CPU was executing):")
for name, n in leaf.most_common(18):
    print(f"  {n:5d}  {100*n/total:5.1f}%  {name}")
print("\ntop frames anywhere in a stack (share of samples inside them):")
for name, n in anywhere.most_common(18):
    print(f"  {n:5d}  {100*n/total:5.1f}%  {name}")
