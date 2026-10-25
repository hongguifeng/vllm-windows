#!/usr/bin/env bash
# Measure NVMe read traffic during decode and the current readahead setting.
echo "readahead sectors: $(blockdev --getra /dev/sdd 2>/dev/null || echo n/a)  (32 sectors = 16 KiB)"
r() { awk '{print $1, $3, $5, $6}' /sys/block/sdd/stat; }
S0=$(cat /sys/block/sdd/stat); T0=$(date +%s.%N)
curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-w4a16","messages":[{"role":"user","content":"Explain in a long paragraph how TCP congestion control works, covering slow start, AIMD, and BBR. Keep writing for a while."}],"max_tokens":300,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' \
  -o /tmp/dec.json
S1=$(cat /sys/block/sdd/stat); T1=$(date +%s.%N)
python3 - "$S0" "$S1" "$T0" "$T1" <<'PY'
import sys
a = [int(x) for x in sys.argv[1].split()]
b = [int(x) for x in sys.argv[2].split()]
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
dt = t1 - t0
reads = b[0] - a[0]
sectors = b[2] - a[2]
mb = sectors * 512 / 1e6
print(f"window {dt:.1f} s: {reads} reads, {mb:.1f} MB read")
print(f"  read IOPS {reads/dt:.0f}, throughput {mb/dt:.1f} MB/s, avg {sectors/max(reads,1)*512/1024:.1f} KiB/read")
PY
python3 -c "import json;d=json.load(open('/tmp/dec.json'));u=d['usage'];print('tokens:',u['completion_tokens'],'wall',round(float(0),2))"
