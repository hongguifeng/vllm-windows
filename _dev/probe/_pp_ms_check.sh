#!/usr/bin/env bash
# Compare the ModelScope copy of the checkpoint against the pinned HF revision,
# and measure ModelScope throughput.
set -uo pipefail
MS=https://modelscope.cn
M=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
curl -s --max-time 40 --noproxy '*' "$MS/api/v1/models/$M/repo/files?Revision=master&Root=" -o /tmp/ms_files.json
python3 - <<'PY'
import json
ms = json.load(open('/tmp/ms_files.json'))['Data']['Files']
ms_map = {f['Path']: (f['Size'], f.get('Sha256')) for f in ms}
hf = {}
for line in open('/tmp/pp_files.txt'):
    size, path = line.split(None, 1)
    hf[path.strip()] = int(size)
print(f"ModelScope files: {len(ms_map)}   HF files: {len(hf)}")
missing = [p for p in hf if p not in ms_map]
extra = [p for p in ms_map if p not in hf]
diff = [(p, hf[p], ms_map[p][0]) for p in hf if p in ms_map and ms_map[p][0] != hf[p]]
print("missing on ModelScope:", missing or "none")
print("extra on ModelScope:", extra or "none")
print("size mismatches:", diff or "none")
print("total GiB: HF %.1f  ModelScope %.1f" % (sum(hf.values()) / 2**30, sum(v[0] for v in ms_map.values()) / 2**30))
big = sorted(((v[0], k, v[1]) for k, v in ms_map.items() if k.endswith('.safetensors')), reverse=True)[:3]
for s, k, sha in big:
    print(f"  {s/2**30:6.2f} GiB  {k}  sha256={sha}")
PY
BIG=$(python3 -c "
import json
ms=json.load(open('/tmp/ms_files.json'))['Data']['Files']
s=sorted(((f['Size'],f['Path']) for f in ms if f['Path'].endswith('.safetensors')),reverse=True)[0][1]
print(s)")
echo "--- speed test on $BIG (direct, 60 MiB range) ---"
curl --noproxy '*' -s -r 0-62914560 -o /dev/null --max-time 40 \
  -w "  direct: %{speed_download} B/s http=%{http_code}\n" \
  "$MS/models/$M/resolve/master/$BIG"
echo "--- speed test (2 parallel ranges) ---"
for i in 0 1; do
  s=$((i * 33554432)); e=$((s + 33554431))
  curl --noproxy '*' -s -r "$s-$e" -o /dev/null --max-time 40 \
    -w "  conn$i: %{speed_download} B/s\n" "$MS/models/$M/resolve/master/$BIG" &
done
wait
