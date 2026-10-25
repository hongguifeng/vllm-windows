#!/usr/bin/env bash
# Switch MTP depth back to 5 draft tokens.
set -euo pipefail
S=/home/hong/pp-recipe/start.sh
python3 - "$S" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
m = re.search(r'num_speculative_tokens":(\d+)', src)
if not m:
    print('spec config not found'); raise SystemExit(1)
cur = m.group(1)
if cur == '5':
    print('already k=5')
else:
    src = src[:m.start(1)] + '5' + src[m.end(1):]
    open(p, 'w').write(src)
    print(f'patched k={cur} -> k=5')
PY
grep -n "speculative-config" "$S"
bash -n "$S" && echo "syntax ok"
