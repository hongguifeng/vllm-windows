#!/usr/bin/env bash
# Switch MTP depth from 5 to 3 draft tokens.
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
if cur == '3':
    print('already k=3')
else:
    src = src[:m.start(1)] + '3' + src[m.end(1):]
    open(p, 'w').write(src)
    print(f'patched k={cur} -> k=3')
PY
grep -n "speculative-config" "$S"
bash -n "$S" && echo "syntax ok"
