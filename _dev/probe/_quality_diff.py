"""Differential quality check for the host-ids (PLE) path.

The engine is not bit-deterministic across runs (repeated greedy requests differ),
so equality of raw outputs cannot prove anything. This runs the same small,
verifiable suite against two ports and compares pass counts: math word problems
with integer answers, coding tasks with unit tests, and instruction/format checks.
A broken PLE id path produces wrong embeddings, which show up here as a collapse,
not as a subtle drift.

    python _dev/probe/_quality_diff.py --a 9393 --b 9394 --math 24
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path

MODEL = "Qwen3.8-Flash-Next"


def post(port: int, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(port: int, prompt: str, max_tokens: int = 512) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return (post(port, body)["choices"][0]["message"].get("content") or "").strip()


def last_number(text: str) -> int | None:
    nums = re.findall(r"-?\d[\d,]*", text.replace(" ", ""))
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", "").rstrip(".,"))
    except ValueError:
        return None


def math_suite(n: int, seed: int = 7) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:
            a, b, c = rng.randint(3, 40), rng.randint(2, 30), rng.randint(2, 20)
            q = (
                f"A shop had {a} boxes. It sold {b} boxes in the morning and then "
                f"received {c} more. How many boxes does it have now?"
            )
            ans = a - b + c
        elif kind == 1:
            price, qty, disc = rng.randint(5, 60), rng.randint(3, 12), rng.randint(1, 15)
            q = (
                f"Each ticket costs {price} dollars. A group buys {qty} tickets and gets "
                f"a discount of {disc} dollars on the total. How much do they pay?"
            )
            ans = price * qty - disc
        elif kind == 2:
            groups, per_bus, extra = rng.randint(2, 8), rng.randint(4, 12), rng.randint(1, 9)
            total = groups * per_bus  # divisible, so "split evenly" is true
            q = (
                f"{total} students are split evenly into {groups} buses, and then "
                f"{extra} more students join one bus. How many students are on that bus?"
            )
            ans = per_bus + extra
        else:
            start, per_day, days = rng.randint(10, 50), rng.randint(3, 12), rng.randint(2, 7)
            q = (
                f"A tank holds {start} litres. It is filled by {per_day} litres per day for "
                f"{days} days, then 7 litres leak out. How many litres are in the tank?"
            )
            ans = start + per_day * days - 7
        items.append((q + " Reply with the final number only.", ans))
    return items


CODE_TASKS = [
    (
        "Write a Python function `add(a, b)` returning the sum. Reply with code only.",
        "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    ),
    (
        "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumerics. "
        "Reply with code only.",
        "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('abc')",
    ),
    (
        "Write a Python function `fact(n)` returning n! iteratively. Reply with code only.",
        "assert fact(0) == 1\nassert fact(5) == 120",
    ),
    (
        "Write a Python function `count_vowels(s)` returning the number of vowels. "
        "Reply with code only.",
        "assert count_vowels('hello') == 2\nassert count_vowels('XYZ') == 0",
    ),
    (
        "Write a Python function `dedupe(xs)` that removes duplicates preserving order. "
        "Reply with code only.",
        "assert dedupe([1,2,1,3]) == [1,2,3]\nassert dedupe([]) == []",
    ),
    (
        "Write a Python function `fizz(n)` returning 'Fizz' for multiples of 3, 'Buzz' for 5, "
        "'FizzBuzz' for both, else str(n). Reply with code only.",
        "assert fizz(3) == 'Fizz'\nassert fizz(5) == 'Buzz'\nassert fizz(15) == 'FizzBuzz'\nassert fizz(7) == '7'",
    ),
]

FORMAT_TASKS = [
    (
        "Reply with a JSON object only, keys name and age, values 'Ada' and 36.",
        lambda s: (lambda d: d.get("name") == "Ada" and d.get("age") == 36)(json.loads(s)),
    ),
    (
        "Reply with exactly three bullet lines, each starting with '- ', listing three colours.",
        lambda s: len([l for l in s.splitlines() if l.strip().startswith("- ")]) == 3,
    ),
    (
        "Reply with the word BANANA repeated exactly four times, space separated, nothing else.",
        lambda s: s.strip().upper() == "BANANA BANANA BANANA BANANA",
    ),
    (
        "Reply with a JSON array of the integers 1 to 5 only.",
        lambda s: json.loads(s) == [1, 2, 3, 4, 5],
    ),
]


def code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_code(src: str, tests: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mod.py"
        p.write_text(src + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, str(p)], capture_output=True, timeout=20, text=True
            )
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def eval_port(port: int, math_items, label: str) -> dict:
    res = {"math": [0, 0], "code": [0, 0], "format": [0, 0]}
    for i, (q, want) in enumerate(math_items):
        got = last_number(ask(port, q, 256))
        res["math"][1] += 1
        res["math"][0] += int(got == want)
        if (i + 1) % 8 == 0:
            print(f"  [{label}] math {i + 1}/{len(math_items)}: {res['math'][0]} ok", flush=True)
    for q, tests in CODE_TASKS:
        src = code_block(ask(port, q, 512))
        res["code"][1] += 1
        res["code"][0] += int(run_code(src, tests))
    for q, check in FORMAT_TASKS:
        out = ask(port, q, 128)
        res["format"][1] += 1
        try:
            ok = bool(check(out))
        except Exception:
            ok = False
        res["format"][0] += int(ok)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="9393", help="baseline port")
    ap.add_argument("--b", default="9394", help="opt-in port")
    ap.add_argument("--math", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    math_items = math_suite(args.math, args.seed)

    print(f"baseline :{args.a} (default path)")
    a = eval_port(int(args.a), math_items, "base")
    print(f"opt-in   :{args.b} (host ids)")
    b = eval_port(int(args.b), math_items, "host")

    rows = []
    for k in ("math", "code", "format"):
        rows.append((k, a[k][0], a[k][1], b[k][0], b[k][1]))
    print("\n| category | baseline | opt-in |")
    print("|---|---:|---:|")
    for k, oka, na, okb, nb in rows:
        print(f"| {k} | {oka}/{na} | {okb}/{nb} |")
    total_a = sum(v[0] for v in a.values())
    total_b = sum(v[0] for v in b.values())
    total_n = sum(v[1] for v in a.values())
    print(f"| **total** | **{total_a}/{total_n}** | **{total_b}/{total_n}** |")
    return 0


if __name__ == "__main__":
    sys.exit(main())


