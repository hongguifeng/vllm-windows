#!/usr/bin/env python3
"""Held-out request table for the MTP depth campaign.

Six classes, weighted toward code because depth is most likely to fail there. A
fast configuration would otherwise pull more requests into the same wall-clock
window and see a different prompt mix than a slow one, so the table is fixed and
one pass over it is one measurement block. Temperature is zero everywhere and the
prompts are written to run out of tokens rather than stop on EOS, so the decode
length is set by the request and not by the model's mood.

Classes:
  edit    change a given snippet and print the changed code, novel tokens
  gen     write a function from a specification, novel tokens
  explain narrate what a snippet does, medium acceptance
  copy    reproduce a block with mechanical edits, high acceptance
  prose   write connected prose, medium acceptance
  short   a handful of tokens, where prefill can erase any decode benefit
"""

REQUESTS = [
    # ---- edit: 14 ----
    ("edit", "Rename every identifier in this Python function to a snake_case form "
     "prefixed with `ax_`, keep behaviour identical, and print the whole "
     "function. Do not add comments.\n```python\n"
     "def CalcCost(q, rate, tax):\n    total = q * rate\n"
     "    return total * (1 + tax)\n```", 420),

    ("edit", "Convert this JavaScript callback into an async/await function, print "
     "the result, and keep the same error handling.\n```js\n"
     "function load(id, cb) { fetch('/x/' + id).then(r => r.json()).then(d "
     "=> cb(null, d), e => cb(e)); }\n```", 460),

    ("edit", "Change this C loop to use an index of type size_t and bounds-checked "
     "access, print the full function, nothing else.\n```c\n"
     "int sum(int *v, int n) { int s = 0; for (int i = 0; i < n; i++) s += "
     "v[i]; return s; }\n```", 430),

    ("edit", "Replace the mutable default argument in this signature with a None "
     "sentinel and print the corrected function plus a two-line docstring.\n"
     "```python\ndef append(item, bucket=[]):\n    bucket.append(item)\n"
     "    return bucket\n```", 400),

    ("edit", "Make this Rust function return Result instead of panicking on the "
     "unwrap, and print the whole function.\n```rust\n"
     "fn parse(s: &str) -> u32 { s.parse::<u32>().unwrap() }\n```", 450),

    ("edit", "Rewrite this Go goroutine so the channel is always closed exactly "
     "once, and print the full function.\n```go\n"
     "func pump(ch chan int) { for i := 0; i < 5; i++ { ch <- i } close(ch) "
     "}\n```", 440),

    ("edit", "Add type hints to every parameter and the return of this function and "
     "print it back in full, unchanged otherwise.\n```python\n"
     "def mix(a, b, n, scale): return [(x * scale + y) % n for x, y in "
     "zip(a, b)]\n```", 410),

    ("edit", "Convert this SQL string building into a parameterized query in Python "
     "and print the whole function.\n```python\ndef find(db, name):\n"
     "    return db.query(\"select id from t where name = '\" + name + "
     "\"'\")\n```", 420),

    ("edit", "Change this bash script to quote every variable expansion and print "
     "the whole script.\n```bash\n#!/bin/bash\ncd $1\ncp $2 $3/ch.log\n"
     "echo done $4\n```", 430),

    ("edit", "Give this Java method a guard clause that returns early on null and "
     "print the whole method.\n```java\n"
     "int area(int w, int h) { return w * h; }\n```", 400),

    ("edit", "Turn this pandas chain into three separate statements with named "
     "intermediates and print the whole block.\n```python\n"
     "out = df.dropna().groupby('k').agg({'v': "
     "'mean'}).reset_index().sort_values('v')\n```", 460),

    ("edit", "Replace the recursion here with an explicit stack and print the whole "
     "function.\n```python\ndef flat(n):\n    if not n: return []\n"
     "    return [n[0]] + flat(n[1:])\n```", 430),

    ("edit", "Add a timeout to this requests call, print the whole function, keep "
     "the retry count at three.\n```python\ndef get(url):\n"
     "    return requests.get(url).text\n```", 410),

    ("edit", "Make this lock acquisition exception-safe with try/finally and print "
     "the whole function.\n```python\ndef work(m, fn):\n    m.acquire()\n"
     "    fn()\n    m.release()\n```", 440),

    # ---- gen: 12 ----
    ("gen", "Write a Python function `chunk(it, n)` that splits any iterable into "
     "lists of at most n items and returns them. Print the whole function "
     "and nothing else.", 500),

    ("gen", "Write a Python function that parses a cron expression into a dict of "
     "five fields and rejects malformed input with ValueError. Print the "
     "whole function.", 520),

    ("gen", "Write a function `moving_average(values, window)` that returns a list "
     "of rolling means without any third-party library. Print the whole "
     "function.", 480),

    ("gen", "Write a function that reads a file in fixed-size blocks and yields "
     "each block, closing the file in a finally block. Print the whole function.", 470),

    ("gen", "Write a function that merges two sorted lists into one sorted list "
     "without using sort or reverse. Print the whole function.", 490),

    ("gen", "Write a function `safe_div(a, b)` that returns None instead of "
     "raising on division by zero, and print it with a one-line docstring.", 430),

    ("gen", "Write a function that converts bytes into a human readable size "
     "string with binary units up to PiB. Print the whole function.", 480),

    ("gen", "Write a function that clamps an integer into a closed interval and "
     "raises ValueError if the interval is empty. Print the whole function.", 440),

    ("gen", "Write a function `dedupe_keep_order(items)` that removes duplicates "
     "while preserving first-seen order. Print the whole function.", 450),

    ("gen", "Write a function that validates an IPv4 dotted quad without using the "
     "ipaddress module and returns True or False. Print the whole function.", 500),

    ("gen", "Write a function that counts how many times a substring appears in a "
     "text allowing overlaps. Print the whole function.", 460),

    ("gen", "Write a function that groups records by a key field into a dict of "
     "lists and prints the whole function.", 480),

    # ---- explain: 8 ----
    ("explain", "Explain line by line what this decorator does, then say what breaks "
     "if it is applied to a generator.\n```python\ndef once(fn):\n"
     "    done = []\n    def inner(*a):\n        if not done:\n"
     "            done.append(fn(*a))\n        return done[0]\n"
     "    return inner\n```", 460),

    ("explain", "Explain what this comprehension produces for a three-element input "
     "and why the inner loop runs in reverse.\n```python\n"
     "[v for row in table for v in reversed(row) if v]\n```", 440),

    ("explain", "Explain what this bitwise expression computes and give two concrete "
     "numeric examples.\n```python\n"
     "((x ^ (x >> 1)) & 0x55555555) + (x & 0xAAAAAAAA)\n```", 470),

    ("explain", "Explain how this context manager behaves when __exit__ returns True "
     "and what the caller sees.\n```python\nclass Swallow:\n"
     "    def __enter__(self): return self\n"
     "    def __exit__(self, *exc): return True\n```", 450),

    ("explain", "Explain what this sort key does and why it is stable for equal "
     "scores.\n```python\nsorted(rows, key=lambda r: (-r.score, r.name))\n```", 430),

    ("explain", "Explain what `__slots__` changes about this class and what it does "
     "not change.\n```python\nclass P:\n    __slots__ = ('a', 'b')\n```", 440),

    ("explain", "Explain why this generator expression is lazily evaluated and what "
     "happens when sum consumes it twice.\n```python\n"
     "g = (i * i for i in range(1000))\nsum(g), sum(g)\n```", 450),

    ("explain", "Explain what this memoization does and why it leaks memory over a "
     "long run.\n```python\ndef fib(n, memo={}):\n"
     "    if n in memo: return memo[n]\n"
     "    memo[n] = fib(n - 1) + fib(n - 2)\n    return memo[n]\n```", 460),

    # ---- copy: 8 ----
    ("copy", "Reproduce this block verbatim, then repeat it once with every `i` "
     "renamed to `j`. Print both blocks.\n```python\nfor i in range(12):\n"
     "    if i % 3 == 0:\n        print(i, i * i)\n    elif i % 3 == 1:\n"
     "        print(i, i + 7)\n    else:\n        print(i, -i)\n```", 500),

    ("copy", "Reproduce this block verbatim, then repeat it once with the range "
     "bound changed from 40 to 60. Print both blocks.\n```python\n"
     "def run():\n    out = []\n    for i in range(40):\n"
     "        out.append(i * 3 - 5)\n    return out\n```", 480),

    ("copy", "Reproduce this table verbatim, then repeat it once with the third "
     "column doubled. Print both tables.\na|7|3.5|yes\nb|11|4.0|no\n"
     "c|5|2.5|yes\nd|19|6.5|no\ne|8|3.0|yes", 460),

    ("copy", "Reproduce this list of twelve settings verbatim, then repeat it once "
     "with every value doubled. Print both lists.\n"
     "timeout=30, retries=2, workers=8, depth=4, buffer=65536, gap=12, "
     "margin=7, scale=3, window=5, offset=9, floor=1, ceil=99", 520),

    ("copy", "Reproduce these twelve commit messages verbatim, then repeat them "
     "with the ticket numbers incremented by one. Print both lists.\n"
     "fix clamp in scaler (#101)\nadd test for router (#104)\n"
     "drop dead flag (#108)\ntune worker count (#112)\n"
     "repair cache key (#115)\nsplit build step (#119)", 500),

    ("copy", "Reproduce this sequence of forty numbers exactly, one per line.\n"
     "3 7 11 5 9 13 6 10 14 8 12 4 6 9 13 5 11 7 10 14 3 8 12 6 9 13 5 11 7 "
     "10 4 12 8 3 9 13 6 11 5 10", 480),

    ("copy", "Reproduce this schema verbatim, then repeat it once with every "
     "integer field made nullable. Print both schemas.\n"
     "id:int, name:str, qty:int, price:float, tag:str, count:int, "
     "flag:bool, delta:int", 440),

    ("copy", "Reproduce this schedule verbatim, then repeat it with every start "
     "time shifted one hour later. Print both schedules.\n08:00 build\n"
     "09:30 test\n11:00 lint\n13:00 package\n15:30 deploy\n17:00 verify", 460),

    # ---- prose: 10 ----
    ("prose", "Write a connected account of why a small team should prefer a boring "
     "database over a novel one, in three paragraphs. No headings.", 420),

    ("prose", "Write three paragraphs on what makes a code review useful to the "
     "author rather than to the reviewer. No headings.", 410),

    ("prose", "Write a short essay on why error messages fail most often at the "
     "boundary between two components. Three paragraphs.", 430),

    ("prose", "Describe the difference between a benchmark and a test, in three "
     "paragraphs, without using the word performance.", 420),

    ("prose", "Write three paragraphs on what breaks when a cache is added to a "
     "system that was correct without it.", 420),

    ("prose", "Write an account of a migration that failed, in three paragraphs, "
     "from the point of view of the person who ran it.", 430),

    ("prose", "Write three paragraphs on how a queue changes the shape of a service "
     "dependency graph.", 420),

    ("prose", "Write three paragraphs on why the first version of a feature is "
     "usually the cheapest and the last the most expensive.", 420),

    ("prose", "Describe what an on-call engineer learns in the first month, in three "
     "paragraphs.", 420),

    ("prose", "Write three paragraphs on the difference between a fallback and a "
     "circuit breaker.", 420),

    # ---- short: 8 ----
    ("short", "Answer with the one word that names this error: trying to access a "
     "memory location that your process does not have permission to access.", 24),

    ("short", "Name the sorting algorithm described here in one word: partition, "
     "recurse on both sides, no merging step.", 24),

    ("short", "Give the single term for a lock that is held for an entire "
     "transaction and released at commit.", 32),

    ("short", "Answer in one word: what is the name of the pattern where a function "
     "returns a new copy of its input rather than mutating it?", 24),

    ("short", "Name in one word the kind of memory where a freed pointer is still "
     "readable and then reused.", 24),

    ("short", "Give the single word for the phase where a model runs all prompt "
     "tokens before it generates any new ones.", 24),

    ("short", "Answer in one word: the term for tokens the draft model proposes and "
     "the target model declines.", 24),

    ("short", "Name in one word the quantity that divides total generated tokens by "
     "wall-clock seconds.", 24),
]

CLASSES = ('edit', 'gen', 'explain', 'copy', 'prose', 'short')


def table() -> list[tuple[str, str, int]]:
    """Return the fixed request table in a fixed order."""
    return list(REQUESTS)
