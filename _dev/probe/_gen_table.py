#!/usr/bin/env python3
"""Regenerate the request table source from a JSON dump of the parsed table.

Every physical line stays inside 88 columns by breaking the prompt at newline
escapes, and because the source is generated from the parsed values the table can
only come back identical to the snapshot. Section comments are rebuilt wherever the
class changes, which is the only thing the dump cannot carry.
"""
import json
import sys

LIMIT = 88
INDENT = "    "
CLASS_ORDER = ("edit", "gen", "explain", "copy", "prose", "short")
HEADER = '''#!/usr/bin/env python3
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

REQUESTS = ['''


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def wrap(body: str, budget: int) -> list[str]:
    """Break an escaped body into physical lines within budget."""
    units = []
    for chunk in body.split("\\n"):
        units.append(chunk + "\\n")
    if units:
        units[-1] = units[-1][:-2]
    split: list[str] = []
    for unit in units:
        while len(unit) > budget:
            cut = unit.rfind(" ", 0, budget)
            keep = 1
            if cut < 0:
                cut = unit.rfind(",", 0, budget)
            if cut < 0:
                cut = unit.rfind("|", 0, budget)
            if cut < 0:
                cut, keep = budget, 0
            while cut > keep and unit[cut - keep] == "\\":
                cut -= 1
            split.append(unit[:cut + keep])
            unit = unit[cut + keep:]
        if unit:
            split.append(unit)
    lines: list[str] = []
    cur = ""
    for piece in split:
        if cur and len(cur) + len(piece) > budget:
            lines.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        lines.append(cur)
    return [line for line in lines if line]


def emit(kind: str, prompt: str, tokens: int) -> str:
    body = escape(prompt)
    tail = f", {tokens}),"
    # One conservative budget for every physical line: the opening prefix is
    # longest on the first line and the numeric tail sits on the last.
    pieces = wrap(body, LIMIT - len(INDENT) - len('("explain", "') - 1)
    first_cap = LIMIT - len(INDENT) - len('("explain", "') - 1
    cont_cap = LIMIT - len(INDENT) - 3 - len(tail)
    lines: list[str] = []
    cur = ""
    for piece in pieces:
        cap = first_cap if not lines else cont_cap
        if cur and len(cur) + len(piece) > cap:
            lines.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        lines.append(cur)
    out = [f'{INDENT}("{kind}", "{lines[0]}"']
    for line in lines[1:]:
        out.append(f'{INDENT} "{line}"')
    out[-1] += tail
    return "\n".join(out)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
    target = sys.argv[2]
    parts = [HEADER]
    previous = None
    for index, (kind, prompt, tokens) in enumerate(data):
        if kind != previous:
            if previous is not None:
                parts.append("")
            total = sum(1 for k, _, _ in data if k == kind)
            parts.append(f"{INDENT}# ---- {kind}: {total} ----")
        elif index:
            parts.append("")
        parts.append(emit(kind, prompt, tokens))
        previous = kind
    parts.append("]\n\nCLASSES = " + repr(CLASS_ORDER) +
                 "\n\n\ndef table() -> list[tuple[str, str, int]]:\n"
                 '    """Return the fixed request table in a fixed order."""\n'
                 "    return list(REQUESTS)\n")
    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(parts))
    print(f"emitted {len(data)} requests to {target}")


if __name__ == "__main__":
    main()
