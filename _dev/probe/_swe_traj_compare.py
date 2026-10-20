"""Compare SWE-bench agent trajectories between two arms (process, not just score).

Reads two arm directories (traj.json per instance), their preds.json, the
subset json (gold patch) and the two swebench reports, then prints per-arm
process statistics, a per-instance table and the behavioural differences on
the instances where the arms disagree.

    python _swe_traj_compare.py --a-run runH9393 --b-run runEH \
        --a-report openai__Qwen3.8-Flash-Next.runH9393.json \
        --b-report openai__qwen3.8-flash-next.runEH.json \
        --subset subset_hard25.json
"""

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

EDIT_PATTERNS = (
    "sed -i",
    "apply_patch",
    "git apply",
    "patch -p",
    "cat >",
    "tee ",
    "python - <<",
    "python3 - <<",
    ">> ",
)
TEST_PATTERNS = ("pytest", "runtests", "manage.py test", "tox", "python -m pytest")


def outcome(report: dict, iid: str) -> str:
    for key, label in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty"),
        ("ambiguous_failure_ids", "ambiguous"),
    ):
        if iid in report.get(key, []):
            return label
    return "unknown"


def patch_files(patch: str) -> set[str]:
    files = set()
    for line in patch.splitlines():
        match = re.match(r"^\+\+\+ b/(.+)$", line)
        if match:
            files.add(match.group(1))
    return files


def load_commands(msgs) -> list[str]:
    commands = []
    for message in msgs:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not calls:
            continue
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except Exception:
                try:
                    calls = ast.literal_eval(calls)
                except Exception:
                    continue
        for call in calls or []:
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                continue
            command = args.get("command")
            if command:
                commands.append(command)
    return commands


def analyze(root: Path, iid: str, preds: dict, gold: dict, report: dict) -> dict:
    path = root / iid / f"{iid}.traj.json"
    row = {
        "iid": iid,
        "outcome": outcome(report, iid),
        "repo": gold.get("repo", "?"),
    }
    if not path.exists():
        row["missing"] = True
        return row
    traj = json.loads(path.read_text())
    msgs = traj.get("messages", [])
    commands = load_commands(msgs)
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    thinking = sum(len(m.get("reasoning_content") or "") for m in msgs)
    patch = (preds.get(iid) or {}).get("model_patch") or ""
    files = patch_files(patch)
    gold_files = patch_files(gold.get("patch") or "")
    first_edit = None
    for index, command in enumerate(commands):
        if any(pattern in command for pattern in EDIT_PATTERNS):
            first_edit = index
            break
    tests = sum(1 for command in commands if any(p in command for p in TEST_PATTERNS))
    row.update(
        steps=len(assistants),
        tool_calls=len(commands),
        thinking=thinking,
        first_edit=first_edit,
        tests=tests,
        patch_len=len(patch),
        files=len(files),
        gold_files=len(gold_files),
        gold_hit=len(files & gold_files),
        gold_frac=(len(files & gold_files) / len(gold_files)) if gold_files else 0.0,
        exit_status=(traj.get("info") or {}).get("exit_status", "?"),
        commands=commands,
        files_set=sorted(files),
        gold_set=sorted(gold_files),
    )
    return row


def describe(label: str, rows: list[dict]) -> None:
    def med(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(values) if values else 0

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    edited = [r for r in rows if r.get("first_edit") is not None]
    print(f"\n== {label}: {len(rows)} episodes ==")
    print(
        f"  steps med {med('steps')} mean {mean('steps'):.1f} | "
        f"tool calls med {med('tool_calls')} mean {mean('tool_calls'):.1f} | "
        f"thinking chars mean {mean('thinking'):.0f}"
    )
    print(
        f"  first edit at call med {med('first_edit')} "
        f"(edited {len(edited)}/{len(rows)}) | tests run mean {mean('tests'):.1f} "
        f"| ran tests {sum(1 for r in rows if r['tests'])}/{len(rows)}"
    )
    print(
        f"  patch len med {med('patch_len')} mean {mean('patch_len'):.0f} | "
        f"files med {med('files')} | touches gold file "
        f"{sum(1 for r in rows if r['gold_hit'])}/{len(rows)} | "
        f"gold files covered mean {mean('gold_frac'):.2f}"
    )


def case_table(title: str, rows_a: list[dict], rows_b: list[dict], only: set[str]) -> None:
    print(f"\n== {title} ==")
    header = (
        f"{'instance':<42}{'A':<11}{'B':<11}{'A steps/edit':<14}"
        f"{'B steps/edit':<14}{'B gold':<8}{'B patch':<9}{'B exit'}"
    )
    print(header)
    for a, b in zip(rows_a, rows_b):
        if a["iid"] not in only:
            continue

        def fmt(row):
            edit = "-" if row.get("first_edit") is None else str(row["first_edit"])
            return row.get("steps", 0), edit

        a_steps, a_edit = fmt(a)
        b_steps, b_edit = fmt(b)
        print(
            f"{a['iid']:<42}{a['outcome']:<11}{b['outcome']:<11}"
            f"{str(a_steps) + '/' + a_edit:<14}{str(b_steps) + '/' + b_edit:<14}"
            f"{str(b.get('gold_hit', 0)) + '/' + str(b.get('gold_files', 0)):<8}"
            f"{b.get('patch_len', 0):<9}{b.get('exit_status', '?')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-run", required=True, type=Path)
    parser.add_argument("--b-run", required=True, type=Path)
    parser.add_argument("--a-report", required=True, type=Path)
    parser.add_argument("--b-report", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=12)
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())
    gold = {entry["instance_id"]: entry for entry in subset}
    preds_a = json.loads((args.a_run / "preds.json").read_text())
    preds_b = json.loads((args.b_run / "preds.json").read_text())
    report_a = json.loads(args.a_report.read_text())
    report_b = json.loads(args.b_report.read_text())

    ids = [entry["instance_id"] for entry in subset]
    rows_a = [analyze(args.a_run, i, preds_a, gold[i], report_a) for i in ids]
    rows_b = [analyze(args.b_run, i, preds_b, gold[i], report_b) for i in ids]

    describe("A (AutoRound 3bpw)", rows_a)
    describe("B (EXL3 5bpw)", rows_b)

    a_res = {r["iid"] for r in rows_a if r["outcome"] == "resolved"}
    b_res = {r["iid"] for r in rows_b if r["outcome"] == "resolved"}
    print(f"\nresolved A={len(a_res)} B={len(b_res)} | A-only={len(a_res - b_res)} B-only={len(b_res - a_res)}")

    case_table("A resolved, B not", rows_a, rows_b, a_res - b_res)
    case_table("B resolved, A not", rows_a, rows_b, b_res - a_res)

    empty_b = {r["iid"] for r in rows_b if r["outcome"] == "empty"}
    if empty_b:
        case_table("B empty patch", rows_a, rows_b, empty_b)

    # Where both resolved: how close is the patch to the gold file set?
    both = a_res & b_res
    if both:
        ga = [r["gold_frac"] for r in rows_a if r["iid"] in both]
        gb = [r["gold_frac"] for r in rows_b if r["iid"] in both]
        print(
            f"\nboth resolved ({len(both)}): gold-file coverage "
            f"A {sum(ga)/len(ga):.2f} vs B {sum(gb)/len(gb):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
