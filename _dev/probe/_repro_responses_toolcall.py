"""Regression harness for `/v1/responses` streamed tool calls.

Replays delta sequences through the real state machine in the same order
`_process_simple_streaming_events` (serving.py:1393-1408) drives it, then
checks the emitted `response.function_call_arguments.done` payload.

Context -- two defects were found here on 2026-09-21:

  * 18:08:50  a delta whose tool_call had `function.name=None` reached
              `open()`, and `ResponseFunctionToolCallItem(name=None)` blew up
              pydantic -> SSE stream aborted + "response already started".
  * 19:0x     the first attempt at a guard required `function.name` on EVERY
              delta.  Naming is one-shot in the stream (the name rides the
              first delta of a call only), so every later argument chunk was
              classified NONE and dropped -> truncated `arguments`.

Case 1 is the regression that proves the guard belongs on the transition into
TOOL_CALL only; case 2 proves the pre-name chunks get buffered and replayed.

Run from OUTSIDE the repo (the source tree would shadow site-packages):
    cd ~ && D:/code/vllm-windows/.venv/Scripts/python.exe D:/code/vllm-windows/_repro_responses_toolcall.py
"""

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
    split_delta,
)

ARGS_DONE = "response.function_call_arguments.done"
ARGS_DELTA = "response.function_call_arguments.delta"
ITEM_ADDED = "response.output_item.added"


def tc_delta(index, name, arguments, call_id="call_1"):
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                id=call_id,
                type="function",
                index=index,
                function=DeltaFunctionCall(name=name, arguments=arguments),
            )
        ]
    )


def run(deltas):
    """Drive the processor exactly like the streaming serving loop does."""
    proc = SimpleStreamingEventProcessor()
    events = []
    for dm in deltas:
        for atom in split_delta(dm):
            target, tool_call = proc.resolve_target_state(atom)
            if target == _StateType.NONE:
                continue
            if proc.needs_transition(target, tool_call):
                events.extend(proc.close_current())
                events.extend(proc.open(target, tool_call))
            events.extend(proc.emit_delta(atom, None))
    events.extend(proc.close_current())
    return events


def summarise(events):
    names = [
        e.item.name
        for e in events
        if e.type == ITEM_ADDED and getattr(e.item, "type", None) == "function_call"
    ]
    done = [e.arguments for e in events if e.type == ARGS_DONE]
    deltas = [e.delta for e in events if e.type == ARGS_DELTA]
    return names, done, deltas


def check(label, deltas, expect_names, expect_args):
    print(f"--- {label} ---")
    try:
        events = run(deltas)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] crashed: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return False

    names, done, deltas = summarise(events)
    print(f"  items opened      : {names}")
    print(f"  arguments deltas  : {deltas}")
    print(f"  arguments (done)  : {done}")
    print(f"  replayed from all deltas: {''.join(deltas)!r}")

    ok = True
    if names != expect_names:
        print(f"  [FAIL] names {names} != {expect_names}")
        ok = False
    if done != expect_args:
        print(f"  [FAIL] arguments {done} != {expect_args}")
        ok = False
    if ok:
        print("  [PASS]")
    return ok


FULL = '{"path": "D:/code/vllm-windows/AGENTS.md", "offset": 12}'

results = []

# 1. normal stream: the name rides the first delta, then argument chunks only.
#    Regression for the 19:0x truncation -- this used to yield '{"path": "'.
results.append(
    check(
        "normal: name first, then argument chunks",
        [
            tc_delta(0, "read_file", ""),
            tc_delta(0, None, '{"path": "D:/code'),
            tc_delta(0, None, "/vllm-windows/AGENTS.md"),
            tc_delta(0, None, '", "offset": 12}'),
        ],
        ["read_file"],
        [FULL],
    )
)

# 2. crash case: argument chunks arrive before the name is streamed.
results.append(
    check(
        "name arrives late (the 18:08:50 crash)",
        [
            tc_delta(0, None, '{"path": "D:/code'),
            tc_delta(0, None, "/vllm-windows/AGENTS.md"),
            tc_delta(0, "read_file", '", "offset": 12}'),
        ],
        ["read_file"],
        [FULL],
    )
)

# 3. the name never shows up at all: no crash, and no half-built item either.
results.append(
    check(
        "name never streamed",
        [tc_delta(0, None, '{"path": "D:/x"}')],
        [],
        [],
    )
)

# 4. two consecutive tool calls, each with its own name.
results.append(
    check(
        "two consecutive tool calls",
        [
            tc_delta(0, "read_file", '{"path": "a"}', call_id="call_a"),
            tc_delta(0, None, ""),
            tc_delta(1, "write_file", '{"path": "b"}', call_id="call_b"),
        ],
        ["read_file", "write_file"],
        ['{"path": "a"}', '{"path": "b"}'],
    )
)

# 5. text before the tool call -> content item, then the function_call item.
results.append(
    check(
        "content then tool call",
        [
            DeltaMessage(content="Let me read that file."),
            tc_delta(0, "read_file", ""),
            tc_delta(0, None, FULL),
        ],
        ["read_file"],
        [FULL],
    )
)

print()
print(f"=== {sum(results)}/{len(results)} passed ===")
raise SystemExit(0 if all(results) else 1)
