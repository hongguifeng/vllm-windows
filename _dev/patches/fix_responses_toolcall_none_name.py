"""Stop `/v1/responses` streaming from 500-ing when a tool_call has no name yet.

Observed on 2026-09-21 18:08:50 (see `.workbuddy/memory/2026-09-21.md`):

    pydantic_core.ValidationError: 1 validation error for ResponseFunctionToolCallItem
    name  Input should be a valid string [type=string_type, input_value=None, ...]
      -> emit_simple_tool_call_open        (streaming_events.py:1040)
      -> _process_simple_streaming_events  (serving.py:1401)
    RuntimeError: Caught handled exception, but response already started.

Root cause (reproduced offline with `_repro_responses_toolcall.py`):

  `SimpleStreamingEventProcessor.resolve_target_state()` only checks that
  `tool_calls[0].function is not None`, so when a streamed delta carries a
  tool_call whose `function.name` is still unset it happily returns TOOL_CALL.
  `open()` then calls

      resolve_responses_tool_call_name(tool_call.function.name, ...)

  which ends in `tool_parsers/utils.py`:

      return name_map.get(name, ResponsesToolCallName(name=name))   # no None guard

  i.e. it rebuilds `name=None` verbatim, and `ResponseFunctionToolCallItem`
  requires `name: str`.

Naming is one-shot in the stream: the parser emits `function.name` on the FIRST
delta of a tool call only, and every later delta carries argument chunks with
`function.name=None`.  So the name may only be required when *entering*
TOOL_CALL -- never on the deltas that belong to a call we are already inside:

  * requiring it on every delta drops every argument chunk after the first
    (`serving.py` skips NONE deltas) and `response.function_call_arguments.done`
    then reports a truncated `arguments` string.  Measured live on 2026-09-21
    19:0x: `arguments = '{"path": "'` while the non-streamed response carried
    the full `{"path": "D:/code/vllm-windows/AGENTS.md", "offset": 12}`.
  * requiring it while entering defers the item until the name shows up.  The
    argument chunks that arrived before it are buffered in `pending_tool_args`
    and replayed right after the open event, so nothing is lost.

Fix -- four idempotent hunks:

  1. `__init__`: buffer for argument chunks seen before the name.
  2. `resolve_target_state`: name required only for the transition into
     TOOL_CALL; deltas of the call we are already inside stay TOOL_CALL.
  3. `open()`: replay the buffered chunks once the item opens.
  4. `emit_simple_tool_call_open`: `name or ""` backstop for any other caller
     that can still hand us None.

Usage:
    python fix_responses_toolcall_none_name.py <vllm_pkg_dir> [<another> ...]

    # e.g. both the installed package and the checked-out source tree
    python fix_responses_toolcall_none_name.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

TARGET = os.path.join("entrypoints", "openai", "responses", "streaming_events.py")

# --- 1. __init__: buffer arguments that arrive before the name ----------------
_N1 = b"        self.tool_call_name_map = build_responses_tool_call_name_map(tools)\n"
_R1 = (
    b"        self.tool_call_name_map = build_responses_tool_call_name_map(tools)\n"
    b"        self.pending_tool_args: dict[int | None, str] = {}\n"
)
_M1 = b"        self.pending_tool_args: dict[int | None, str] = {}\n"

# --- 2. resolve_target_state: name only gates the transition into TOOL_CALL ---
# two states of this method exist in the wild: pristine, and the 2026-09-21
# naive guard that broke argument streaming.  Both become the same block.
_N2_PRISTINE = (
    b"        if (\n"
    b"            delta_message.tool_calls\n"
    b"            and delta_message.tool_calls[0].function is not None\n"
    b"        ):\n"
    b"            return _StateType.TOOL_CALL, delta_message.tool_calls[0]\n"
)
_N2_NAIVE = (
    b"        if (\n"
    b"            delta_message.tool_calls\n"
    b"            and delta_message.tool_calls[0].function is not None\n"
    b"            and delta_message.tool_calls[0].function.name\n"
    b"        ):\n"
    b"            return _StateType.TOOL_CALL, delta_message.tool_calls[0]\n"
)
_R2 = (
    b"        tool_calls = delta_message.tool_calls\n"
    b"        if tool_calls and tool_calls[0].function is not None:\n"
    b"            tool_call = tool_calls[0]\n"
    b"            # The name is streamed once, on the first delta of a call; the\n"
    b"            # argument chunks that follow carry function.name=None.\n"
    b"            inside_this_call = (\n"
    b"                self.state.current_state == _StateType.TOOL_CALL\n"
    b"                and self.state.tool_call_index == tool_call.index\n"
    b"            )\n"
    b"            if tool_call.function.name or inside_this_call:\n"
    b"                return _StateType.TOOL_CALL, tool_call\n"
    b"            # Not inside the call yet and no name: hold the chunk back so a\n"
    b"            # NONE delta does not swallow it.\n"
    b"            self.pending_tool_args[tool_call.index] = self.pending_tool_args.get(\n"
    b'                tool_call.index, ""\n'
    b"            ) + (tool_call.function.arguments or \"\")\n"
)
_M2 = b"            if tool_call.function.name or inside_this_call:\n"

# --- 3. open(): replay the buffered chunks after the open event ---------------
_N3 = (
    b"            return handlers.open_fn(\n"
    b"                self.state,\n"
    b"                call_name.name,\n"
    b"                tool_call.index,\n"
    b"                call_name.namespace,\n"
    b"            )\n"
    b"        return handlers.open_fn(self.state)\n"
)
_R3 = (
    b"            events = handlers.open_fn(\n"
    b"                self.state,\n"
    b"                call_name.name,\n"
    b"                tool_call.index,\n"
    b"                call_name.namespace,\n"
    b"            )\n"
    b'            pending = self.pending_tool_args.pop(tool_call.index, "")\n'
    b"            if pending:\n"
    b"                events.extend(handlers.delta_fn(self.state, pending))\n"
    b"            return events\n"
    b"        return handlers.open_fn(self.state)\n"
)
_M3 = b'            pending = self.pending_tool_args.pop(tool_call.index, "")\n'

# --- 4. emit_simple_tool_call_open: backstop against name=None ----------------
_N4 = b"                call_id=state.tool_call_id,\n                name=name,\n"
_R4 = b"                call_id=state.tool_call_id,\n                name=name or \"\",\n"
_M4 = b'                name=name or "",\n'

_EDITS = (
    (
        "init pending_tool_args",
        _M1,
        ((_N1, _R1),),
    ),
    (
        "resolve_target_state guard",
        _M2,
        ((_N2_PRISTINE, _R2), (_N2_NAIVE, _R2)),
    ),
    (
        "open() replay buffered args",
        _M3,
        ((_N3, _R3),),
    ),
    (
        "open() name backstop",
        _M4,
        ((_N4, _R4),),
    ),
)


def patch(root: str) -> None:
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        return

    with open(path, mode="rb") as file:
        raw = file.read()

    eol = b"\r\n" if b"\r\n" in raw else b"\n"

    for label, marker_lf, variants in _EDITS:
        if marker_lf.replace(b"\n", eol) in raw:
            print(f"already patched [{label}]: {path}")
            continue

        for needle_lf, repl_lf in variants:
            needle = needle_lf.replace(b"\n", eol)
            if needle in raw:
                raw = raw.replace(
                    needle, repl_lf.replace(b"\n", eol), 1
                )
                print(f"patched [{label}]: {path}")
                break
        else:
            print(f"WARN: anchor not found [{label}], left untouched: {path}")

    with open(path, mode="wb") as file:
        file.write(raw)


def main() -> int:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        return 2
    for root in roots:
        patch(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
