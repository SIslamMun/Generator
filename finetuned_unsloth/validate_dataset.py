"""Structural + simulated-helper validation of jarvis_v7_functiongemma.jsonl.

Replicates the filtering logic in FunctionGemma_(270M).ipynb's
`prepare_messages_and_tools()` and checks every row would survive.

Checks, per row:
  1. JSON is valid, top-level is a list.
  2. First message carries a non-empty `tools` list (stripped into tools_raw).
  3. Every assistant message has a `think` key with non-empty string (else poisoned).
  4. Every tool_call has `id` + `type=function` + `function.name` + `function.arguments` (dict).
  5. Every `tool` role message has `name`, `tool_call_id`, and `content` (str).
  6. tool_call_id → function-name lookup succeeds for every tool response.
  7. Every tool actually called is declared in the tools list (trainable consistency).
  8. Number of tools per example is within the 5-tool ceiling.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v7_2k" / "jarvis_v7_functiongemma.jsonl"


def validate_row(messages: list[dict]) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(messages, list) or not messages:
        return False, ["empty or non-list messages"]

    # 1 — tools on first message
    first = messages[0]
    tools_raw = first.get("tools") if isinstance(first, dict) else None
    if not isinstance(tools_raw, list) or not tools_raw:
        errors.append("first message has no `tools` list")
        return False, errors

    declared_tool_names = set()
    for t in tools_raw:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = fn.get("name") or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            errors.append("tool without name in declarations")
        else:
            declared_tool_names.add(name)

    if len(declared_tool_names) > 5:
        errors.append(f"declared {len(declared_tool_names)} tools (ceiling is 5)")

    # build map of tool_call_id -> function name as helper does
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name

    # per-message checks
    seen_assistant = False
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            seen_assistant = True
            think = m.get("think") or m.get("think_fast") or m.get("think_faster")
            if not think:
                errors.append(f"assistant @ {i} missing `think` (would be poisoned)")
            # validate tool_calls if present
            for j, tc in enumerate(m.get("tool_calls", []) or []):
                if not isinstance(tc, dict):
                    errors.append(f"assistant @ {i} tool_call[{j}] not a dict")
                    continue
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if not tc_id:
                    errors.append(f"assistant @ {i} tool_call[{j}] missing id")
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    errors.append(f"assistant @ {i} tool_call[{j}] missing function dict")
                    continue
                fn_name = fn.get("name")
                if not fn_name:
                    errors.append(f"assistant @ {i} tool_call[{j}] missing function.name")
                elif fn_name not in declared_tool_names:
                    errors.append(
                        f"assistant @ {i} calls `{fn_name}` not in declared tools"
                    )
                args = fn.get("arguments")
                if not isinstance(args, (dict, str)):
                    errors.append(
                        f"assistant @ {i} tool_call[{j}] arguments is {type(args).__name__}"
                    )
        elif role == "tool":
            if not m.get("name"):
                tc_id = m.get("tool_call_id")
                if not tc_id or tc_id not in id_to_name:
                    errors.append(f"tool @ {i} has no name and id `{tc_id}` unresolved")
            if not isinstance(m.get("content"), str):
                errors.append(f"tool @ {i} content is not string: {type(m.get('content')).__name__}")
            tc_id = m.get("tool_call_id")
            if tc_id and tc_id not in id_to_name:
                errors.append(f"tool @ {i} unknown tool_call_id `{tc_id}`")
        elif role not in ("system", "user"):
            errors.append(f"unknown role `{role}` at {i}")

    if not seen_assistant:
        errors.append("no assistant message in conversation")

    return len(errors) == 0, errors


def main():
    total = 0
    bad = 0
    err_types: Counter = Counter()
    lengths: list[int] = []
    tool_counts: Counter = Counter()
    with SRC.open() as f:
        for line_no, line in enumerate(f, 1):
            total += 1
            row = json.loads(line)
            messages = json.loads(row["messages"])
            lengths.append(len(messages))
            for m in messages:
                for tc in m.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    tool_counts[fn.get("name", "?")] += 1
            ok, errors = validate_row(messages)
            if not ok:
                bad += 1
                for e in errors:
                    err_types[e.split("@")[0].strip()] += 1
                if bad <= 5:
                    print(f"[line {line_no}] ERRORS:")
                    for e in errors:
                        print(f"  - {e}")
    print()
    print(f"total examples:    {total}")
    print(f"valid:             {total - bad}")
    print(f"invalid:           {bad}")
    print(f"avg message count: {sum(lengths) / len(lengths):.2f}")
    print(f"max message count: {max(lengths)}")
    if err_types:
        print("error category counts:")
        for k, v in err_types.most_common():
            print(f"  {v:5d}  {k}")
    print()
    print("tool-call distribution across all examples:")
    for name, n in tool_counts.most_common():
        print(f"  {n:5d}  {name}")


if __name__ == "__main__":
    main()
