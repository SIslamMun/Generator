"""Hard validator for the NDP tool-calling dataset.

Every tool call in every example must be a structurally valid, schema-perfect
JSON call against the ndp_mcp catalog — that is the whole point of the rebuild.
This script is the gate: it checks each call and FAILS (exit 1) if any row is
invalid, so a broken dataset can never reach training.

Checks per tool call:
  - tool name is in the catalog
  - every argument name is in that tool's schema (no cross-tool leakage)
  - every required parameter is present
  - enum parameters hold only legal values (server, identifier_type)
  - JSON types are correct (array params are arrays, integer params are ints,
    string params are strings) — what a strict MCP client enforces
  - search_datasets never mixes search_terms with advanced field filters
    (the server silently ignores the advanced fields if search_terms is set)
  - no phantom args (None / "" / "none" / "null")
  - every tool result (expected_result) is a real JSON object, not prose

Usage: python3.11 validate_ndp_dataset.py [dataset.json] [catalog.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "ndp_tool_examples_curated.json"
DEFAULT_CATALOG = HERE.parent.parent / "configs" / "tools" / "ndp_tools.json"

# search_datasets simple-mode vs advanced-mode fields (mutually exclusive)
SIMPLE_FIELDS = {"search_terms", "search_keys"}
ADVANCED_FIELDS = {
    "dataset_name", "dataset_title", "owner_org", "resource_url", "resource_name",
    "dataset_description", "resource_description", "resource_format",
    "search_term", "filter_list", "timestamp",
}
JSON_TYPE = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict,
}


def load_catalog(path: Path) -> dict:
    raw = json.loads(path.read_text())
    tools = raw["tools"] if isinstance(raw, dict) and "tools" in raw else raw
    schema = {}
    for t in tools:
        params = {}
        required = []
        for p in t.get("parameters", []):
            params[p["name"]] = {"type": p["type"], "enum": p.get("enum")}
            if p.get("required", False):
                required.append(p["name"])
        schema[t["name"]] = {"params": params, "required": required}
    return schema


def _phantom(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
        return True
    return False


def validate_call(tool: str, args: dict, schema: dict) -> list[str]:
    errs: list[str] = []
    if tool not in schema:
        return [f"unknown tool '{tool}'"]
    spec = schema[tool]
    if not isinstance(args, dict):
        return [f"{tool}: args is not an object"]

    for k, v in args.items():
        if k not in spec["params"]:
            errs.append(f"{tool}: arg '{k}' not in schema")
            continue
        if _phantom(v):
            errs.append(f"{tool}: arg '{k}' is a phantom/None value ({v!r})")
            continue
        ptype = spec["params"][k]["type"]
        want = JSON_TYPE.get(ptype, object)
        # bool is a subclass of int — reject bool where int expected
        if ptype == "integer" and isinstance(v, bool):
            errs.append(f"{tool}: arg '{k}' must be integer, got bool")
        elif not isinstance(v, want):
            errs.append(f"{tool}: arg '{k}' must be {ptype}, got {type(v).__name__}")
        penum = spec["params"][k]["enum"]
        if penum and v not in penum:
            errs.append(f"{tool}: arg '{k}'={v!r} not in enum {penum}")

    for req in spec["required"]:
        if req not in args or _phantom(args.get(req)):
            errs.append(f"{tool}: required arg '{req}' missing")

    if tool == "search_datasets":
        used_simple = SIMPLE_FIELDS & set(args)
        used_adv = ADVANCED_FIELDS & set(args)
        if used_simple & {"search_terms"} and used_adv:
            errs.append(f"search_datasets: search_terms mixed with advanced "
                        f"fields {sorted(used_adv)} (server ignores the latter)")
        if "search_keys" in args and "search_terms" not in args:
            errs.append("search_datasets: search_keys given without search_terms")
        if not used_simple and not used_adv and "limit" not in args \
                and set(args) <= {"server"}:
            errs.append("search_datasets: no search criteria given")
    return errs


def main() -> int:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    cat_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CATALOG

    schema = load_catalog(cat_path)
    data = json.loads(data_path.read_text())
    print(f"[validate] {len(data)} examples  |  catalog: {sorted(schema)}")

    total_calls = 0
    bad_rows = 0
    tool_counts: Counter = Counter()
    failures: list[str] = []

    for idx, ex in enumerate(data):
        sol = ex.get("solution") or {}
        steps = sol.get("reasoning_path") or []
        row_errs: list[str] = []

        if not (ex.get("instruction") or "").strip():
            row_errs.append("empty instruction")
        if not (sol.get("final_answer") or "").strip():
            row_errs.append("empty final_answer")

        for step in steps:
            tool = step.get("tool")
            args = step.get("args") or {}
            total_calls += 1
            tool_counts[tool] += 1
            row_errs += validate_call(tool, args, schema)
            res = step.get("expected_result")
            if res is None or not isinstance(res, dict):
                row_errs.append(f"{tool}: expected_result is not a JSON object")

        if row_errs:
            bad_rows += 1
            for e in row_errs[:4]:
                failures.append(f"  row {idx}: {e}")

    print(f"[validate] tool calls: {total_calls}  by tool: {dict(tool_counts)}")
    notool = sum(1 for e in data if not (e.get('solution') or {}).get('reasoning_path'))
    print(f"[validate] no-tool examples: {notool}")

    if failures:
        print(f"\n[validate] FAIL — {bad_rows} bad row(s):")
        for f in failures[:60]:
            print(f)
        if len(failures) > 60:
            print(f"  … and {len(failures) - 60} more")
        return 1

    print("\n[validate] PASS — every tool call is schema-valid, "
          "correctly typed, and JSON-clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
