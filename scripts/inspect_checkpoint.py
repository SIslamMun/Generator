#!/usr/bin/env python3
"""
Inspect tool-use checkpoint quality against real Jarvis MCP server returns.

Usage:
    python scripts/inspect_checkpoint.py outputs/v7_2k/jarvis_v7_2000_intermediate.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Real return shapes from jarvis_handler.py (ground truth)
REAL_PIPELINE_SHAPES = {
    "create_pipeline": {"pipeline_id", "status"},
    "load_pipeline": {"pipeline_id", "status"},
    "update_pipeline": {"pipeline_id", "status"},
    "build_pipeline_env": {"pipeline_id", "status"},
    "run_pipeline": {"pipeline_id", "status"},
    "destroy_pipeline": {"pipeline_id", "status"},
    "append_pkg": {"pipeline_id", "appended"},
    "configure_pkg": {"pipeline_id", "configured"},
    "get_pkg_config": {"pipeline_id", "pkg_id", "config"},
    "unlink_pkg": {"pipeline_id", "unlinked"},
    "remove_pkg": {"pipeline_id", "removed"},
}

REAL_STATUS_VALUES = {
    "create_pipeline": "created",
    "load_pipeline": "loaded",
    "update_pipeline": "updated",
    "build_pipeline_env": "environment_built",
    "run_pipeline": "running",
    "destroy_pipeline": "destroyed",
}

# Manager tools return plain strings
MANAGER_TOOLS = {
    "jm_create_config", "jm_load_config", "jm_save_config",
    "jm_set_hostfile", "jm_bootstrap_from", "jm_bootstrap_list",
    "jm_reset", "jm_list_pipelines", "jm_cd",
    "jm_list_repos", "jm_add_repo", "jm_remove_repo",
    "jm_promote_repo", "jm_get_repo", "jm_construct_pkg",
    "jm_graph_show", "jm_graph_build", "jm_graph_modify",
}


def check_result_shape(tool_name, expected_result):
    """Check if expected_result matches real server shape."""
    issues = []

    if tool_name in REAL_PIPELINE_SHAPES:
        if isinstance(expected_result, dict):
            expected_keys = REAL_PIPELINE_SHAPES[tool_name]
            actual_keys = set(expected_result.keys())
            extra = actual_keys - expected_keys
            missing = expected_keys - actual_keys
            if extra:
                issues.append(f"extra keys: {extra}")
            if missing:
                issues.append(f"missing keys: {missing}")
            # Check status value
            if tool_name in REAL_STATUS_VALUES and "status" in expected_result:
                real_status = REAL_STATUS_VALUES[tool_name]
                if expected_result["status"] != real_status:
                    issues.append(f"wrong status: '{expected_result['status']}' should be '{real_status}'")
        elif isinstance(expected_result, str) and ("404:" in str(expected_result) or "500:" in str(expected_result)):
            pass  # Error string is OK
        else:
            issues.append(f"expected dict, got {type(expected_result).__name__}")

    elif tool_name in MANAGER_TOOLS:
        if isinstance(expected_result, dict):
            # Manager tools should NOT return dicts — they return strings
            if "status" in expected_result and expected_result.get("status") == "success":
                issues.append("invented {'status': 'success'} — manager tools return plain strings")
        # Strings and lists are fine for manager tools

    return issues


def inspect(path):
    with open(path) as f:
        data = json.load(f)

    print(f"═══ Checkpoint Quality Report ═══")
    print(f"File: {path}")
    print(f"Total examples: {len(data)}")
    print()

    # Method distribution
    methods = Counter(e["solution"]["method"] for e in data)
    print(f"By method: {dict(methods)}")

    # Tool usage
    tool_counts = Counter()
    for e in data:
        for s in e["solution"]["reasoning_path"]:
            tool_counts[s["tool"]] += 1
    print(f"Unique tools used: {len(tool_counts)}")
    print(f"Tool distribution: {dict(tool_counts.most_common(10))}")
    print()

    # Quality checks
    empty_thoughts = 0
    empty_args = 0
    shape_issues = []
    unknown_tools = []
    all_tools = set(REAL_PIPELINE_SHAPES.keys()) | MANAGER_TOOLS
    short_instructions = 0
    generic_finals = 0
    hallucinated_tools = 0

    for i, e in enumerate(data):
        # Instruction quality
        if len(e["instruction"]) < 20:
            short_instructions += 1

        # Final answer quality
        fa = e["solution"].get("final_answer", "")
        if any(phrase in fa.lower() for phrase in ["called the tool", "i have executed", "the operation was"]):
            generic_finals += 1

        for s in e["solution"]["reasoning_path"]:
            # Thought quality
            if not s.get("thought") or len(s.get("thought", "")) < 10:
                empty_thoughts += 1

            # Args check
            if not s.get("args") and s.get("tool") not in {"jm_load_config", "jm_save_config", "jm_reset", "jm_list_pipelines", "jm_list_repos", "jm_bootstrap_list", "jm_graph_show"}:
                empty_args += 1

            # Tool exists?
            if s.get("tool") and s["tool"] not in all_tools:
                hallucinated_tools += 1
                unknown_tools.append((i, s["tool"]))

            # Result shape check
            if s.get("expected_result"):
                issues = check_result_shape(s["tool"], s["expected_result"])
                if issues:
                    shape_issues.append((i, s["tool"], issues))

    print("═══ Quality Metrics ═══")
    print(f"  Short instructions (<20 chars): {short_instructions}/{len(data)}")
    print(f"  Empty/shallow thoughts: {empty_thoughts}")
    print(f"  Empty args (non-zero-param tools): {empty_args}")
    print(f"  Generic final answers: {generic_finals}")
    print(f"  Hallucinated tool names: {hallucinated_tools}")
    print(f"  Wrong result shapes: {len(shape_issues)}")
    print()

    if unknown_tools:
        print("═══ Hallucinated Tools ═══")
        for idx, tool in unknown_tools[:10]:
            print(f"  Example {idx}: '{tool}'")
        print()

    if shape_issues:
        print("═══ Shape Issues (first 10) ═══")
        for idx, tool, issues in shape_issues[:10]:
            print(f"  Example {idx}, {tool}: {'; '.join(issues)}")
        print()

    # Show 3 sample examples
    print("═══ Sample Examples ═══")
    for i in range(min(3, len(data))):
        e = data[i]
        print(f"\n--- Example {i} ({e['solution']['method']}) ---")
        print(f"Instruction: {e['instruction'][:120]}")
        for s in e["solution"]["reasoning_path"]:
            print(f"  Step {s['step']}: thought={s.get('thought', '')[:80]}...")
            print(f"           tool={s['tool']}  args={json.dumps(s.get('args', {}))[:80]}")
            if s.get("expected_result"):
                print(f"           result={json.dumps(s['expected_result'])[:80]}")
        print(f"  Final: {e['solution'].get('final_answer', '')[:120]}")

    # Overall score
    total_steps = sum(len(e["solution"]["reasoning_path"]) for e in data)
    issues_pct = len(shape_issues) / max(total_steps, 1) * 100
    halluc_pct = hallucinated_tools / max(total_steps, 1) * 100
    print(f"\n═══ Overall ═══")
    print(f"  Total steps: {total_steps}")
    print(f"  Shape issue rate: {issues_pct:.1f}%")
    print(f"  Hallucination rate: {halluc_pct:.1f}%")
    grade = "GOOD" if issues_pct < 10 and halluc_pct < 5 else "NEEDS_WORK" if issues_pct < 30 else "BAD"
    print(f"  Grade: {grade}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_checkpoint.py <checkpoint.json>")
        sys.exit(1)
    inspect(sys.argv[1])
