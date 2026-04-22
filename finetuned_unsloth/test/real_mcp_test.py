"""End-to-end test against the REAL Jarvis MCP server, graded by category.

Categories exercised (mirrors the training-data methods):
  - single        : one tool call, no follow-ups
  - multi         : >=2 independent tools in one turn
  - chain_first   : 2+ dependent tools (output of A feeds B's args)
  - error_recovery: first tool fails, model must recover with a second call

Grading per case:
  TOOL    = every expected tool appears, in the right order
  ARG     = every expected arg key/value appears on its call
  MCP_OK  = no MCP `error`/`isError` in the returned tool responses
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/u/sislam3/Generator")

from inference.mcp_client import JarvisMCP
from inference.ollama_backend import OllamaBackend
from inference.render_and_parse import (
    initial_messages, render_prompt, split_think_and_calls,
    append_tool_call, append_tool_result, mcp_tools_to_hf_schema,
)
from transformers import AutoTokenizer

TOKENIZER_DIR = "/u/sislam3/Generator/finetuned_unsloth/artifacts/v8/model_merged_16bit"
MODEL = "jarvis-v8"


CASES = [
    # ─────────── SINGLE ───────────
    {"cat": "single", "prompt": "Create a pipeline named demo_pipeline.",
     "expect_tools": ["create_pipeline"],
     "expect_args":  [{"pipeline_id": "demo_pipeline"}]},
    {"cat": "single", "prompt": "List every Jarvis pipeline I currently have.",
     "expect_tools": ["jm_list_pipelines"], "expect_args": [{}]},
    {"cat": "single", "prompt": "Bootstrap my Jarvis setup for the summit machine.",
     "expect_tools": ["jm_bootstrap_from"],
     "expect_args":  [{"machine": "summit"}]},
    {"cat": "single", "prompt": "Reset the whole Jarvis system.",
     "expect_tools": ["jm_reset"], "expect_args": [{}]},
    {"cat": "single", "prompt": "Set my current pipeline to gpu_training.",
     "expect_tools": ["jm_cd"],
     "expect_args":  [{"pipeline_id": "gpu_training"}]},
    {"cat": "single", "prompt": "Build the resource graph with a half-second sleep between operations.",
     "expect_tools": ["jm_graph_build"],
     "expect_args":  [{"net_sleep": 0.5}]},

    # ─────────── MULTI (independent calls) ───────────
    {"cat": "multi", "prompt": "Create a pipeline named bench_a, then destroy the deprecated_test pipeline.",
     "expect_tools": ["create_pipeline", "destroy_pipeline"],
     "expect_args":  [{"pipeline_id": "bench_a"}, {"pipeline_id": "deprecated_test"}]},
    {"cat": "multi", "prompt": "List my pipelines, then show me the resource graph.",
     "expect_tools": ["jm_list_pipelines", "jm_graph_show"],
     "expect_args":  [{}, {}]},

    # ─────────── CHAIN (dependent calls — create → cd → attach → configure) ───────────
    {"cat": "chain_first", "prompt": "Create a pipeline called bench_v2, switch to it, and attach an IOR package with 16 procs.",
     "expect_tools": ["create_pipeline", "jm_cd", "append_pkg", "configure_pkg"],
     "expect_args":  [
         {"pipeline_id": "bench_v2"},
         {"pipeline_id": "bench_v2"},
         {"pipeline_id": "bench_v2", "pkg_type": "ior"},
         {"pipeline_id": "bench_v2", "pkg_id": "ior", "extra_args": {"nprocs": 16}},
     ]},
    {"cat": "chain_first", "prompt": "Load the pipeline climate_forecast_2026 and make it my current pipeline.",
     "expect_tools": ["load_pipeline", "jm_cd"],
     "expect_args":  [{"pipeline_id": "climate_forecast_2026"}, {"pipeline_id": "climate_forecast_2026"}]},

    # ─────────── ERROR RECOVERY ───────────
    # load_pipeline on a non-existent id → should fall back to create_pipeline.
    {"cat": "error_recovery", "prompt": "Load the pipeline fresh_pipeline so I can use it; if it doesn't exist, create it first.",
     "expect_tools_any_of": [
         ["load_pipeline", "create_pipeline"],
         ["create_pipeline", "load_pipeline"],
         ["load_pipeline"],
     ]},
    # append_pkg to a non-existent pipeline → should create it first.
    {"cat": "error_recovery", "prompt": "Append an mdtest package to pipeline io_bench — if the pipeline is missing, create it and then attach.",
     "expect_tools_any_of": [
         ["append_pkg", "create_pipeline", "append_pkg"],
         ["create_pipeline", "append_pkg"],
         ["load_pipeline", "create_pipeline", "append_pkg"],
     ]},
]


def grade_tools(expected_names, actual_names) -> bool:
    """Every expected name appears in order (actual may include extras)."""
    ai = 0
    for name in expected_names:
        while ai < len(actual_names) and actual_names[ai] != name:
            ai += 1
        if ai >= len(actual_names):
            return False
        ai += 1
    return True


def grade_any_of(expect_any_of, actual_names) -> bool:
    for option in expect_any_of:
        if grade_tools(option, actual_names):
            return True
    return False


def _args_subset(expected_args: dict, actual_args: dict) -> bool:
    """Every expected key/value matches something in actual."""
    for k, v in expected_args.items():
        if k not in actual_args:
            return False
        a = actual_args[k]
        if isinstance(v, dict) and isinstance(a, dict):
            if not _args_subset(v, a):
                return False
        elif isinstance(v, float) and isinstance(a, (int, float)):
            if abs(float(a) - v) > 1e-6:
                return False
        else:
            if str(a) != str(v):
                return False
    return True


def grade_args(expected_args_list, actual_tool_calls) -> bool:
    """Match each expected arg dict to the first call with the same tool name."""
    idx = 0
    for exp in expected_args_list:
        found = False
        while idx < len(actual_tool_calls):
            if _args_subset(exp, actual_tool_calls[idx]["arguments"]):
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            return False
    return True


def run_case(tokenizer, backend, mcp, case):
    messages = initial_messages(case["prompt"])
    tools = mcp_tools_to_hf_schema(mcp.list_tools())
    all_calls = []
    mcp_errors = 0

    for step in range(6):
        rendered = render_prompt(tokenizer, messages, tools)
        raw = backend.generate(rendered)
        thought, calls, trailing = split_think_and_calls(raw)
        if not calls:
            return {"final_answer": trailing or raw.strip(),
                    "calls": all_calls, "mcp_errors": mcp_errors}
        all_calls.extend(calls)
        append_tool_call(messages, thought, calls)
        for call in calls:
            result = mcp.call_tool(call["name"], call["arguments"])
            if '"error"' in result.lower() or '"iserror": true' in result.lower():
                mcp_errors += 1
            append_tool_result(messages, call["name"], result)
    return {"final_answer": "(loop exceeded)", "calls": all_calls, "mcp_errors": mcp_errors}


def main():
    print(f"Loading tokenizer from {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    backend = OllamaBackend(model=MODEL, temperature=0.0, top_p=1.0, top_k=1,
                            num_predict=512, num_ctx=8192)

    rows = []
    with JarvisMCP() as mcp:
        print(f"Connected to real MCP: {len(mcp.list_tools())} tools")
        for i, case in enumerate(CASES, 1):
            mcp.call_tool("jm_reset", {})  # fresh state each case
            print(f"\n[{i}/{len(CASES)}] {case['cat']}: {case['prompt']}")
            t0 = time.time()
            try:
                out = run_case(tokenizer, backend, mcp, case)
            except Exception as e:
                out = {"final_answer": f"EXCEPTION: {e}",
                       "calls": [], "mcp_errors": -1}
                traceback.print_exc(file=sys.stdout)
            dt = time.time() - t0

            actual_names = [c["name"] for c in out["calls"]]
            if "expect_tools_any_of" in case:
                tool_ok = grade_any_of(case["expect_tools_any_of"], actual_names)
                arg_ok = tool_ok  # for error-recovery, presence is what we grade
            else:
                tool_ok = grade_tools(case["expect_tools"], actual_names)
                arg_ok = tool_ok and grade_args(case["expect_args"], out["calls"])
            mcp_ok = out["mcp_errors"] == 0

            print(f"    calls:   {actual_names}")
            print(f"    TOOL={'✓' if tool_ok else '✗'}  ARG={'✓' if arg_ok else '✗'}  MCP_OK={'✓' if mcp_ok else '✗'}  ({dt:.1f}s, mcp_errors={out['mcp_errors']})")
            print(f"    answer:  {out['final_answer'][:140]}")

            rows.append({
                "cat": case["cat"], "prompt": case["prompt"],
                "tool": tool_ok, "arg": arg_ok, "mcp": mcp_ok,
                "calls": actual_names, "mcp_errors": out["mcp_errors"],
                "seconds": round(dt, 2),
            })

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    cats = {}
    for r in rows:
        c = cats.setdefault(r["cat"], {"n": 0, "tool": 0, "arg": 0, "mcp": 0})
        c["n"] += 1
        c["tool"] += r["tool"]
        c["arg"] += r["arg"]
        c["mcp"] += r["mcp"]
    print(f"{'category':<18}{'cases':>8}{'TOOL':>8}{'ARG':>8}{'MCP_OK':>10}")
    for cat, c in cats.items():
        print(f"{cat:<18}{c['n']:>8}{c['tool']:>8}{c['arg']:>8}{c['mcp']:>10}")
    n = len(rows)
    tool = sum(r["tool"] for r in rows)
    arg = sum(r["arg"] for r in rows)
    mcp_ok = sum(r["mcp"] for r in rows)
    print("-" * 52)
    print(f"{'TOTAL':<18}{n:>8}{tool:>8}{arg:>8}{mcp_ok:>10}")
    print(f"\n{'rate (%)':<18}{'':>8}{100*tool/n:>7.0f}%{100*arg/n:>7.0f}%{100*mcp_ok/n:>9.0f}%")

    out_json = Path("/u/sislam3/Generator/finetuned_unsloth/logs/last_real_mcp_report.json")
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nper-case report → {out_json}")


if __name__ == "__main__":
    main()
