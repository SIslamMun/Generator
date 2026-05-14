"""Full-trace benchmark run for jarvis-v10 only.

For every (case, temperature, run_idx) we capture EVERYTHING:
  - the rendered prompt fed into the model at each step
  - the raw model output (full token stream)
  - the parsed `thought`, `calls`, and trailing text
  - each MCP call (name + arguments) and the raw response
  - the final answer
  - latency per step
  - scoring breakdown

Outputs two artifacts under inference/results/v10_traces/:
  - traces.jsonl   : 1 row per (case, T, run_idx), with a 'steps' array
  - trace_<cid>_T<T>_r<r>.md : human-readable markdown per trace

This doesn't change the benchmark runner — it reimports the same helpers
and the same CASES/SYSTEM_PROMPT so the scoring is identical.
"""

import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, "/u/sislam3/Generator")
sys.path.insert(0, "/u/sislam3/Generator/finetuned_unsloth/test")

from inference.mcp_client import JarvisMCP
from inference.render_and_parse import (
    SYSTEM_PROMPT, mcp_tools_to_hf_schema, render_prompt,
    split_think_and_calls, append_tool_call, append_tool_result, initial_messages,
)

# Re-use benchmark CASES and grader so scoring matches the main table.
import benchmark_all as bench

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
MODEL_TAG   = "jarvis-v10:latest"
TOKENIZER   = "/work/hdd/bekn/sislam3/jarvis_v10_lora/merged_16bit"
TEMPS       = [0.0, 0.3, 0.7]
REPEATS_NZ  = 3   # repeats at T > 0 (T=0 always 1)
MAX_STEPS   = 6   # matches bench.OllamaFunctionGemmaProvider

OUT_DIR = Path("/u/sislam3/Generator/inference/results/v10_traces")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ollama_generate(prompt: str, temperature: float, seed: int) -> tuple[str, float]:
    payload = {
        "model": MODEL_TAG, "prompt": prompt, "raw": True, "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0 else 0.95,
            "top_k": 1 if temperature == 0 else 64,
            "seed": seed, "num_predict": 512, "num_ctx": 8192,
        },
    }
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", ""), time.time() - t0


def run_one(mcp, tokenizer, tools, catalog_names: set[str], case: dict, T: float, seed: int) -> dict:
    """Returns a dict with complete trace info for this (case, T, seed)."""
    messages = initial_messages(case["prompt"])
    steps = []
    calls_flat = []        # for grader
    mcp_responses = []
    mcp_errors = 0
    final_answer = ""
    gen_s = 0.0
    t0 = time.time()

    for step_idx in range(MAX_STEPS):
        rendered = render_prompt(tokenizer, messages, tools)
        raw, dt = ollama_generate(rendered, T, seed)
        gen_s += dt
        thought, calls, trailing = split_think_and_calls(raw)

        step_trace = {
            "step": step_idx,
            "rendered_prompt": rendered,
            "raw_output": raw,
            "parsed": {"thought": thought, "calls": calls, "trailing": trailing},
            "gen_latency_s": round(dt, 3),
            "mcp_interactions": [],
        }

        if not calls:
            final_answer = trailing or raw.strip()
            step_trace["ended_with_answer"] = final_answer
            steps.append(step_trace)
            break

        calls_flat.extend(calls)
        append_tool_call(messages, thought, calls)
        for call in calls:
            t_mcp = time.time()
            try:
                result = mcp.call_tool(call["name"], call["arguments"])
                err_raised = None
            except Exception as e:
                result = f"(exception) {e}"
                err_raised = str(e)
            mcp_dt = time.time() - t_mcp
            is_error = (
                '"error"' in result.lower()
                or '"iserror": true' in result.lower()
                or err_raised is not None
            )
            if is_error:
                mcp_errors += 1
            mcp_responses.append(result)
            append_tool_result(messages, call["name"], result)
            step_trace["mcp_interactions"].append({
                "tool":   call["name"],
                "args":   call["arguments"],
                "result": result,
                "is_error": is_error,
                "mcp_latency_s": round(mcp_dt, 3),
            })
        steps.append(step_trace)
    else:
        final_answer = "(max iterations)"

    total_s = time.time() - t0

    # ── score using the benchmark's own grader ──
    actual_names = [c["name"] for c in calls_flat]
    if "expect_tools_any_of" in case:
        tool_ok = bench._grade_any_of(case["expect_tools_any_of"], actual_names)
        arg_ok = tool_ok
    else:
        tool_ok = bench._grade_ordered(case["expect_tools"], actual_names)
        arg_ok = tool_ok and bench._grade_args(case["expect_args"], calls_flat)
    answer_ok = all(kw.lower() in final_answer.lower() for kw in case.get("expect_answer", []))
    # hallucinations: calls for tools not in the catalog
    hallucinations = sum(1 for n in actual_names if n not in catalog_names)
    task_success = bool(tool_ok and arg_ok and mcp_errors == 0)

    return {
        "case_id": case["id"],
        "category": case["cat"],
        "prompt":   case["prompt"],
        "expected_tools": case.get("expect_tools") or case.get("expect_tools_any_of"),
        "expected_args":  case.get("expect_args"),
        "expected_answer_keywords": case.get("expect_answer"),
        "temperature": T,
        "seed": seed,
        "steps": steps,
        "flat_calls": calls_flat,
        "final_answer": final_answer,
        "grade": {
            "tool_ok": tool_ok, "arg_ok": arg_ok, "answer_ok": answer_ok,
            "task_success": task_success,
            "hallucinations": hallucinations,
            "mcp_errors": mcp_errors,
            "n_calls": len(calls_flat),
        },
        "latency_gen_s": round(gen_s, 3),
        "latency_total_s": round(total_s, 3),
    }


def render_markdown(trace: dict, out_path: Path):
    """Human-readable single-page markdown per trace."""
    g = trace["grade"]
    ok_mark = "✓" if g["task_success"] else ("~" if g["tool_ok"] else "✗")
    lines = [
        f"# Trace · case `{trace['case_id']}` ({trace['category']}) · T={trace['temperature']} · seed={trace['seed']}",
        "",
        f"**Task success:** {ok_mark}  ·  tool_ok={int(g['tool_ok'])}  arg_ok={int(g['arg_ok'])}  "
        f"answer_ok={int(g['answer_ok'])}  halluc={g['hallucinations']}  mcp_err={g['mcp_errors']}  "
        f"·  gen={trace['latency_gen_s']}s  total={trace['latency_total_s']}s",
        "",
        "## Prompt",
        f"> {trace['prompt']}",
        "",
        "## Expected",
        f"- tools: `{trace['expected_tools']}`",
        f"- args:  `{trace['expected_args']}`",
        f"- answer keywords: `{trace['expected_answer_keywords']}`",
        "",
        "## Step-by-step model trace",
    ]
    for step in trace["steps"]:
        lines += [
            f"### Step {step['step']}  (gen_latency={step['gen_latency_s']}s)",
            "",
            "**Raw model output:**",
            "```",
            step["raw_output"],
            "```",
            "",
            f"**Parsed thought:**",
            "```",
            step["parsed"]["thought"] or "(empty)",
            "```",
            "",
            f"**Parsed tool calls:**  {len(step['parsed']['calls'])}",
        ]
        for c in step["parsed"]["calls"]:
            lines.append(f"- `{c['name']}`  args=`{json.dumps(c['arguments'])}`")
        lines.append("")
        if step["mcp_interactions"]:
            lines += ["**MCP round-trips:**", ""]
            for mi in step["mcp_interactions"]:
                err = " ← ERROR" if mi["is_error"] else ""
                lines += [
                    f"- **{mi['tool']}**{err} (mcp_latency={mi['mcp_latency_s']}s)",
                    f"  args: `{json.dumps(mi['args'])}`",
                    f"  result:",
                    "  ```",
                    "  " + mi["result"].replace("\n", "\n  ")[:1000],
                    "  ```",
                    "",
                ]
        if "ended_with_answer" in step:
            lines += ["**Final answer:**", "", "> " + step["ended_with_answer"].replace("\n", "\n> "), ""]
    out_path.write_text("\n".join(lines))


def main():
    print(f"tokenizer = {TOKENIZER}")
    print(f"ollama    = {OLLAMA_HOST}  tag={MODEL_TAG}")
    print(f"output    = {OUT_DIR}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    out_jsonl = (OUT_DIR / "traces.jsonl").open("w")

    with JarvisMCP(startup_timeout=60.0) as mcp:
        mcp_tools_raw = mcp.list_tools()
        tools = mcp_tools_to_hf_schema(mcp_tools_raw)
        catalog_names = {t["name"] for t in mcp_tools_raw}
        print(f"MCP: {len(mcp_tools_raw)} tools visible\n")

        for case in bench.CASES:
            for T in TEMPS:
                n = 1 if T == 0.0 else REPEATS_NZ
                for run_idx in range(n):
                    seed = 42 + run_idx
                    mcp.call_tool("jm_reset", {})
                    try:
                        trace = run_one(mcp, tokenizer, tools, catalog_names, case, T, seed)
                    except Exception as e:
                        traceback.print_exc()
                        trace = {
                            "case_id": case["id"], "category": case["cat"],
                            "prompt": case["prompt"],
                            "expected_tools": case.get("expect_tools") or case.get("expect_tools_any_of"),
                            "expected_args":  case.get("expect_args"),
                            "expected_answer_keywords": case.get("expect_answer"),
                            "temperature": T, "seed": seed,
                            "exception": str(e),
                            "grade": {"tool_ok": False, "arg_ok": False, "answer_ok": False,
                                      "task_success": False, "hallucinations": 0, "mcp_errors": 0, "n_calls": 0},
                            "latency_gen_s": 0, "latency_total_s": 0, "steps": [], "flat_calls": [],
                            "final_answer": f"EXCEPTION: {e}",
                        }

                    out_jsonl.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    out_jsonl.flush()

                    # per-trace markdown
                    md_path = OUT_DIR / f"trace_{case['id']}_T{T:.1f}_r{run_idx}.md"
                    render_markdown(trace, md_path)

                    g = trace["grade"]
                    flag = "✓" if g["task_success"] else ("~" if g["tool_ok"] else "✗")
                    print(f"{flag} {case['id']} T={T} r{run_idx}  "
                          f"tool={int(g['tool_ok'])} arg={int(g['arg_ok'])} "
                          f"ans={int(g['answer_ok'])} halluc={g['hallucinations']} "
                          f"mcp_err={g['mcp_errors']} t={trace['latency_total_s']}s")

    out_jsonl.close()
    print(f"\ntraces.jsonl   → {OUT_DIR / 'traces.jsonl'}")
    print(f"per-case .md  → {OUT_DIR}/trace_<cid>_T*_r*.md")


if __name__ == "__main__":
    main()
