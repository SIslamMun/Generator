"""Prepare training data for Nemotron-3 Nano 4B from generator output.

The same script handles any combination of:
  --types qa            → just QA pairs (each row: user/assistant)
  --types qa,cot        → QA + CoT (assistant content carries `**Reasoning:**` + `**Answer:**`)
  --types tool          → tool-use traces (user / assistant-with-tool_calls / tool-result / …)
  --types qa,cot,tool   → mix all three

For each chosen type, point at a generator output file with the matching flag:
  --in-qa   runs/<topic>/data/qa_curated.json
  --in-cot  runs/<topic>/data/cot_curated.json
  --in-tool runs/<topic>/data/tool_examples_curated.json

The script:
  1. Loads + parses each requested source
  2. Builds universal `conversations` lists per row
  3. Loads the Nemotron tokenizer and applies its chat template (with tools= for tool rows)
  4. Writes ONE JSONL where every row has both:
       - `conversations`: raw structured turns (audit-friendly)
       - `text`: the tokenizer-rendered string (what SFTTrainer trains on)
       - `_source`: "qa" | "cot" | "tool" (for ratio inspection later)
  5. Prints a summary of row counts per type
"""
from __future__ import annotations

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Iterable

# ─────────────────────────── universal conversation builders ───────

def qa_to_convo(row: dict) -> list[dict] | None:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    if not (q and a):
        return None
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]


def cot_to_convo(row: dict) -> list[dict] | None:
    """QA + reasoning → assistant content with the canonical format."""
    q = (row.get("question") or "").strip()
    reasoning = (row.get("reasoning") or "").strip()
    a = (row.get("answer") or "").strip()
    if not (q and reasoning and a):
        return None
    assistant = (
        "<think>\n" + reasoning + "\n</think>\n\n" + a
    )
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": assistant},
    ]


ANTI_HALLUCINATION_SYSTEM = (
    "Tool-call discipline:\n"
    "- ONLY include parameters that you are actually setting to a value.\n"
    "- NEVER include parameters whose value would be None, null, empty string, or unset.\n"
    "- NEVER invent parameter names that are not in the tool's schema.\n"
    "- If a parameter is optional and you're not using it, OMIT IT ENTIRELY (do not emit it with a None placeholder)."
)


# ─────────────────────────── tool-call arg sanitizer ────────────────
# The ANTI_HALLUCINATION_SYSTEM prompt above *tells* the model not to emit
# phantom params — but that only sticks if the training TARGETS obey it too.
# The generator's `reasoning_path` steps frequently carry args that are
# None-valued or that don't belong to the called tool; left unsanitized,
# every SFT target teaches the model exactly the behaviour we forbid.
# (Confirmed downstream: the fine-tuned model floods calls with the union
# of all tools' params set to "None", which strict validators — e.g. the
# LM Studio MCP client — reject outright.)
#
# This mirrors test_inference.parse_tool_call's inference-time cleanup, but
# applied to the data so the model never learns the pattern.

def _is_phantom_value(v) -> bool:
    """True if an arg value is a None-placeholder that must be dropped."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
        return True
    return False


def _catalog_param_names(catalog_tools: list[dict]) -> dict[str, set]:
    """Map each tool name → the set of parameter names in its schema."""
    names: dict[str, set] = {}
    for t in catalog_tools:
        fn = t.get("function", t)
        props = (fn.get("parameters") or {}).get("properties") or {}
        names[fn.get("name", "")] = set(props)
    return names


def sanitize_tool_args(
    tool_name: str, args: dict, param_names: dict[str, set]
) -> tuple[dict, int]:
    """Strip phantom args from a training target's tool call.

    Drops an arg when either:
      - its value is a None placeholder (None / "None" / "null" / ""), or
      - its name is not in the called tool's schema (cross-tool leakage).

    Returns (clean_args, n_dropped).
    """
    if not isinstance(args, dict):
        return {}, 0
    schema = param_names.get(tool_name)
    clean: dict = {}
    dropped = 0
    for k, v in args.items():
        if _is_phantom_value(v):
            dropped += 1
            continue
        if schema is not None and k not in schema:
            dropped += 1
            continue
        clean[k] = v
    return clean, dropped


def tool_to_convo(
    ex: dict, catalog_tools: list[dict]
) -> tuple[list[dict], list[dict], int] | None:
    """Convert a generator tool-use example to (conversations, tools, n_dropped).

    Output conversations include user/assistant-with-tool_calls/tool/final-assistant.
    The `tools` list is the JSON-schema form for each catalog tool — apply_chat_template
    will render it inside the system message according to Nemotron's tool-calling format.

    A leading system message enforces tool-call discipline (no phantom None params),
    and every tool call's args are run through `sanitize_tool_args` so the SFT
    targets actually obey that discipline. `n_dropped` is the count of phantom
    args removed across this example (for the run summary).
    """
    instr = (ex.get("instruction") or "").strip()
    sol = ex.get("solution") or {}
    steps = sol.get("reasoning_path") or []
    if not instr:
        return None

    param_names = _catalog_param_names(catalog_tools)
    n_dropped = 0

    msgs: list[dict] = [
        {"role": "system", "content": ANTI_HALLUCINATION_SYSTEM},
        {"role": "user", "content": instr},
    ]
    for step in steps:
        tool = step.get("tool")
        if not tool:
            continue
        thought = (step.get("thought") or "").strip()
        args = step.get("args") or {}
        args, dropped = sanitize_tool_args(tool, args, param_names)
        n_dropped += dropped
        actual = step.get("actual_result")
        expected = step.get("expected_result")
        # Assistant turn: <think>thought</think> + a tool call.
        assistant_content = f"<think>\n{thought}\n</think>" if thought else ""
        msgs.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}],
        })
        # Tool turn: the result the tool returned.
        result_value = actual if actual is not None else expected
        msgs.append({
            "role": "tool",
            "name": tool,
            "content": json.dumps(result_value, default=str) if result_value is not None else "",
        })
    final = (sol.get("final_answer") or "").strip()
    if not steps:
        # No-tool example: the tool catalog is still rendered for the model,
        # but the correct move is to answer directly (or ask for clarification)
        # rather than force a call. This teaches the model when NOT to call.
        if not final:
            return None
        thought = (sol.get("no_tool_thought") or "").strip()
        content = f"<think>\n{thought}\n</think>\n\n{final}" if thought else final
        msgs.append({"role": "assistant", "content": content})
    elif final:
        # Final assistant turn = the natural-language answer grounded in results.
        msgs.append({"role": "assistant", "content": final})

    return msgs, catalog_tools, n_dropped


# ─────────────────────────── loaders ────────────────────────────────

def load_json(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "tools" in data:
        return data["tools"]
    return data


def load_tool_catalog(path: Path) -> list[dict]:
    """Render the tools.json catalog into the JSON-schema form
    apply_chat_template expects."""
    raw = json.loads(path.read_text())
    tools = raw["tools"] if isinstance(raw, dict) and "tools" in raw else raw
    out = []
    for t in tools:
        props = {}
        required = []
        for p in t.get("parameters", []):
            prop = {"type": p["type"], "description": p.get("description", "")}
            if "enum" in p:
                prop["enum"] = p["enum"]
            if "default" in p:
                prop["default"] = p["default"]
            props[p["name"]] = prop
            if p.get("required", True):
                required.append(p["name"])
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return out


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--types", required=True,
                    help="Comma-separated subset of {qa,cot,tool}. Examples: 'qa' / 'qa,cot' / 'tool' / 'qa,cot,tool'")
    ap.add_argument("--in-qa",   type=Path, help="QA pairs JSON (rows: {question, answer})")
    ap.add_argument("--in-cot",  type=Path, help="CoT pairs JSON (rows: {question, reasoning, answer})")
    ap.add_argument("--in-tool", type=Path, help="Tool examples JSON (ToolExample dicts)")
    ap.add_argument("--tool-catalog", type=Path,
                    default=Path("configs/tools/ndp_tools.json"),
                    help="Tool catalog (only used when types includes 'tool')")
    ap.add_argument("--hf-model-id", default="unsloth/granite-4.1-8b",
                    help="HF model whose tokenizer renders the chat template.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows after concat")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap total rows after concat (0 = no cap)")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Skip loading the HF tokenizer and emit conversations without the "
                         "`text` field. Useful for verifying the conversion logic on a node "
                         "that can't reach HuggingFace, or before the model's venv exists. "
                         "Training itself REQUIRES the text field, so re-run without --no-tokenizer "
                         "on a compute node before submitting.")
    args = ap.parse_args()

    types_req = [t.strip().lower() for t in args.types.split(",") if t.strip()]
    valid_types = {"qa", "cot", "tool"}
    bad = [t for t in types_req if t not in valid_types]
    if bad:
        print(f"ERROR: unknown type(s): {bad}. Valid: {sorted(valid_types)}", file=sys.stderr)
        sys.exit(2)

    # Validate inputs match requested types
    for need_type, path_attr in [("qa", "in_qa"), ("cot", "in_cot"), ("tool", "in_tool")]:
        if need_type in types_req and getattr(args, path_attr) is None:
            print(f"ERROR: --types includes '{need_type}' but --in-{need_type} was not provided",
                  file=sys.stderr)
            sys.exit(2)

    # Load tokenizer (unless dry-run)
    tokenizer = None
    if not args.no_tokenizer:
        print(f"[prepare] loading tokenizer: {args.hf_model_id}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
        print(f"[prepare] tokenizer ok: {tokenizer.__class__.__name__}")
    else:
        print(f"[prepare] --no-tokenizer: skipping HF tokenizer load (text field will be empty)")

    # Build conversation rows
    rows: list[dict] = []
    counts = {"qa": 0, "cot": 0, "tool": 0, "skipped_qa": 0, "skipped_cot": 0,
              "skipped_tool": 0, "dropped_tool_args": 0}

    def _render(convo, tools=None):
        """Apply the chat template if tokenizer is loaded; else return empty string."""
        if tokenizer is None:
            return ""
        if tools is not None:
            try:
                return tokenizer.apply_chat_template(
                    convo, tools=tools, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                pass
        return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)

    if "qa" in types_req:
        data = load_json(args.in_qa)
        print(f"[prepare] QA  loaded {len(data)} rows from {args.in_qa}")
        for r in data:
            convo = qa_to_convo(r)
            if convo is None:
                counts["skipped_qa"] += 1
                continue
            rows.append({"conversations": convo, "text": _render(convo), "_source": "qa"})
            counts["qa"] += 1

    if "cot" in types_req:
        data = load_json(args.in_cot)
        print(f"[prepare] CoT loaded {len(data)} rows from {args.in_cot}")
        for r in data:
            convo = cot_to_convo(r)
            if convo is None:
                counts["skipped_cot"] += 1
                continue
            rows.append({"conversations": convo, "text": _render(convo), "_source": "cot"})
            counts["cot"] += 1

    if "tool" in types_req:
        catalog = load_tool_catalog(args.tool_catalog)
        print(f"[prepare] tool catalog: {len(catalog)} tools from {args.tool_catalog}")
        data = load_json(args.in_tool)
        print(f"[prepare] Tool loaded {len(data)} rows from {args.in_tool}")
        for ex in data:
            result = tool_to_convo(ex, catalog)
            if result is None:
                counts["skipped_tool"] += 1
                continue
            convo, tools, dropped = result
            counts["dropped_tool_args"] += dropped
            rows.append({
                "conversations": convo,
                "tools": tools,
                "text": _render(convo, tools=tools),
                "_source": "tool",
            })
            counts["tool"] += 1

    if not rows:
        print("ERROR: no rows produced. Check inputs.", file=sys.stderr)
        sys.exit(1)

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"[prepare] wrote {len(rows)} rows → {args.out}")
    print(f"[prepare]   kept    : qa={counts['qa']}  cot={counts['cot']}  tool={counts['tool']}")
    print(f"[prepare]   skipped : qa={counts['skipped_qa']}  cot={counts['skipped_cot']}  tool={counts['skipped_tool']}")
    if "tool" in types_req:
        print(f"[prepare]   sanitized: dropped {counts['dropped_tool_args']} phantom tool-call arg(s) from targets")
    # Mix-ratio hint per Unsloth's Nemotron docs (75% reasoning / 25% non-reasoning ideal)
    reasoning_n = counts["cot"] + counts["tool"]    # tool traces contain <think> reasoning
    plain_n = counts["qa"]
    total = reasoning_n + plain_n
    if total:
        pct_reason = 100 * reasoning_n / total
        print(f"[prepare]   reasoning mix: {pct_reason:.1f}% (cot+tool / total)")
        if reasoning_n and pct_reason < 60:
            print("[prepare]   note: Unsloth's Nemotron guide recommends ~75% reasoning.")


if __name__ == "__main__":
    main()
