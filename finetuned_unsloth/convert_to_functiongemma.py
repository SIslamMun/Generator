"""Convert v7 reasoning_path dataset to FunctionGemma chat-template format.

Source: outputs/v7_2k/jarvis_v7_2000.json
Output: finetuned_unsloth/v7_2k/jarvis_v7_functiongemma.jsonl

Each output row: {"messages": "<JSON STRING>"} where the decoded list matches what
`prepare_messages_and_tools()` in FunctionGemma_(270M).ipynb expects:
  - First message carries a `tools` list (stripped by the notebook helper).
  - Every assistant message has a `think` field (required; examples without it are
    filtered as "poison" by the helper).
  - `tool_calls` follow HF-style: {id, type:"function", function:{name, arguments}}.
  - Tool responses carry `name` and `tool_call_id`.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # Generator/
SRC = ROOT / "outputs" / "v7_2k" / "jarvis_v7_2000.json"
CATALOG = ROOT / "configs" / "tools" / "jarvis_tools.json"
OUT_DIR = HERE / "v7_2k"
OUT = OUT_DIR / "jarvis_v7_functiongemma.jsonl"

SYSTEM_PROMPT = (
    "You are a Jarvis-CD HPC workflow assistant. Use the provided tools to "
    "create and manage pipelines, attach and configure packages, and operate "
    "the JarvisManager. Think briefly before each tool call. Call one tool "
    "at a time unless the user asks for multiple actions."
)

TOOLS_PER_EXAMPLE = 5  # training ceiling documented in CLAUDE.md
FINAL_THINK = (
    "The requested operations are complete; I'll summarize the outcome for the user."
)


def load_catalog(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text())
    return {t["name"]: t for t in raw["tools"]}


def catalog_to_function_schema(tool: dict) -> dict:
    props: dict[str, dict] = {}
    required: list[str] = []
    for p in tool.get("parameters", []):
        schema = {"type": p.get("type", "string")}
        if p.get("description"):
            schema["description"] = p["description"]
        props[p["name"]] = schema
        if p.get("required"):
            required.append(p["name"])
    description = tool.get("description", "").split(" Returns EXACTLY:")[0].strip()
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def pick_tools(required_names: list[str], catalog_names: list[str], seed: int) -> list[str]:
    """Return a deterministic list of tool names including the required ones first,
    padded with random other catalog tools up to TOOLS_PER_EXAMPLE.
    """
    rng = random.Random(seed)
    selected = list(dict.fromkeys(required_names))  # dedupe, preserve order
    pool = [n for n in catalog_names if n not in selected]
    rng.shuffle(pool)
    while len(selected) < TOOLS_PER_EXAMPLE and pool:
        selected.append(pool.pop())
    return selected[:TOOLS_PER_EXAMPLE]


def format_tool_response(step: dict) -> str:
    """JSON-encode the tool's observed output. Error steps carry an error string."""
    if step.get("status") == "failure":
        actual = step.get("actual_result")
        if actual is not None:
            return actual if isinstance(actual, str) else json.dumps(actual)
        err = step.get("error_message") or "Unknown error"
        return json.dumps({"error": err})
    result = step.get("actual_result") if step.get("actual_result") is not None else step.get("expected_result", {})
    return result if isinstance(result, str) else json.dumps(result)


def convert_example(ex: dict, catalog: dict[str, dict], idx: int) -> dict | None:
    reasoning = ex["solution"]["reasoning_path"]
    # collect required tools (skip None tool steps)
    required_tools = [s["tool"] for s in reasoning if s.get("tool")]
    if not required_tools:
        return None  # can't train a tool-call example with zero tool calls
    # make sure every required tool is in the catalog
    if any(t not in catalog for t in required_tools):
        return None

    tool_names = pick_tools(required_tools, sorted(catalog.keys()), seed=idx)
    tools_list = [catalog_to_function_schema(catalog[n]) for n in tool_names]

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT, "tools": tools_list},
        {"role": "user", "content": ex["instruction"]},
    ]

    call_counter = 0
    for step in reasoning:
        tool_name = step.get("tool")
        thought = step.get("thought", "")
        if not tool_name:
            # a "give up" / speak-only step: append an assistant turn with only a think block
            # (no tool_calls, no content). The notebook's helper needs think present.
            messages.append({"role": "assistant", "think": thought, "content": ""})
            continue
        call_counter += 1
        call_id = f"call_{call_counter}"
        args = step.get("args", {}) or {}
        messages.append(
            {
                "role": "assistant",
                "think": thought,
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": call_id,
                "content": format_tool_response(step),
            }
        )

    # final assistant turn with the natural-language answer
    final_answer = ex["solution"].get("final_answer", "").strip()
    if not final_answer:
        final_answer = "Done."
    messages.append({"role": "assistant", "think": FINAL_THINK, "content": final_answer})

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(CATALOG)
    data = json.loads(SRC.read_text())
    kept = 0
    dropped = 0
    method_counts: dict[str, int] = {}
    with OUT.open("w") as f:
        for idx, ex in enumerate(data):
            row = convert_example(ex, catalog, idx)
            if row is None:
                dropped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            m = ex["solution"].get("method", "?")
            method_counts[m] = method_counts.get(m, 0) + 1
    print(f"Wrote {OUT}")
    print(f"  kept   = {kept}")
    print(f"  dropped= {dropped}")
    print(f"  by method: {method_counts}")


if __name__ == "__main__":
    main()
