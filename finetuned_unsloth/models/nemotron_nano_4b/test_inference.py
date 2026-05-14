"""Test the fine-tuned Nemotron-3 Nano 4B on NDP tool-use queries.

Loads the merged_16bit checkpoint, prompts with a handful of NDP-style
queries (with the tools catalog in the system message), and prints the
model's tool calls. We're checking whether fine-tuning taught the model
to (a) pick the right tool, (b) pass valid args from the catalog enum,
(c) format the call correctly.
"""
from __future__ import annotations

# ── COMPAT SHIM (must run BEFORE mamba_ssm is imported) ────────────
# mamba_ssm 2.2.5 imports GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
# from transformers.generation, but transformers v5 removed these. Without
# this shim, `from mamba_ssm import ...` crashes at module import time —
# which then crashes Nemotron's modeling_nemotron_h.py:56 dynamic-remote-code
# load. Stub the names so import succeeds; we never actually use the classes.
import transformers.generation as _gen
import transformers.generation.utils as _gen_utils
if not hasattr(_gen, "GreedySearchDecoderOnlyOutput"):
    _gen.GreedySearchDecoderOnlyOutput = getattr(
        _gen_utils, "GenerateDecoderOnlyOutput", _gen_utils.ModelOutput
    )
if not hasattr(_gen, "SampleDecoderOnlyOutput"):
    _gen.SampleDecoderOnlyOutput = getattr(
        _gen_utils, "GenerateDecoderOnlyOutput", _gen_utils.ModelOutput
    )
# ───────────────────────────────────────────────────────────────────

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MERGED_DIR = HERE / "artifacts" / "merged_16bit"
NDP_TOOLS  = HERE.parent.parent.parent / "configs" / "tools" / "ndp_tools.json"


def load_tools_for_chat_template():
    """Render configs/tools/ndp_tools.json into the JSON-schema list that
    apply_chat_template(tools=...) expects (matches prepare_data.py)."""
    raw = json.loads(NDP_TOOLS.read_text())
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


import re

_TOOL_CALL_RE = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>",
    re.DOTALL,
)
# Tolerant fallback: model sometimes loops on <parameter> blocks and never
# emits </function>. Match an opening <function=NAME> then take everything
# after it (we still need a corresponding </function> for the strict regex).
_TOOL_CALL_OPEN_RE = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<body>.*)",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\s*(?P<val>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_call(model_output: str, tools_catalog: list[dict]) -> dict | None:
    """Extract the FIRST clean tool call from noisy model output.

    Strips:
      - parameters whose value is 'None', 'null', or whitespace-only
      - parameters whose name is not in the called tool's actual schema
      - duplicate parameter names (keeps the first non-None occurrence)
    Tolerant of truncated output (no closing </function>) — falls back to
    matching just <function=NAME> and parsing all <parameter> blocks after.

    Returns: {"name": tool_name, "arguments": {clean_args}} or None.
    """
    m = _TOOL_CALL_RE.search(model_output)
    truncated = False
    if not m:
        m = _TOOL_CALL_OPEN_RE.search(model_output)
        if not m:
            return None
        truncated = True
    name = m.group("name").strip()
    body = m.group("body")
    schema = next(
        (t["function"]["parameters"]["properties"] for t in tools_catalog
         if t["function"]["name"] == name),
        None,
    )
    if schema is None:
        return {"name": name, "arguments": {}, "_error": f"tool '{name}' not in catalog"}
    args: dict = {}
    for pm in _PARAM_RE.finditer(body):
        k, v = pm.group("key").strip(), pm.group("val").strip()
        if k not in schema:           continue   # phantom param name
        if v in ("None", "null", ""): continue   # phantom None value
        if k in args:                 continue   # dedup, keep first non-None
        # try to coerce value to the schema's type
        ptype = schema[k].get("type")
        if ptype == "array":
            try:
                args[k] = json.loads(v)
                continue
            except Exception:
                pass
        if ptype == "integer":
            try:
                args[k] = int(v); continue
            except ValueError: pass
        if ptype == "boolean":
            args[k] = v.lower() in ("true", "1", "yes"); continue
        args[k] = v
    result = {"name": name, "arguments": args}
    if truncated:
        result["_truncated"] = True
    return result


PROBES = [
    # easy: single-tool, exact match for catalog example
    "Which organizations publish datasets on the National Data Platform?",
    # filter argument
    "Show me NASA-related organizations.",
    # enum value
    "List organizations from the pre-CKAN staging catalog.",
    # search by term
    "Find datasets about climate.",
    # advanced search
    "List all datasets owned by NASA FIRMS in the global catalog.",
    # get details by ID
    "Get the full details for dataset 1f29f678-924c-456d-95d4-aa0bc7de7037.",
    # get details by name slug
    "Look up the dataset named clm-full-climate-connectivity-network.",
    # multi-step chain
    "Find NASA organizations, then list 3 CSV datasets they own.",
]


def main():
    print(f"=== loading model: {MERGED_DIR}")
    if not MERGED_DIR.exists():
        sys.exit(f"ERROR: {MERGED_DIR} not found. Did training save merged_16bit?")

    # Use Unsloth's loader (same path as training). Bare transformers fails
    # because mamba_ssm 2.2.5 imports `GreedySearchDecoderOnlyOutput` which
    # was removed in transformers v5; Unsloth has compat shims for this.
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = str(MERGED_DIR),
        max_seq_length    = 4096,
        load_in_4bit      = False,
        load_in_8bit      = False,
        trust_remote_code = True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  tokenizer: {tokenizer.__class__.__name__}")
    print(f"  model loaded on {next(model.parameters()).device}")

    tools = load_tools_for_chat_template()
    print(f"  tools: {[t['function']['name'] for t in tools]}")
    print()

    # Find <|im_end|> token id for proper stop — Nemotron emits this after
    # each turn; without it as eos, generate() keeps sampling and produces
    # hundreds of duplicate <|im_end|> tokens.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id] if im_end_id and im_end_id != tokenizer.eos_token_id \
              else tokenizer.eos_token_id
    print(f"  eos_token_id: {tokenizer.eos_token_id}  <|im_end|>={im_end_id}  using={eos_ids}")
    print()

    # Same anti-hallucination system message used during training — keep
    # train-time and inference-time prompts identical.
    ANTI_HALLUCINATION_SYSTEM = (
        "Tool-call discipline:\n"
        "- ONLY include parameters that you are actually setting to a value.\n"
        "- NEVER include parameters whose value would be None, null, empty string, or unset.\n"
        "- NEVER invent parameter names that are not in the tool's schema.\n"
        "- If a parameter is optional and you're not using it, OMIT IT ENTIRELY (do not emit it with a None placeholder)."
    )

    for i, query in enumerate(PROBES, 1):
        print("=" * 80)
        print(f"[{i}/{len(PROBES)}] USER: {query}")
        messages = [
            {"role": "system", "content": ANTI_HALLUCINATION_SYSTEM},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # Greedy decoding — structured output (tool calls) doesn't need
            # sampling; sampling at top_k=0 was letting the model emit phantom
            # schema params with None values.
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=False,    # Nemotron-H modeling_nemotron_h.py:1535 crashes
                                    # with `cache_position[-1]` on None when caching
                                    # is enabled; the Unsloth notebook also sets this.
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
                # Stop the moment the tool call closes. Without this the model
                # gets stuck in a loop emitting more <parameter> blocks until
                # max_new_tokens — and the parser can't extract a call that
                # never closes its </function> tag.
                stop_strings=["</function>"],
                tokenizer=tokenizer,        # required when stop_strings is set
            )
        # strip the prompt from output
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        parsed = parse_tool_call(decoded, tools)
        print(f"PARSED CALL: {json.dumps(parsed) if parsed else 'NONE FOUND'}")
        # show raw only on parse failure or for debugging
        if not parsed:
            print(f"RAW MODEL:\n{decoded[:300]}...")
        print()


if __name__ == "__main__":
    main()
