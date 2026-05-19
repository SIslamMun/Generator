"""Bake-off probe — does a fine-tuned model emit CLEAN NDP tool calls?

Model-agnostic: loads any merged checkpoint with Unsloth's FastModel (the same
loader used for training — handles Gemma 4 / Granite / Nemotron uniformly),
renders the 8 NDP probes with the model's own chat template + the 3-tool
catalog, generates greedily, and prints the RAW output.

The raw `<tool_call>` block is the verdict: a clean call lists only the
parameters actually used; a flooded call enumerates every catalog parameter
with `None`.

Usage:  MODEL_DIR=<merged checkpoint> python bakeoff_probe.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch
from transformers import AutoTokenizer
from unsloth import FastModel

HERE = Path(__file__).resolve().parent
MODEL_DIR = os.environ["MODEL_DIR"]
CATALOG = HERE.parent / "configs" / "tools" / "ndp_tools.json"

ANTI_HALLUCINATION_SYSTEM = (
    "Tool-call discipline:\n"
    "- ONLY include parameters that you are actually setting to a value.\n"
    "- NEVER include parameters whose value would be None, null, empty string, or unset.\n"
    "- NEVER invent parameter names that are not in the tool's schema.\n"
    "- If a parameter is optional and you're not using it, OMIT IT ENTIRELY."
)

PROBES = [
    "Which organizations publish datasets on the National Data Platform?",
    "Show me NASA-related organizations.",
    "List organizations from the pre-CKAN staging catalog.",
    "Find datasets about climate.",
    "List all datasets owned by NASA FIRMS in the global catalog.",
    "Get the full details for dataset 1f29f678-924c-456d-95d4-aa0bc7de7037.",
    "Look up the dataset named clm-full-climate-connectivity-network.",
    "Find NASA organizations, then list 3 CSV datasets they own.",
]


def load_catalog() -> list[dict]:
    """ndp_tools.json → JSON-schema `tools` form for apply_chat_template."""
    raw = json.loads(CATALOG.read_text())
    tools = raw["tools"] if isinstance(raw, dict) and "tools" in raw else raw
    out = []
    for t in tools:
        props, required = {}, []
        for p in t.get("parameters", []):
            prop = {"type": p["type"], "description": p.get("description", "")}
            if "enum" in p:
                prop["enum"] = p["enum"]
            props[p["name"]] = prop
            if p.get("required", False):
                required.append(p["name"])
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required},
        }})
    return out


def main() -> None:
    print(f"=== bake-off probe: {MODEL_DIR}")
    model, _proc = FastModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=4096,
        load_in_4bit=False, load_in_8bit=False, full_finetuning=False)
    FastModel.for_inference(model)
    # Use the plain text tokenizer for chat templating — multimodal models
    # return a processor whose apply_chat_template wants list-of-dict content;
    # the plain tokenizer handles string content + `tools=`, matching training.
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tools = load_catalog()
    print(f"  tools: {[t['function']['name'] for t in tools]}\n")

    for i, query in enumerate(PROBES, 1):
        msgs = [{"role": "system", "content": ANTI_HALLUCINATION_SYSTEM},
                {"role": "user", "content": query}]
        try:
            ids = tok.apply_chat_template(
                msgs, tools=tools, add_generation_prompt=True,
                tokenize=True, return_tensors="pt")
        except Exception:
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt")
        ids = ids.to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=512,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        print("=" * 78)
        print(f"[{i}/8] {query}")
        print("  RAW:", " ".join(gen.split())[:560])

    print("=" * 78)
    print("done — inspect each RAW block: clean = only real params; "
          "flood = every catalog param set to None.")


if __name__ == "__main__":
    main()
