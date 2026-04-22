"""Quick smoke-test: does the fine-tuned FunctionGemma actually emit valid
Jarvis tool calls?

Runs a handful of held-out prompts through the merged 16-bit model via HF
transformers and checks that the output includes
  `<start_function_call>call:<tool>{...}<end_function_call>`
with the right tool name.
"""

import json
import re
import sys
from pathlib import Path

MODEL_DIR = "/u/sislam3/Generator/finetuned_unsloth/artifacts/model_merged_16bit"
CATALOG = Path("/u/sislam3/Generator/configs/tools/jarvis_tools.yaml")

SYSTEM_PROMPT = (
    "You are a Jarvis-CD HPC workflow assistant. Use the provided tools to "
    "create and manage pipelines, attach and configure packages, and operate "
    "the JarvisManager. Think briefly before each tool call. Call one tool "
    "at a time unless the user asks for multiple actions."
)

# Held-out test queries with expected tool calls (not seen during training)
TESTS = [
    {
        "query": "Spin up a new pipeline named astrophysics_sim for star formation modeling.",
        "expected_tool": "create_pipeline",
        "expected_arg_values": {"pipeline_id": "astrophysics_sim"},
    },
    {
        "query": "Load the pipeline climate_forecast_2026 so I can work with it.",
        "expected_tool": "load_pipeline",
        "expected_arg_values": {"pipeline_id": "climate_forecast_2026"},
    },
    {
        "query": "Run my existing quantum_chem pipeline.",
        "expected_tool": "run_pipeline",
        "expected_arg_values": {"pipeline_id": "quantum_chem"},
    },
    {
        "query": "Destroy the old deprecated_test pipeline and clean up everything.",
        "expected_tool": "destroy_pipeline",
        "expected_arg_values": {"pipeline_id": "deprecated_test"},
    },
    {
        "query": "Add an IOR package to my bench_suite pipeline.",
        "expected_tool": "append_pkg",
        "expected_arg_values": {"pipeline_id": "bench_suite", "pkg_type": "ior"},
    },
    {
        "query": "Show me the config of the hdf5 package in the data_pipe pipeline.",
        "expected_tool": "get_pkg_config",
        "expected_arg_values": {"pipeline_id": "data_pipe", "pkg_id": "hdf5"},
    },
    {
        "query": "List all the pipelines in my system.",
        "expected_tool": "jm_list_pipelines",
        "expected_arg_values": {},
    },
    {
        "query": "Set my current pipeline to gpu_training.",
        "expected_tool": "jm_cd",
        "expected_arg_values": {"pipeline_id": "gpu_training"},
    },
    {
        "query": "Reset the whole Jarvis system to a clean state.",
        "expected_tool": "jm_reset",
        "expected_arg_values": {},
    },
    {
        "query": "Bootstrap my Jarvis setup for the summit machine.",
        "expected_tool": "jm_bootstrap_from",
        "expected_arg_values": {"machine": "summit"},
    },
]


def load_catalog() -> list[dict]:
    import yaml
    raw = yaml.safe_load(CATALOG.read_text())
    tools = []
    for t in raw["tools"]:
        props = {}
        required = []
        for p in t.get("parameters", []):
            props[p["name"]] = {"type": p.get("type", "string"),
                                "description": p.get("description", "")}
            if p.get("required"):
                required.append(p["name"])
        tools.append({
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
    return tools


def pick_10(all_tools, target_name):
    """Pick a 10-tool subset including target + 9 others. Uses stable seed for reproducibility."""
    import random
    # Use a stable hash (Python's hash() is randomized across processes)
    import zlib
    seed = zlib.adler32(target_name.encode()) & 0xFFFF
    rng = random.Random(seed)
    target = next(t for t in all_tools if t["function"]["name"] == target_name)
    others = [t for t in all_tools if t["function"]["name"] != target_name]
    rng.shuffle(others)
    subset = [target] + others[:9]
    rng.shuffle(subset)
    return subset


def parse_tool_call(raw: str):
    """Extract {name, args} from the raw output."""
    m = re.search(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", raw, re.DOTALL)
    if not m:
        return None, None
    name = m.group(1)
    args_blob = m.group(2)
    args = {}
    for key, v_esc, v_plain in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args_blob):
        raw_v = v_esc if v_esc or v_esc == "" else v_plain
        val = raw_v.strip("'\" ")
        # Try to cast
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        args[key] = val
    return name, args


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded: {model.__class__.__name__}, params={sum(p.numel() for p in model.parameters()):,}")

    catalog = load_catalog()

    correct_tool = 0
    correct_args = 0

    for i, test in enumerate(TESTS):
        tools = pick_10(catalog, test["expected_tool"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["query"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        ).removeprefix("<bos>")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # Greedy decode for deterministic tool calls
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        name, args = parse_tool_call(raw)

        tool_match = name == test["expected_tool"]
        args_match = bool(args) and all(
            str(args.get(k)) == str(v) for k, v in test["expected_arg_values"].items()
        )

        if tool_match:
            correct_tool += 1
        if tool_match and args_match:
            correct_args += 1

        status = "OK" if (tool_match and args_match) else ("TOOL" if tool_match else "FAIL")
        print(f"\n[{status}] test {i+1}: {test['query'][:70]}")
        print(f"  visible tools: {[t['function']['name'] for t in tools]}")
        print(f"  expected: {test['expected_tool']}({test['expected_arg_values']})")
        print(f"  got:      {name}({args})")
        if status == "FAIL":
            # Show raw output for debug
            print(f"  raw: {raw[:300]}")

    print(f"\n{'='*70}")
    print(f"Tool correct: {correct_tool}/{len(TESTS)} ({100*correct_tool/len(TESTS):.0f}%)")
    print(f"Args correct: {correct_args}/{len(TESTS)} ({100*correct_args/len(TESTS):.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
