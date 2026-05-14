"""Smoke-test the Jarvis FunctionGemma model served via Ollama.

Same 10 held-out prompts as test_model.py, but routed through
Ollama's /api/generate with raw=true. Prompt rendering uses the HF
tokenizer so the byte sequence matches training exactly.
"""

import json
import sys
import time
import urllib.request
import zlib
import random
from pathlib import Path

sys.path.insert(0, "/u/sislam3/Generator")

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "jarvis-v7"
TOKENIZER_DIR = "/u/sislam3/Generator/finetuned_unsloth/artifacts/model_merged_16bit"
CATALOG = Path("/u/sislam3/Generator/configs/tools/jarvis_tools.yaml")

SYSTEM_PROMPT = (
    "You are a Jarvis-CD HPC workflow assistant. Use the provided tools to "
    "create and manage pipelines, attach and configure packages, and operate "
    "the JarvisManager. Think briefly before each tool call. Call one tool "
    "at a time unless the user asks for multiple actions."
)

TESTS = [
    {"query": "Spin up a new pipeline named astrophysics_sim for star formation modeling.",
     "expected_tool": "create_pipeline",
     "expected_arg_values": {"pipeline_id": "astrophysics_sim"}},
    {"query": "Load the pipeline climate_forecast_2026 so I can work with it.",
     "expected_tool": "load_pipeline",
     "expected_arg_values": {"pipeline_id": "climate_forecast_2026"}},
    {"query": "Run my existing quantum_chem pipeline.",
     "expected_tool": "run_pipeline",
     "expected_arg_values": {"pipeline_id": "quantum_chem"}},
    {"query": "Destroy the old deprecated_test pipeline and clean up everything.",
     "expected_tool": "destroy_pipeline",
     "expected_arg_values": {"pipeline_id": "deprecated_test"}},
    {"query": "Add an IOR package to my bench_suite pipeline.",
     "expected_tool": "append_pkg",
     "expected_arg_values": {"pipeline_id": "bench_suite", "pkg_type": "ior"}},
    {"query": "Show me the config of the hdf5 package in the data_pipe pipeline.",
     "expected_tool": "get_pkg_config",
     "expected_arg_values": {"pipeline_id": "data_pipe", "pkg_id": "hdf5"}},
    {"query": "List all the pipelines in my system.",
     "expected_tool": "jm_list_pipelines",
     "expected_arg_values": {}},
    {"query": "Set my current pipeline to gpu_training.",
     "expected_tool": "jm_cd",
     "expected_arg_values": {"pipeline_id": "gpu_training"}},
    {"query": "Reset the whole Jarvis system to a clean state.",
     "expected_tool": "jm_reset",
     "expected_arg_values": {}},
    {"query": "Bootstrap my Jarvis setup for the summit machine.",
     "expected_tool": "jm_bootstrap_from",
     "expected_arg_values": {"machine": "summit"}},
]


def load_catalog():
    import yaml
    raw = yaml.safe_load(CATALOG.read_text())
    tools = []
    for t in raw["tools"]:
        props, required = {}, []
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
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return tools


def pick_10(all_tools, target_name):
    seed = zlib.adler32(target_name.encode()) & 0xFFFF
    rng = random.Random(seed)
    target = next(t for t in all_tools if t["function"]["name"] == target_name)
    others = [t for t in all_tools if t["function"]["name"] != target_name]
    rng.shuffle(others)
    subset = [target] + others[:9]
    rng.shuffle(subset)
    return subset


from inference.render_and_parse import extract_tool_calls


def parse_tool_call(text):
    calls = extract_tool_calls(text)
    if not calls:
        return None, None
    c = calls[0]
    return c["name"], c["arguments"]


def ollama_generate(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST):
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "num_predict": 256,
            "num_ctx": 4096,
        },
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


def main():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    catalog = load_catalog()
    print(f"Catalog: {len(catalog)} tools | Backend: Ollama ({OLLAMA_MODEL} @ {OLLAMA_HOST})")

    correct_tool = 0
    correct_args = 0
    total_time = 0.0

    for i, test in enumerate(TESTS):
        tools = pick_10(catalog, test["expected_tool"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["query"]},
        ]
        prompt = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        ).removeprefix("<bos>")

        t0 = time.time()
        raw = ollama_generate(prompt)
        dt = time.time() - t0
        total_time += dt

        name, args = parse_tool_call(raw)
        tool_match = name == test["expected_tool"]
        # empty-dict-aware arg match
        expected = test["expected_arg_values"]
        if not expected:
            args_match = (args == {} or args is None) if tool_match else False
            # for tools with no args, just require tool_match + no-arg call
            args_match = tool_match and (args is None or args == {})
        else:
            args_match = bool(args) and all(
                str(args.get(k)) == str(v) for k, v in expected.items()
            )

        if tool_match:
            correct_tool += 1
        if tool_match and args_match:
            correct_args += 1

        status = "OK" if (tool_match and args_match) else ("TOOL" if tool_match else "FAIL")
        print(f"\n[{status}] test {i+1} ({dt:.1f}s): {test['query'][:70]}")
        print(f"  visible tools: {[t['function']['name'] for t in tools]}")
        print(f"  expected: {test['expected_tool']}({expected})")
        print(f"  got:      {name}({args})")
        if status == "FAIL":
            print(f"  raw: {raw[:250]}")

    n = len(TESTS)
    print(f"\n{'='*70}")
    print(f"Tool correct: {correct_tool}/{n} ({100*correct_tool/n:.0f}%)")
    print(f"Args correct: {correct_args}/{n} ({100*correct_args/n:.0f}%)")
    print(f"Total gen time: {total_time:.1f}s  ({total_time/n:.1f}s avg)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
