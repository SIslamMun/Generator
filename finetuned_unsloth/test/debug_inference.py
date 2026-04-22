"""Debug: show exactly what prompt the model sees for a failing case."""
import json, sys
from pathlib import Path

MODEL_DIR = "/u/sislam3/Generator/finetuned_unsloth/artifacts/model_merged_16bit"
CATALOG = Path("/u/sislam3/Generator/configs/tools/jarvis_tools.yaml")
SYSTEM_PROMPT = (
    "You are a Jarvis-CD HPC workflow assistant. Use the provided tools to "
    "create and manage pipelines, attach and configure packages, and operate "
    "the JarvisManager. Think briefly before each tool call. Call one tool "
    "at a time unless the user asks for multiple actions."
)

def load_catalog():
    import yaml
    raw = yaml.safe_load(CATALOG.read_text())
    tools = []
    for t in raw["tools"]:
        props = {}
        required = []
        for p in t.get("parameters", []):
            props[p["name"]] = {"type": p.get("type", "string"), "description": p.get("description", "")}
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
    import random
    rng = random.Random(hash(target_name) & 0xFFFF)
    target = next(t for t in all_tools if t["function"]["name"] == target_name)
    others = [t for t in all_tools if t["function"]["name"] != target_name]
    rng.shuffle(others)
    subset = [target] + others[:9]
    rng.shuffle(subset)
    return subset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(f"Loading...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
model.eval()

catalog = load_catalog()

# Test 3: "Run my existing quantum_chem pipeline" expected run_pipeline
query = "Run my existing quantum_chem pipeline."
tools = pick_10(catalog, "run_pipeline")
print(f"\n=== VISIBLE TOOLS ({len(tools)}) ===")
for t in tools:
    print(f"  - {t['function']['name']}: {t['function']['description'][:60]}")

messages = [{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":query}]
prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False).removeprefix("<bos>")

# Print the last 2000 chars of the prompt (to see end near generation point)
print(f"\n=== PROMPT (last 1500 chars) ===")
print(prompt[-1500:])
print("\n=== GENERATION ===")

inputs = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=256, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(generated[:800])
