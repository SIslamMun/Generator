"""Test the LoRA the Unsloth reference run produced.

Loads gemma_4_lora/ via Unsloth (base + adapter) and runs several probes —
sanity, chat, reasoning, and an NDP-style question — using the notebook's
inference recipe (apply_chat_template return_dict=True). Prints each response.
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from pathlib import Path
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

HERE = Path(__file__).resolve().parent
LORA_DIR = str(HERE / "gemma_4_lora")

print(f"=== loading base + adapter from {LORA_DIR}")
model, tokenizer = FastModel.from_pretrained(
    model_name = LORA_DIR,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)
tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")
FastModel.for_inference(model)

PROBES = [
    "Why is the sky blue?",
    "What is 2 + 2? Explain briefly.",
    "Write a 3-line haiku about the ocean.",
    "Summarize the National Data Platform in 1 sentence.",
    "Continue the sequence: 1, 1, 2, 3, 5, 8,",
]

for i, prompt in enumerate(PROBES, 1):
    print("\n" + "=" * 72)
    print(f"[{i}/{len(PROBES)}] {prompt}")
    print("-" * 72)
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 200,
            temperature = 1.0, top_p = 0.95, top_k = 64,
            use_cache = True,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens = True)
    print(gen.strip())

print("\n" + "=" * 72)
print("DONE")
