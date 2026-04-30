"""Compare baseline Llama-3.1-8B vs LoRA-tuned on 8 Aurora holdout questions.

Loads each model serially on a single PVC tile, generates answers, prints
side-by-side. Designed for inside a PBS job on Aurora.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


HOLDOUT = [
    {
        "id": "pbs-script",
        "q": ("Write a minimal Aurora PBS batch script that requests 4 nodes "
              "for 60 minutes on the debug queue, charges project gpu_hack, "
              "with filesystems=home:flare, then runs mpiexec over 96 ranks."),
    },
    {
        "id": "ze-affinity",
        "q": ("What does the ZE_AFFINITY_MASK environment variable do on "
              "Aurora, and what value would I set to expose only tile 0 "
              "of GPU 1?"),
    },
    {
        "id": "pytorch-xpu",
        "q": ("How do I load PyTorch with Intel GPU support on Aurora and "
              "move a tensor to the GPU? Show the module load and Python "
              "code."),
    },
    {
        "id": "vllm-aurora",
        "q": ("How do I launch vLLM on a single Aurora node, and which "
              "argument controls tensor-parallel sharding across the 12 "
              "PVC tiles?"),
    },
    {
        "id": "daos-pool",
        "q": ("How do I create a POSIX container in my DAOS pool on Aurora "
              "and mount it via dfuse on a compute node?"),
    },
    {
        "id": "sycl-compile",
        "q": ("Which compiler and flags compile a SYCL source file for "
              "Aurora's PVC GPUs using the Intel oneAPI toolchain?"),
    },
    {
        "id": "profiler",
        "q": ("I want a quick GPU-kernel timeline of my SYCL app on Aurora "
              "without instrumenting the source. Which tool, and what's "
              "the one-line invocation?"),
    },
    {
        "id": "flare-path",
        "q": ("What's the path to my Aurora project directory on the "
              "Lustre flare filesystem, and what is a recommended `lfs "
              "setstripe` setting for large files?"),
    },
]

SYSTEM = (
    "You are an expert assistant for users of the ALCF Aurora supercomputer "
    "(Intel Xeon Sapphire Rapids + Intel GPU Max 1550 / Ponte Vecchio, oneAPI, "
    "PBS scheduler). For every question, first explain your reasoning step by "
    "step, then give a concise, accurate answer.\n"
    "Use this format:\n**Reasoning:**\n<analysis>\n\n**Answer:**\n<answer>"
)


def generate_for_model(model_path: str, label: str, max_new_tokens: int = 600):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*88}\n[{label}] loading {model_path}\n{'='*88}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("xpu").eval()
    print(f"[{label}] loaded in {time.time()-t0:.1f}s")

    out = []
    for ex in HOLDOUT:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": ex["q"]},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to("xpu")
        t1 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        dt = time.time() - t1
        out.append({"id": ex["id"], "q": ex["q"], "a": text, "elapsed_s": dt})
        print(f"\n[{label}] {ex['id']}  ({dt:.1f}s, {len(gen_ids)} tokens)")
        print(text[:400] + ("..." if len(text) > 400 else ""))

    # free memory before loading next model
    del model
    torch.xpu.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--tuned", required=True, help="Path to merged_16bit model dir")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max-new-tokens", type=int, default=600)
    args = p.parse_args()

    base = generate_for_model(args.baseline, "BASE", args.max_new_tokens)
    tuned = generate_for_model(args.tuned,    "LORA", args.max_new_tokens)

    # Pair up
    paired = []
    for b, t in zip(base, tuned):
        paired.append({
            "id":       b["id"],
            "question": b["q"],
            "baseline": b["a"],
            "tuned":    t["a"],
            "elapsed_baseline_s": b["elapsed_s"],
            "elapsed_tuned_s":    t["elapsed_s"],
        })

    import json
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(paired, ensure_ascii=False, indent=2))
    print(f"\n[saved] {args.output}")

    # Print side-by-side summary
    print("\n" + "="*88 + "\nSUMMARY\n" + "="*88)
    for r in paired:
        print(f"\n--- [{r['id']}] {r['question']}")
        print(f"\n[BASE  {r['elapsed_baseline_s']:.1f}s]\n{r['baseline'][:500]}")
        print(f"\n[LORA  {r['elapsed_tuned_s']:.1f}s]\n{r['tuned'][:500]}")
        print()


if __name__ == "__main__":
    main()
