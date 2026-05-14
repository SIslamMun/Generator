"""Fine-tune Nemotron-3 Nano 4B on NDP tool-use data.

This script is a 1:1 port of Unsloth's official notebook:
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Nemotron-3-Nano-30B-A3B_A100.ipynb

Only THREE things differ from the notebook:
  1. model_name           — 4B variant instead of the 30B-A3B
  2. dataset              — our JSONL of conversations, not OpenMathReasoning-mini
  3. minimal CLI flags    — --dataset, --output, --max-steps (everything else hardcoded
                            to match the notebook so it stays a known-good recipe)

The dataset JSONL must contain rows of shape:
  {"conversations": [{"role": "user|assistant|tool", "content": "...", ...}, ...]}
  optional: {"tools": [...]}   (rendered via apply_chat_template(tools=...))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

HERE = Path(__file__).resolve().parent

# ────────── notebook cell parity ──────────
MODEL_NAME       = "unsloth/NVIDIA-Nemotron-3-Nano-4B"   # cell 6 (variant: 4B not 30B-A3B)
MAX_SEQ_LENGTH   = 4096                                   # cell 6: bumped from notebook
                                                          # default of 2048 because our tool
                                                          # rows go up to 2822 tokens (long
                                                          # tool schemas + reasoning chains);
                                                          # 2048 truncated 100% of assistant
                                                          # turns → all labels became -100.
LORA_R           = 8                                      # cell 8: reverted to notebook default
LORA_ALPHA       = 16                                     # cell 8: 2*r convention
                                                          # r=32 made hallucination WORSE
                                                          # (model invented param names beyond
                                                          # the schema). Anti-hallucination
                                                          # signal now lives in the system
                                                          # message (see prepare_data.py).
LORA_DROPOUT     = 0                                      # cell 8
TARGET_MODULES   = [                                      # cell 8 — Mamba+Attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "out_proj",
]
INSTRUCTION_PART = "<|im_start|>user\n"                   # cell 20
RESPONSE_PART    = "<|im_start|>assistant\n"              # cell 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True,
                    help="JSONL with rows of {conversations, tools?, ...}")
    ap.add_argument("--output", type=Path, default=HERE / "artifacts")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="Match notebook default (60). Use small value for smoke test.")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"=== Nemotron-3 Nano 4B fine-tune ===")
    print(f"  model     : {MODEL_NAME}")
    print(f"  dataset   : {args.dataset}")
    print(f"  output    : {args.output}")
    print(f"  max_steps : {args.max_steps}")

    # ── cell 6: load model + tokenizer ─────────────────────────────
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = MODEL_NAME,
        max_seq_length    = MAX_SEQ_LENGTH,
        load_in_4bit      = False,
        load_in_8bit      = False,
        full_finetuning   = False,
        trust_remote_code = True,
    )

    # ── cell 8: LoRA adapters ──────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        target_modules = TARGET_MODULES,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
        use_rslora     = False,
        loftq_config   = None,
    )

    # ── cells 10–14: dataset → conversations → chat template → text ──
    # Our JSONL already has `conversations` (and optionally `tools`) per row.
    ds = load_dataset("json", data_files=str(args.dataset), split="train")
    print(f"[data] loaded {len(ds)} rows; columns: {ds.column_names}")
    assert "conversations" in ds.column_names, "JSONL must have a `conversations` field"

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        tools_list = examples.get("tools") if "tools" in ds.column_names else [None] * len(convos)
        texts = []
        for convo, tools in zip(convos, tools_list):
            kwargs = {"tokenize": False, "add_generation_prompt": False}
            if tools:
                try:
                    text = tokenizer.apply_chat_template(convo, tools=tools, **kwargs)
                except Exception:
                    text = tokenizer.apply_chat_template(convo, **kwargs)
            else:
                text = tokenizer.apply_chat_template(convo, **kwargs)
            texts.append(text)
        return {"text": texts}

    ds = ds.map(formatting_prompts_func, batched=True)
    print(f"[data] applied chat template; sample text (first 500 chars):")
    print(ds[0]["text"][:500])
    print()

    # ── cell 18: SFTTrainer + SFTConfig (notebook values verbatim) ──
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field           = "text",
            per_device_train_batch_size  = 4,
            gradient_accumulation_steps  = 2,
            warmup_steps                 = 5,
            max_steps                    = args.max_steps,
            learning_rate                = 2e-4,
            logging_steps                = 1,
            optim                        = "adamw_8bit",
            weight_decay                 = 0.001,
            lr_scheduler_type            = "linear",
            seed                         = 3407,
            report_to                    = "none",
            output_dir                   = str(args.output / "checkpoints"),
        ),
    )

    # ── cell 20: train on responses only ───────────────────────────
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = INSTRUCTION_PART,
        response_part    = RESPONSE_PART,
    )

    # ── cell 25: GPU memory ────────────────────────────────────────
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory       = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # ── cell 27: train ─────────────────────────────────────────────
    trainer_stats = trainer.train()

    # ── cell 28: post-train stats ──────────────────────────────────
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

    # ── cell 32: save LoRA ─────────────────────────────────────────
    lora_dir = args.output / "lora"
    print(f"[save] LoRA → {lora_dir}")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # ── cell 36: save merged_16bit (for vLLM / Ollama import) ──────
    merged_dir = args.output / "merged_16bit"
    print(f"[save] merged-16bit → {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # Summary
    summary = {
        "model":      MODEL_NAME,
        "dataset":    str(args.dataset),
        "n_rows":     len(ds),
        "max_steps":  args.max_steps,
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "train_runtime_s": trainer_stats.metrics.get("train_runtime"),
        "peak_vram_gb": used_memory,
        "output_dir": str(args.output),
    }
    (args.output / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary → {args.output / 'train_summary.json'}")


if __name__ == "__main__":
    main()
