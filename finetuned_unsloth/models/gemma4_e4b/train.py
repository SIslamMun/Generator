"""Fine-tune Gemma 4 E4B IT on NDP tool-use data (bake-off entry).

Gemma 4 is loaded with Unsloth's `FastModel`. Tool conversations are rendered
with Gemma's own chat template — `apply_chat_template` with `tools=` — so the
model learns Gemma's native tool-call format.

Dataset JSONL rows: {"conversations": [...], "tools": [...]?}
"""
from __future__ import annotations

import os
# Gemma 4 + Unsloth FastModel triggers torch.compile / inductor, which on this
# stack does `from triton.compiler.compiler import triton_key` — a symbol
# removed in triton 3.7. Disable TorchDynamo: Unsloth's own Triton kernels run
# independently of torch.compile, so training is unaffected.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

HERE = Path(__file__).resolve().parent

MODEL_NAME       = "unsloth/gemma-4-e4b-it"
MAX_SEQ_LENGTH   = 4096
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0
TARGET_MODULES   = [                       # Gemma 3 — pure-attention transformer
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART    = "<|turn>model\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=HERE / "artifacts")
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print("=== Gemma 3 4B IT fine-tune ===")
    print(f"  model     : {MODEL_NAME}")
    print(f"  dataset   : {args.dataset}")
    print(f"  output    : {args.output}")
    print(f"  max_steps : {args.max_steps}")

    # ── load model + tokenizer (FastModel — Unsloth's Gemma 3 path) ──
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name        = MODEL_NAME,
        max_seq_length    = MAX_SEQ_LENGTH,
        load_in_4bit      = False,
        load_in_8bit      = False,
        full_finetuning   = False,
    )

    # ── LoRA adapters ───────────────────────────────────────────────
    model = FastModel.get_peft_model(
        model,
        r              = LORA_R,
        target_modules = TARGET_MODULES,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
        use_rslora     = False,
    )

    # ── dataset → chat-template text ────────────────────────────────
    ds = load_dataset("json", data_files=str(args.dataset), split="train")
    print(f"[data] loaded {len(ds)} rows; columns: {ds.column_names}")
    assert "conversations" in ds.column_names, "JSONL must have a `conversations` field"

    has_tools = "tools" in ds.column_names

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        tools_list = examples.get("tools") if has_tools else [None] * len(convos)
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
    print("[data] sample text (first 600 chars):")
    print(ds[0]["text"][:600])
    print()

    # ── SFTTrainer ──────────────────────────────────────────────────
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
            warmup_steps                 = 30,
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

    # ── train on responses only ─────────────────────────────────────
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = INSTRUCTION_PART,
        response_part    = RESPONSE_PART,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu_stats.name}. Max memory = "
          f"{round(gpu_stats.total_memory/1e9,1)} GB.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")

    # ── save LoRA + merged ──────────────────────────────────────────
    lora_dir = args.output / "lora"
    print(f"[save] LoRA → {lora_dir}")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    merged_dir = args.output / "merged_16bit"
    print(f"[save] merged-16bit → {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

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
