"""Aurora XPU LoRA fine-tune of a chat model on QA+CoT ChatML data.

Drop-in replacement for train_qa.py that runs on Intel GPUs (Ponte Vecchio /
Max series) via intel-extension-for-pytorch + standard transformers + peft.
No unsloth, no bitsandbytes, no triton-cuda. Same CLI flags so the
`generator train-chat` pipeline can swap to it without other changes.

Usage:
    python train_qa_xpu.py \
        --model-name unsloth/gemma-3-270m-it \
        --dataset    runs/aurora-v1/data/chat/train.jsonl \
        --output     runs/aurora-v1/artifacts/chat \
        --epochs 2 --batch-size 8 --lr 2e-4 \
        --lora-r 64 --lora-alpha 128 --bf16 --save-merged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="XPU LoRA fine-tune (chat / QA+CoT).")
    p.add_argument("--model-name", default="unsloth/gemma-3-270m-it")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=3407)
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true")
    prec.add_argument("--fp16", action="store_true")
    p.add_argument("--save-merged", action="store_true")
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                   help="HF attention implementation. Use 'eager' to bypass XPU SDPA bug "
                        "with grouped-query attention (Gemma 3 family).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] model      = {args.model_name}")
    print(f"[config] dataset    = {dataset_path}")
    print(f"[config] output     = {output_dir}")
    so_e = f"max_steps={args.max_steps}" if args.max_steps > 0 else f"epochs={args.epochs}"
    print(f"[config] {so_e}")
    print(f"[config] batch      = {args.batch_size} (grad_accum={args.grad_accum})")
    print(f"[config] lr         = {args.lr}")
    print(f"[config] lora_r/a   = {args.lora_r}/{args.lora_alpha}")
    print(f"[config] precision  = {'bf16' if args.bf16 else ('fp16' if args.fp16 else 'auto')}")

    import torch
    import intel_extension_for_pytorch  # noqa: F401  (registers XPU backend)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    if torch.xpu.is_available():
        device = "xpu"
        print(f"[xpu] count={torch.xpu.device_count()}  current=xpu:{torch.xpu.current_device()}")
    else:
        device = "cpu"
        print("[xpu] not available — falling back to CPU (training will be slow)")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Llama-3.1 has no default pad token. Use the dedicated finetune-pad id (128004)
    # rather than re-using EOS (128009) which would mask real EOS during training.
    if tokenizer.pad_token_id is None:
        if "llama-3.1" in args.model_name.lower() or "meta-llama-3.1" in args.model_name.lower():
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128004)
            tokenizer.pad_token_id = 128004
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # TRL's get_training_chat_template auto-patches Llama-3 templates with
    # {% generation %} markers when assistant_only_loss=True is set on SFTConfig.
    # Don't inject our own — it confuses TRL's prefix-preservation check.

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,  # default "sdpa"; "eager" for Gemma GQA bug
    )
    model.config.use_cache = False  # incompatible with grad-ckpt; silences warning

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    # IMPORTANT: enable_input_require_grads must be called AFTER get_peft_model;
    # PEFT re-wraps forward and would silently drop the input-grad passthrough,
    # causing "element 0 of tensors does not require grad" on first backward
    # when gradient checkpointing is on.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    model.to(device)

    rows = [json.loads(line) for line in open(dataset_path)]
    raw = Dataset.from_list(rows)
    print(f"[data] loaded {len(raw)} conversations")

    def to_text(batch):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in batch["conversations"]
        ]
        return {"text": texts}

    dataset = raw.map(to_text, batched=True, remove_columns=raw.column_names)

    sft_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,                 # ~3% warmup matches Llama-3.1 LoRA recipes
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_torch",                # bitsandbytes adamw_8bit is CUDA-only
        weight_decay=0.0,                   # LoRA papers use 0
        lr_scheduler_type="cosine",         # cosine > linear for ~1k+ step runs
        seed=args.seed,
        max_length=args.max_seq_length,            # trl >=1.3 renamed from max_seq_length
        dataset_text_field="text",
        report_to="none",
        save_strategy="no",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # avoid XPU multi-worker pickling issues
        packing=False,
        # assistant_only_loss=True omitted: NousResearch's Llama-3.1 template
        # isn't recognized by TRL's auto-patcher, and our manual injection
        # fails prefix-preservation. Wastes ~30% capacity but trains correctly.
    )
    if args.max_steps > 0:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,    # TRL 1.3 renamed from tokenizer=
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )

    print("[train] starting")
    stats = trainer.train()
    print(f"[train] done in {stats.metrics['train_runtime']/60:.1f} min  "
          f"train_loss={stats.metrics.get('train_loss'):.4f}")

    lora_dir = output_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[save] LoRA adapter → {lora_dir}")

    if args.save_merged:
        merged_dir = output_dir / "merged_16bit"
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[save] merged 16-bit → {merged_dir}")


if __name__ == "__main__":
    main()
