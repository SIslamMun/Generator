"""Delta-AI headless LoRA fine-tune of Gemma3-270M-it on Jarvis QA+CoT data.

Expects the JSONL produced by `data/qa_cot_to_chatml.py` (conversations-style).
Uses Unsloth's Gemma3 chat template and `train_on_responses_only` masking so we
only train on the assistant turn.

Usage:
    python train_qa.py \
        --dataset finetuned_unsloth/data/qa_v1/jarvis_qa_v1_cot.jsonl \
        --output  /work/hdd/bekn/sislam3/jarvis_qa_v1_lora \
        --epochs  3 \
        --batch-size 16 \
        --grad-accum 1 \
        --lr 2e-4 \
        --lora-r 128 --lora-alpha 256 \
        --bf16 --save-merged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_parser():
    p = argparse.ArgumentParser(description="Gemma3-270m-it LoRA fine-tune on Jarvis QA+CoT")
    p.add_argument("--model-name", default="unsloth/gemma-3-270m-it")
    p.add_argument("--dataset", required=True, help="JSONL produced by qa_cot_to_chatml.py")
    p.add_argument("--output", required=True, help="Run artefacts (checkpoints, LoRA)")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=0,
                   help="If >0, overrides --epochs. If 0, use --epochs.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=3407)
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    p.add_argument("--save-merged", action="store_true",
                   help="After training, save merged 16-bit HF checkpoint alongside LoRA.")
    return p


def main():
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] model      = {args.model_name}")
    print(f"[config] dataset    = {dataset_path}")
    print(f"[config] output     = {output_dir}")
    steps_or_epochs = f"max_steps={args.max_steps}" if args.max_steps > 0 else f"epochs={args.epochs}"
    print(f"[config] {steps_or_epochs}")
    print(f"[config] batch      = {args.batch_size} (grad_accum={args.grad_accum})")
    print(f"[config] lr         = {args.lr}")
    print(f"[config] lora_r/a   = {args.lora_r}/{args.lora_alpha}")
    print(f"[config] precision  = {'bf16' if args.bf16 else ('fp16' if args.fp16 else 'auto')}")

    # Lazy imports so --help works without unsloth.
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

    # Load JSONL of conversations and render to `text` field using Gemma3 chat template.
    rows = [json.loads(line) for line in open(dataset_path)]
    dataset = Dataset.from_list(rows)
    print(f"[data] loaded {len(dataset)} conversations")

    def to_text(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                     .removeprefix("<bos>")
            for c in convos
        ]
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True)

    # Build SFT config.
    sft_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(output_dir / "checkpoints"),
        report_to="none",
        save_strategy="no",
    )
    if args.max_steps > 0:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.epochs
    if args.bf16:
        sft_kwargs["bf16"] = True
    elif args.fp16:
        sft_kwargs["fp16"] = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(**sft_kwargs),
    )
    # Mask out the system+user turns — train only on the assistant response.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        print(f"[gpu] {g.name} / {round(g.total_memory / 1024**3, 2)} GB")

    print("[train] starting")
    stats = trainer.train()
    print(f"[train] done in {stats.metrics['train_runtime']/60:.1f} min  "
          f"train_loss={stats.metrics.get('train_loss'):.4f}")

    lora_dir = output_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[save] LoRA adapter → {lora_dir}")

    if args.save_merged:
        merged = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        print(f"[save] merged 16bit → {merged}")


if __name__ == "__main__":
    main()
