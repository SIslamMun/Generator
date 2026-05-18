"""Unsloth LoRA fine-tuning backend.

Unsloth (https://github.com/unslothai/unsloth) — fast, memory-efficient
LoRA training. Requires a CUDA GPU. The heavy deps (unsloth, torch, trl,
datasets) are imported lazily inside :meth:`run` so importing this module
is cheap on a CPU-only host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import FinetuneConfig
from .base import Backend, load_conversations, write_summary

# A broad LoRA target set — covers Llama/Qwen/Mistral-style attention+MLP.
_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class UnslothBackend(Backend):
    """LoRA fine-tune via Unsloth's FastLanguageModel + TRL SFTTrainer."""

    name = "unsloth"

    def run(self, cfg: FinetuneConfig) -> dict[str, Any]:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        print(f"[unsloth] loading base model: {cfg.base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            trust_remote_code=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=_TARGET_MODULES,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )

        rows = load_conversations(cfg.dataset)
        print(f"[unsloth] dataset: {len(rows)} conversations")
        dataset = Dataset.from_list(rows)

        def _format(batch):
            texts = []
            for convo in batch["conversations"]:
                texts.append(
                    tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                )
            return {"text": texts}

        dataset = dataset.map(_format, batched=True)

        args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps or -1,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=cfg.seed,
            report_to="none",
            bf16=cfg.bf16,
            fp16=not cfg.bf16,
            output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        )
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=dataset, args=args
        )
        stats = trainer.train()

        lora_dir = Path(cfg.output_dir) / "lora"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        print(f"[unsloth] LoRA adapter → {lora_dir}")

        result: dict[str, Any] = {
            "backend": "unsloth",
            "base_model": cfg.base_model,
            "n_rows": len(rows),
            "lora_dir": str(lora_dir),
            "train_loss": stats.metrics.get("train_loss"),
            "train_runtime_s": stats.metrics.get("train_runtime"),
        }
        if cfg.save_merged:
            merged = Path(cfg.output_dir) / "merged_16bit"
            model.save_pretrained_merged(
                str(merged), tokenizer, save_method="merged_16bit"
            )
            result["merged_dir"] = str(merged)
            print(f"[unsloth] merged 16-bit → {merged}")
        if cfg.export_gguf:
            gguf = Path(cfg.output_dir) / "gguf"
            model.save_pretrained_gguf(str(gguf), tokenizer)
            result["gguf_dir"] = str(gguf)
            print(f"[unsloth] GGUF → {gguf}")

        write_summary(cfg, result)
        return result
