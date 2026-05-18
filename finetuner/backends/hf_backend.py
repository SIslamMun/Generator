"""HuggingFace LoRA fine-tuning backend (transformers + peft + TRL).

The portable path: plain `transformers` + `peft` LoRA + TRL's SFTTrainer.
Slower and heavier than Unsloth but has no custom kernels — useful when
Unsloth doesn't support a model architecture. Requires a CUDA GPU for any
practical run. Heavy deps are imported lazily inside :meth:`run`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import FinetuneConfig
from .base import Backend, load_conversations, write_summary

_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class HFBackend(Backend):
    """LoRA fine-tune via transformers + peft + TRL SFTTrainer."""

    name = "hf"

    def run(self, cfg: FinetuneConfig) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        print(f"[hf] loading base model: {cfg.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        )
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_TARGET_MODULES,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        rows = load_conversations(cfg.dataset)
        print(f"[hf] dataset: {len(rows)} conversations")
        dataset = Dataset.from_list(
            [
                {
                    "text": tokenizer.apply_chat_template(
                        r["conversations"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                }
                for r in rows
            ]
        )

        args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps or -1,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=cfg.seed,
            report_to="none",
            bf16=cfg.bf16,
            fp16=not cfg.bf16,
            max_seq_length=cfg.max_seq_length,
            output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        )
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=dataset, args=args
        )
        stats = trainer.train()

        lora_dir = Path(cfg.output_dir) / "lora"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        print(f"[hf] LoRA adapter → {lora_dir}")

        result: dict[str, Any] = {
            "backend": "hf",
            "base_model": cfg.base_model,
            "n_rows": len(rows),
            "lora_dir": str(lora_dir),
            "train_loss": stats.metrics.get("train_loss"),
            "train_runtime_s": stats.metrics.get("train_runtime"),
        }
        if cfg.save_merged:
            merged = Path(cfg.output_dir) / "merged_16bit"
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged))
            tokenizer.save_pretrained(str(merged))
            result["merged_dir"] = str(merged)
            print(f"[hf] merged 16-bit → {merged}")

        write_summary(cfg, result)
        return result
