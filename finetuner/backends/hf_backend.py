"""HuggingFace LoRA fine-tuning backend (transformers + peft + TRL).

The portable path: plain `transformers` + `peft` LoRA + TRL's SFTTrainer — no
custom kernels, useful when Unsloth doesn't support an architecture. Slower
and heavier than the unsloth backend; requires a CUDA GPU for any real run.

Like the unsloth backend, it is model-agnostic: LoRA target modules and
tool-rendering are auto-resolved from the chosen model by
`model_profiles.resolve()`. It does NOT do response-only masking (that path
needs Unsloth) — it trains on full sequences; use the unsloth backend when
masking matters or for multimodal models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import FinetuneConfig
from ..model_profiles import resolve
from .base import Backend, load_conversations, write_summary


class HFBackend(Backend):
    """LoRA fine-tune via transformers + peft + TRL SFTTrainer."""

    name = "hf"

    def run(self, cfg: FinetuneConfig) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        # ── per-model deltas, auto-resolved from the model ──────────────
        profile = resolve(cfg.base_model, cfg)
        print(f"[hf] model profile:\n{profile.summary()}")

        print(f"[hf] loading base model: {cfg.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=profile.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=profile.trust_remote_code,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        )
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=profile.target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # ── dataset → chat-template text (with tools when present) ──────
        rows = load_conversations(cfg.dataset)
        n_with_tools = sum(1 for r in rows if r.get("tools"))
        print(f"[hf] dataset: {len(rows)} conversations "
              f"({n_with_tools} carry a tool catalog)")
        render_tools = cfg.render_tools

        def _render(r):
            kw = {"tokenize": False, "add_generation_prompt": False}
            if r.get("tools") and render_tools:
                try:
                    return tokenizer.apply_chat_template(
                        r["conversations"], tools=r["tools"], **kw)
                except Exception:
                    pass
            return tokenizer.apply_chat_template(r["conversations"], **kw)

        dataset = Dataset.from_list([{"text": _render(r)} for r in rows])

        sft_kwargs = dict(
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
            output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        )
        if getattr(tokenizer, "eos_token", None):
            sft_kwargs["eos_token"] = tokenizer.eos_token
        # SFTConfig's sequence-length kwarg was renamed max_seq_length →
        # max_length across TRL versions; try the modern name, then the old.
        args = None
        for seq_kw in ("max_length", "max_seq_length", None):
            kw = dict(sft_kwargs)
            if seq_kw:
                kw[seq_kw] = cfg.max_seq_length
            try:
                args = SFTConfig(**kw)
                break
            except TypeError:
                kw.pop("eos_token", None)            # also retry without eos_token
                try:
                    args = SFTConfig(**kw)
                    break
                except TypeError:
                    continue
        if args is None:
            args = SFTConfig(output_dir=str(Path(cfg.output_dir) / "checkpoints"))

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=dataset, args=args,
        )
        stats = trainer.train()

        lora_dir = Path(cfg.output_dir) / "lora"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        print(f"[hf] LoRA adapter → {lora_dir}")

        result: dict[str, Any] = {
            "backend": "hf",
            "base_model": cfg.base_model,
            "model_type": profile.model_type,
            "target_modules": profile.target_modules,
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
