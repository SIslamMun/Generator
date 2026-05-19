"""Unsloth LoRA fine-tuning backend — model-universal.

Unsloth (https://github.com/unslothai/unsloth) — fast, memory-efficient LoRA
training. Requires a CUDA GPU.

This backend is model-agnostic by design: the caller (CLI / web UI) supplies
only the flat, model-independent FinetuneConfig (model id, epochs, lr, LoRA
rank/alpha/dropout, batch …). Everything model-SPECIFIC — the Unsloth loader
class, LoRA target modules, response-masking turn markers, trust_remote_code
— is auto-resolved from the chosen model by `model_profiles.resolve()`. So
changing the model needs no other input.

Heavy deps (unsloth, torch, trl, datasets) are imported lazily inside
:meth:`run` so importing this module stays cheap on a CPU-only host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import FinetuneConfig
from ..model_profiles import resolve
from .base import Backend, load_conversations, write_summary


class UnslothBackend(Backend):
    """LoRA fine-tune via Unsloth — loader/targets/masking auto-resolved."""

    name = "unsloth"

    def run(self, cfg: FinetuneConfig) -> dict[str, Any]:
        # Import unsloth FIRST so its patches apply to trl/transformers.
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel, FastModel
        from unsloth.chat_templates import train_on_responses_only, get_chat_template

        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        # ── resolve the per-model deltas from the model itself ──────────
        profile = resolve(cfg.base_model, cfg)
        print(f"[unsloth] model profile:\n{profile.summary()}")

        Loader = FastLanguageModel if profile.loader == "language" else FastModel
        print(f"[unsloth] loading {cfg.base_model} via {Loader.__name__}")
        model, _tok = Loader.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            trust_remote_code=profile.trust_remote_code,
        )
        # A plain tokenizer for chat templating: multimodal loaders hand back
        # a processor whose apply_chat_template wants list-content, not the
        # plain-string conversations our datasets use.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=profile.trust_remote_code)

        # FIX 1: install the model's canonical chat template (e.g. "gemma-4")
        # when the family has a known-good one. This matches the official
        # Unsloth notebook recipe and avoids template-quirk bugs.
        if profile.chat_template_name:
            try:
                tokenizer = get_chat_template(
                    tokenizer, chat_template=profile.chat_template_name)
                print(f"[unsloth] installed chat template "
                      f"'{profile.chat_template_name}'")
            except Exception as e:                       # noqa: BLE001
                print(f"[unsloth] WARN: get_chat_template "
                      f"'{profile.chat_template_name}' failed ({e}); "
                      f"using the tokenizer's built-in template")

        # FIX 2: LoRA via finetune_*_layers flags for multimodal models
        # (Gemma 3/4), via target_modules for everything else. This is what
        # Unsloth's notebooks do; mis-using target_modules on a multimodal
        # model is what broke our Gemma 4 merge.
        if profile.multimodal:
            print("[unsloth] LoRA via finetune_*_layers flags (multimodal)")
            model = Loader.get_peft_model(
                model,
                finetune_vision_layers     = False,
                finetune_language_layers   = True,
                finetune_attention_modules = True,
                finetune_mlp_modules       = True,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                random_state=cfg.seed,
            )
        else:
            model = Loader.get_peft_model(
                model,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=profile.target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=cfg.seed,
            )

        # ── dataset → chat-template text ────────────────────────────────
        rows = load_conversations(cfg.dataset)
        n_with_tools = sum(1 for r in rows if r.get("tools"))
        print(f"[unsloth] dataset: {len(rows)} conversations "
              f"({n_with_tools} carry a tool catalog)")
        dataset = Dataset.from_list(rows)
        render_tools = cfg.render_tools
        # FIX 3: strip the tokenizer's BOS token from rendered text so the
        # SFTTrainer's tokenizer (which adds BOS again) doesn't produce a
        # double-BOS sequence — Unsloth's notebooks do .removeprefix('<bos>').
        bos = getattr(tokenizer, "bos_token", None) or ""

        def _format(batch):
            convos = batch["conversations"]
            tools_col = batch["tools"] if "tools" in batch else [None] * len(convos)
            texts = []
            for convo, tools in zip(convos, tools_col):
                kw = {"tokenize": False, "add_generation_prompt": False}
                rendered = None
                if tools and render_tools:
                    try:
                        rendered = tokenizer.apply_chat_template(
                            convo, tools=tools, **kw)
                    except Exception:               # template lacks tools= / patch bug
                        rendered = None
                if rendered is None:
                    try:
                        rendered = tokenizer.apply_chat_template(convo, **kw)
                    except Exception:               # unrenderable — dropped below
                        rendered = ""
                if bos and rendered.startswith(bos):
                    rendered = rendered[len(bos):]
                texts.append(rendered)
            return {"text": texts}

        dataset = dataset.map(_format, batched=True)
        n_before = len(dataset)
        dataset = dataset.filter(lambda r: bool((r["text"] or "").strip()))
        n_dropped = n_before - len(dataset)
        if n_dropped:
            print(f"[unsloth] WARNING: {n_dropped}/{n_before} rows could not be "
                  f"rendered by this model's chat template and were dropped")
        if len(dataset) == 0:
            raise RuntimeError(
                f"No rows could be rendered with {cfg.base_model}'s chat "
                f"template. The conversations likely use tool-call structures "
                f"this model's template (as patched by Unsloth) cannot render. "
                f"Try a different model, or pre-render the dataset's `text`.")
        print(f"[unsloth] {len(dataset)} trainable rows")
        print("[unsloth] sample rendered text (first 400 chars):")
        print(dataset[0]["text"][:400])

        # ── trainer ─────────────────────────────────────────────────────
        sft_kwargs = dict(
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
        # TRL's SFTConfig may default `eos_token` to a sentinel that isn't in
        # the vocabulary; pin it to the model's real EOS token.
        if getattr(tokenizer, "eos_token", None):
            sft_kwargs["eos_token"] = tokenizer.eos_token
        try:
            args = SFTConfig(**sft_kwargs)
        except TypeError:                            # older TRL without eos_token
            sft_kwargs.pop("eos_token", None)
            args = SFTConfig(**sft_kwargs)

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=dataset, args=args,
        )

        # Response-only masking when the model's turn markers are known.
        if profile.instruction_part and profile.response_part:
            trainer = train_on_responses_only(
                trainer,
                instruction_part=profile.instruction_part,
                response_part=profile.response_part,
            )
            print("[unsloth] response-only masking: ON")
        else:
            print("[unsloth] response-only masking: OFF "
                  "(no turn markers for this model — training full sequences)")

        stats = trainer.train()

        # ── save ────────────────────────────────────────────────────────
        lora_dir = Path(cfg.output_dir) / "lora"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        print(f"[unsloth] LoRA adapter → {lora_dir}")

        result: dict[str, Any] = {
            "backend": "unsloth",
            "base_model": cfg.base_model,
            "model_type": profile.model_type,
            "loader": Loader.__name__,
            "target_modules": profile.target_modules,
            "response_masking": bool(profile.instruction_part),
            "n_rows": len(rows),
            "lora_dir": str(lora_dir),
            "train_loss": stats.metrics.get("train_loss"),
            "train_runtime_s": stats.metrics.get("train_runtime"),
        }
        if cfg.save_merged:
            merged = Path(cfg.output_dir) / "merged_16bit"
            model.save_pretrained_merged(
                str(merged), tokenizer, save_method="merged_16bit")
            result["merged_dir"] = str(merged)
            print(f"[unsloth] merged 16-bit → {merged}")
        if cfg.export_gguf:
            gguf = Path(cfg.output_dir) / "gguf"
            model.save_pretrained_gguf(str(gguf), tokenizer)
            result["gguf_dir"] = str(gguf)
            print(f"[unsloth] GGUF → {gguf}")

        write_summary(cfg, result)
        return result
