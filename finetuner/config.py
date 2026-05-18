"""Shared fine-tune configuration.

``FinetuneConfig`` is the single parameter set the dispatcher and every
backend agree on. It mirrors the inputs the Phagocyte web UI collects for
Phase 6 (base model, LoRA rank/alpha/dropout, epochs, lr, batch size,
output dir) — see issue grc-iit/Phagocyte#4.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("unsloth", "hf", "ollama")


@dataclass
class FinetuneConfig:
    """One fine-tune job's configuration."""

    # ── required ──────────────────────────────────────────────────────
    backend: str          # unsloth | hf | ollama
    base_model: str       # HF model id (unsloth/hf); base to build on (ollama)
    dataset: str          # path to training data — JSONL of conversations
    output_dir: str       # directory for artifacts (adapter / merged / GGUF)

    # ── LoRA adapter ──────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── training hyperparameters ──────────────────────────────────────
    epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 1
    warmup_steps: int = 10
    max_seq_length: int = 2048
    max_steps: int = 0    # 0 → train by epochs; >0 → cap steps (smoke test)
    seed: int = 3407
    bf16: bool = True     # bf16 on Ampere+; backends fall back to fp16

    # ── outputs ───────────────────────────────────────────────────────
    save_merged: bool = False   # also write a merged fp16 checkpoint
    export_gguf: bool = False   # also export a GGUF (for Ollama import)

    # ── ollama backend ────────────────────────────────────────────────
    model_name: str = "phagocyte-finetuned"   # name for `ollama create`

    def validate(self) -> None:
        """Raise ValueError on an unusable config."""
        if self.backend not in BACKENDS:
            raise ValueError(
                f"unknown backend '{self.backend}' (choose one of {BACKENDS})"
            )
        if not self.base_model:
            raise ValueError("base_model is required")
        # `dataset` is only consumed by training backends; the ollama
        # backend just packages an already-trained checkpoint.
        if self.backend in ("unsloth", "hf"):
            if not self.dataset:
                raise ValueError("dataset is required for training backends")
            if self.lora_rank <= 0:
                raise ValueError("lora_rank must be positive")
            if self.epochs <= 0 and self.max_steps <= 0:
                raise ValueError("set epochs > 0 or max_steps > 0")

    def summary(self) -> str:
        """A one-block human-readable summary."""
        lines = [
            f"  backend       : {self.backend}",
            f"  base_model    : {self.base_model}",
            f"  dataset       : {self.dataset}",
            f"  output_dir    : {self.output_dir}",
            f"  lora          : r={self.lora_rank} alpha={self.lora_alpha} "
            f"dropout={self.lora_dropout}",
            f"  train         : epochs={self.epochs} lr={self.learning_rate} "
            f"batch={self.batch_size} grad_accum={self.grad_accum}",
        ]
        if self.max_steps:
            lines.append(f"  max_steps     : {self.max_steps}")
        return "\n".join(lines)
