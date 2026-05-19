"""finetuner CLI — dispatch a Phase 6 fine-tune job to a backend.

    finetuner run --backend unsloth --base-model unsloth/llama-3.1-8b \\
        --dataset training_data.jsonl --output runs/llama-ft \\
        --lora-rank 16 --lora-alpha 32 --epochs 1 --lr 1e-5 --batch-size 4

The same flags drive every backend; `--backend` picks the implementation
(see issue grc-iit/Phagocyte#4).
"""

from __future__ import annotations

import sys

import click

from .backends import get_backend
from .config import BACKENDS, FinetuneConfig


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Multi-backend fine-tuning (Phase 6): Unsloth / HuggingFace / Ollama."""


@main.command("list")
def list_backends():
    """List the available fine-tune backends."""
    rows = [
        ("unsloth", "Fast LoRA training (CUDA). Recommended for training."),
        ("hf", "Portable LoRA training via transformers+peft (CUDA)."),
        ("ollama", "Package a trained checkpoint into an Ollama model."),
    ]
    for name, desc in rows:
        click.echo(f"  {name:<9} {desc}")


@main.command("run")
@click.option("--backend", type=click.Choice(BACKENDS), required=True,
              help="Fine-tune backend")
@click.option("--base-model", required=True,
              help="HF model id (unsloth/hf), or a merged-checkpoint/GGUF path (ollama)")
@click.option("--dataset", default="",
              help="Training data — JSONL/JSON of conversations (training backends)")
@click.option("-o", "--output", "output_dir", required=True,
              help="Output directory for artifacts")
@click.option("--lora-rank", type=int, default=16, show_default=True)
@click.option("--lora-alpha", type=int, default=32, show_default=True)
@click.option("--lora-dropout", type=float, default=0.05, show_default=True)
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--lr", "learning_rate", type=float, default=1e-5, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--grad-accum", type=int, default=1, show_default=True)
@click.option("--max-seq-length", type=int, default=2048, show_default=True)
@click.option("--max-steps", type=int, default=0,
              help="Cap training steps (0 = train by epochs)")
@click.option("--bf16/--fp16", default=True, help="Training precision")
@click.option("--save-merged", is_flag=True, help="Also write a merged fp16 checkpoint")
@click.option("--export-gguf", is_flag=True, help="Also export a GGUF (unsloth only)")
@click.option("--model-name", default="phagocyte-finetuned", show_default=True,
              help="Name for the Ollama model (ollama backend)")
@click.option("--no-render-tools", is_flag=True,
              help="Do not render a tool catalog even if the dataset carries one")
# ── developer overrides — normally auto-resolved from the model; the UI
#    never sets these (see finetuner/model_profiles.py) ──────────────────
@click.option("--loader", default="", help="[override] language | general | vision")
@click.option("--target-modules", default="",
              help="[override] comma-separated LoRA target modules")
@click.option("--instruction-part", default="",
              help="[override] response-masking instruction marker")
@click.option("--response-part", default="",
              help="[override] response-masking response marker")
def run(backend, base_model, dataset, output_dir, lora_rank, lora_alpha,
        lora_dropout, epochs, learning_rate, batch_size, grad_accum,
        max_seq_length, max_steps, bf16, save_merged, export_gguf, model_name,
        no_render_tools, loader, target_modules, instruction_part, response_part):
    """Run a fine-tune job."""
    cfg = FinetuneConfig(
        backend=backend, base_model=base_model, dataset=dataset,
        output_dir=output_dir, lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, grad_accum=grad_accum,
        max_seq_length=max_seq_length, max_steps=max_steps, bf16=bf16,
        save_merged=save_merged, export_gguf=export_gguf, model_name=model_name,
        render_tools=not no_render_tools,
        loader=loader,
        target_modules=[m.strip() for m in target_modules.split(",") if m.strip()],
        instruction_part=instruction_part, response_part=response_part,
    )
    try:
        cfg.validate()
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"\n=== finetuner: {backend} backend ===")
    click.echo(cfg.summary())
    click.echo("")

    backend_impl = get_backend(backend)
    try:
        result = backend_impl.run(cfg)
    except Exception as e:  # noqa: BLE001 — surface any backend failure cleanly
        import traceback
        traceback.print_exc()
        raise click.ClickException(f"{backend} backend failed: {e}") from e

    click.echo(f"\n[done] {result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
