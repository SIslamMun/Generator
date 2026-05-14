"""Train Nemotron-3 Nano 4B with Unsloth + TRL on a prepared JSONL.

Reads config.yaml (alongside this file) for all knobs, so users tune via
YAML, not CLI. Only one CLI flag: --dataset (path to the JSONL produced
by prepare_data.py).

Adapted from:
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Nemotron-3-Nano-30B-A3B_A100.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset

HERE = Path(__file__).resolve().parent


def load_config() -> dict:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True,
                    help="JSONL produced by prepare_data.py (each row has a `text` field)")
    ap.add_argument("--output", type=Path, default=HERE / "artifacts",
                    help="Where to write LoRA + merged checkpoints")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override config.train.max_steps (smoke testing)")
    args = ap.parse_args()

    cfg = load_config()
    print(f"=== {cfg['display_name']} — fine-tune ===")
    print(f"  model:   {cfg['hf_model_id']}")
    print(f"  dataset: {args.dataset}")
    print(f"  output:  {args.output}")
    print()

    # ── Load model ─────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    load = cfg["load"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = cfg["hf_model_id"],
        max_seq_length    = load["max_seq_length"],
        load_in_4bit      = load.get("load_in_4bit", False),
        load_in_8bit      = load.get("load_in_8bit", False),
        full_finetuning   = load.get("full_finetuning", False),
        trust_remote_code = load.get("trust_remote_code", True),
    )

    # ── LoRA ───────────────────────────────────────────────────────
    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora["r"],
        lora_alpha     = lora["alpha"],
        lora_dropout   = lora.get("dropout", 0.0),
        bias           = lora.get("bias", "none"),
        use_gradient_checkpointing = lora.get("use_gradient_checkpointing", "unsloth"),
        target_modules = lora["target_modules"],
        use_rslora     = lora.get("use_rslora", False),
        loftq_config   = None,
        random_state   = cfg["train"].get("seed", 3407),
    )

    # ── Dataset ────────────────────────────────────────────────────
    ds = load_dataset("json", data_files=str(args.dataset), split="train")
    print(f"[data] loaded {len(ds)} rows")
    if "text" not in ds.column_names:
        raise RuntimeError(f"dataset missing `text` column. Saw: {ds.column_names}. "
                           "Did prepare_data.py run cleanly?")

    # Show one sample so the train-on-responses_only masking can be sanity-checked
    print("\n[data] sample row text (first 600 chars):")
    print(ds[0]["text"][:600])
    print()

    # ── Trainer ────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    tcfg = cfg["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps", 0)
    epochs    = tcfg.get("epochs", 1)
    if max_steps and max_steps > 0:
        epochs_arg = {"max_steps": int(max_steps)}
    else:
        epochs_arg = {"num_train_epochs": float(epochs)}

    sft_args = SFTConfig(
        dataset_text_field           = "text",
        per_device_train_batch_size  = tcfg["batch_size"],
        gradient_accumulation_steps  = tcfg.get("grad_accum", 1),
        warmup_steps                 = tcfg.get("warmup_steps", 5),
        learning_rate                = float(tcfg["lr"]),
        logging_steps                = tcfg.get("logging_steps", 5),
        save_steps                   = tcfg.get("save_steps", 0) or None,
        optim                        = tcfg.get("optim", "adamw_8bit"),
        weight_decay                 = float(tcfg.get("weight_decay", 0.0)),
        lr_scheduler_type            = tcfg.get("scheduler", "linear"),
        seed                         = int(tcfg.get("seed", 3407)),
        report_to                    = tcfg.get("report_to", "none"),
        packing                      = tcfg.get("packing", False),
        output_dir                   = str(args.output / "checkpoints"),
        **epochs_arg,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = ds,
        eval_dataset  = None,
        args          = sft_args,
    )

    # ── train_on_responses_only ────────────────────────────────────
    from unsloth.chat_templates import train_on_responses_only
    mask = cfg["masking"]
    trainer = train_on_responses_only(
        trainer,
        instruction_part = mask["instruction_part"],
        response_part    = mask["response_part"],
    )

    # ── Pre-train memory probe ─────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_mem   = round(gpu.total_memory / 1024**3, 3)
    print(f"[gpu] {gpu.name} — max {max_mem} GB, reserved {start_mem} GB pre-train")

    # ── Train ──────────────────────────────────────────────────────
    stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print()
    print(f"[done] train_runtime: {stats.metrics.get('train_runtime', '?')} s")
    print(f"[done] train_loss   : {stats.metrics.get('train_loss', '?')}")
    print(f"[done] peak VRAM    : {used_mem} GB / {max_mem} GB")

    # ── Save ───────────────────────────────────────────────────────
    save = cfg["save"]
    out  = args.output
    out.mkdir(parents=True, exist_ok=True)

    if save.get("lora", True):
        p = out / "lora"
        print(f"[save] LoRA → {p}")
        model.save_pretrained(str(p))
        tokenizer.save_pretrained(str(p))

    if save.get("merged_16bit", False):
        p = out / "merged_16bit"
        print(f"[save] merged-16bit → {p}")
        model.save_pretrained_merged(str(p), tokenizer, save_method="merged_16bit")

    if save.get("merged_4bit", False):
        p = out / "merged_4bit"
        print(f"[save] merged-4bit → {p}")
        model.save_pretrained_merged(str(p), tokenizer, save_method="merged_4bit")

    gguf = save.get("gguf", {})
    if gguf.get("enabled", False):
        for quant in gguf.get("quantizations", ["q8_0"]):
            p = out / f"gguf_{quant}"
            print(f"[save] GGUF {quant} → {p}")
            model.save_pretrained_gguf(str(p), tokenizer, quantization_method=quant)

    # Final summary
    summary = {
        "model":        cfg["hf_model_id"],
        "dataset":      str(args.dataset),
        "n_rows":       len(ds),
        "epochs":       epochs if not (max_steps and max_steps > 0) else None,
        "max_steps":    max_steps if max_steps and max_steps > 0 else None,
        "train_loss":   stats.metrics.get("train_loss"),
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "peak_vram_gb": used_mem,
        "output_dir":   str(out),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary → {out / 'train_summary.json'}")


if __name__ == "__main__":
    main()
