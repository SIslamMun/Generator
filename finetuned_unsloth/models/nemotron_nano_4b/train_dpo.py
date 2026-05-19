"""DPO stage for Nemotron-3 Nano 4B — kill the phantom-parameter flood.

SFT (even on a 100%-clean dataset, train loss 0.03) leaves the model flooding
`None` parameters at inference. That is exposure bias: SFT only ever scores the
model in clean token states, so the base model's "enumerate every parameter"
prior survives. DPO fixes it directly — it contrasts a clean call (chosen)
against the flooded call (rejected) and trains the model to prefer the former.

This continues from the SFT model: it loads the SFT-merged checkpoint, attaches
a fresh LoRA, and runs DPO. The frozen SFT model is the implicit reference
(Unsloth uses the LoRA-disabled model when ref_model=None).

Inputs : --sft-model  (the SFT merged_16bit dir)
         --dpo-data   (data/dpo.jsonl from prepare_dpo.py)
Outputs: <output>/merged_16bit  (DPO model — drop-in for push/verify)
         <output>/lora_dpo, <output>/dpo_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()  # must run before importing TRL's DPOTrainer

import torch                                    # noqa: E402
from datasets import load_dataset               # noqa: E402

HERE = Path(__file__).resolve().parent

MAX_SEQ_LENGTH    = 4096
MAX_PROMPT_LENGTH = 3072          # tool schemas are long; leave room for completion
LORA_R            = 32
LORA_ALPHA        = 64
TARGET_MODULES    = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "out_proj",
]
DPO_BETA          = 0.1           # standard; lower = more aggressive divergence
DPO_LR            = 5e-6          # DPO uses a far lower LR than SFT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sft-model", type=Path, required=True,
                    help="SFT merged_16bit checkpoint to continue from")
    ap.add_argument("--dpo-data", type=Path, required=True,
                    help="dpo.jsonl with {prompt, chosen, rejected}")
    ap.add_argument("--output", type=Path, default=HERE / "artifacts")
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print("=== Nemotron-3 Nano 4B — DPO stage ===")
    print(f"  sft-model : {args.sft_model}")
    print(f"  dpo-data  : {args.dpo_data}")
    print(f"  output    : {args.output}")
    print(f"  epochs    : {args.epochs}  beta={DPO_BETA}  lr={DPO_LR}")

    # ── load the SFT model ─────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = str(args.sft_model),
        max_seq_length    = MAX_SEQ_LENGTH,
        load_in_4bit      = False,
        load_in_8bit      = False,
        full_finetuning   = False,
        trust_remote_code = True,
    )

    # ── fresh LoRA for the DPO update ──────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        target_modules = TARGET_MODULES,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
        use_rslora     = False,
        loftq_config   = None,
    )

    # ── preference dataset ─────────────────────────────────────────
    ds = load_dataset("json", data_files=str(args.dpo_data), split="train")
    print(f"[data] {len(ds)} preference pairs; columns: {ds.column_names}")
    assert {"prompt", "chosen", "rejected"} <= set(ds.column_names), \
        "dpo.jsonl must have prompt/chosen/rejected"

    # ── DPO trainer ────────────────────────────────────────────────
    from trl import DPOTrainer, DPOConfig
    trainer = DPOTrainer(
        model      = model,
        ref_model  = None,                       # Unsloth uses LoRA-off as reference
        train_dataset = ds,
        tokenizer  = tokenizer,
        args = DPOConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio                = 0.1,
            num_train_epochs            = args.epochs,
            learning_rate               = DPO_LR,
            logging_steps               = 1,
            optim                       = "adamw_8bit",
            weight_decay                = 0.0,
            lr_scheduler_type           = "linear",
            seed                        = 3407,
            beta                        = DPO_BETA,
            max_length                  = MAX_SEQ_LENGTH,
            max_prompt_length           = MAX_PROMPT_LENGTH,
            report_to                   = "none",
            output_dir                  = str(args.output / "checkpoints_dpo"),
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu.name}. Max memory = {round(gpu.total_memory/1e9,1)} GB.")

    stats = trainer.train()

    used = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    print(f"{stats.metrics.get('train_runtime')} s used for DPO.")
    print(f"Peak reserved memory = {used} GB.")

    # ── save: LoRA + merged_16bit (drop-in for push.sbatch / verify) ─
    lora_dir = args.output / "lora_dpo"
    print(f"[save] DPO LoRA → {lora_dir}")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    merged_dir = args.output / "merged_16bit"
    print(f"[save] DPO merged-16bit → {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    summary = {
        "stage":           "dpo",
        "sft_model":       str(args.sft_model),
        "dpo_data":        str(args.dpo_data),
        "n_pairs":         len(ds),
        "epochs":          args.epochs,
        "beta":            DPO_BETA,
        "learning_rate":   DPO_LR,
        "train_loss":      stats.metrics.get("train_loss"),
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "peak_vram_gb":    used,
        "output_dir":      str(args.output),
    }
    (args.output / "dpo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary → {args.output / 'dpo_summary.json'}")


if __name__ == "__main__":
    main()
