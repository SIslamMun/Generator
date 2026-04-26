"""Stage: train-chat — Gemma3 + LoRA on ChatML JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

from ._subprocess import run
from .gen_cot import chat_data_paths


def chat_artifact_paths(cfg: dict):
    out = Path(cfg["output_dir"])
    return {
        "lora":   out / "artifacts" / "chat" / "lora",
        "merged": out / "artifacts" / "chat" / "merged_16bit",
        "logs":   out / "logs",
    }


def run_train_chat(cfg: dict) -> None:
    pdata = chat_data_paths(cfg)
    part  = chat_artifact_paths(cfg)
    part["lora"].parent.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).resolve().parent.parent.parent.parent.parent / \
                   "finetuned_unsloth" / "train" / "train_qa.py"
    if not train_script.exists():
        raise FileNotFoundError(f"chat trainer missing: {train_script}")

    tcfg = cfg["chat"]["train"]
    cmd = [
        sys.executable, str(train_script),
        "--model-name",      tcfg["base_model"],
        "--dataset",         str(pdata["train"]),
        "--output",          str(part["lora"].parent),
        "--epochs",          str(tcfg["epochs"]),
        "--batch-size",      str(tcfg["batch_size"]),
        "--grad-accum",      str(tcfg.get("grad_accum", 1)),
        "--warmup-steps",    str(tcfg.get("warmup_steps", 30)),
        "--lr",              str(tcfg["lr"]),
        "--lora-r",          str(tcfg["lora_r"]),
        "--lora-alpha",      str(tcfg["lora_alpha"]),
        "--max-seq-length",  str(tcfg.get("max_seq_length", 2048)),
    ]
    if tcfg.get("bf16"):         cmd.append("--bf16")
    if tcfg.get("save_merged"):  cmd.append("--save-merged")
    if tcfg.get("max_steps", 0): cmd += ["--max-steps", str(tcfg["max_steps"])]

    run(cmd, log_path=part["logs"] / "train_chat.log")
