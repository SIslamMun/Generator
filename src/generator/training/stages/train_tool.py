"""Stage: train-tool — FunctionGemma + LoRA on FunctionGemma-format JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

from ._subprocess import run
from .gen_tool import tool_data_paths


def tool_artifact_paths(cfg: dict):
    out = Path(cfg["output_dir"])
    return {
        "lora":   out / "artifacts" / "tool" / "lora",
        "merged": out / "artifacts" / "tool" / "merged_16bit",
        "logs":   out / "logs",
    }


def run_train_tool(cfg: dict) -> None:
    pdata = tool_data_paths(cfg)
    part  = tool_artifact_paths(cfg)
    part["lora"].parent.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent.parent.parent.parent.parent / \
             "finetuned_unsloth" / "legacy" / "train" / "train.py"
    if not script.exists():
        raise FileNotFoundError(f"train.py missing: {script}")

    tcfg = cfg["tool"]["train"]
    cmd = [
        sys.executable, str(script),
        "--model-name", tcfg["base_model"],
        "--dataset",    str(pdata["train"]),
        "--output",     str(part["lora"].parent),
        "--max-seq-length", str(tcfg.get("max_seq_length", 4096)),
        "--batch-size", str(tcfg["batch_size"]),
        "--grad-accum", str(tcfg.get("grad_accum", 1)),
        "--warmup-steps", str(tcfg.get("warmup_steps", 30)),
        "--lr",         str(tcfg["lr"]),
        "--lora-r",     str(tcfg["lora_r"]),
        "--lora-alpha", str(tcfg["lora_alpha"]),
    ]
    if tcfg.get("max_steps", 0) > 0:
        cmd += ["--max-steps", str(tcfg["max_steps"])]
    else:
        cmd += ["--epochs", str(tcfg["epochs"])]
    if tcfg.get("bf16"):        cmd.append("--bf16")
    if tcfg.get("save_merged"): cmd.append("--save-merged")

    run(cmd, log_path=part["logs"] / "train_tool.log")
