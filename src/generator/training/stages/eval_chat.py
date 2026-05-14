"""Stage: eval-chat — score the trained chat model on the held-out val set."""
from __future__ import annotations

import sys
from pathlib import Path

from ._subprocess import run
from .gen_cot import chat_data_paths
from .train_chat import chat_artifact_paths


def run_eval_chat(cfg: dict) -> None:
    pdata = chat_data_paths(cfg)
    part  = chat_artifact_paths(cfg)
    out_dir = Path(cfg["output_dir"]) / "reports" / "chat_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdata["val"].exists():
        print(f"[eval_chat] no val file (val_split=0?), skipping eval")
        return
    if not part["merged"].exists():
        raise FileNotFoundError(f"merged model not found: {part['merged']}")

    eval_script = Path(__file__).resolve().parent.parent.parent.parent.parent / \
                  "finetuned_unsloth" / "legacy" / "test" / "eval_qa.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"chat eval script missing: {eval_script}")

    ecfg = cfg["chat"]["eval"]
    cmd = [
        sys.executable, str(eval_script),
        "--model", str(part["merged"]),
        "--val",   str(pdata["val"]),
        "--out",   str(out_dir),
        "--max-new-tokens", "700",
    ]
    if ecfg.get("max_examples"):
        cmd += ["--max-examples", str(ecfg["max_examples"])]
    if ecfg.get("baseline"):
        cmd += ["--baseline", ecfg["baseline"]]

    run(cmd, log_path=Path(cfg["output_dir"]) / "logs" / "eval_chat.log")
