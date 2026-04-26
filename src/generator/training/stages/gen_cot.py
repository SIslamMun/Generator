"""Stage: generate-cot — LanceDB chunks → QA+CoT pairs."""
from __future__ import annotations

from pathlib import Path
import sys

from ._subprocess import run


def chat_data_paths(cfg: dict):
    out = Path(cfg["output_dir"])
    return {
        "lancedb":  Path(cfg["chat"]["data"]["lancedb_path"]),
        "raw_cot":  out / "data" / "chat_cot_raw.json",
        "fixed":    out / "data" / "chat_cot_fixed.json",
        "curated":  out / "data" / "chat_cot_curated.json",
        "train":    out / "data" / "chat_cot.train.jsonl",
        "val":      out / "data" / "chat_cot.val.jsonl",
        "logs":     out / "logs",
    }


def run_gen_cot(cfg: dict) -> None:
    p = chat_data_paths(cfg)
    p["raw_cot"].parent.mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "generator.cli", "generate-cot",
        str(p["lancedb"]),
        "-o", str(p["raw_cot"]),
        "--table",        cfg["chat"]["data"]["table"],
        "--target-pairs", str(cfg["chat"]["data"]["target_pairs"]),
        "--provider",     cfg["llm"]["provider"],
        "--model",        cfg["llm"]["model"],
        "--workers",      str(cfg["llm"]["workers"]),
    ]
    if cfg["chat"]["data"].get("max_chunks"):
        cmd += ["--max-chunks", str(cfg["chat"]["data"]["max_chunks"])]
    if cfg.get("topic"):
        cmd += ["--topic", str(cfg["topic"])]

    run(cmd, log_path=p["logs"] / "gen_cot.log")
