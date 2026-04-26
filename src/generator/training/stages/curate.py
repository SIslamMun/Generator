"""Stage: curate — LLM-as-judge filter, threshold-based."""
from __future__ import annotations

import sys

from ._subprocess import run
from .gen_cot import chat_data_paths


def run_curate(cfg: dict) -> None:
    p = chat_data_paths(cfg)
    cmd = [
        sys.executable, "-m", "generator.cli", "curate",
        str(p["fixed"]),
        "-o", str(p["curated"]),
        "--threshold", str(cfg["chat"]["curate"]["threshold"]),
        "--provider",  cfg["llm"]["provider"],
        "--model",     cfg["llm"]["model"],
        "--workers",   str(cfg["llm"]["workers"]),
    ]
    if cfg.get("topic"):
        cmd += ["--topic", str(cfg["topic"])]
    run(cmd, log_path=p["logs"] / "curate.log")
