"""Stage: fix-cot — re-attempt empty-reasoning rows from raw CoT output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ._subprocess import run
from .gen_cot import chat_data_paths


def has_empty_cot(raw_path: Path) -> bool:
    """Cheap probe: open the raw CoT JSON and return True if any reasoning is empty."""
    try:
        data = json.loads(raw_path.read_text())
    except Exception:
        return False
    if not isinstance(data, list):
        return False
    return any(not (r.get("reasoning") or "").strip() for r in data)


def run_fix_cot(cfg: dict) -> None:
    p = chat_data_paths(cfg)
    if not p["raw_cot"].exists():
        raise FileNotFoundError(f"raw CoT not found: {p['raw_cot']}")

    if not has_empty_cot(p["raw_cot"]):
        # nothing to fix — copy through so downstream stage finds the file
        p["fixed"].write_text(p["raw_cot"].read_text())
        print(f"[fix_cot] no empty CoT found, copying through → {p['fixed']}")
        return

    cmd = [
        sys.executable, "-m", "generator.cli", "fix-cot",
        str(p["raw_cot"]),
        "-o", str(p["fixed"]),
        "--provider",  cfg["llm"]["provider"],
        "--model",     cfg["llm"]["model"],
        "--workers",   str(cfg["llm"]["workers"]),
    ]
    run(cmd, log_path=p["logs"] / "fix_cot.log")
