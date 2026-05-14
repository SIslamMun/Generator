"""Stage: schema-filter — drop tool-use rows whose arg shapes violate the catalog."""
from __future__ import annotations

import sys
from pathlib import Path

from ._subprocess import run
from .gen_tool import tool_data_paths


def run_schema_filter(cfg: dict) -> None:
    p = tool_data_paths(cfg)
    script = Path(__file__).resolve().parent.parent.parent.parent.parent / \
             "finetuned_unsloth" / "legacy" / "data" / "schema_filter.py"
    if not script.exists():
        raise FileNotFoundError(f"schema_filter.py missing: {script}")

    cmd = [
        sys.executable, str(script),
        "--input",   str(p["raw"]),
        "--output",  str(p["clean"]),
        "--catalog", str(p["tools_path"]),
    ]
    run(cmd, log_path=p["logs"] / "schema_filter.log")
