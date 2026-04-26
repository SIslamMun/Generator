"""Stage: gen-tool — generate balanced tool-use examples via tool-generate-full."""
from __future__ import annotations

import sys
from pathlib import Path

from ._subprocess import run


def tool_data_paths(cfg: dict):
    out = Path(cfg["output_dir"])
    return {
        "tools_path": Path(cfg["tool"]["data"]["tools_path"]),
        "raw":        out / "data" / "tool_examples_raw.json",
        "clean":      out / "data" / "tool_examples_clean.json",
        "train":      out / "data" / "tool_train_functiongemma.jsonl",
        "logs":       out / "logs",
    }


def run_gen_tool(cfg: dict) -> None:
    p = tool_data_paths(cfg)
    p["raw"].parent.mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)
    d = cfg["tool"]["data"]
    r = d.get("ratios", {"single": 0.10, "multi": 0.15, "chain": 0.45, "error": 0.30})

    cmd = [
        sys.executable, "-m", "generator.cli", "tool-generate-full",
        str(p["tools_path"]),
        "-o", str(p["raw"]),
        "--target-pairs",    str(d["target_pairs"]),
        "--ratio-single",    str(r["single"]),
        "--ratio-multi",     str(r["multi"]),
        "--ratio-chain",     str(r["chain"]),
        "--ratio-error",     str(r["error"]),
        "--tools-per-example", str(d.get("tools_per_example", 10)),
        "--distractor-strategy", str(d.get("distractor_strategy", "mixed")),
        "--save-every", "50",
        "--provider",  cfg["llm"]["provider"],
        "--model",     cfg["llm"]["model"],
    ]
    run(cmd, log_path=p["logs"] / "gen_tool.log")
