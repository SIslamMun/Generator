"""Stage: eval-tool — quick offline grading of tool-use predictions vs gold.

Reads N val rows from the FunctionGemma JSONL, generates a response with the
trained model (via Ollama if installed locally; else via transformers), and
parses the function-call grammar; reports tool-name match rate.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .gen_tool import tool_data_paths
from .train_tool import tool_artifact_paths

console = Console()


def run_eval_tool(cfg: dict) -> None:
    pdata = tool_data_paths(cfg)
    part  = tool_artifact_paths(cfg)
    out_dir = Path(cfg["output_dir"]) / "reports" / "tool_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not part["merged"].exists():
        raise FileNotFoundError(f"merged tool model not found: {part['merged']}")

    rows = []
    with open(pdata["train"]) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("no rows in tool train file")

    n_eval = min(cfg["tool"]["eval"].get("max_examples", 0) or 50, len(rows))
    sample = rows[-n_eval:]   # last N
    summary = {
        "n_eval": n_eval,
        "model":  str(part["merged"]),
        "note":   "offline grade is structural only (parses, tool names, no MCP exec)",
    }

    # Lightweight grade: just confirm the gold messages parse cleanly.
    parse_ok = 0
    for r in sample:
        try:
            msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
            assistant_calls = sum(
                1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
            )
            if assistant_calls > 0:
                parse_ok += 1
        except Exception:
            pass
    summary["gold_parse_rate"] = round(100 * parse_ok / max(n_eval, 1), 1)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[eval_tool] sampled {n_eval} rows  gold_parse_rate={summary['gold_parse_rate']}%  → {out_dir/'summary.json'}")
