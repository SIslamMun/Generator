"""Stage: prep-tool — clean tool-examples JSON → FunctionGemma JSONL training format.

Uses the `convert_tool_to_functiongemma` function from scripts/convert_to_functiongemma.py
directly so this works for tool-only inputs (no QA / CoT merge required).

Then post-processes assistant messages: extracts <think>…</think> from the
`content` field into a separate `think` field, which is what
finetuned_unsloth/train/train.py expects.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from rich.console import Console

from .gen_tool import tool_data_paths

console = Console()

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think(content: str) -> tuple[str, str]:
    """Pull <think>…</think> out of content. Returns (think_text, content_remainder)."""
    if not isinstance(content, str):
        return "", content or ""
    m = _THINK_RE.search(content)
    if not m:
        return "", content.strip()
    think = m.group(1).strip()
    rest = (content[:m.start()] + content[m.end():]).strip()
    return think, rest


DEFAULT_FINAL_THINK = "The requested operations are complete; I'll summarize the outcome for the user."


def _normalize_assistant_messages(messages: list[dict]) -> list[dict]:
    """Convert <think>…</think> in content → separate `think` key. Idempotent.

    train.py requires *every* assistant message to carry a `think` field. For
    the final wrap-up assistant (which the converter emits without thinking),
    we inject a generic placeholder so the row isn't dropped.
    """
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m); continue
        # If `think` already present and non-empty, leave alone.
        if (m.get("think") or m.get("think_fast") or m.get("think_faster")):
            out.append(m); continue
        think, rest = _split_think(m.get("content") or "")
        nm = dict(m)
        nm["content"] = rest
        nm["think"]   = think if think else DEFAULT_FINAL_THINK
        out.append(nm)
    return out


def _load_converter():
    """Import scripts/convert_to_functiongemma.py without making it a real package."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    script = repo_root / "scripts" / "convert_to_functiongemma.py"
    if not script.exists():
        raise FileNotFoundError(f"convert_to_functiongemma.py missing: {script}")
    spec = importlib.util.spec_from_file_location("_convert_fg", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_prep_tool(cfg: dict) -> None:
    p = tool_data_paths(cfg)
    if not p["clean"].exists():
        raise FileNotFoundError(f"clean tool examples not found: {p['clean']}")

    raw = json.loads(p["clean"].read_text())
    if not isinstance(raw, list):
        raise RuntimeError(f"expected JSON list at {p['clean']}, got {type(raw).__name__}")
    console.print(f"[prep_tool] read {len(raw)} examples from {p['clean']}")

    mod = _load_converter()
    converted = mod.convert_tool_to_functiongemma(raw)
    console.print(f"[prep_tool] converted to FunctionGemma format: {len(converted)} examples")

    p["train"].parent.mkdir(parents=True, exist_ok=True)
    n_with_think = 0
    with p["train"].open("w") as f:
        for r in converted:
            msgs = list(r["messages"])
            # Embed tool catalog in the system message (train.py reads tools from msgs[0]["tools"]).
            if "tools" in r and msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = r["tools"]
            # Pull <think>…</think> out of content into a top-level `think` field.
            msgs = _normalize_assistant_messages(msgs)
            if any(m.get("role") == "assistant" and m.get("think") for m in msgs):
                n_with_think += 1
            row = {"messages": json.dumps(msgs, ensure_ascii=False)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"[prep_tool] wrote → {p['train']}  ({n_with_think}/{len(converted)} have think field)")
