"""Stage: prep-chat — curated CoT JSON → ChatML JSONL split into train/val.

Mirrors finetuned_unsloth/data/qa_cot_to_chatml.py but kept in-package so the
training pipeline is self-contained.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .gen_cot import chat_data_paths

SYSTEM_PROMPT = (
    "You are a domain documentation assistant. For every question, first explain "
    "your reasoning step by step, then provide a concise, accurate final answer. "
    "Use the exact format:\n**Reasoning:**\n<your step-by-step analysis>\n\n"
    "**Answer:**\n<concise final answer>"
)


def _row_to_conversation(row: dict) -> dict | None:
    q = (row.get("question") or "").strip()
    reasoning = (row.get("reasoning") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not (q and reasoning and answer):
        return None
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant",
             "content": f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{answer}"},
        ],
        "type": row.get("type", "unknown"),
    }


def run_prep_chat(cfg: dict) -> None:
    p = chat_data_paths(cfg)
    raw = json.loads(p["curated"].read_text())
    converted = [c for c in (_row_to_conversation(r) for r in raw) if c is not None]
    if not converted:
        raise RuntimeError(f"no valid rows in {p['curated']}")

    rng = random.Random(3407)
    rng.shuffle(converted)

    val_split = float(cfg["chat"]["eval"].get("val_split", 0.05))
    n_val = max(1, int(len(converted) * val_split)) if val_split > 0 else 0
    val, train = converted[:n_val], converted[n_val:]

    p["train"].parent.mkdir(parents=True, exist_ok=True)
    p["train"].write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
    if n_val:
        p["val"].write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n")
    print(f"[prep_chat] train={len(train)} val={n_val}  →  {p['train']}")
