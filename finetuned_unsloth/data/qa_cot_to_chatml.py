"""Convert jarvis_qa_cot_curated.json → JSONL with {"conversations": [...]}.

Input row shape (from jarvis_qa_cot_24w/jarvis_qa_cot_curated.json):
    {
      "question":  str,
      "reasoning": str,
      "answer":    str,
      "type":      "code" | "text"
    }

Output row shape (ChatML conversations, what Unsloth's chat_template wants):
    {
      "conversations": [
        {"role": "system",    "content": "<Jarvis QA assistant persona>"},
        {"role": "user",      "content": "<question>"},
        {"role": "assistant", "content": "**Reasoning:**\n<reasoning>\n\n**Answer:**\n<answer>"}
      ]
    }

Usage:
    python qa_cot_to_chatml.py \
        -i /u/sislam3/Generator/jarvis_qa_cot_24w/jarvis_qa_cot_curated.json \
        -o /u/sislam3/Generator/finetuned_unsloth/data/qa_v1/jarvis_qa_v1_cot.jsonl \
        [--val-split 0.05] [--seed 3407]
"""

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Jarvis-CD documentation assistant. For every question, first explain "
    "your reasoning step by step, then provide a concise, accurate final answer. "
    "Use the exact format:\n**Reasoning:**\n<your step-by-step analysis>\n\n"
    "**Answer:**\n<concise final answer>"
)


def row_to_conversation(row: dict) -> dict:
    q = (row.get("question") or "").strip()
    reasoning = (row.get("reasoning") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not (q and reasoning and answer):
        return None
    assistant = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{answer}"
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant},
        ],
        "type": row.get("type", "unknown"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Path to jarvis_qa_cot_curated.json")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL (train split)")
    ap.add_argument("--val-output", default=None,
                    help="If --val-split > 0, write validation JSONL here (default: <output>.val.jsonl)")
    ap.add_argument("--val-split", type=float, default=0.05,
                    help="Fraction of the data held out for validation (default 0.05)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    val_dst = Path(args.val_output).resolve() if args.val_output else dst.with_suffix(".val.jsonl")

    raw = json.loads(src.read_text())
    print(f"loaded {len(raw)} raw rows from {src}")

    converted, dropped = [], 0
    for row in raw:
        c = row_to_conversation(row)
        if c is None:
            dropped += 1
            continue
        converted.append(c)

    random.Random(args.seed).shuffle(converted)
    n_val = int(len(converted) * args.val_split) if args.val_split > 0 else 0
    val_rows = converted[:n_val]
    train_rows = converted[n_val:]

    with dst.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train → {dst}  ({len(train_rows)} rows)")

    if n_val > 0:
        with val_dst.open("w") as f:
            for r in val_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"val   → {val_dst}  ({len(val_rows)} rows)")

    print(f"dropped empty rows: {dropped}")

    # quick sanity-check: print one example
    if train_rows:
        print("\n--- sample train row ---")
        print(json.dumps(train_rows[0], indent=2, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
