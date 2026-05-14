"""Validate the JSONL produced by prepare_data.py before training fires.

Checks every row has:
  - non-empty `text`
  - `text` parses through the tokenizer without OOV-token errors
  - presence of the model's instruction/response markers (so train_on_responses_only can mask)
  - reasonable length (not absurdly short / over max_seq_length)

Prints a summary table; exits non-zero if any HARD failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--hf-model-id", default="unsloth/NVIDIA-Nemotron-3-Nano-4B")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--instruction-part", default="<|im_start|>user\n")
    ap.add_argument("--response-part",    default="<|im_start|>assistant\n")
    ap.add_argument("--min-tokens", type=int, default=16,
                    help="rows shorter than this in token count are flagged")
    args = ap.parse_args()

    print(f"[validate] loading tokenizer: {args.hf_model_id}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        print("ERROR: empty jsonl", file=sys.stderr)
        sys.exit(1)
    print(f"[validate] {len(rows)} rows")

    sources = Counter(r.get("_source", "?") for r in rows)
    errors = 0
    warnings = 0
    too_short = 0
    too_long = 0
    missing_instr = 0
    missing_resp = 0
    token_lens = []

    for i, r in enumerate(rows):
        text = r.get("text") or ""
        if not text.strip():
            print(f"  row {i}: ERROR empty text")
            errors += 1
            continue
        # Marker presence (required for train_on_responses_only)
        if args.instruction_part not in text:
            missing_instr += 1
            if missing_instr <= 3:
                print(f"  row {i}: WARN missing instruction marker {args.instruction_part!r}")
            warnings += 1
        if args.response_part not in text:
            missing_resp += 1
            if missing_resp <= 3:
                print(f"  row {i}: WARN missing response marker {args.response_part!r}")
            warnings += 1
        # Tokenize length
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_lens.append(len(ids))
        if len(ids) < args.min_tokens:
            too_short += 1
        if len(ids) > args.max_seq_length:
            too_long += 1

    print()
    print(f"[validate] sources : {dict(sources)}")
    if token_lens:
        token_lens.sort()
        n = len(token_lens)
        print(f"[validate] token len: min={token_lens[0]} p50={token_lens[n//2]} p95={token_lens[int(n*0.95)]} max={token_lens[-1]}")
    print(f"[validate] too short (<{args.min_tokens}): {too_short}")
    print(f"[validate] too long  (>{args.max_seq_length}): {too_long}    ← will be silently truncated by SFTTrainer")
    print(f"[validate] missing instruction marker: {missing_instr}")
    print(f"[validate] missing response marker:    {missing_resp}")
    print(f"[validate] errors={errors}  warnings={warnings}")

    if errors:
        sys.exit(1)
    # If too many rows lack the response marker, training-on-responses-only will mask everything → useless.
    if rows and (missing_resp / len(rows)) > 0.10:
        print(f"\nERROR: >10% of rows missing the response marker — train_on_responses_only would mask too much.",
              file=sys.stderr)
        sys.exit(1)
    print("\n[validate] OK — dataset is training-ready.")


if __name__ == "__main__":
    main()
