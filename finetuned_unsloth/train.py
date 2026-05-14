"""Top-level dispatcher for model-agnostic fine-tuning.

Flow:
  1. User picks a model (must exist under finetuned_unsloth/models/<name>/)
  2. User picks data types (--types tool / qa / qa,cot / qa,cot,tool ...)
  3. We call <model>/prepare_data.py with the matching --in-* flags to
     produce <model>/data/train.jsonl
  4. We call <model>/validate_data.py to confirm the JSONL is sane
  5. We (optionally) submit <model>/submit.sbatch to actually train

Run without --submit to just do the prep+validate (handy for iterating
on the dataset). Use --submit to actually queue the training job.

Example (NDP tool-use, Nemotron Nano 4B):

  python finetuned_unsloth/train.py \
      --model nemotron_nano_4b \
      --types tool \
      --in-tool runs/ndp/data/ndp_tool_examples_curated.json \
      --tool-catalog configs/tools/ndp_tools.json \
      --submit

To just list available models:
  python finetuned_unsloth/train.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_common"))
import registry  # type: ignore[import-not-found]


def cmd_list():
    registry.print_table()


def cmd_run(args):
    entry = registry.get(args.model)
    if entry is None:
        print(f"ERROR: model '{args.model}' not found.", file=sys.stderr)
        print("\nAvailable:", file=sys.stderr)
        registry.print_table()
        sys.exit(2)

    print(f"=== model: {entry.display_name} ({entry.name})")
    print(f"=== path : {entry.path}")
    print(f"=== hf   : {entry.hf_model_id}")

    if not entry.has_prepare:
        print(f"ERROR: {entry.path}/prepare_data.py missing", file=sys.stderr)
        sys.exit(2)

    # ── --submit: hand the WHOLE pipeline (prepare+validate+train) to SLURM ──
    # We do NOT run prepare/validate locally when submitting, because the
    # login node can stall on Lustre while transformers imports, and the
    # compute node has the tokenizer-equipped venv anyway.
    if args.submit:
        if not entry.has_sbatch:
            print(f"ERROR: no submit.sbatch in {entry.path} — can't --submit", file=sys.stderr)
            sys.exit(2)
        env = os.environ.copy()
        env["TYPES"] = args.types
        if args.in_qa:        env["IN_QA"]        = str(args.in_qa)
        if args.in_cot:       env["IN_COT"]       = str(args.in_cot)
        if args.in_tool:      env["IN_TOOL"]      = str(args.in_tool)
        if args.tool_catalog: env["TOOL_CATALOG"] = str(args.tool_catalog)
        if args.shuffle:      env["SHUFFLE"]      = "1"
        if args.max_rows:     env["MAX_ROWS"]     = str(args.max_rows)
        if args.max_steps:    env["MAX_STEPS"]    = str(args.max_steps)
        if args.output:       env["OUTPUT"]       = str(args.output)

        submit_cmd = ["sbatch", str(entry.path / "submit.sbatch")]
        print(f"\n=== submit (everything runs on compute node) ===")
        for k in ("TYPES","IN_QA","IN_COT","IN_TOOL","TOOL_CATALOG","SHUFFLE","MAX_ROWS","MAX_STEPS","OUTPUT"):
            if k in env: print(f"  {k}={env[k]}")
        print("  $", " ".join(submit_cmd))
        rc = subprocess.call(submit_cmd, env=env)
        sys.exit(rc)

    # ── Local dry-run mode (no --submit): runs prepare locally for inspection ──
    out_jsonl = entry.path / "data" / "train.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prep_cmd = [
        sys.executable, str(entry.path / "prepare_data.py"),
        "--types", args.types,
        "--out", str(out_jsonl),
    ]
    for src_type, flag, val in [("qa", "--in-qa", args.in_qa),
                                 ("cot", "--in-cot", args.in_cot),
                                 ("tool", "--in-tool", args.in_tool)]:
        if val:
            prep_cmd += [flag, str(val)]
    if args.tool_catalog: prep_cmd += ["--tool-catalog", str(args.tool_catalog)]
    if args.shuffle:      prep_cmd.append("--shuffle")
    if args.max_rows:     prep_cmd += ["--max-rows", str(args.max_rows)]
    if args.dry_run:      prep_cmd.append("--no-tokenizer")

    print(f"\n=== STEP 1: prepare → {out_jsonl}")
    print("  $", " ".join(prep_cmd))
    rc = subprocess.call(prep_cmd)
    if rc != 0:
        print(f"\nERROR: prepare_data.py exited {rc}", file=sys.stderr)
        sys.exit(rc)

    # 2. VALIDATE — skipped in dry-run (needs tokenizer for length checks)
    if args.dry_run:
        print(f"\n=== STEP 2: validate — skipped (--dry-run)")
    elif entry.has_validate:
        val_cmd = [sys.executable, str(entry.path / "validate_data.py"), "--jsonl", str(out_jsonl)]
        print(f"\n=== STEP 2: validate")
        print("  $", " ".join(val_cmd))
        rc = subprocess.call(val_cmd)
        if rc != 0:
            print(f"\nERROR: validate_data.py exited {rc}", file=sys.stderr)
            sys.exit(rc)
    else:
        print(f"\n=== STEP 2: validate — skipped (no validate_data.py in {entry.path})")

    print(f"\n=== STEP 3: skipped (no --submit).")
    print(f"To actually train, re-run with --submit (everything will run on compute node).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="List available models and exit")
    ap.add_argument("--model", help="Model folder name under finetuned_unsloth/models/")
    ap.add_argument("--types", default="",
                    help="Comma-list of {qa, cot, tool}. Examples: qa | qa,cot | tool | qa,cot,tool")
    ap.add_argument("--in-qa",   type=Path, help="QA JSON file")
    ap.add_argument("--in-cot",  type=Path, help="CoT JSON file")
    ap.add_argument("--in-tool", type=Path, help="Tool examples JSON file")
    ap.add_argument("--tool-catalog", type=Path,
                    help="Tools catalog JSON (needed when --types includes tool)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the prepared rows")
    ap.add_argument("--max-rows", type=int, default=0, help="Cap total rows (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip tokenizer load + validation. Just verify the data-conversion logic. "
                         "Incompatible with --submit (training needs the rendered text).")
    ap.add_argument("--submit", action="store_true",
                    help="After prepare+validate, sbatch the model's submit.sbatch")
    ap.add_argument("--output", type=Path,
                    help="Override OUTPUT env var passed to submit.sbatch")
    ap.add_argument("--max-steps", type=int,
                    help="Pass MAX_STEPS env var to submit.sbatch (smoke testing)")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return
    if not args.model:
        ap.error("--model is required (or use --list)")
    if not args.types:
        ap.error("--types is required (e.g. 'tool' or 'qa,cot,tool')")
    if args.dry_run and args.submit:
        ap.error("--dry-run and --submit are mutually exclusive (training needs rendered text)")
    cmd_run(args)


if __name__ == "__main__":
    main()
