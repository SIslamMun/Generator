"""Build DPO preference pairs that fix Nemotron-3 Nano 4B's param-flooding tic.

SFT on a 100%-clean dataset still leaves the model flooding phantom `None`
parameters at inference (exposure bias — the base model's prior wins in token
states the clean SFT targets never visit). DPO fixes exactly that: it shows the
model, side by side, the same call done two ways and trains it to PREFER one.

For every single-call tool example in the curated dataset we emit:
  prompt   — system + user turn, with the tool catalog rendered
  chosen   — the clean tool call (only the parameters actually used)
  rejected — the SAME call, byte-identical, plus the phantom-`None` flood the
             model currently emits (every other catalog parameter = None,
             including params from other tools — the real observed failure)

chosen and rejected differ ONLY by the flood, so the DPO gradient isolates
precisely the behaviour to kill.

Output: data/dpo.jsonl  with rows {prompt, chosen, rejected} (plain text).
Run on a compute node (needs the HF tokenizer for the chat template).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import load_tool_catalog, ANTI_HALLUCINATION_SYSTEM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-tool", type=Path, required=True,
                    help="Curated tool-examples JSON (same file SFT used)")
    ap.add_argument("--tool-catalog", type=Path,
                    default=Path("/u/sislam3/Generator/configs/tools/ndp_tools.json"))
    ap.add_argument("--hf-model-id", default="unsloth/NVIDIA-Nemotron-3-Nano-4B")
    ap.add_argument("--out", type=Path, required=True, help="Output dpo.jsonl")
    args = ap.parse_args()

    print(f"[dpo-prep] loading tokenizer: {args.hf_model_id}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)

    catalog = load_tool_catalog(args.tool_catalog)
    # union of every parameter name across all tools — the flood pulls in
    # params from other tools too, so the rejected sample must as well.
    all_params: list[str] = []
    for t in catalog:
        for p in (t["function"]["parameters"]["properties"]):
            if p not in all_params:
                all_params.append(p)
    print(f"[dpo-prep] catalog: {[t['function']['name'] for t in catalog]}")
    print(f"[dpo-prep] union of params ({len(all_params)}): {all_params}")

    data = json.loads(args.in_tool.read_text())
    rows: list[dict] = []
    skipped = 0

    for ex in data:
        instr = (ex.get("instruction") or "").strip()
        sol = ex.get("solution") or {}
        steps = sol.get("reasoning_path") or []
        if not instr or len(steps) != 1:        # single-call examples only
            continue
        step = steps[0]
        tool = step.get("tool")
        call_args = step.get("args") or {}
        thought = (step.get("thought") or "").strip()
        if not tool:
            continue

        sys_msg = {"role": "system", "content": ANTI_HALLUCINATION_SYSTEM}
        user_msg = {"role": "user", "content": instr}
        asst = {
            "role": "assistant",
            "content": f"<think>\n{thought}\n</think>" if thought else "",
            "tool_calls": [{"type": "function",
                            "function": {"name": tool, "arguments": call_args}}],
        }

        prompt = tok.apply_chat_template(
            [sys_msg, user_msg], tools=catalog,
            tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(
            [sys_msg, user_msg, asst], tools=catalog,
            tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt) or "</function>" not in full:
            skipped += 1
            continue
        chosen = full[len(prompt):]

        # rejected = chosen + the phantom-None flood, injected before </function>
        phantoms = [p for p in all_params if p not in call_args]
        flood = "".join(f"<parameter={p}>\nNone\n</parameter>\n" for p in phantoms)
        rejected = chosen.replace("</function>", flood + "</function>", 1)
        if rejected == chosen:                  # nothing injected → useless pair
            skipped += 1
            continue

        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not rows:
        print("ERROR: no DPO pairs produced", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[dpo-prep] wrote {len(rows)} preference pairs → {args.out}  "
          f"(skipped {skipped})")
    # show one pair so the flood is visible in the log
    ex = rows[0]
    print("\n--- sample chosen (tail) ---")
    print(ex["chosen"][-300:])
    print("--- sample rejected (tail) ---")
    print(ex["rejected"][-500:])


if __name__ == "__main__":
    main()
