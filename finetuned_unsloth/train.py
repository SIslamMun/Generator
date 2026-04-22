"""Delta-AI / headless training script for FunctionGemma 270M on Jarvis-CD.

Same recipe as `train_jarvis_functiongemma.ipynb` — stripped of Colab
assumptions and driven by CLI flags so a SLURM job can fire it off with
one `sbatch submit_delta.sbatch`.

Key differences vs. the notebook:
  - No `!pip install` — assume the venv is activated by the batch script.
  - Prefer bf16 on A100s (drop `load_in_16bit`); still supports T4 via `--fp16`.
  - Dataset/output paths are CLI flags; defaults resolve relative to this file.
  - Optional GGUF export at the end via `--export-gguf`.

Usage:
  python train.py \
      --dataset v7_2k/jarvis_v7_functiongemma.jsonl \
      --output  $SCRATCH/jarvis_v7_lora \
      --max-steps 500 \
      --batch-size 16 \
      --grad-accum 1 \
      --bf16 \
      --export-gguf
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"
HERE = Path(__file__).resolve().parent


def build_parser():
    p = argparse.ArgumentParser(description="FunctionGemma 270M LoRA fine-tune on Jarvis-CD v7 corpus")
    p.add_argument("--model-name", default="unsloth/functiongemma-270m-it")
    p.add_argument("--dataset", default=str(HERE / "v7_2k" / "jarvis_v7_functiongemma.jsonl"),
                   help="Path to the JSONL produced by convert_to_functiongemma.py")
    p.add_argument("--output", default=str(HERE / "outputs" / "jarvis_v7_lora"),
                   help="Run artefacts (checkpoints, LoRA adapter)")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=None,
                   help="If set, overrides --max-steps with num_train_epochs.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=3407)
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on A100/H100)")
    precision.add_argument("--fp16", action="store_true", help="Use fp16 (T4 fallback)")
    p.add_argument("--export-gguf", action="store_true",
                   help="After training, merge LoRA and export Q8_0 GGUF next to --output")
    p.add_argument("--save-merged", action="store_true",
                   help="Also save a merged 16-bit HF checkpoint")
    return p


def prepare_messages_and_tools(example):
    raw = json.loads(example["messages"])
    msgs = [dict(m) for m in raw]

    tools_raw = []
    if msgs and isinstance(msgs[0], dict):
        tlist = msgs[0].get("tools")
        if isinstance(tlist, list) and tlist:
            tools_raw = tlist
            msgs[0].pop("tools", None)

    THINK_KEYS = ["think", "think_fast", "think_faster"]
    has_valid_thought = False
    for m in msgs:
        if m.get("role") == "assistant":
            found_key = next((k for k in THINK_KEYS if m.get(k)), None)
            if found_key:
                think_text = m[found_key]
                content = m.get("content")
                block = f"{THINK_TAG_OPEN}{think_text}{THINK_TAG_CLOSE}"
                m["content"] = (block + "\n" + content) if isinstance(content, str) and content else block
                has_valid_thought = True
                for k in THINK_KEYS:
                    m.pop(k, None)
            else:
                return None, None
    if not has_valid_thought:
        return None, None

    for m in msgs:
        if "tool_calls" not in m or not m["tool_calls"]:
            continue
        new_tool_calls = []
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            if "function" in tc and isinstance(tc["function"], dict):
                new_tool_calls.append(tc)
                continue
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            new_tool_calls.append({
                "id": tc.get("id") or tc.get("tool_call_id"),
                "type": tc.get("type", "function"),
                "function": {"name": tc.get("name", ""), "arguments": args},
            })
        m["tool_calls"] = new_tool_calls

    id_to_name = {}
    for m in msgs:
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name

    for m in msgs:
        if m.get("role") == "tool" and not m.get("name"):
            tc_id = m.get("tool_call_id")
            m["name"] = id_to_name.get(tc_id) or "unknown_tool"

    adapted_tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            adapted_tools.append(t)
            continue
        adapted_tools.append({
            "type": t.get("type", "function"),
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        })

    if msgs and msgs[0].get("role") == "system" and "content" not in msgs[0]:
        msgs.pop(0)
    return msgs, adapted_tools


def main():
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] model      = {args.model_name}")
    print(f"[config] dataset    = {dataset_path}")
    print(f"[config] output     = {output_dir}")
    print(f"[config] max_steps  = {args.max_steps} (epochs={args.epochs})")
    print(f"[config] batch      = {args.batch_size} (grad_accum={args.grad_accum})")
    print(f"[config] lr         = {args.lr}")
    print(f"[config] lora_r/a   = {args.lora_r}/{args.lora_alpha}")
    print(f"[config] precision  = {'bf16' if args.bf16 else ('fp16' if args.fp16 else 'auto')}")

    # — lazy imports so `--help` works without torch installed —
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    # On A100+ prefer bf16 weights; Unsloth's `load_in_16bit` already picks the right half
    # format at load time, but we make it explicit in SFTConfig too.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
        load_in_8bit = False,
        load_in_16bit = True,
        full_finetuning = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
        use_rslora = False,
        loftq_config = None,
    )

    # — load + render dataset —
    rows = [json.loads(line) for line in open(dataset_path)]
    dataset = Dataset.from_list(rows)
    print(f"[data] loaded {len(dataset)} raw examples")

    def format_example(example):
        messages, tools = prepare_messages_and_tools(example)
        if messages is None or not messages:
            return {"text": None}
        chat_str = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, tokenize=False,
        ).removeprefix("<bos>")
        return {"text": chat_str}

    train_dataset = dataset.map(format_example).filter(lambda x: x["text"] is not None)
    print(f"[data] after filtering: {len(train_dataset)} examples")

    # — SFT config —
    sft_kwargs = dict(
        dataset_text_field = "text",
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps = args.warmup_steps,
        learning_rate = args.lr,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = args.seed,
        output_dir = str(output_dir / "checkpoints"),
        report_to = "none",
        save_strategy = "no",
    )
    if args.epochs is not None:
        sft_kwargs["num_train_epochs"] = args.epochs
    else:
        sft_kwargs["max_steps"] = args.max_steps
    if args.bf16:
        sft_kwargs["bf16"] = True
    elif args.fp16:
        sft_kwargs["fp16"] = True

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = None,
        args = SFTConfig(**sft_kwargs),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part    = "<start_of_turn>model\n",
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"[gpu] {gpu.name} / {round(gpu.total_memory / 1024**3, 2)} GB")

    print("[train] starting")
    stats = trainer.train()
    print(f"[train] done in {stats.metrics['train_runtime']/60:.1f} min")

    adapter_dir = output_dir / "lora"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[save] LoRA adapter → {adapter_dir}")

    if args.save_merged:
        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"[save] merged 16bit → {merged_dir}")

    if args.export_gguf:
        gguf_dir = output_dir / "gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="Q8_0")
        print(f"[save] GGUF Q8_0 → {gguf_dir}")


if __name__ == "__main__":
    main()
