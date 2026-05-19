#!/bin/bash
#SBATCH --job-name=ft-finetome-test
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/finetome-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/finetome-%j.err

# Run the SAME dataset (mlabonne/FineTome-100k train[:3000]) the official
# Unsloth Gemma-4 notebook used, but through OUR finetuner CLI. Then load
# the resulting LoRA and probe it. Lets us compare to the reference run
# (job 2301142: train_loss=1.02, clean inference).

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
OUT=$REPO/runs/ft-finetome-test
DATA=$REPO/runs/finetome-3k.jsonl
mkdir -p "$REPO/runs" "$REPO/finetuner/logs"

echo "=== finetuner FineTome end-to-end  host=$(hostname) job=$SLURM_JOB_ID  $(date)"
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME"
source $VENV/bin/activate
cd $REPO

# ── STEP 1: build the FineTome JSONL the finetuner can consume ──────
echo ""; echo "=== STEP 1: build FineTome JSONL"
python - <<EOF
import json
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
ds = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
ds = standardize_data_formats(ds)
with open("$DATA", "w") as f:
    for ex in ds:
        f.write(json.dumps({"conversations": ex["conversations"]}, ensure_ascii=False) + "\n")
print("wrote", len(ds), "rows ->", "$DATA")
EOF

# ── STEP 2: train via OUR finetuner — notebook-matching hyperparams ─
echo ""; echo "=== STEP 2: finetuner.cli run (Gemma 4 E4B, 60 steps)"
time python -m finetuner.cli run \
    --backend unsloth \
    --base-model unsloth/gemma-4-E4B-it \
    --dataset "$DATA" \
    --output  "$OUT" \
    --lora-rank 8 --lora-alpha 8 --lora-dropout 0 \
    --epochs 1 --max-steps 60 --lr 2e-4 \
    --batch-size 1 --grad-accum 4 --max-seq-length 1024

# ── STEP 3: load the LoRA and probe — same prompts the reference used ──
echo ""; echo "=== STEP 3: inference probes on the finetuner output"
python - <<EOF
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

LORA = "$OUT/lora"
print("loading", LORA)
model, tok = FastModel.from_pretrained(model_name=LORA, max_seq_length=1024,
                                       load_in_4bit=False, full_finetuning=False)
tok = get_chat_template(tok, chat_template="gemma-4")
FastModel.for_inference(model)

PROBES = [
    "Why is the sky blue?",
    "What is 2 + 2? Explain briefly.",
    "Write a 3-line haiku about the ocean.",
    "Continue the sequence: 1, 1, 2, 3, 5, 8,",
]
for i, p in enumerate(PROBES, 1):
    msgs = [{"role":"user","content":[{"type":"text","text":p}]}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
        return_tensors="pt", tokenize=True, return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160,
            temperature=1.0, top_p=0.95, top_k=64, use_cache=True)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{i}/4] {p}\n{'-'*60}\n{gen.strip()}")
EOF
echo ""; echo "=== DONE  $(date)"
