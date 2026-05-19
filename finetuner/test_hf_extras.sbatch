#!/bin/bash
#SBATCH --job-name=ft-hf-extras
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/hf-extras-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/hf-extras-%j.err

# Extra hf-backend cases: confirm it works on a Qwen family (not just Llama).

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
DATA=$REPO/runs/finetuner-smoke/smoke.jsonl
OUT=$REPO/runs/ft-hf-extras
mkdir -p "$OUT" "$REPO/finetuner/logs"

module load python/miniforge3_pytorch/2.7.0 || true
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME"

# Rebuild smoke jsonl if missing (was deleted in cleanup)
[ -f "$DATA" ] || python -c "
import json
qa = [('What is NDP?','NDP is the National Data Platform.'),
      ('What is LoRA?','A parameter-efficient fine-tuning method.'),
      ('What is a tool call?','A structured request for a function.'),
      ('What is GGUF?','A quantized model format for llama.cpp.'),
      ('What is bf16?','Brain-float 16-bit precision.'),
      ('What does finetuner do?','Dispatches a LoRA fine-tune to a backend.')]
import os; os.makedirs(os.path.dirname('$DATA'), exist_ok=True)
with open('$DATA','w') as f:
    for q,a in qa:
        f.write(json.dumps({'conversations':[{'role':'user','content':q},{'role':'assistant','content':a}]})+chr(10))
"

source $VENV/bin/activate
cd $REPO
echo ""; echo "### hf backend on Qwen2.5-0.5B (small, different family) ###"
time python -m finetuner.cli run \
    --backend hf --base-model unsloth/Qwen2.5-0.5B-Instruct \
    --dataset "$DATA" --output "$OUT/qwen" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024 \
    && echo "QWEN-HF: PASS" || echo "QWEN-HF: FAIL"

echo ""; echo "### hf backend on Llama-3.2-1B with --save-merged ###"
time python -m finetuner.cli run \
    --backend hf --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$DATA" --output "$OUT/llama-merged" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024 \
    --save-merged \
    && echo "LLAMA-HF-MERGED: PASS" || echo "LLAMA-HF-MERGED: FAIL"

echo "=== DONE  $(date)"
