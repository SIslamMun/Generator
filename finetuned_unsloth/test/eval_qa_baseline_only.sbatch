#!/bin/bash
#SBATCH --job-name=jarvis-qa-v1-eval-base
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Baseline-only re-run: the original eval_qa job ran out of wall-clock
# at 320/365 baseline rows. This sweep uses the base model as the "model"
# arg so the script writes its predictions + summary, and we skip the FT
# (already saved from the first run).

set -eu
module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd /u/sislam3/Generator
nvidia-smi | head -15

echo "=== baseline-only eval: unsloth/gemma-3-270m-it on 365 val rows ==="
python finetuned_unsloth/test/eval_qa.py \
    --model unsloth/gemma-3-270m-it \
    --val   /u/sislam3/Generator/finetuned_unsloth/data/qa_v1/jarvis_qa_v1_cot.val.jsonl \
    --out   /u/sislam3/Generator/inference/results/qa_v1_eval_base \
    --max-new-tokens 700

echo ""
echo "=== DONE — $(date) ==="
ls -la /u/sislam3/Generator/inference/results/qa_v1_eval_base/
