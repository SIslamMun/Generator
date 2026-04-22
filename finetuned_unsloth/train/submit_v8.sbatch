#!/bin/bash
#SBATCH --job-name=jarvis-v8
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

set -eu

REPO_DIR="/u/sislam3/Generator/finetuned_unsloth"
PROJECT_DIR="/u/sislam3/Generator"
VENV="${VENV:-$PROJECT_DIR/.venv-delta}"
SCRATCH_OUT="${JARVIS_FT_OUT:-/work/hdd/bekn/$USER/jarvis_v8_lora}"
DATASET="$REPO_DIR/data/v7_merged/jarvis_v7_merged_functiongemma.jsonl"

mkdir -p "$REPO_DIR/logs" "$SCRATCH_OUT"

module load python/anaconda3/2.10.0
source "$VENV/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "═══ Jarvis v8 Fine-tune (merged 7637 examples) ═══"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "Dataset: $DATASET ($(wc -l < $DATASET) rows)"
echo "Output:  $SCRATCH_OUT"
echo ""

nvidia-smi | head -15
echo ""

cd "$REPO_DIR"
# Fresh train from base model (not LoRA continuation — cleaner results).
# 3500 steps (~7.3 epochs over 7637 examples with batch 16).
srun python train/train.py \
    --dataset "$DATASET" \
    --output  "$SCRATCH_OUT" \
    --max-seq-length 4096 \
    --batch-size 16 \
    --grad-accum 1 \
    --max-steps 3500 \
    --warmup-steps 30 \
    --lr 2e-4 \
    --lora-r 128 --lora-alpha 256 \
    --bf16 \
    --save-merged

echo ""
echo "═══ DONE — $(date) ═══"
ls -la "$SCRATCH_OUT"
