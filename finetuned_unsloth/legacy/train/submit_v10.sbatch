#!/bin/bash
#SBATCH --job-name=jarvis-v10
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

set -eu
REPO_DIR="/u/sislam3/Generator/finetuned_unsloth"
PROJECT_DIR="/u/sislam3/Generator"
VENV="${VENV:-$PROJECT_DIR/.venv-delta}"
SCRATCH_OUT="${JARVIS_FT_OUT:-/work/hdd/bekn/$USER/jarvis_v10_lora}"
DATASET="$REPO_DIR/data/v10_merged/jarvis_v10_merged_functiongemma.jsonl"

mkdir -p "$REPO_DIR/logs" "$SCRATCH_OUT"

module load python/anaconda3/2.10.0
source "$VENV/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "═══ Jarvis v10 Fine-tune (merged 10616 = v7_merged 7637 + gpt-oss-120b-generated 2979) ═══"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "Dataset: $DATASET ($(wc -l < $DATASET) rows)"
echo "Output:  $SCRATCH_OUT"

nvidia-smi | head -15
echo ""

cd "$REPO_DIR"
# 4800 steps ≈ 7.2 epochs over 10616 rows with batch 16.
srun python train/train.py \
    --dataset "$DATASET" \
    --output  "$SCRATCH_OUT" \
    --max-seq-length 4096 \
    --batch-size 16 \
    --grad-accum 1 \
    --max-steps 4800 \
    --warmup-steps 30 \
    --lr 2e-4 \
    --lora-r 128 --lora-alpha 256 \
    --bf16 \
    --save-merged

echo ""
echo "═══ DONE — $(date) ═══"
ls -la "$SCRATCH_OUT"
