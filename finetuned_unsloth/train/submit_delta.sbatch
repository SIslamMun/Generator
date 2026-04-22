#!/bin/bash
#SBATCH --job-name=jarvis-ft
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# ── Usage ─────────────────────────────────────────────────────────────
#   1. Pre-build the venv on a login node (see README_delta.md).
#   2. sbatch submit_delta.sbatch
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
REPO_DIR="/u/sislam3/Generator/finetuned_unsloth"
PROJECT_DIR="/u/sislam3/Generator"
VENV="${VENV:-$PROJECT_DIR/.venv-delta}"
SCRATCH_OUT="${JARVIS_FT_OUT:-/work/hdd/bekn/$USER/jarvis_v7_lora}"
DATASET="$REPO_DIR/data/v7_10k_clean/jarvis_v7_functiongemma.jsonl"

mkdir -p "$REPO_DIR/logs" "$SCRATCH_OUT"

# ── Environment ───────────────────────────────────────────────────────
# Do NOT `module purge` — removes default compilers that python/anaconda3 depends on
module load python/anaconda3/2.10.0

source "$VENV/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "═══ Jarvis FunctionGemma Fine-tune ═══"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "Dataset: $DATASET"
echo "Output:  $SCRATCH_OUT"
echo ""

nvidia-smi | head -20
echo ""
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
echo ""

# ── Train ─────────────────────────────────────────────────────────────
# GH200 has 120GB VRAM — batch 16, grad_accum 1, bf16 is safe.
# v2: 3000 steps (~7 epochs over 6977 examples) for robust tool-semantic mapping
cd "$REPO_DIR"
srun python train/train.py \
    --dataset "$DATASET" \
    --output  "$SCRATCH_OUT" \
    --max-seq-length 4096 \
    --batch-size 16 \
    --grad-accum 1 \
    --max-steps 3000 \
    --warmup-steps 30 \
    --lr 2e-4 \
    --lora-r 128 --lora-alpha 256 \
    --bf16 \
    --save-merged

echo ""
echo "═══ DONE — $(date) ═══"
echo "Artifacts: $SCRATCH_OUT"
ls -la "$SCRATCH_OUT"
