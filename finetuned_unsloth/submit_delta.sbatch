#!/bin/bash
#SBATCH --job-name=jarvis-ft
#SBATCH --account=REPLACE_ME_ACCOUNT       # e.g. bbka-delta-gpu
#SBATCH --partition=gpuA100x4              # or ghx4 on Delta-AI — edit per cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ── Usage ─────────────────────────────────────────────────────────────
#   1. Edit SBATCH --account above with your allocation name.
#   2. Pre-build the venv on a login node (one-time, see README_delta.md).
#   3. sbatch submit_delta.sbatch
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # finetuned_unsloth/
PROJECT_DIR="$(cd "$REPO_DIR/.." && pwd)"                      # Generator/
VENV="${VENV:-$PROJECT_DIR/.venv-delta}"
SCRATCH_OUT="${JARVIS_FT_OUT:-$SCRATCH/jarvis_v7_lora}"
DATASET="$REPO_DIR/v7_2k/jarvis_v7_functiongemma.jsonl"

mkdir -p "$REPO_DIR/logs" "$SCRATCH_OUT"

# ── Modules ───────────────────────────────────────────────────────────
# Adjust module names to whatever Delta currently ships. Check: `module avail cuda`.
module purge
module load cuda/12.4.1 || module load cuda
module load python/3.11 || module load python

# ── Environment ───────────────────────────────────────────────────────
source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# no internet on some compute nodes — pre-cache model+tokenizer on the login node
# then force offline mode here to fail fast instead of hanging:
# export HF_HUB_OFFLINE=1

nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ── Train ─────────────────────────────────────────────────────────────
# Conservative A100-40GB defaults: batch 16, grad_accum 1, bf16.
# Scale batch up/down based on `nvidia-smi` memory footprint in the first 10 steps.
cd "$REPO_DIR"
srun python train.py \
    --dataset "$DATASET" \
    --output  "$SCRATCH_OUT" \
    --max-seq-length 4096 \
    --batch-size 16 \
    --grad-accum 1 \
    --max-steps 500 \
    --lr 2e-4 \
    --lora-r 128 --lora-alpha 256 \
    --bf16 \
    --save-merged \
    --export-gguf

echo "[done] artefacts under $SCRATCH_OUT"
