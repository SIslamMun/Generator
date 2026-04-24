#!/bin/bash
#SBATCH --job-name=jarvis-qa-v1
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
SCRATCH_OUT="${JARVIS_QA_OUT:-/work/hdd/bekn/$USER/jarvis_qa_v1_lora}"
DATASET="$REPO_DIR/data/qa_v1/jarvis_qa_v1_cot.jsonl"

mkdir -p "$REPO_DIR/logs" "$SCRATCH_OUT"

module load python/anaconda3/2.10.0
source "$VENV/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "═══ Jarvis QA v1 Fine-tune (gemma-3-270m-it on 7315 CoT rows) ═══"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "Dataset: $DATASET  ($(wc -l < $DATASET) rows)"
echo "Output:  $SCRATCH_OUT"

nvidia-smi | head -15
echo ""

cd "$REPO_DIR"
# 3 epochs over ~7000 rows with batch 16 ≈ 1300 steps.
srun python train/train_qa.py \
    --model-name unsloth/gemma-3-270m-it \
    --dataset  "$DATASET" \
    --output   "$SCRATCH_OUT" \
    --max-seq-length 2048 \
    --epochs 3 \
    --batch-size 16 \
    --grad-accum 1 \
    --warmup-steps 30 \
    --lr 2e-4 \
    --lora-r 128 --lora-alpha 256 \
    --bf16 \
    --save-merged

echo ""
echo "═══ DONE — $(date) ═══"
ls -la "$SCRATCH_OUT"
