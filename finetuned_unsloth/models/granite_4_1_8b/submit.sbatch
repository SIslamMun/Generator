#!/bin/bash
#SBATCH --job-name=granite-4-1-8b-ft
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/models/granite_4_1_8b/logs/ft-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/models/granite_4_1_8b/logs/ft-%j.err

# Gemma 3 4B fine-tune — prepare → validate → train on the compute node.
# Standard transformer: reuses the Nemotron venv (Unsloth/TRL/transformers
# already there), so there is no install step.
#
# ENV: TYPES, IN_TOOL, TOOL_CATALOG (required); MAX_STEPS, SHUFFLE (optional)

set -eu
export PYTHONUNBUFFERED=1

MODEL_DIR=/u/sislam3/Generator/finetuned_unsloth/models/granite_4_1_8b
VENV=/u/sislam3/Generator/finetuned_unsloth/.venv-bakeoff
TYPES="${TYPES:?TYPES=qa|cot|tool[,...] is required}"
OUTPUT="${OUTPUT:-$MODEL_DIR/artifacts}"

mkdir -p $MODEL_DIR/logs $MODEL_DIR/data

echo "=== ENV ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID  date=$(date)"
echo "types=$TYPES  output=$OUTPUT"
nvidia-smi | head -12

cd $MODEL_DIR
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true

# HF cache → NVMe scratch (module default HF_HOME is a full shared disk).
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=0
export TMPDIR=/work/nvme/bekn/sislam3/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
echo "=== HF_HOME = $HF_HOME"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: shared venv $VENV missing — run the Nemotron install first." >&2
    exit 1
fi
source $VENV/bin/activate
echo "=== venv: $(which python)"

DATASET=$MODEL_DIR/data/train.jsonl

# ── STEP 1: prepare ────────────────────────────────────────────────
PREP_ARGS=(--types "$TYPES" --out "$DATASET" --hf-model-id unsloth/granite-4.1-8b)
[ -n "${IN_TOOL:-}" ]      && PREP_ARGS+=(--in-tool "$IN_TOOL")
[ -n "${IN_QA:-}" ]        && PREP_ARGS+=(--in-qa "$IN_QA")
[ -n "${IN_COT:-}" ]       && PREP_ARGS+=(--in-cot "$IN_COT")
[ -n "${TOOL_CATALOG:-}" ] && PREP_ARGS+=(--tool-catalog "$TOOL_CATALOG")
[ -n "${SHUFFLE:-}" ]      && PREP_ARGS+=(--shuffle)
[ -n "${MAX_ROWS:-}" ]     && PREP_ARGS+=(--max-rows "$MAX_ROWS")

echo ""; echo "=== STEP 1: prepare_data.py ==="
time python prepare_data.py "${PREP_ARGS[@]}"

# ── STEP 2: validate ───────────────────────────────────────────────
echo ""; echo "=== STEP 2: validate_data.py ==="
time python validate_data.py --jsonl "$DATASET" --max-seq-length 4096 \
    --hf-model-id unsloth/granite-4.1-8b \
    --instruction-part "<|start_of_role|>user<|end_of_role|>
" --response-part "<|start_of_role|>assistant<|end_of_role|>
" || true

# ── STEP 3: train ──────────────────────────────────────────────────
TRAIN_ARGS=(--dataset "$DATASET" --output "$OUTPUT")
[ -n "${MAX_STEPS:-}" ] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")

echo ""; echo "=== STEP 3: train.py ==="
time python train.py "${TRAIN_ARGS[@]}"

echo ""; echo "=== DONE  $(date) ==="
ls -la $OUTPUT
