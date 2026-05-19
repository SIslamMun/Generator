#!/bin/bash
#SBATCH --job-name=nemotron-nano-4b-ft
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/ft-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/ft-%j.err

# Self-contained Nemotron-3 Nano 4B fine-tune pipeline.
# Runs prepare → validate → train on the compute node. The login-node Lustre
# can stall on transformers imports, so the dispatcher offloads everything
# here.
#
# ENV (set by `finetuned_unsloth/train.py --submit`):
#   TYPES        : comma-list of {qa,cot,tool}             (required)
#   IN_TOOL      : path to tool examples JSON              (if TYPES has tool)
#   IN_QA        : path to QA pairs JSON                   (if TYPES has qa)
#   IN_COT       : path to CoT JSON                        (if TYPES has cot)
#   TOOL_CATALOG : path to tools.json                      (if TYPES has tool)
#   MAX_STEPS    : (optional) cap training steps for smoke
#   MAX_ROWS     : (optional) cap dataset rows for smoke
#   SHUFFLE      : (optional) set to 1 to shuffle rows
#   OUTPUT       : (optional) override artifacts dir

set -eu

# Force line-buffered stdout/stderr in this script + every Python child.
export PYTHONUNBUFFERED=1

MODEL_DIR=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b
TYPES="${TYPES:?TYPES=qa|cot|tool[,...] is required}"
OUTPUT="${OUTPUT:-$MODEL_DIR/artifacts}"

mkdir -p $MODEL_DIR/logs $MODEL_DIR/data

echo "=== ENV ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID  date=$(date)"
echo "model_dir=$MODEL_DIR"
echo "types=$TYPES  output=$OUTPUT"
[ -n "${IN_TOOL:-}" ]      && echo "in_tool=$IN_TOOL"
[ -n "${IN_QA:-}" ]        && echo "in_qa=$IN_QA"
[ -n "${IN_COT:-}" ]       && echo "in_cot=$IN_COT"
[ -n "${TOOL_CATALOG:-}" ] && echo "tool_catalog=$TOOL_CATALOG"
[ -n "${MAX_STEPS:-}" ]    && echo "max_steps=$MAX_STEPS"
[ -n "${MAX_ROWS:-}" ]     && echo "max_rows=$MAX_ROWS"
nvidia-smi | head -15

cd $MODEL_DIR

# uv on PATH
export PATH="$HOME/.local/bin:$PATH"

# Load modules BEFORE the venv check.
#   - python/miniforge3_pytorch/2.7.0 → torch 2.7+cu126 prebuilt for aarch64
#   - cuda/12.9 → nvcc for mamba_ssm build
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
echo "=== python3 = $(which python3 || echo NOT_FOUND)"
echo "=== nvcc    = $(which nvcc    || echo NOT_FOUND)"

# CRITICAL: the miniforge3_pytorch module sets HF_HOME=/projects/bekn/hf
# (a shared 500G cache that is currently 100% FULL). Override AFTER the
# module load so our exports actually win. /work/nvme is the local NVMe
# scratch — fast and has space.
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
unset HUGGINGFACE_HUB_CACHE     # legacy, can override the above
unset TRANSFORMERS_CACHE        # legacy, can override the above
export HF_HUB_ENABLE_HF_TRANSFER=0
export TMPDIR=/work/nvme/bekn/sislam3/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR"
echo "=== HF_HOME      = $HF_HOME"
echo "=== HF_HUB_CACHE = $HF_HUB_CACHE"

# Build venv on first run; subsequent runs skip the install step.
if [ ! -x "$MODEL_DIR/.venv-nemotron/bin/python" ]; then
    echo ""; echo "=== first run: building venv with uv ==="
    bash install.sh
fi

source $MODEL_DIR/.venv-nemotron/bin/activate
echo "=== venv active: $(which python) ==="

DATASET=$MODEL_DIR/data/train.jsonl

# ── STEP 1: prepare ────────────────────────────────────────────
PREP_ARGS=(--types "$TYPES" --out "$DATASET")
[ -n "${IN_TOOL:-}" ]      && PREP_ARGS+=(--in-tool "$IN_TOOL")
[ -n "${IN_QA:-}" ]        && PREP_ARGS+=(--in-qa "$IN_QA")
[ -n "${IN_COT:-}" ]       && PREP_ARGS+=(--in-cot "$IN_COT")
[ -n "${TOOL_CATALOG:-}" ] && PREP_ARGS+=(--tool-catalog "$TOOL_CATALOG")
[ -n "${SHUFFLE:-}" ]      && PREP_ARGS+=(--shuffle)
[ -n "${MAX_ROWS:-}" ]     && PREP_ARGS+=(--max-rows "$MAX_ROWS")

echo ""; echo "=== STEP 1: prepare_data.py ==="
time python prepare_data.py "${PREP_ARGS[@]}"

# ── STEP 2: validate ───────────────────────────────────────────
echo ""; echo "=== STEP 2: validate_data.py ==="
# report truncation against the real training seq length (train.py uses 4096)
time python validate_data.py --jsonl "$DATASET" --max-seq-length 4096

# ── STEP 3: train ──────────────────────────────────────────────
TRAIN_ARGS=(--dataset "$DATASET" --output "$OUTPUT")
[ -n "${MAX_STEPS:-}" ] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")

echo ""; echo "=== STEP 3: train.py ==="
time python train.py "${TRAIN_ARGS[@]}"

echo ""; echo "=== DONE  $(date) ==="
ls -la $OUTPUT
