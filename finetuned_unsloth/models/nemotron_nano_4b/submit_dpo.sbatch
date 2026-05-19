#!/bin/bash
#SBATCH --job-name=nemotron-nano-4b-dpo
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/dpo-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/dpo-%j.err

# DPO stage — fixes the phantom-parameter flood that SFT alone cannot.
# Runs prepare_dpo → snapshot SFT model → train_dpo on the compute node.
#
# ENV (set at submit time):
#   IN_TOOL      : curated tool-examples JSON (same file SFT used)   (required)
#   TOOL_CATALOG : tools.json                                        (required)
#   DPO_EPOCHS   : (optional) DPO epochs, default 2

set -eu
export PYTHONUNBUFFERED=1

MODEL_DIR=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b
IN_TOOL="${IN_TOOL:?IN_TOOL=<curated tool JSON> is required}"
TOOL_CATALOG="${TOOL_CATALOG:?TOOL_CATALOG=<tools.json> is required}"
DPO_EPOCHS="${DPO_EPOCHS:-2}"

mkdir -p $MODEL_DIR/logs $MODEL_DIR/data

echo "=== ENV ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID  date=$(date)"
echo "in_tool=$IN_TOOL  tool_catalog=$TOOL_CATALOG  dpo_epochs=$DPO_EPOCHS"
nvidia-smi | head -15

cd $MODEL_DIR
export PATH="$HOME/.local/bin:$PATH"

# Modules: torch 2.7+cu126 for aarch64, plus nvcc.
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
echo "=== python3 = $(which python3 || echo NOT_FOUND)"

# HF cache → NVMe scratch (the module's default HF_HOME is a full shared disk).
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
unset HUGGINGFACE_HUB_CACHE
unset TRANSFORMERS_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=0
export TMPDIR=/work/nvme/bekn/sislam3/tmp
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR"
echo "=== HF_HOME = $HF_HOME"

if [ ! -x "$MODEL_DIR/.venv-nemotron/bin/python" ]; then
    echo "ERROR: .venv-nemotron missing — run an SFT job first (it builds the venv)." >&2
    exit 1
fi
source $MODEL_DIR/.venv-nemotron/bin/activate
echo "=== venv active: $(which python) ==="

SFT_MODEL=$MODEL_DIR/artifacts/merged_16bit
SFT_BASE=$MODEL_DIR/artifacts/sft_base
if [ ! -f "$SFT_MODEL/model.safetensors" ]; then
    echo "ERROR: SFT model $SFT_MODEL not found — run SFT (submit.sbatch) first." >&2
    exit 1
fi

# ── STEP 1: build DPO preference pairs ─────────────────────────────
echo ""; echo "=== STEP 1: prepare_dpo.py ==="
time python prepare_dpo.py \
    --in-tool "$IN_TOOL" \
    --tool-catalog "$TOOL_CATALOG" \
    --out "$MODEL_DIR/data/dpo.jsonl"

# ── STEP 2: snapshot the SFT model (DPO loads here, writes merged_16bit) ──
echo ""; echo "=== STEP 2: snapshot SFT model → sft_base ==="
rm -rf "$SFT_BASE"
cp -r "$SFT_MODEL" "$SFT_BASE"
ls -la "$SFT_BASE"

# ── STEP 3: DPO training ───────────────────────────────────────────
echo ""; echo "=== STEP 3: train_dpo.py ==="
time python train_dpo.py \
    --sft-model "$SFT_BASE" \
    --dpo-data  "$MODEL_DIR/data/dpo.jsonl" \
    --output    "$MODEL_DIR/artifacts" \
    --epochs    "$DPO_EPOCHS"

# modeling_nemotron_h.py is preserved across save_pretrained_merged; confirm it.
echo ""; echo "=== DONE  $(date) ==="
ls -la $MODEL_DIR/artifacts/merged_16bit
