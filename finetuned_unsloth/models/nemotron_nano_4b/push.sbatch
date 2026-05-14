#!/bin/bash
#SBATCH --job-name=nemotron-push-hf
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/push-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/push-%j.err

# Pre-flight: HF_TOKEN must be set in the env when sbatch is run.
# We pass it through with --export but DO NOT print it.
#   HF_TOKEN=hf_xxxx sbatch --export=ALL,HF_TOKEN push.sbatch
# Optional override:
#   HF_REPO=user/custom-name HF_TOKEN=hf_xxx sbatch ...

set -euo pipefail
export PYTHONUNBUFFERED=1

MODEL_DIR=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b
mkdir -p $MODEL_DIR/logs

cd $MODEL_DIR
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true

# Override the module's full /projects/bekn/hf cache with our local NVMe scratch
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

source $MODEL_DIR/.venv-nemotron/bin/activate
echo "=== venv: $(which python)"

# Sanity-check HF_TOKEN presence WITHOUT printing it
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var is not set. Run with:"
    echo "  HF_TOKEN=hf_xxx sbatch --export=ALL,HF_TOKEN $0"
    exit 1
fi
echo "=== HF_TOKEN present (length: ${#HF_TOKEN})"

# Ensure huggingface_hub is installed (it pulls in via transformers, but just in case)
python -c "import huggingface_hub" 2>/dev/null || pip install --quiet huggingface_hub

# Build llama.cpp from source (no sudo needed). Idempotent — skips if built.
echo ""; echo "=== ensure llama.cpp built ==="
export LLAMACPP_DIR=/work/nvme/bekn/sislam3/llama.cpp
bash $MODEL_DIR/build_llamacpp.sh

echo ""; echo "=== convert + upload ==="
time python push_to_hf.py

echo ""; echo "=== DONE  $(date) ==="
