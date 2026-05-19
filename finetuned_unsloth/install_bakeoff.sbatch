#!/bin/bash
#SBATCH --job-name=bakeoff-venv-install
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/install-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/install-%j.err

# Build a fresh venv for the Gemma 4 / Granite 4.1 bake-off.
# Both are standard transformers (Granite 4.1 is dense decoder-only, NOT the
# 4.0 Mamba-hybrid) — so, unlike the Nemotron venv, there are NO Mamba
# kernels to build. torch comes from the miniforge module via
# --system-site-packages; everything else is pinned to the latest releases
# (Gemma 4 and Granite 4.1 are April-2026 models and need recent transformers).

set -euo pipefail
export PYTHONUNBUFFERED=1

VENV=/u/sislam3/Generator/finetuned_unsloth/.venv-bakeoff
mkdir -p /u/sislam3/Generator/finetuned_unsloth/logs

echo "=== install: host=$(hostname) job=$SLURM_JOB_ID date=$(date)"
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
echo "module python: $(which python3)  nvcc: $(which nvcc || echo none)"

# HF cache + tmp → NVMe scratch (override the module's full shared HF_HOME)
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp
export PIP_CACHE_DIR=/work/nvme/bekn/sislam3/pip
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TMPDIR" "$PIP_CACHE_DIR"

# fresh venv, inheriting the module's aarch64 torch build
rm -rf "$VENV"
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
echo "=== venv python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install --upgrade pip setuptools wheel

echo ""; echo "=== phase 1: core HF / training stack (latest) ==="
pip install --no-cache-dir -U \
    "transformers" "tokenizers" "accelerate" "datasets" \
    "trl" "peft" "bitsandbytes" "torchao" "triton" \
    "sentencepiece" "protobuf" "timm" "hf_transfer" "huggingface_hub"

echo ""; echo "=== phase 2: Unsloth (latest from git) ==="
pip install --no-cache-dir "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo.git"
pip install --no-cache-dir "unsloth[base] @ git+https://github.com/unslothai/unsloth.git"

echo ""; echo "=== versions ==="
python - <<'PY'
import torch, transformers, trl, peft
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
try:
    import unsloth; print("unsloth     ", getattr(unsloth, "__version__", "?"))
except Exception as e:
    print("unsloth import FAILED:", e)
PY

echo ""; echo "=== DONE  $(date) ==="
