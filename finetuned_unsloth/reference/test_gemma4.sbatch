#!/bin/bash
#SBATCH --job-name=ref-gemma4-test
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/reference/test-gemma4-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/reference/test-gemma4-%j.err

set -u
export PYTHONUNBUFFERED=1
REF=/u/sislam3/Generator/finetuned_unsloth/reference
VENV=/u/sislam3/Generator/finetuned_unsloth/.venv-bakeoff

module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME"

source $VENV/bin/activate
cd $REF
time python test_gemma4_reference.py
echo "=== DONE  $(date)"
