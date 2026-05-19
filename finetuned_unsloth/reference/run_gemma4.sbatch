#!/bin/bash
#SBATCH --job-name=unsloth-ref-gemma4
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/reference/ref-gemma4-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/reference/ref-gemma4-%j.err

# Run the official Unsloth Gemma 4 E4B notebook recipe verbatim — reference
# baseline for comparison against our gemma4_e4b/ pipeline.

set -u
export PYTHONUNBUFFERED=1
REF=/u/sislam3/Generator/finetuned_unsloth/reference
VENV=/u/sislam3/Generator/finetuned_unsloth/.venv-bakeoff

echo "=== unsloth Gemma 4 reference run  host=$(hostname) job=$SLURM_JOB_ID  $(date)"
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true

# HF caches → /work/nvme, xet disabled (lesson from the quota failure)
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

source $VENV/bin/activate
echo "=== venv: $(which python)"
cd $REF
time python run_gemma4_reference.py
echo "=== DONE  $(date)"
