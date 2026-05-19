#!/bin/bash
#SBATCH --job-name=bakeoff-probe
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/probe-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/probe-%j.err

# Raw-output bake-off probe for one model. Pass MODEL_DIR at submit time:
#   MODEL_DIR=<merged checkpoint> sbatch --export=ALL,MODEL_DIR bakeoff_probe.sbatch

set -eu
export PYTHONUNBUFFERED=1
MODEL_DIR="${MODEL_DIR:?MODEL_DIR=<merged checkpoint dir> is required}"

echo "=== probe: host=$(hostname) job=$SLURM_JOB_ID  model=$MODEL_DIR"
module load python/miniforge3_pytorch/2.7.0 || true

export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp
export TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME" "$TMPDIR" /u/sislam3/Generator/finetuned_unsloth/logs

source /u/sislam3/Generator/finetuned_unsloth/.venv-bakeoff/bin/activate
echo "=== venv: $(which python)"

cd /u/sislam3/Generator/finetuned_unsloth
time python bakeoff_probe.py
echo "=== DONE  $(date) ==="
