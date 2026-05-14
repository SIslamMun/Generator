#!/bin/bash
#SBATCH --job-name=nemotron-test
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/test-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b/logs/test-%j.err

set -eu
export PYTHONUNBUFFERED=1
MODEL_DIR=/u/sislam3/Generator/finetuned_unsloth/models/nemotron_nano_4b
mkdir -p $MODEL_DIR/logs
cd $MODEL_DIR

module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true

# Same HF-cache override as submit.sbatch (module sets it to a full /projects/bekn/hf)
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp

source $MODEL_DIR/.venv-nemotron/bin/activate
echo "=== venv: $(which python)"
nvidia-smi | head -10

echo ""; echo "=== test_inference.py ==="
time python test_inference.py
