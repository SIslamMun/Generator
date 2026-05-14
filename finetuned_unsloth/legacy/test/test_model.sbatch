#!/bin/bash
#SBATCH --job-name=jarvis-test
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

set -euo pipefail

module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd /u/sislam3/Generator
python finetuned_unsloth/test/test_model.py
