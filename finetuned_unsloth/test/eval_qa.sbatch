#!/bin/bash
#SBATCH --job-name=jarvis-qa-v1-eval
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

set -eu
module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd /u/sislam3/Generator

nvidia-smi | head -15

echo "=== offline eval: jarvis-qa-v1 + base gemma-3-270m-it on 365 val rows ==="
python finetuned_unsloth/test/eval_qa.py \
    --model /work/hdd/bekn/sislam3/jarvis_qa_v1_lora/merged_16bit \
    --val   /u/sislam3/Generator/finetuned_unsloth/data/qa_v1/jarvis_qa_v1_cot.val.jsonl \
    --out   /u/sislam3/Generator/inference/results/qa_v1_eval \
    --max-new-tokens 700 \
    --baseline unsloth/gemma-3-270m-it

echo ""
echo "=== DONE — $(date) ==="
ls -la /u/sislam3/Generator/inference/results/qa_v1_eval/
