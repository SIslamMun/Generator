#!/bin/bash
#SBATCH --job-name=finetuner-retest
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=00:50:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/retest-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/retest-%j.err

# Re-test the 2 cases that failed in fulltest 2300219:
#   - unsloth/llama/tools : was 'NoneType has no len()' (cli.py now prints the
#                           full traceback so we see where)
#   - hf/llama/plain      : was SFTConfig max_seq_length kwarg — now fixed

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
SMOKE=$REPO/runs/finetuner-smoke/smoke.jsonl
TOOLDATA=$REPO/finetuned_unsloth/models/gemma4_e4b/data/train.jsonl
OUT=$REPO/runs/ft-retest
mkdir -p $REPO/finetuner/logs "$OUT"

echo "=== finetuner re-test  host=$(hostname) job=$SLURM_JOB_ID  $(date)"
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
# Pin EVERY HF cache to the spacious /work/nvme allocation and disable xet
# (its downloader's cache resolution caused a 'Disk quota exceeded' error).
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
export XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME"
source $VENV/bin/activate
python -c "import click" 2>/dev/null || pip install -q click
cd $REPO

echo ""; echo "############ RETEST 1: unsloth/llama/tools ############"
python -m finetuner.cli run \
    --backend unsloth --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$TOOLDATA" --output "$OUT/uns-llama-tools" \
    --lora-rank 8 --max-steps 5 --batch-size 1 --max-seq-length 4096 \
    && echo "RETEST1: PASS" || echo "RETEST1: FAIL"

echo ""; echo "############ RETEST 2: hf/llama/plain ############"
python -m finetuner.cli run \
    --backend hf --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$SMOKE" --output "$OUT/hf-llama" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024 \
    && echo "RETEST2: PASS" || echo "RETEST2: FAIL"

echo ""; echo "=== DONE  $(date)"
