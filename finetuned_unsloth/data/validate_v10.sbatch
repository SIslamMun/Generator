#!/bin/bash
#SBATCH --job-name=jarvis-v10-val
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Validate the 2980 raw v10 examples against real jarvis-mcp.
# Uses 20s per-call + 60s per-example timeouts so a single pathological
# run_pipeline can't hang the whole job.

set -eu
module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

cd /u/sislam3/Generator

RAW="/u/sislam3/Generator/finetuned_unsloth/data/v10/v10_delta_raw.json"
CLEAN="/u/sislam3/Generator/finetuned_unsloth/data/v10/v10_delta_clean.json"

echo "=== validating v10 ==="
time python finetuned_unsloth/data/validate_against_mcp.py \
    --input  "$RAW" \
    --output "$CLEAN" \
    --stats  "${CLEAN%.json}.stats.json"

echo ""
echo "=== stats ==="
cat "${CLEAN%.json}.stats.json"
