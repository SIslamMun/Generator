#!/bin/bash
#SBATCH --job-name=finetuner-fulltest
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:30:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/fulltest-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/fulltest-%j.err

# Full real test of the finetuner — all 3 backends, 2 model families, plain
# + tool datasets, 5 short runs in one GPU allocation. Each run is 5 steps.

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
SMOKE=$REPO/runs/finetuner-smoke/smoke.jsonl
TOOLDATA=$REPO/finetuned_unsloth/models/gemma4_e4b/data/train.jsonl
OUT=$REPO/runs/ft-fulltest
mkdir -p $REPO/finetuner/logs "$OUT"

echo "=== finetuner full test  host=$(hostname) job=$SLURM_JOB_ID  $(date)"
module load python/miniforge3_pytorch/2.7.0 || true
module load cuda/12.9 || true
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp
export TORCHDYNAMO_DISABLE=1
mkdir -p "$HF_HOME" "$TMPDIR"
source $VENV/bin/activate
python -c "import click" 2>/dev/null || pip install -q click
cd $REPO

declare -a NAMES RESULTS
run_test () {       # $1 = test name, rest = finetuner.cli args
    local name="$1"; shift
    echo ""; echo "############################################################"
    echo "### TEST: $name"
    echo "###   args: $*"
    echo "############################################################"
    if python -m finetuner.cli run "$@"; then
        NAMES+=("$name"); RESULTS+=("PASS")
    else
        NAMES+=("$name"); RESULTS+=("FAIL")
    fi
}

# 1) unsloth · Llama 1B · plain QA · --save-merged   (language path)
run_test "unsloth/llama/plain" \
    --backend unsloth --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$SMOKE" --output "$OUT/uns-llama" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024 --save-merged

# 2) unsloth · Llama 1B · NDP tool dataset           (tools-rendering path)
run_test "unsloth/llama/tools" \
    --backend unsloth --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$TOOLDATA" --output "$OUT/uns-llama-tools" \
    --lora-rank 8 --max-steps 5 --batch-size 1 --max-seq-length 4096

# 3) unsloth · Qwen2.5-0.5B · plain QA               (second model family)
run_test "unsloth/qwen/plain" \
    --backend unsloth --base-model unsloth/Qwen2.5-0.5B-Instruct \
    --dataset "$SMOKE" --output "$OUT/uns-qwen" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024

# 4) hf · Llama 1B · plain QA                        (portable backend)
run_test "hf/llama/plain" \
    --backend hf --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$SMOKE" --output "$OUT/hf-llama" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024

# 5) ollama · package the merged checkpoint from test 1
module load ollama 2>/dev/null || true
if command -v ollama >/dev/null 2>&1; then
    run_test "ollama/package" \
        --backend ollama --base-model "$OUT/uns-llama/merged_16bit" \
        --dataset "" --output "$OUT/oll-llama" --model-name ft-fulltest-llama
else
    NAMES+=("ollama/package"); RESULTS+=("SKIP (ollama not on PATH)")
fi

echo ""; echo "============================================================"
echo "=== FINETUNER FULL TEST SUMMARY"
for i in "${!NAMES[@]}"; do
    printf "  %-22s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"
done
echo "============================================================"
echo "=== DONE  $(date)"
