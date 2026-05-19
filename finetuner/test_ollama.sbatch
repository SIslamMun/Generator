#!/bin/bash
#SBATCH --job-name=ft-ollama-test
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/u/sislam3/Generator/finetuner/logs/ollama-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuner/logs/ollama-%j.err

# End-to-end test of the ollama backend:
#   1. start ollama serve (local daemon)
#   2. train Llama-3.2-1B via the finetuner with --save-merged
#   3. package the merged checkpoint via `finetuner --backend ollama`
#   4. confirm `ollama list` shows it

set -u
export PYTHONUNBUFFERED=1
REPO=/u/sislam3/Generator
VENV=$REPO/finetuned_unsloth/.venv-bakeoff
OUT=$REPO/runs/ft-ollama-test
DATA=$REPO/runs/finetuner-smoke/smoke.jsonl
mkdir -p "$OUT" "$REPO/finetuner/logs"

module load python/miniforge3_pytorch/2.7.0 || true
export PATH=$HOME/.local/ollama/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/ollama/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/work/nvme/bekn/sislam3/hf_cache HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet XDG_CACHE_HOME=/work/nvme/bekn/sislam3/xdg_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export TMPDIR=/work/nvme/bekn/sislam3/tmp TORCHDYNAMO_DISABLE=1
# put ollama's model store on /work/nvme (it can be huge)
export OLLAMA_MODELS=/work/nvme/bekn/sislam3/ollama_models
mkdir -p "$HF_HOME/hub" "$HF_HOME/xet" "$TMPDIR" "$XDG_CACHE_HOME" "$OLLAMA_MODELS"

echo "=== ollama version ==="
ollama --version 2>&1 | head -3

# ── 1) start ollama serve daemon ────────────────────────────────────
echo ""; echo "=== starting ollama serve ==="
ollama serve > $REPO/finetuner/logs/ollama-daemon-$SLURM_JOB_ID.log 2>&1 &
DAEMON=$!
trap "kill $DAEMON 2>/dev/null" EXIT
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    ollama list >/dev/null 2>&1 && { echo "daemon up after ${i}*2s"; break; }
done
ollama list 2>&1

# ── 2) train + merge a small model via the finetuner ────────────────
source $VENV/bin/activate
cd $REPO
echo ""; echo "=== 2) finetuner train+merge (Llama-3.2-1B, 5 steps) ==="
time python -m finetuner.cli run \
    --backend unsloth \
    --base-model unsloth/Llama-3.2-1B-Instruct \
    --dataset "$DATA" --output "$OUT/llama-merged" \
    --lora-rank 8 --max-steps 5 --batch-size 2 --max-seq-length 1024 \
    --save-merged

# ── 3) package the merged checkpoint via the ollama backend ─────────
echo ""; echo "=== 3) finetuner --backend ollama (package) ==="
python -m finetuner.cli run \
    --backend ollama \
    --base-model "$OUT/llama-merged/merged_16bit" \
    --output "$OUT/ollama-pkg" \
    --dataset "" --model-name ft-test-llama

# ── 4) verify with ollama list + a quick run ────────────────────────
echo ""; echo "=== 4) ollama list ==="
ollama list 2>&1
echo "--- ollama show ft-test-llama ---"
ollama show ft-test-llama 2>&1 | head -20
echo "--- ollama generate (3 tokens) ---"
echo "Hello" | timeout 60 ollama run ft-test-llama --verbose 2>&1 | head -30 || true

echo ""; echo "=== DONE  $(date)"
