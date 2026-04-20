#!/bin/bash
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=toolgen-batch-a
#SBATCH --output=/u/sislam3/Generator/slurm/logs/batch_a_%j.out
#SBATCH --error=/u/sislam3/Generator/slurm/logs/batch_a_%j.err

# ═══════════════════════════════════════════════════════════════════
#  BATCH A: Generate 4000 + 6000 tool-use examples (sequential)
#  2000 already generated in earlier run (outputs/v7_2k/)
#  Model: gpt-oss:120b via Ollama on compute node
#  Checkpoints every 100 examples — safe to resume on resubmit
# ═══════════════════════════════════════════════════════════════════

set -e

# ── Environment ─────────────────────────────────────────────────
module load python/anaconda3/2.10.0
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama-models
export PATH=/work/hdd/bekn/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/work/hdd/bekn/sislam3/ollama/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /u/sislam3/Generator
export PYTHONPATH=/u/sislam3/Generator/src:/u/sislam3/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

echo "═══ BATCH A — $(hostname) — $(date) ═══"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# ── Start Ollama ────────────────────────────────────────────────
echo "Starting Ollama server..."
nohup ollama serve > /tmp/ollama_batchA_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready (max 60s)
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i}x2 seconds"
        break
    fi
    sleep 2
done

# Verify model is available
echo "Available models:"
ollama list
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up Ollama (PID $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null || true
    wait $OLLAMA_PID 2>/dev/null || true
    echo "Done. $(date)"
}
trap cleanup EXIT

# ── Common flags ────────────────────────────────────────────────
TOOLS=configs/tools/jarvis_tools.yaml
COMMON="--provider ollama --model gpt-oss:120b \
  --tools-per-example 10 --distractor-strategy mixed \
  --ratio-single 0.30 --ratio-multi 0.30 \
  --ratio-chain 0.25 --ratio-error 0.15 \
  --save-every 100"

# ── Run 1: 4000 examples ───────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo " RUN 1/2: 4000 examples — $(date)"
echo "═══════════════════════════════════════════════════"
python -m generator.cli tool-generate-full $TOOLS \
  -o outputs/v7_4k/jarvis_v7_4000.json \
  --target-pairs 4000 $COMMON

echo ""
echo "✓ Run 1 complete: $(date)"
echo ""

# ── Run 2: 6000 examples ───────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo " RUN 2/2: 6000 examples — $(date)"
echo "═══════════════════════════════════════════════════"
python -m generator.cli tool-generate-full $TOOLS \
  -o outputs/v7_6k/jarvis_v7_6000.json \
  --target-pairs 6000 $COMMON

echo ""
echo "═══ BATCH A COMPLETE — $(date) ═══"
