#!/bin/bash
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=toolgen-v7-2k
#SBATCH --output=/u/sislam3/Generator/slurm/logs/v7_2k_%j.out
#SBATCH --error=/u/sislam3/Generator/slurm/logs/v7_2k_%j.err

# ═══════════════════════════════════════════════════════════════════
#  v7 PILOT: Generate 2000 tool-use examples with gpt-oss:120b
#  - 10 tools per example (target + distractors)
#  - 60/40 semantic/random distractor mix
#  - Checkpoints every 100 examples (auto-resume on resubmit)
# ═══════════════════════════════════════════════════════════════════

set -e

# ── Environment ─────────────────────────────────────────────────
module load python/anaconda3/2.10.0
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama-models
export PATH=/work/hdd/bekn/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/work/hdd/bekn/sislam3/ollama/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/u/sislam3/Generator/src:/u/sislam3/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

cd /u/sislam3/Generator

echo "═══════════════════════════════════════════════════════════"
echo " v7 PILOT: 2000 tool-use examples"
echo " Host: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Start Ollama ────────────────────────────────────────────────
echo "Starting Ollama server..."
nohup ollama serve > /tmp/ollama_v7_2k_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i}x2 seconds"
        break
    fi
    sleep 2
done

echo "Available models:"
ollama list
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up Ollama (PID $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null || true
    wait $OLLAMA_PID 2>/dev/null || true
    echo "Job finished: $(date)"
}
trap cleanup EXIT

# ── Generate 2000 examples ─────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Generating 2000 examples — $(date)"
echo "═══════════════════════════════════════════════════════════"

python -m generator.cli tool-generate-full \
  configs/tools/jarvis_tools.yaml \
  -o outputs/v7_2k/jarvis_v7_2000.json \
  --target-pairs 2000 \
  --tools-per-example 10 \
  --distractor-strategy mixed \
  --ratio-single 0.30 --ratio-multi 0.30 \
  --ratio-chain 0.25 --ratio-error 0.15 \
  --save-every 100 \
  --provider ollama --model gpt-oss:120b

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " v7 PILOT COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════════"

# Print summary
python -c "
import json
from collections import Counter
data = json.load(open('outputs/v7_2k/jarvis_v7_2000.json'))
methods = Counter(e['solution']['method'] for e in data)
print(f'Total examples: {len(data)}')
print(f'By method: {dict(methods)}')
tools_used = Counter()
for e in data:
    for s in e['solution']['reasoning_path']:
        tools_used[s['tool']] += 1
print(f'Unique tools used: {len(tools_used)}')
print(f'Top 5 tools: {tools_used.most_common(5)}')
"
