#!/bin/bash
#SBATCH --job-name=jarvis-gen-v8d
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

set -eu

REPO=/u/sislam3/Generator
module load python/anaconda3/2.10.0
source "$REPO/.venv-delta/bin/activate"

export PATH=/u/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/u/sislam3/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_KEEP_ALIVE=120m

cd "$REPO"

echo "=== GPU ==="
nvidia-smi | head -15
echo ""

# Start ollama server (bound to compute-node GPU so gpt-oss uses CUDA)
nohup ollama serve > "$REPO/finetuned_unsloth/logs/gen-ollama-serve-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done

TEACHER="${TEACHER:-qwen2.5:7b-instruct}"

echo ""
echo "=== pulling teacher model $TEACHER ==="
time ollama pull "$TEACHER" 2>&1 | tail -20
ollama list

echo ""
echo "=== generating ==="
# Ratios biased toward the failing categories (chain_first + error_recovery).
TARGET=${TARGET:-500}
OUT="$REPO/finetuned_unsloth/data/v8_delta_raw.json"

time python -m generator.cli tool-generate-full \
    "$REPO/configs/tools/jarvis_tools.json" \
    -o "$OUT" \
    --target-pairs "$TARGET" \
    --ratio-single 0.10 \
    --ratio-multi  0.15 \
    --ratio-chain  0.40 \
    --ratio-error  0.35 \
    --tools-per-example 10 \
    --distractor-strategy mixed \
    --save-every 50 \
    --provider ollama \
    --model "$TEACHER"

echo ""
echo "=== done ==="
ls -la "$OUT"
python3 -c "
import json
d = json.load(open('$OUT'))
from collections import Counter
c = Counter(e.get('solution',{}).get('method','?') for e in d)
print('examples:', len(d))
print('by method:', dict(c))
"
