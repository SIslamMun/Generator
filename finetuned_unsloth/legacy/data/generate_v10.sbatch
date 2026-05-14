#!/bin/bash
#SBATCH --job-name=jarvis-v10-gen
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Validated data generation for v10.
#   1. Pull gpt-oss:120b as teacher (all weights to shared /work/hdd scratch).
#   2. Run generator tool-generate-full with ratios biased to chain_first + error_recovery.
#   3. Replay every generated trace against real jarvis-mcp; drop failing ones.
#   4. Emit v10_delta_clean.json for the training step.

set -eu

REPO=/u/sislam3/Generator
module load python/anaconda3/2.10.0
source "$REPO/.venv-delta/bin/activate"

export PATH=/u/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/u/sislam3/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama_models      # shared /work
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=8h

cd "$REPO"
mkdir -p finetuned_unsloth/data/v10 finetuned_unsloth/logs

echo "=== GPU ==="
nvidia-smi | head -15
df -h /work/hdd/bekn | tail -1

echo ""
echo "=== start ollama on compute node ==="
nohup ollama serve > "finetuned_unsloth/logs/v10-ollama-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done

TEACHER="${TEACHER:-gpt-oss:120b}"
if ! ollama list | awk '{print $1}' | grep -qx "$TEACHER"; then
  echo "=== pulling $TEACHER (~65 GB; pulled into $OLLAMA_MODELS) ==="
  time ollama pull "$TEACHER" 2>&1 | tail -8
fi
ollama list

echo ""
echo "=== warm-up teacher ==="
time ollama run "$TEACHER" "Generate a realistic Jarvis-CD request." 2>&1 | tail -3 || true

echo ""
echo "=== reset Jarvis MCP state before generation ==="
python3 -c "
from inference.mcp_client import JarvisMCP
with JarvisMCP() as m: print(m.call_tool('jm_reset', {})[:80])
"

RAW="$REPO/finetuned_unsloth/data/v10/v10_delta_raw.json"
CLEAN="$REPO/finetuned_unsloth/data/v10/v10_delta_clean.json"
TARGET=${TARGET:-3000}

echo ""
echo "=== generating $TARGET raw examples with $TEACHER ==="
time python -m generator.cli tool-generate-full \
    "$REPO/configs/tools/jarvis_tools.json" \
    -o "$RAW" \
    --target-pairs "$TARGET" \
    --ratio-single 0.10 \
    --ratio-multi  0.15 \
    --ratio-chain  0.45 \
    --ratio-error  0.30 \
    --tools-per-example 10 \
    --distractor-strategy mixed \
    --save-every 50 \
    --provider ollama \
    --model "$TEACHER"

echo ""
echo "=== raw stats ==="
python3 -c "
import json
d = json.load(open('$RAW'))
from collections import Counter
c = Counter(e.get('solution',{}).get('method','?') for e in d)
print('raw examples:', len(d))
print('by method:', dict(c))
"

echo ""
echo "=== validating against real MCP (drops invalid / schema-violating examples) ==="
time python finetuned_unsloth/data/validate_against_mcp.py \
    --input "$RAW" \
    --output "$CLEAN" \
    --stats "${CLEAN%.json}.stats.json"

echo ""
echo "=== done ==="
ls -la "$CLEAN" "${CLEAN%.json}.stats.json"
