#!/bin/bash
#SBATCH --job-name=jarvis-nemo
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Follow-up sweep for the two NVIDIA Nemotron models only.
# Runs after the main sweep so we don't fight over disk / NFS.

set -eu
module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

export PATH=/u/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/u/sislam3/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=30m
export HF_HOME="$HOME/.cache/huggingface"

cd /u/sislam3/Generator

echo "=== GPU ==="
nvidia-smi | head -15

echo ""
echo "=== start ollama ==="
nohup ollama serve > /u/sislam3/Generator/finetuned_unsloth/logs/nemo-ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
ollama list
df -h /work/hdd/bekn | tail -1

echo ""
echo "=== pulling nemotron models (into shared work fs) ==="
echo "OLLAMA_MODELS=$OLLAMA_MODELS"
for m in "nemotron-3-nano:30b" "nemotron-cascade-2:30b" "nemotron-3-super:120b"; do
  if ollama list | awk '{print $1}' | grep -qx "$m"; then
    echo "  ✓ $m already present"
  else
    echo "  pulling $m ..."
    # Try both likely tag variants if the first fails.
    if ! time ollama pull "$m" 2>&1 | tail -5; then
      alt="${m/nemotron-3-/nemotron:}"
      echo "  first tag failed, trying $alt"
      time ollama pull "$alt" 2>&1 | tail -5 || echo "  $m still unavailable"
    fi
  fi
done
ollama list

echo ""
echo "=== warm-ups ==="
for m in nemotron-3-nano:30b nemotron-cascade-2:30b nemotron-3-super:120b; do
  echo "--- warm $m ---"
  time ollama run "$m" "hi" 2>&1 | tail -2 || echo "(load failed)"
done

echo ""
echo "=== reset jarvis MCP state ==="
python3 -c "
from inference.mcp_client import JarvisMCP
with JarvisMCP() as m:
    print(m.call_tool('jm_reset', {})[:80])
"

echo ""
echo "=== nemotron-only benchmark ==="
python finetuned_unsloth/test/benchmark_all.py \
    --models nemotron-nano-30b nemotron-cascade-30b nemotron-super-120b \
    --temperatures 0.0 0.3 0.7 --repeats 3 \
    --tag nemotron_$SLURM_JOB_ID
