#!/bin/bash
#SBATCH --job-name=jarvis-bench
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

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
nohup ollama serve > /u/sislam3/Generator/finetuned_unsloth/logs/bench-ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
ollama list

echo ""
echo "=== pull missing models ==="
for m in "gpt-oss:20b" "llama3.1:8b" "mistral:7b-instruct"; do
  if ollama list | awk '{print $1}' | grep -qx "$m"; then
    echo "  ✓ $m already present"
  else
    echo "  pulling $m ..."
    time ollama pull "$m" 2>&1 | tail -3
  fi
done
ollama list

echo ""
echo "=== warm-ups (force model load onto GPU) ==="
for m in jarvis-v7 jarvis-v8 jarvis-v9 qwen2.5:7b-instruct gpt-oss:20b llama3.1:8b mistral:7b-instruct; do
  echo "--- warm $m ---"
  ollama list | awk '{print $1}' | grep -qx "$m:latest\|$m" && \
    time ollama run "$m" "hi" 2>&1 | tail -2 || echo "(not installed)"
done

echo ""
echo "=== reset jarvis MCP state ==="
python3 -c "
from inference.mcp_client import JarvisMCP
with JarvisMCP() as m:
    print(m.call_tool('jm_reset', {})[:80])
"

echo ""
echo "=== benchmark (T sweep: 0.0, 0.3, 0.7 — 3 repeats at non-zero) ==="
python finetuned_unsloth/test/benchmark_all.py "$@"
