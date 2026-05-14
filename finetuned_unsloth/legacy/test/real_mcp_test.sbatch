#!/bin/bash
#SBATCH --job-name=jarvis-real-mcp
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:20:00
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
export OLLAMA_KEEP_ALIVE=10m

cd /u/sislam3/Generator

echo "=== GPU ==="
nvidia-smi | head -15

echo ""
echo "=== starting ollama server ==="
nohup ollama serve > /u/sislam3/Generator/finetuned_unsloth/logs/ollama-serve-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

for i in $(seq 1 30); do
  if ollama list >/dev/null 2>&1; then echo "server up after ${i}s"; break; fi
  sleep 1
done

ollama list
echo ""

echo "=== warm-up ==="
time ollama run jarvis-v8 "hi" --verbose 2>&1 | tail -20 || true

echo ""
echo "=== real MCP test ==="
python finetuned_unsloth/test/real_mcp_test.py
