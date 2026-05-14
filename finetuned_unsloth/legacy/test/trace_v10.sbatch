#!/bin/bash
#SBATCH --job-name=jarvis-v10-trace
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
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
nohup ollama serve > /u/sislam3/Generator/finetuned_unsloth/logs/v10-trace-ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done

echo ""
echo "=== ensure jarvis-v10 is imported ==="
if ! ollama list | awk '{print $1}' | grep -qx "jarvis-v10:latest"; then
    ollama create jarvis-v10 -f /u/sislam3/Generator/finetuned_unsloth/artifacts/v10/Modelfile
fi
ollama list | grep -E '^(NAME|jarvis-v10)'

echo ""
echo "=== warm-up ==="
time ollama run jarvis-v10 "hi" 2>&1 | tail -2 || true

echo ""
echo "=== trace v10 over all 12 cases × 3 temperatures ==="
python finetuned_unsloth/test/trace_v10.py

echo ""
echo "=== DONE — $(date) ==="
ls -la /u/sislam3/Generator/inference/results/v10_traces/ | head
