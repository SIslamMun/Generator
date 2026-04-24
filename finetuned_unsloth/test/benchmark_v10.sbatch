#!/bin/bash
#SBATCH --job-name=jarvis-v10-bench
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
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
nohup ollama serve > /u/sislam3/Generator/finetuned_unsloth/logs/v10-bench-ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done

echo ""
echo "=== import jarvis-v10 from merged_16bit ==="
# Re-create each run so the tag always points at the latest merge.
ollama create jarvis-v10 -f /u/sislam3/Generator/finetuned_unsloth/artifacts/v10/Modelfile
ollama list | grep -E '^(NAME|jarvis-v10)'

echo ""
echo "=== warm-up ==="
time ollama run jarvis-v10 "hi" 2>&1 | tail -2 || true

echo ""
echo "=== reset jarvis MCP state (retry on cold-start init timeout) ==="
python3 -c "
import time
from inference.mcp_client import JarvisMCP, MCPError
last = None
for attempt in range(1, 4):
    try:
        with JarvisMCP(startup_timeout=60.0) as m:
            print(m.call_tool('jm_reset', {})[:80])
        break
    except MCPError as e:
        print(f'MCP init attempt {attempt} failed: {e}', flush=True)
        last = e
        time.sleep(5)
else:
    raise SystemExit(f'MCP never came up: {last}')
"

echo ""
echo "=== benchmark v10 only (T sweep: 0.0, 0.3, 0.7) ==="
python finetuned_unsloth/test/benchmark_all.py --models jarvis-v10 --tag v10 "$@"

echo ""
echo "=== DONE — $(date) ==="
