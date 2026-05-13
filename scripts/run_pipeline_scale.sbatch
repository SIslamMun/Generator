#!/bin/bash
#SBATCH --job-name=pipeline-scale
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Real-target validation: 1000 chat + 1000 tool examples.
# Idempotent — re-runs skip cached stages.

set -eu
REPO=/u/sislam3/Generator
module load python/anaconda3/2.10.0
source $REPO/.venv-delta/bin/activate

export PATH=/u/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/u/sislam3/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=8h

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

cd $REPO
mkdir -p $REPO/finetuned_unsloth/logs

echo "=== HW ==="; nvidia-smi | head -15

echo ""; echo "=== ollama ==="
nohup ollama serve > $REPO/finetuned_unsloth/logs/scale-ollama-$SLURM_JOB_ID.log 2>&1 &
trap "kill $! 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
TEACHER="${TEACHER:-gpt-oss:20b}"
ollama list | awk '{print $1}' | grep -qx "$TEACHER" || ollama pull "$TEACHER"
ollama list

echo ""; echo "=== STEP 0: build LanceDB ==="
[ -d ./lancedb_scale ] || python -m generator.cli ingest /u/sislam3/clio-kit \
    -o ./lancedb_scale --table code_chunks

echo ""; echo "=== ONE COMMAND: chat + tool pipelines (target=1000 each) ==="
python -m generator.cli train -c $REPO/examples/pipeline-scale.yaml

echo ""; echo "=== DONE — $(date) ==="
ls -la ./runs/scale_v1/artifacts/
