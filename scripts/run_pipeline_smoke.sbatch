#!/bin/bash
#SBATCH --job-name=pipeline-smoke
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/u/sislam3/Generator/slurm/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/slurm/logs/%x-%j.err

# End-to-end smoke test of `generator train-chat` + `generator train-tool`
# at target=100 each. Verifies the autonomous pipeline works.

set -eu
REPO=/u/sislam3/Generator
module load python/anaconda3/2.10.0
source $REPO/.venv-delta/bin/activate

# Ollama on compute node — needed by gen-cot + gen-tool stages
export PATH=/u/sislam3/ollama/bin:$PATH
export LD_LIBRARY_PATH=/u/sislam3/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=/work/hdd/bekn/sislam3/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=8h

# HF cache — point at $HOME so we use whatever was downloaded by prior runs.
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

cd $REPO
mkdir -p $REPO/slurm/logs

echo "=== HW ==="
nvidia-smi | head -15

echo ""; echo "=== start ollama ==="
nohup ollama serve > $REPO/slurm/logs/smoke-ollama-$SLURM_JOB_ID.log 2>&1 &
trap "kill $! 2>/dev/null || true" EXIT
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done

# Pull the teacher if missing
TEACHER="${TEACHER:-gpt-oss:20b}"
if ! ollama list | awk '{print $1}' | grep -qx "$TEACHER"; then
    echo "pulling $TEACHER ..."; time ollama pull "$TEACHER" 2>&1 | tail -5
fi
ollama list

echo ""; echo "=== STEP 0: build LanceDB from clio-kit source (only if missing) ==="
if [ ! -d ./lancedb_smoke ]; then
    python -m generator.cli ingest /u/sislam3/clio-kit \
        -o ./lancedb_smoke --table code_chunks --max-files 60
else
    echo "lancedb_smoke/ already exists, skipping ingest"
fi

echo ""; echo "=== STEP 1: chat pipeline (target=100) ==="
python -m generator.cli train-chat \
    --config $REPO/examples/pipeline-smoke.yaml \
    --lancedb ./lancedb_smoke \
    --target 100

echo ""; echo "=== STEP 2: tool pipeline (target=100) ==="
python -m generator.cli train-tool \
    --config $REPO/examples/pipeline-smoke.yaml \
    --target 100

echo ""; echo "=== DONE — $(date) ==="
ls -la ./runs/smoke_v1/
