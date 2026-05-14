#!/bin/bash
#SBATCH --job-name=ndp-curate
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/u/sislam3/Generator/runs/ndp/logs/curate-%j.out
#SBATCH --error=/u/sislam3/Generator/runs/ndp/logs/curate-%j.err

# LLM-as-judge curation of NDP tool-use examples. Scores each row on
# tool_selection / parameter_correctness / reasoning_quality / result_realism
# (sum 0-10) using gpt-oss:120b, then filters by --threshold.
#
# Input:  runs/ndp/data/ndp_tool_examples_clean.json     (1879 schema-passed rows)
# Output: runs/ndp/data/ndp_tool_examples_curated.json
#
# Overrides:
#   THRESHOLD=7.5 sbatch scripts/run_ndp_curate.sbatch
#   TEACHER=gpt-oss:20b sbatch ...                       # faster, lower quality
#   INPUT=path/to/something.json OUTPUT=path sbatch ...

set -eu

REPO=/u/sislam3/Generator
TEACHER="${TEACHER:-gpt-oss:120b}"
THRESHOLD="${THRESHOLD:-7.0}"
WORKERS="${WORKERS:-4}"
INPUT="${INPUT:-$REPO/runs/ndp/data/ndp_tool_examples_clean.json}"
OUTPUT="${OUTPUT:-$REPO/runs/ndp/data/ndp_tool_examples_curated.json}"

umask 002
OLLAMA_ROOT=/work/hdd/bekn/ollama
export OLLAMA_MODELS=$OLLAMA_ROOT/models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=$WORKERS
export OLLAMA_KEEP_ALIVE=8h
export PATH=$OLLAMA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH:-}

OUT_DIR=$REPO/runs/ndp
mkdir -p $OUT_DIR/logs

cd $REPO
module load python/anaconda3/2.10.0
source $REPO/.venv-delta/bin/activate

echo "=== ENV ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID  date=$(date)"
echo "teacher=$TEACHER  threshold=$THRESHOLD  workers=$WORKERS"
echo "input  = $INPUT"
echo "output = $OUTPUT"
nvidia-smi | head -15

# ── 1. Ensure Ollama is available (binary should already be installed) ────
if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama not on PATH after sourcing $OLLAMA_ROOT/bin"
    echo "Re-run scripts/run_ndp_toolgen.sbatch first to install Ollama."
    exit 1
fi
ollama --version

# ── 2. Start ollama serve ─────────────────────────────────────
echo ""; echo "=== start ollama serve ==="
nohup ollama serve > $OUT_DIR/logs/ollama-curate-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && { echo "ollama up after ${i}s"; break; }
    sleep 1
done

# ── 3. Verify teacher is present ──────────────────────────────
if ! ollama list | awk '{print $1}' | grep -qx "$TEACHER"; then
    echo "ERROR: teacher $TEACHER not in $OLLAMA_MODELS"
    echo "Re-run scripts/run_ndp_toolgen.sbatch first to pull the model."
    exit 1
fi
ollama list | head -5

# ── 4. Curate ────────────────────────────────────────────────
echo ""; echo "=== generator tool-curate → $OUTPUT ==="
time python -m generator.cli tool-curate \
    "$INPUT" \
    -o "$OUTPUT" \
    --tools $REPO/configs/tools/ndp_tools.json \
    --threshold $THRESHOLD \
    --min-success-rate 0.0 \
    --provider ollama \
    --model "$TEACHER"

echo ""; echo "=== DONE  $(date) ==="
ls -la $OUT_DIR/data/
