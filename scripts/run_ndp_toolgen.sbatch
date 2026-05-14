#!/bin/bash
#SBATCH --job-name=ndp-toolgen
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/u/sislam3/Generator/runs/ndp/logs/toolgen-%j.out
#SBATCH --error=/u/sislam3/Generator/runs/ndp/logs/toolgen-%j.err

# Generate NDP tool-use training data with gpt-oss:120b on a GH200.
# Self-contained: installs Ollama into the bekn project's shared storage
# (group-readable by anyone in delta_bekn), pulls gpt-oss:120b on first
# run, then runs `generator tool-generate-full`. No fine-tuning — only
# synthetic-data generation.
#
# SHARED LOCATION (anyone in delta_bekn group can use):
#   binary:  /work/hdd/bekn/ollama/bin/ollama
#   models:  /work/hdd/bekn/ollama/models/
#
# Override defaults via env:
#   TARGET_PAIRS=200 sbatch scripts/run_ndp_toolgen.sbatch   # smoke
#   TARGET_PAIRS=2000 sbatch scripts/run_ndp_toolgen.sbatch  # full
#   TEACHER=gpt-oss:20b sbatch ...                            # smaller teacher

set -eu

REPO=/u/sislam3/Generator
TEACHER="${TEACHER:-gpt-oss:120b}"
TARGET_PAIRS="${TARGET_PAIRS:-2000}"
WORKERS="${WORKERS:-4}"

# Shared Ollama install on /work/hdd/bekn (group: delta_bekn, setgid).
# umask 002 so any new files inherit group-write — required for other
# bekn members to share the model store.
umask 002

OLLAMA_ROOT=/work/hdd/bekn/ollama
export OLLAMA_MODELS=$OLLAMA_ROOT/models
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=$WORKERS
export OLLAMA_KEEP_ALIVE=8h
export PATH=$OLLAMA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH:-}

OUT_DIR=$REPO/runs/ndp
mkdir -p $OUT_DIR/logs $OUT_DIR/data $OLLAMA_MODELS

cd $REPO
module load python/anaconda3/2.10.0
source $REPO/.venv-delta/bin/activate

echo "=== ENV ==="
echo "host=$(hostname)  job=$SLURM_JOB_ID  date=$(date)"
echo "teacher=$TEACHER  target_pairs=$TARGET_PAIRS  workers=$WORKERS"
echo "ollama_root=$OLLAMA_ROOT  models=$OLLAMA_MODELS  (shared, group=delta_bekn)"
nvidia-smi | head -15

# ── 1. Ensure Ollama is installed ─────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    echo ""; echo "=== installing Ollama → $OLLAMA_ROOT (shared) ==="
    mkdir -p $OLLAMA_ROOT
    TARBALL=ollama-linux-arm64.tar.zst
    curl -fL -o /tmp/$TARBALL "https://ollama.com/download/${TARBALL}"
    ls -la /tmp/$TARBALL
    tar --zstd -xf /tmp/$TARBALL -C $OLLAMA_ROOT
    rm /tmp/$TARBALL
    # make the install group-readable+executable for other bekn members
    chmod -R g+rwX $OLLAMA_ROOT
    hash -r
    command -v ollama && ollama --version
fi

# ── 2. Start Ollama server ────────────────────────────────────
echo ""; echo "=== start ollama serve ==="
nohup ollama serve > $OUT_DIR/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && { echo "ollama up after ${i}s"; break; }
    sleep 1
done

# ── 3. Pull the teacher model (idempotent) ────────────────────
if ! ollama list | awk '{print $1}' | grep -qx "$TEACHER"; then
    echo ""; echo "=== pulling $TEACHER (this can take 20-40 min for 120b) ==="
    time ollama pull "$TEACHER"
    # ensure other delta_bekn members can read the model blobs
    chmod -R g+rwX $OLLAMA_MODELS
fi
echo "available models:"; ollama list

# ── 4. Generate ───────────────────────────────────────────────
RAW_OUT=$OUT_DIR/data/ndp_tool_examples_raw.json
echo ""; echo "=== generator tool-generate-full → $RAW_OUT ==="
# Ratios tuned for NDP's small 3-tool surface:
#   - multi=0 because the multi section produces 90% single-step output
#     when the instruction generator yields predominantly single-tool
#     instructions (most NDP operations are single-call by nature).
#     chain_first already covers the only natural multi-step pattern
#     (search_datasets → get_dataset_details).
#   - For richer tool catalogs (e.g. Jarvis with 29 tools), bump multi
#     back up to 0.25-0.30 once the code fix for the multi-section
#     instruction filter lands.
time python -m generator.cli tool-generate-full \
    $REPO/configs/tools/ndp_tools.json \
    -o $RAW_OUT \
    --target-pairs $TARGET_PAIRS \
    --ratio-single 0.55 \
    --ratio-multi  0.00 \
    --ratio-chain  0.30 \
    --ratio-error  0.15 \
    --tools-per-example 0 \
    --save-every 50 \
    --provider ollama \
    --model "$TEACHER"

echo ""; echo "=== DONE  $(date) ==="
ls -la $OUT_DIR/data/
