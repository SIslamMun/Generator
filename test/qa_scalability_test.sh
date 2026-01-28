#!/bin/bash
# QA Generator Scalability Test (similar to run_full_benchmark.sh)
# Tests QA generation with different OLLAMA_NUM_PARALLEL values

set -e

OVERRIDE_FILE="/etc/systemd/system/ollama.service.d/override.conf"
RESULTS_DIR="/home/cc/Generator/test/scalability_results"
PARALLELISM_VALUES=(1 2 4 6 8)

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "QA Generator Scalability Test"
echo "Testing OLLAMA_NUM_PARALLEL: ${PARALLELISM_VALUES[*]}"
echo "=============================================="
echo ""

cd /home/cc/Generator

for PARALLEL in "${PARALLELISM_VALUES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Testing OLLAMA_NUM_PARALLEL=$PARALLEL"
    echo "=============================================="

    # Update Ollama service configuration
    echo "Updating service configuration..."
    sudo tee "$OVERRIDE_FILE" > /dev/null << EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=$PARALLEL"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF

    # Reload and restart
    echo "Restarting Ollama service..."
    sudo systemctl daemon-reload
    sudo systemctl restart ollama

    # Wait for service
    echo "Waiting for service..."
    sleep 5
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Service ready after ${i}s"
            break
        fi
        sleep 1
    done

    echo ""
    echo "Test 1: Sequential (workers=1) with PARALLEL=$PARALLEL"
    time uv run generator generate \
        --lancedb-path /home/cc/Jarvis_QA_Generator/lancedb \
        --table code_chunks \
        --output "$RESULTS_DIR/parallel${PARALLEL}_workers1.json" \
        --provider ollama \
        --model gpt-oss:20b \
        --max-chunks 10 \
        --workers 1 \
        --n-pairs 2 \
        2>&1 | tee "$RESULTS_DIR/parallel${PARALLEL}_workers1.log"

    sleep 3

    echo ""
    echo "Test 2: Parallel (workers=$PARALLEL) with PARALLEL=$PARALLEL"
    time uv run generator generate \
        --lancedb-path /home/cc/Jarvis_QA_Generator/lancedb \
        --table code_chunks \
        --output "$RESULTS_DIR/parallel${PARALLEL}_workers${PARALLEL}.json" \
        --provider ollama \
        --model gpt-oss:20b \
        --max-chunks 10 \
        --workers "$PARALLEL" \
        --n-pairs 2 \
        2>&1 | tee "$RESULTS_DIR/parallel${PARALLEL}_workers${PARALLEL}.log"

    sleep 3
done

echo ""
echo "=============================================="
echo "Scalability test completed!"
echo "Results in: $RESULTS_DIR"
echo "=============================================="
