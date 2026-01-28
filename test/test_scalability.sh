#!/bin/bash
# QA Generator Scalability Test
# Tests different parallelism configurations similar to run_full_benchmark.sh

set -e

cd /home/cc/Generator

echo "=============================================="
echo "QA Generator Scalability Test"
echo "=============================================="
echo ""

# Test 1: Single instance with workers=4 (relies on OLLAMA_NUM_PARALLEL)
echo "Test 1: Single instance (http://localhost:11434) with 4 workers"
echo "  Ensure OLLAMA_NUM_PARALLEL=2 or 4 for best results"
echo ""

time uv run generator generate \
    --lancedb-path /home/cc/Jarvis_QA_Generator/lancedb \
    --table code_chunks \
    --output outputs/test_single_w4.json \
    --provider ollama \
    --model gpt-oss:20b \
    --max-chunks 20 \
    --workers 4 \
    --n-pairs 2

echo ""
echo "=============================================="
echo ""

# Test 2: Dual instances with workers=8 (load balanced)
echo "Test 2: Dual instances (load balanced) with 8 workers"
echo "  Check both instances are running:"
echo "    Instance 1: http://localhost:11434"
echo "    Instance 2: http://localhost:11435"
echo ""

# Check if second instance is running
if ! curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "⚠ Warning: Instance 2 (port 11435) not running"
    echo "  Start it with: OLLAMA_HOST=0.0.0.0:11435 ollama serve &"
    echo "  Skipping dual instance test..."
else
    time uv run generator generate \
        --lancedb-path /home/cc/Jarvis_QA_Generator/lancedb \
        --table code_chunks \
        --output outputs/test_dual_w8.json \
        --provider ollama \
        --model gpt-oss:20b \
        --ollama-instances "http://localhost:11434,http://localhost:11435" \
        --max-chunks 20 \
        --workers 8 \
        --n-pairs 2
fi

echo ""
echo "=============================================="
echo "Test completed!"
echo "Check outputs/ directory for results"
echo "=============================================="
