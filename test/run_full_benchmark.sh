#!/bin/bash
# Full Ollama Scalability Benchmark
# Tests different OLLAMA_NUM_PARALLEL values by restarting the service

set -e

OVERRIDE_FILE="/etc/systemd/system/ollama.service.d/override.conf"
RESULTS_FILE="full_benchmark_results.txt"
JSON_FILE="full_benchmark_results.json"

# Parallelism values to test
PARALLELISM_VALUES=(1 2 4 6 8 10 12)

# Queries per parallelism multiplier
QUERIES_MULTIPLIER=5

echo "=============================================="
echo "Ollama Full Scalability Benchmark"
echo "=============================================="
echo "Testing OLLAMA_NUM_PARALLEL values: ${PARALLELISM_VALUES[*]}"
echo ""

# Initialize results file
echo "timestamp: $(date -Iseconds)" > "$RESULTS_FILE"
echo "results:" >> "$RESULTS_FILE"

# Initialize JSON
echo '{"timestamp": "'$(date -Iseconds)'", "results": [' > "$JSON_FILE"

FIRST_RESULT=true

for PARALLEL in "${PARALLELISM_VALUES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Testing OLLAMA_NUM_PARALLEL=$PARALLEL"
    echo "=============================================="

    # Update override.conf
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

    # Wait for service to be ready
    echo "Waiting for service to be ready..."
    sleep 5
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Service ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Calculate number of queries
    N_QUERIES=$((QUERIES_MULTIPLIER * PARALLEL))

    echo ""
    echo "Running benchmark with $N_QUERIES queries..."

    # Run Python benchmark for this parallelism level
    python3 << PYTHON_SCRIPT
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import threading

MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434"
PARALLEL = $PARALLEL
N_QUERIES = $N_QUERIES

# Load queries
queries_file = Path("prompts/long_queries.txt")
content = queries_file.read_text()
all_queries = [q.strip() for q in content.split("---") if q.strip()]
queries = [all_queries[i % len(all_queries)] for i in range(N_QUERIES)]

def make_request(query):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": MODEL, "prompt": query, "stream": False},
                         timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Sequential test
print(f"  [6a] Sequential test ({N_QUERIES} queries)...")
start = time.perf_counter()
for i, q in enumerate(queries):
    make_request(q)
    print(f"      Completed {i+1}/{N_QUERIES}")
seq_time = time.perf_counter() - start
seq_throughput = N_QUERIES / seq_time
print(f"      Time: {seq_time:.2f}s, Throughput: {seq_throughput:.4f} q/s")

time.sleep(2)

# Threadpool test
print(f"  [6b] Threadpool test ({N_QUERIES} queries, {PARALLEL} workers)...")
completed = [0]
lock = threading.Lock()

def worker(q):
    result = make_request(q)
    with lock:
        completed[0] += 1
        print(f"      Completed {completed[0]}/{N_QUERIES}")
    return result

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
    list(executor.map(worker, queries))
tp_time = time.perf_counter() - start
tp_throughput = N_QUERIES / tp_time
print(f"      Time: {tp_time:.2f}s, Throughput: {tp_throughput:.4f} q/s")

# Output results for shell script to capture
print(f"RESULT_SEQ:{PARALLEL}:{N_QUERIES}:{seq_time:.2f}:{seq_throughput:.4f}")
print(f"RESULT_TP:{PARALLEL}:{N_QUERIES}:{tp_time:.2f}:{tp_throughput:.4f}")
PYTHON_SCRIPT

done

# Close JSON array
echo ']}' | sed 's/,\]/]/' >> "$JSON_FILE"

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
