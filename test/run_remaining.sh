#!/bin/bash
set -e

OVERRIDE_FILE="/etc/systemd/system/ollama.service.d/override.conf"
PARALLELISM_VALUES=(20 24)

echo "Testing remaining OLLAMA_NUM_PARALLEL: ${PARALLELISM_VALUES[*]}"
echo "Queries = 5 * OLLAMA_NUM_PARALLEL"

for PARALLEL in "${PARALLELISM_VALUES[@]}"; do
    N_QUERIES=$((5 * PARALLEL))
    
    echo ""
    echo "=============================================="
    echo "Testing OLLAMA_NUM_PARALLEL=$PARALLEL ($N_QUERIES queries)"
    echo "=============================================="

    sudo tee "$OVERRIDE_FILE" > /dev/null << CONF
[Service]
Environment="OLLAMA_NUM_PARALLEL=$PARALLEL"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
CONF

    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    sleep 5
    
    for i in {1..30}; do
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break
        sleep 1
    done
    echo "Service ready"

    python3 -u << PYTHON
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import sys

sys.stdout.reconfigure(line_buffering=True)

MODEL = "gpt-oss:20b"
URL = "http://localhost:11434"
PARALLEL = $PARALLEL
N = $N_QUERIES

queries = [q.strip() for q in Path("prompts/long_queries.txt").read_text().split("---") if q.strip()]
test_queries = [queries[i % len(queries)] for i in range(N)]

def req(q):
    try:
        r = requests.post(f"{URL}/api/generate", json={"model": MODEL, "prompt": q, "stream": False}, timeout=600)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Sequential
print(f"  [SEQ] {N} queries sequentially...", flush=True)
t0 = time.perf_counter()
for i, q in enumerate(test_queries):
    req(q)
    print(f"      {i+1}/{N}", flush=True)
seq_time = time.perf_counter() - t0
print(f"      Seq Time: {seq_time:.1f}s, Throughput: {N/seq_time:.4f} q/s", flush=True)

time.sleep(2)

# Threadpool
print(f"  [TP]  {N} queries with {PARALLEL} workers...", flush=True)
done = [0]
lock = threading.Lock()
def work(q):
    r = req(q)
    with lock:
        done[0] += 1
        print(f"      {done[0]}/{N}", flush=True)
    return r

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
    list(ex.map(work, test_queries))
tp_time = time.perf_counter() - t0
print(f"      TP Time: {tp_time:.1f}s, Throughput: {N/tp_time:.4f} q/s", flush=True)

print(f"RESULT:{PARALLEL}:{N}:{seq_time:.2f}:{N/seq_time:.4f}:{tp_time:.2f}:{N/tp_time:.4f}", flush=True)
PYTHON

done

echo ""
echo "Remaining tests complete!"
