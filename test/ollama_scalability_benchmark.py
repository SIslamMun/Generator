#!/usr/bin/env python3
"""
Ollama Batch Performance Scalability Benchmark

Evaluates Ollama's batch performance scalability by testing OLLAMA_NUM_PARALLEL
values to find optimal parallelism before CPU/memory bottlenecks cause degradation.

Experiment Types:
- Sequential (6a): Single thread issues all queries, relies on Ollama's internal queue
- ThreadPool (6b): ThreadPoolExecutor where each thread waits for response before next
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Configuration
MODEL = "gpt-oss:20b"
OLLAMA_API_URL = "http://localhost:11434"
PARALLELISM_VALUES = [1, 2, 4, 6, 8, 10, 12]
QUERIES_PER_PARALLEL = 5
SERVER_STARTUP_TIMEOUT = 60  # seconds
REQUEST_TIMEOUT = 300  # seconds per request

# Paths to prompt files
SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
LONG_QUERIES_FILE = PROMPTS_DIR / "long_queries.txt"


def load_queries() -> list[str]:
    """Load queries from the long_queries.txt file (separated by ---)."""
    if not LONG_QUERIES_FILE.exists():
        raise RuntimeError(f"Queries file not found: {LONG_QUERIES_FILE}")

    content = LONG_QUERIES_FILE.read_text()
    queries = [q.strip() for q in content.split("---") if q.strip()]
    return queries


def wait_for_server_ready(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll Ollama server until it's ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def restart_ollama_with_parallel(num_parallel: int) -> bool:
    """Restart Ollama server with OLLAMA_NUM_PARALLEL env var."""
    print(f"  Stopping existing Ollama process...")
    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    time.sleep(3)

    print(f"  Starting Ollama with OLLAMA_NUM_PARALLEL={num_parallel}...")
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)

    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"  Waiting for server to be ready...")
    if wait_for_server_ready():
        print(f"  Server ready.")
        return True
    else:
        print(f"  ERROR: Server failed to start within {SERVER_STARTUP_TIMEOUT}s")
        return False


def make_ollama_request(query: str) -> dict[str, Any]:
    """Make a single request to Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": query,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"    Request error: {e}")
        return {"error": str(e)}


def run_sequential(queries: list[str]) -> dict[str, Any]:
    """Run sequential experiment - single thread issues all queries."""
    start = time.perf_counter()

    for i, query in enumerate(queries):
        make_ollama_request(query)
        print(f"    Completed query {i + 1}/{len(queries)}")

    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "total_time_sec": round(elapsed, 2),
        "time_per_query_sec": round(elapsed / len(queries), 2),
        "throughput_qps": round(len(queries) / elapsed, 3)
    }


def run_threadpool(queries: list[str], num_workers: int) -> dict[str, Any]:
    """Run threadpool experiment - multiple threads, each waits for response."""
    completed = [0]
    lock = __import__('threading').Lock()

    def worker(query: str) -> dict:
        result = make_ollama_request(query)
        with lock:
            completed[0] += 1
            print(f"    Completed query {completed[0]}/{len(queries)}")
        return result

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(worker, queries))

    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "total_time_sec": round(elapsed, 2),
        "time_per_query_sec": round(elapsed / len(queries), 2),
        "throughput_qps": round(len(queries) / elapsed, 3)
    }


def verify_ollama_connectivity() -> bool:
    """Verify Ollama server is reachable."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def verify_model_available(model: str) -> bool:
    """Verify the specified model is available."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            model_full_names = [m.get("name", "") for m in models]
            return model in model_names or model in model_full_names or model.split(":")[0] in model_names
        return False
    except requests.RequestException:
        return False


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    header = (
        f"{'Parallelism':^12}|{'Experiment':^12}|{'Queries':^8}|"
        f"{'Total Time (s)':^16}|{'Time/Query (s)':^16}|{'Throughput':^14}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        row = (
            f"{r['parallelism']:^12}|{r['experiment']:^12}|{r['query_count']:^8}|"
            f"{r['total_time_sec']:^16.2f}|{r['time_per_query_sec']:^16.2f}|"
            f"{r['throughput_qps']:^12.3f} q/s"
        )
        print(row)

    print("=" * 90)


def save_results_json(results: list[dict[str, Any]], filename: str = "ollama_benchmark_results.json") -> None:
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "parallelism_values": PARALLELISM_VALUES,
        "queries_per_parallel": QUERIES_PER_PARALLEL,
        "results": results
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    print("=" * 60)
    print("Ollama Batch Performance Scalability Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Parallelism values: {PARALLELISM_VALUES}")
    print(f"Queries per parallelism: {QUERIES_PER_PARALLEL} x parallelism")
    print()

    # Pre-flight checks
    print("Running pre-flight checks...")

    # Check Ollama is installed
    result = subprocess.run(["which", "ollama"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: Ollama is not installed or not in PATH")
        sys.exit(1)
    print("  [OK] Ollama is installed")

    # Check server connectivity
    if not verify_ollama_connectivity():
        print("  Ollama server not running, attempting to start...")
        if not restart_ollama_with_parallel(1):
            print("ERROR: Could not start Ollama server")
            sys.exit(1)
    print("  [OK] Ollama server is reachable")

    # Check model availability
    if not verify_model_available(MODEL):
        print(f"ERROR: Model '{MODEL}' is not available")
        print("  Run: ollama pull " + MODEL)
        sys.exit(1)
    print(f"  [OK] Model '{MODEL}' is available")

    # Test single request
    print("  Testing single request...")
    test_response = make_ollama_request("Hello, respond with just 'OK'.")
    if "error" in test_response:
        print(f"ERROR: Test request failed: {test_response['error']}")
        sys.exit(1)
    print("  [OK] Test request successful")

    print("\nPre-flight checks passed. Starting benchmark...\n")

    # Load queries
    print("Loading queries...")
    queries = load_queries()
    print(f"  Loaded {len(queries)} queries from {LONG_QUERIES_FILE.name}")

    # Print query length stats
    lengths = [len(q) for q in queries]
    print(f"  Query lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")

    results = []

    # Run experiments for each parallelism value
    for parallel in PARALLELISM_VALUES:
        print(f"\n{'=' * 60}")
        print(f"Testing OLLAMA_NUM_PARALLEL={parallel}")
        print("=" * 60)

        # Restart server with new parallelism
        if not restart_ollama_with_parallel(parallel):
            print(f"ERROR: Failed to restart server with parallelism={parallel}")
            continue

        # Select queries: n = 5 * parallelism (cycle through if needed)
        n_queries = QUERIES_PER_PARALLEL * parallel
        test_queries = []
        for i in range(n_queries):
            test_queries.append(queries[i % len(queries)])

        # Run sequential experiment (6a)
        print(f"\n[6a] Running SEQUENTIAL test with {n_queries} queries...")
        seq_result = run_sequential(test_queries)
        results.append({
            "parallelism": parallel,
            "experiment": "sequential",
            **seq_result
        })
        print(f"  Total time: {seq_result['total_time_sec']}s, "
              f"Throughput: {seq_result['throughput_qps']} q/s")

        # Small pause between experiments
        time.sleep(2)

        # Run threadpool experiment (6b)
        print(f"\n[6b] Running THREADPOOL test with {n_queries} queries, {parallel} workers...")
        tp_result = run_threadpool(test_queries, parallel)
        results.append({
            "parallelism": parallel,
            "experiment": "threadpool",
            **tp_result
        })
        print(f"  Total time: {tp_result['total_time_sec']}s, "
              f"Throughput: {tp_result['throughput_qps']} q/s")

    # Output results
    print_results_table(results)
    save_results_json(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
