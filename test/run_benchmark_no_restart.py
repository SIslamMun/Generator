#!/usr/bin/env python3
"""
Ollama Benchmark - No Server Restart Version

Runs benchmark with current Ollama configuration.
Set OLLAMA_NUM_PARALLEL before starting Ollama to test different values.
"""

import json
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
REQUEST_TIMEOUT = 300

# Paths
SCRIPT_DIR = Path(__file__).parent
LONG_QUERIES_FILE = SCRIPT_DIR / "prompts" / "long_queries.txt"


def load_queries() -> list[str]:
    """Load queries from file."""
    content = LONG_QUERIES_FILE.read_text()
    return [q.strip() for q in content.split("---") if q.strip()]


def make_request(query: str) -> dict[str, Any]:
    """Make a single request to Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": MODEL, "prompt": query, "stream": False},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"    Error: {e}")
        return {"error": str(e)}


def run_sequential(queries: list[str]) -> dict[str, Any]:
    """Sequential test."""
    start = time.perf_counter()
    for i, query in enumerate(queries):
        make_request(query)
        print(f"    Completed {i + 1}/{len(queries)}")
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "total_time_sec": round(elapsed, 2),
        "time_per_query_sec": round(elapsed / len(queries), 2),
        "throughput_qps": round(len(queries) / elapsed, 3)
    }


def run_threadpool(queries: list[str], workers: int) -> dict[str, Any]:
    """Threadpool test."""
    completed = [0]
    lock = __import__('threading').Lock()

    def worker(query: str) -> dict:
        result = make_request(query)
        with lock:
            completed[0] += 1
            print(f"    Completed {completed[0]}/{len(queries)}")
        return result

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker, queries))
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "total_time_sec": round(elapsed, 2),
        "time_per_query_sec": round(elapsed / len(queries), 2),
        "throughput_qps": round(len(queries) / elapsed, 3)
    }


def print_table(results: list[dict]) -> None:
    """Print results table."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Experiment':<15}|{'Workers':<10}|{'Queries':<10}|{'Total(s)':<12}|{'Per Query(s)':<14}|{'Throughput':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment']:<15}|{r['workers']:<10}|{r['query_count']:<10}|{r['total_time_sec']:<12.2f}|{r['time_per_query_sec']:<14.2f}|{r['throughput_qps']:<10.3f} q/s")
    print("=" * 80)


def main():
    print("=" * 50)
    print("Ollama Benchmark (No Server Restart)")
    print("=" * 50)

    # Check connectivity
    try:
        r = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            print("ERROR: Ollama not responding")
            sys.exit(1)
    except:
        print("ERROR: Cannot connect to Ollama")
        sys.exit(1)
    print("[OK] Ollama connected")

    # Load queries
    queries = load_queries()
    print(f"[OK] Loaded {len(queries)} queries")

    # Test configurations
    configs = [
        ("sequential", 1, 5),
        ("threadpool", 2, 10),
        ("threadpool", 4, 16),
        ("threadpool", 6, 16),
        ("threadpool", 8, 16),
    ]

    results = []

    for exp_type, workers, n_queries in configs:
        test_queries = [queries[i % len(queries)] for i in range(n_queries)]

        print(f"\n--- {exp_type.upper()} (workers={workers}, queries={n_queries}) ---")

        if exp_type == "sequential":
            result = run_sequential(test_queries)
        else:
            result = run_threadpool(test_queries, workers)

        results.append({"experiment": exp_type, "workers": workers, **result})
        print(f"  Time: {result['total_time_sec']}s, Throughput: {result['throughput_qps']} q/s")

    print_table(results)

    # Save JSON
    with open("benchmark_results.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"\nSaved to benchmark_results.json")


if __name__ == "__main__":
    main()
