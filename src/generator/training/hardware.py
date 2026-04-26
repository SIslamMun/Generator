"""Detect runtime environment so the runner adapts to CPU / single GPU / SLURM."""
from __future__ import annotations

import os
import shutil
import subprocess


def detect() -> dict:
    info = {
        "cpu_count": os.cpu_count() or 1,
        "has_cuda": False,
        "gpu_name": None,
        "gpu_mem_gb": 0.0,
        "has_slurm": shutil.which("sbatch") is not None,
        "platform": os.uname().sysname.lower(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            name, mem = [s.strip() for s in line.split(",")]
            info["has_cuda"] = True
            info["gpu_name"] = name
            info["gpu_mem_gb"] = round(float(mem) / 1024, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return info


def summarize(info: dict) -> str:
    bits = []
    bits.append(
        f"GPU: {info['gpu_name']} ({info['gpu_mem_gb']} GB)"
        if info["has_cuda"] else "GPU: none (CPU-only)"
    )
    bits.append(f"CPU cores: {info['cpu_count']}")
    if info["has_slurm"]:
        bits.append("SLURM: detected")
    return " · ".join(bits)
