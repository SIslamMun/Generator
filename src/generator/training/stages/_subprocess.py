"""Helper: run a subprocess streamed to stdout, raise on non-zero exit."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run(cmd: Sequence[str], *, cwd: str | Path | None = None,
        log_path: Path | None = None) -> None:
    """Run cmd. Tee output to stdout AND log_path if given."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    log_fh = open(log_path, "w") if log_path else None
    proc = subprocess.Popen(
        list(cmd), cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_fh:
            log_fh.write(line)
            log_fh.flush()
    proc.wait()
    if log_fh:
        log_fh.close()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")
