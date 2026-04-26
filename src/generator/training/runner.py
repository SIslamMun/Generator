"""Stage execution loop.

Each stage declares its inputs/outputs; the runner skips stages whose outputs
already exist and are newer than every input (idempotent). On failure it stops.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

console = Console()


@dataclass
class Stage:
    name: str
    description: str
    inputs:  list[Path]
    outputs: list[Path]
    fn:      Callable[[dict], None]   # called as fn(cfg) — receives the full pipeline config
    skip_if_disabled_path: Optional[str] = None  # e.g. "chat.enabled"

    def is_disabled(self, cfg: dict) -> bool:
        if not self.skip_if_disabled_path:
            return False
        cur: object = cfg
        for k in self.skip_if_disabled_path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return False
            cur = cur[k]
        return cur is False

    def needs_run(self) -> bool:
        if not all(p.exists() for p in self.outputs):
            return True
        last_input  = max((p.stat().st_mtime for p in self.inputs  if p.exists()), default=0)
        first_output = min((p.stat().st_mtime for p in self.outputs), default=0)
        return last_input > first_output


def run_stages(stages: list[Stage], cfg: dict, *, force: bool = False,
               start_from: Optional[str] = None, only: Optional[str] = None,
               state_path: Optional[Path] = None) -> dict:
    """Execute stages in order. Returns a status dict written to state_path."""
    status: dict = {"started_at": time.time(), "stages": {}}
    started = start_from is None

    for i, st in enumerate(stages, 1):
        if only and st.name != only:
            continue
        if not started and st.name == start_from:
            started = True
        if not started:
            continue

        if st.is_disabled(cfg):
            status["stages"][st.name] = {"status": "disabled"}
            console.print(f"[dim]\\[{i}/{len(stages)}] {st.name:<22} disabled (skipped)[/dim]")
            continue

        if not force and not st.needs_run():
            status["stages"][st.name] = {"status": "cached"}
            console.print(f"[green]\\[{i}/{len(stages)}] {st.name:<22} ✓ cached[/green]   ({', '.join(str(p) for p in st.outputs)})")
            continue

        console.rule(f"[bold cyan]\\[{i}/{len(stages)}] {st.name}[/bold cyan] — {st.description}")
        t0 = time.time()
        try:
            st.fn(cfg)
            dt = time.time() - t0
            status["stages"][st.name] = {"status": "ok", "elapsed_s": round(dt, 1)}
            console.print(f"[green]\\[{i}/{len(stages)}] {st.name:<22} ✓ done[/green]   ({dt:.1f}s)")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            status["stages"][st.name] = {"status": "failed", "error": str(e), "traceback": tb}
            console.print(f"[bold red]\\[{i}/{len(stages)}] {st.name:<22} ✗ FAILED[/bold red]")
            console.print(tb)
            if state_path:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(status, indent=2))
            raise

    status["finished_at"] = time.time()
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(status, indent=2))
    return status
