"""Backend interface + shared dataset loading.

Every fine-tune backend implements :class:`Backend` — given a
:class:`~finetuner.config.FinetuneConfig`, it produces checkpoints under
``config.output_dir`` and returns a small result dict.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import FinetuneConfig


class Backend(ABC):
    """A fine-tuning backend (Unsloth / HuggingFace / Ollama)."""

    name: str = "base"

    @abstractmethod
    def run(self, cfg: FinetuneConfig) -> dict[str, Any]:
        """Execute the job. Returns a summary dict (also written to disk)."""
        raise NotImplementedError


def load_conversations(dataset_path: str) -> list[dict]:
    """Load a training dataset into a list of conversation rows.

    Accepts JSON (a list) or JSONL (one object per line). Each row is
    expected to carry a ``conversations`` or ``messages`` list; rows of
    plain ``{question, answer}`` (optionally ``reasoning``) are converted
    to a user/assistant conversation so older QA dumps still train.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    # Content-based, not extension-based: a `.json` file can hold JSONL
    # (e.g. a ChatML export written to a .json path), so try a whole-file
    # JSON parse first and fall back to line-by-line JSONL.
    raw: list[dict]
    text = path.read_text()
    try:
        loaded = json.loads(text)
        raw = loaded if isinstance(loaded, list) else loaded.get("examples", [])
    except json.JSONDecodeError:
        raw = [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    rows: list[dict] = []
    for r in raw:
        if "conversations" in r:
            rows.append({"conversations": r["conversations"],
                         **({"tools": r["tools"]} if "tools" in r else {})})
        elif "messages" in r:
            rows.append({"conversations": r["messages"]})
        elif "question" in r and "answer" in r:
            # plain QA (optionally with CoT reasoning) → conversation
            answer = r["answer"]
            if r.get("reasoning"):
                answer = f"<think>\n{r['reasoning']}\n</think>\n\n{answer}"
            rows.append({"conversations": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": answer},
            ]})
    if not rows:
        raise ValueError(
            f"no usable rows in {dataset_path} — expected `conversations`, "
            "`messages`, or `question`/`answer` fields"
        )
    return rows


def write_summary(cfg: FinetuneConfig, summary: dict[str, Any]) -> Path:
    """Write a train_summary.json into the output dir; return its path."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path
