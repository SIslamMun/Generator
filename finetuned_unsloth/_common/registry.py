"""Model registry — discovers every model folder under finetuned_unsloth/models/.

A model folder is recognized iff it contains a `config.yaml` AND a
`train.py`. Each registered model exposes:
  - name          : the folder basename (CLI-friendly slug)
  - display_name  : from config.yaml.display_name
  - hf_model_id   : from config.yaml.hf_model_id
  - path          : Path to the folder
  - has_prepare   : whether prepare_data.py is present
  - has_validate  : whether validate_data.py is present
  - has_install   : whether install.sh is present
  - has_sbatch    : whether submit.sbatch is present

Intended as the single source of truth for any CLI/UI that needs to
enumerate available models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:                       # registry should work even before venv exists
    yaml = None


MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


@dataclass
class ModelEntry:
    name:         str
    path:         Path
    display_name: str
    hf_model_id:  str
    family:       str
    has_prepare:  bool
    has_validate: bool
    has_install:  bool
    has_sbatch:   bool


def _read_minimal(cfg_path: Path) -> dict:
    """Read just enough of config.yaml to populate the registry entry.

    Falls back to a regex extract if pyyaml isn't importable yet (e.g. the
    model's dedicated venv hasn't been created — but we still want to list it).
    """
    if yaml is not None:
        try:
            return yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            pass
    # Cheap fallback: grep for top-level keys we care about.
    import re
    txt = cfg_path.read_text()
    out: dict = {}
    for key in ("display_name", "hf_model_id", "family"):
        m = re.search(rf'^\s*{key}\s*:\s*["\']?([^"\'\n]+)', txt, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


def discover() -> list[ModelEntry]:
    """Return every model folder with a config.yaml + train.py, sorted by name."""
    if not MODELS_ROOT.exists():
        return []
    entries: list[ModelEntry] = []
    for child in sorted(MODELS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
            continue
        cfg = child / "config.yaml"
        trn = child / "train.py"
        if not (cfg.exists() and trn.exists()):
            continue
        meta = _read_minimal(cfg)
        entries.append(ModelEntry(
            name         = child.name,
            path         = child,
            display_name = meta.get("display_name") or child.name,
            hf_model_id  = meta.get("hf_model_id")  or "?",
            family       = meta.get("family")       or "?",
            has_prepare  = (child / "prepare_data.py").exists(),
            has_validate = (child / "validate_data.py").exists(),
            has_install  = (child / "install.sh").exists(),
            has_sbatch   = (child / "submit.sbatch").exists(),
        ))
    return entries


def get(name: str) -> ModelEntry | None:
    """Get a single model entry by folder name, or None if not present."""
    for e in discover():
        if e.name == name:
            return e
    return None


def print_table(entries: Iterable[ModelEntry] | None = None) -> None:
    """Print a human-readable table of registered models."""
    entries = list(entries) if entries is not None else discover()
    if not entries:
        print("(no models registered — see finetuned_unsloth/README.md)")
        return
    headers = ["name", "display_name", "hf_model_id", "prepare?", "validate?", "install?", "sbatch?"]
    rows = [[
        e.name, e.display_name, e.hf_model_id,
        "✓" if e.has_prepare else "✗",
        "✓" if e.has_validate else "✗",
        "✓" if e.has_install else "✗",
        "✓" if e.has_sbatch else "✗",
    ] for e in entries]
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


if __name__ == "__main__":
    print_table()
