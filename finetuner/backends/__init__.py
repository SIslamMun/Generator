"""Fine-tune backends — Unsloth / HuggingFace / Ollama.

`get_backend(name)` returns the backend instance for a config's
``backend`` field. Backend modules import their heavy deps lazily, so
importing this package is cheap.
"""

from __future__ import annotations

from .base import Backend

_REGISTRY = {
    "unsloth": ("finetuner.backends.unsloth_backend", "UnslothBackend"),
    "hf": ("finetuner.backends.hf_backend", "HFBackend"),
    "ollama": ("finetuner.backends.ollama_backend", "OllamaBackend"),
}


def get_backend(name: str) -> Backend:
    """Instantiate the backend registered under ``name``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown backend '{name}' (choose one of {sorted(_REGISTRY)})"
        )
    module_path, class_name = _REGISTRY[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


__all__ = ["Backend", "get_backend"]
