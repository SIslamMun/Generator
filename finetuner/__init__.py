"""finetuner — multi-backend Phase 6 fine-tuning.

A thin dispatcher over three back-ends — Unsloth, HuggingFace, Ollama —
sharing one parameter set (:class:`~finetuner.config.FinetuneConfig`).
See issue grc-iit/Phagocyte#4 and README.md.
"""

from .config import BACKENDS, FinetuneConfig

__all__ = ["FinetuneConfig", "BACKENDS"]
