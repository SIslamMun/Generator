"""Generator package for synthetic QA pair generation from LanceDB."""

__version__ = "0.1.0"

from .clients import get_client, BaseLLMClient
from .qa_generator import generate_qa_from_lancedb
from .curate import curate_qa_pairs
from .formatters import export_to_format

__all__ = [
    "get_client",
    "BaseLLMClient",
    "generate_qa_from_lancedb",
    "curate_qa_pairs",
    "export_to_format",
]
