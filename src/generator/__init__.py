"""Generator package for synthetic QA pair generation from LanceDB."""

__version__ = "0.1.0"

from .clients import get_client, BaseLLMClient
from .qa_generator import generate_qa_from_lancedb
from .curate import curate_qa_pairs
from .formatters import export_to_format
from .coverage_selector import CoverageSelector, select_by_coverage
from .dependency_graph import (
    DependencyGraph,
    DependencyEdge,
    ToolNode,
    build_graph_from_tools,
    is_type_compatible,
)
from .outcome_evaluator import (
    OutcomeEvaluator,
    OutcomeEvaluation,
    OutcomeStatus,
    evaluate_tool_examples,
)
from .multi_scorer import (
    MultiDimensionalScorer,
    MultiScore,
    ScoreWeights,
    score_qa_pairs,
)

__all__ = [
    "get_client",
    "BaseLLMClient",
    "generate_qa_from_lancedb",
    "curate_qa_pairs",
    "export_to_format",
    "CoverageSelector",
    "select_by_coverage",
    "DependencyGraph",
    "DependencyEdge",
    "ToolNode",
    "build_graph_from_tools",
    "is_type_compatible",
    "OutcomeEvaluator",
    "OutcomeEvaluation",
    "OutcomeStatus",
    "evaluate_tool_examples",
    "MultiDimensionalScorer",
    "MultiScore",
    "ScoreWeights",
    "score_qa_pairs",
]
