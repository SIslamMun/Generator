"""Generator package for synthetic QA pair generation from LanceDB."""

__version__ = "0.1.0"

# Core utilities (common across pipelines)
from .clients import get_client, BaseLLMClient
from .formatters import export_to_format
from .prompt_loader import load_prompts

# QA Pipeline
from .qa.qa_generator import generate_qa_from_lancedb
from .qa.curate import curate_qa_pairs
from .qa.enrich import enrich_qa_pairs, load_qa_pairs, save_qa_pairs
from .qa.compare import DatasetComparator, compare_datasets
from .qa.multi_scorer import (
    MultiDimensionalScorer,
    MultiScore,
    ScoreWeights,
    score_qa_pairs,
)

# CoT Pipeline
from .cot.cot_generator import generate_cot_pairs
from .cot.cot_enhancer import enhance_with_cot

# Tool Pipeline
from .tool.tool_schemas import Tool, Solution, ReasoningStep, ToolExample
from .tool.tool_generator import ToolGenerator
from .tool.tool_curator import ToolCurator
from .tool.tool_parser import ToolParser
from .tool.tool_executor import ToolExecutor
from .tool.coverage_selector import CoverageSelector, select_by_coverage
from .tool.dependency_graph import (
    DependencyGraph,
    DependencyEdge,
    ToolNode,
    build_graph_from_tools,
    is_type_compatible,
)
from .tool.outcome_evaluator import (
    OutcomeEvaluator,
    OutcomeEvaluation,
    OutcomeStatus,
    evaluate_tool_examples,
)

__all__ = [
    # Core
    "get_client",
    "BaseLLMClient",
    "export_to_format",
    "load_prompts",
    # QA Pipeline
    "generate_qa_from_lancedb",
    "curate_qa_pairs",
    "enrich_qa_pairs",
    "load_qa_pairs",
    "save_qa_pairs",
    "DatasetComparator",
    "compare_datasets",
    "MultiDimensionalScorer",
    "MultiScore",
    "ScoreWeights",
    "score_qa_pairs",
    # CoT Pipeline
    "generate_cot_pairs",
    "enhance_with_cot",
    # Tool Pipeline
    "Tool",
    "Solution",
    "ReasoningStep",
    "ToolExample",
    "ToolGenerator",
    "ToolCurator",
    "ToolParser",
    "ToolExecutor",
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
]
