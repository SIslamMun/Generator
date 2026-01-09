"""
Outcome-oriented evaluation for tool-use examples.

Based on MCP-AgentBench v2 (Sep 2025): Goes beyond execution success
to verify actual task completion and instruction satisfaction.

Paper: https://arxiv.org/abs/2509.09734
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .tool_schemas import ToolExample, Solution, ReasoningStep
from .clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


class OutcomeStatus(Enum):
    """Outcome evaluation status."""
    FULLY_SATISFIED = "fully_satisfied"  # Task completed correctly
    PARTIALLY_SATISFIED = "partially_satisfied"  # Some aspects addressed
    NOT_SATISFIED = "not_satisfied"  # Task not completed
    EXECUTION_FAILED = "execution_failed"  # Technical failure
    CANNOT_EVALUATE = "cannot_evaluate"  # Missing info to evaluate


@dataclass
class OutcomeEvaluation:
    """Result of outcome-oriented evaluation."""
    status: OutcomeStatus
    score: float  # 0.0 to 1.0
    reasoning: str
    instruction_understood: bool = True
    key_requirements: List[str] = field(default_factory=list)
    satisfied_requirements: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    execution_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "instruction_understood": self.instruction_understood,
            "key_requirements": self.key_requirements,
            "satisfied_requirements": self.satisfied_requirements,
            "missing_requirements": self.missing_requirements,
            "execution_issues": self.execution_issues,
        }


class OutcomeEvaluator:
    """
    Evaluate tool-use examples based on actual task completion.
    
    Unlike execution validation which only checks if tools run successfully,
    this evaluator verifies that the task was actually accomplished.
    
    Evaluation criteria:
    1. Instruction Understanding - Did the model understand what was asked?
    2. Requirement Coverage - Are all requirements addressed?
    3. Output Correctness - Do the outputs satisfy the request?
    4. Completeness - Is the solution complete (no missing steps)?
    """
    
    # Default prompts if not provided via config
    DEFAULT_PROMPTS = {
        "outcome_evaluation": """You are an expert evaluator assessing whether a tool-use solution correctly completes the user's task.

**User Instruction:**
{instruction}

**Solution Executed:**
{solution_summary}

**Final Output:**
{final_output}

**Evaluation Task:**
Analyze whether this solution ACTUALLY accomplishes what the user requested. Don't just check if tools executed - verify the task is DONE.

Consider:
1. Did the solution understand the instruction correctly?
2. What are the key requirements from the instruction?
3. Which requirements were satisfied by the solution?
4. Which requirements were NOT addressed?
5. Are there any execution issues that prevented completion?

Respond with JSON:
{{
    "instruction_understood": true/false,
    "key_requirements": ["requirement 1", "requirement 2", ...],
    "satisfied_requirements": ["satisfied 1", "satisfied 2", ...],
    "missing_requirements": ["missing 1", ...],
    "execution_issues": ["issue 1", ...],
    "overall_score": 0.0 to 1.0,
    "reasoning": "Brief explanation of the evaluation"
}}

Be STRICT. A score of 1.0 means ALL requirements were fully satisfied.
Return ONLY valid JSON.""",

        "requirement_extraction": """Extract the key requirements from this user instruction.

**Instruction:** {instruction}

List ALL the things the user wants done, in order of importance.
Be specific - include data types, formats, conditions mentioned.

Respond with JSON:
{{
    "requirements": [
        {{"requirement": "description", "priority": "high/medium/low"}},
        ...
    ]
}}

Return ONLY valid JSON."""
    }
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            llm_config: LLM configuration for evaluation
            prompts: Custom prompt templates
            strict_mode: If True, require all requirements to be satisfied
        """
        self.prompts = {**self.DEFAULT_PROMPTS, **(prompts or {})}
        self.strict_mode = strict_mode
        self.llm: Optional[BaseLLMClient] = None
        
        if llm_config:
            config = llm_config.copy()
            provider = config.pop("provider", "ollama")
            self.llm = get_client(provider, config)
    
    def _get_prompt(self, key: str) -> str:
        """Get prompt template."""
        return self.prompts.get(key, self.DEFAULT_PROMPTS.get(key, ""))
    
    def _summarize_solution(self, solution: Solution) -> str:
        """Create a summary of the solution execution."""
        lines = []
        for step in solution.reasoning_path:
            args_str = json.dumps(step.args) if step.args else "{}"
            result_str = ""
            if step.actual_result:
                result_str = f" -> {json.dumps(step.actual_result)[:200]}"
            status = f" [{step.status}]" if step.status else ""
            error = f" ERROR: {step.error_message}" if step.error_message else ""
            lines.append(f"Step {step.step}: {step.tool}({args_str}){result_str}{status}{error}")
        
        return "\n".join(lines)
    
    def evaluate_example(
        self,
        example: ToolExample,
    ) -> OutcomeEvaluation:
        """
        Evaluate a single example for task completion.
        
        Args:
            example: Tool example to evaluate
            
        Returns:
            OutcomeEvaluation with detailed results
        """
        # Check execution status first
        if example.execution_result and not example.execution_result.success:
            return OutcomeEvaluation(
                status=OutcomeStatus.EXECUTION_FAILED,
                score=0.0,
                reasoning="Execution failed - could not evaluate task completion",
                execution_issues=[str(example.execution_result.error) if example.execution_result.error else "Unknown execution error"],
            )
        
        # If no LLM, return cannot evaluate
        if not self.llm:
            return OutcomeEvaluation(
                status=OutcomeStatus.CANNOT_EVALUATE,
                score=0.5,
                reasoning="No LLM available for outcome evaluation",
            )
        
        # Build evaluation prompt
        solution_summary = self._summarize_solution(example.solution)
        final_output = example.solution.final_answer or (
            example.execution_result.output if example.execution_result else "No output"
        )
        
        prompt = self._get_prompt("outcome_evaluation").format(
            instruction=example.instruction,
            solution_summary=solution_summary,
            final_output=json.dumps(final_output, indent=2) if isinstance(final_output, (dict, list)) else str(final_output),
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.1)
            result = self._parse_json_response(response)
            
            if not result:
                return OutcomeEvaluation(
                    status=OutcomeStatus.CANNOT_EVALUATE,
                    score=0.5,
                    reasoning="Failed to parse evaluation response",
                )
            
            # Extract results
            score = float(result.get("overall_score", 0.5))
            instruction_understood = result.get("instruction_understood", True)
            key_requirements = result.get("key_requirements", [])
            satisfied = result.get("satisfied_requirements", [])
            missing = result.get("missing_requirements", [])
            issues = result.get("execution_issues", [])
            reasoning = result.get("reasoning", "")
            
            # Determine status based on score
            if score >= 0.9:
                status = OutcomeStatus.FULLY_SATISFIED
            elif score >= 0.5:
                status = OutcomeStatus.PARTIALLY_SATISFIED
            else:
                status = OutcomeStatus.NOT_SATISFIED
            
            # In strict mode, any missing requirement = not satisfied
            if self.strict_mode and missing:
                status = OutcomeStatus.NOT_SATISFIED
                score = min(score, 0.5)
            
            return OutcomeEvaluation(
                status=status,
                score=score,
                reasoning=reasoning,
                instruction_understood=instruction_understood,
                key_requirements=key_requirements,
                satisfied_requirements=satisfied,
                missing_requirements=missing,
                execution_issues=issues,
            )
            
        except Exception as e:
            logger.error(f"Outcome evaluation failed: {e}")
            return OutcomeEvaluation(
                status=OutcomeStatus.CANNOT_EVALUATE,
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
            )
    
    def evaluate_examples(
        self,
        examples: List[ToolExample],
        min_score: float = 0.0,
    ) -> List[tuple[ToolExample, OutcomeEvaluation]]:
        """
        Evaluate multiple examples.
        
        Args:
            examples: List of examples to evaluate
            min_score: Minimum score to include in results
            
        Returns:
            List of (example, evaluation) tuples
        """
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Evaluating outcomes...", 
                total=len(examples)
            )
            
            for example in examples:
                evaluation = self.evaluate_example(example)
                if evaluation.score >= min_score:
                    results.append((example, evaluation))
                progress.advance(task)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def filter_by_outcome(
        self,
        examples: List[ToolExample],
        min_score: float = 0.7,
        require_all_satisfied: bool = False,
    ) -> List[ToolExample]:
        """
        Filter examples by outcome evaluation.
        
        Args:
            examples: Examples to filter
            min_score: Minimum outcome score to keep
            require_all_satisfied: Require FULLY_SATISFIED status
            
        Returns:
            Filtered list of examples
        """
        evaluations = self.evaluate_examples(examples)
        
        filtered = []
        for example, eval_result in evaluations:
            if eval_result.score < min_score:
                continue
            if require_all_satisfied and eval_result.status != OutcomeStatus.FULLY_SATISFIED:
                continue
            
            # Attach evaluation to example metadata
            if not example.metadata:
                example.metadata = {}
            example.metadata["outcome_evaluation"] = eval_result.to_dict()
            
            filtered.append(example)
        
        console.print(f"[green]✓ Filtered to {len(filtered)}/{len(examples)} examples by outcome[/green]")
        
        return filtered
    
    def _print_summary(
        self,
        results: List[tuple[ToolExample, OutcomeEvaluation]],
    ) -> None:
        """Print evaluation summary table."""
        if not results:
            console.print("[yellow]No results to summarize[/yellow]")
            return
        
        # Count statuses
        status_counts = {}
        scores = []
        
        for _, eval_result in results:
            status_counts[eval_result.status.value] = status_counts.get(eval_result.status.value, 0) + 1
            scores.append(eval_result.score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        table = Table(title="Outcome Evaluation Summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        for status, count in sorted(status_counts.items()):
            pct = (count / len(results)) * 100
            table.add_row(status, str(count), f"{pct:.1f}%")
        
        table.add_row("", "", "")
        table.add_row("Total", str(len(results)), "100%")
        table.add_row("Avg Score", "", f"{avg_score:.2f}")
        
        console.print(table)
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json"):
                lines = lines[1:]
            elif lines[0] == "```":
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None
    
    def extract_requirements(
        self,
        instruction: str,
    ) -> List[Dict[str, str]]:
        """
        Extract key requirements from an instruction.
        
        Args:
            instruction: User instruction text
            
        Returns:
            List of requirements with priorities
        """
        if not self.llm:
            return [{"requirement": instruction, "priority": "high"}]
        
        prompt = self._get_prompt("requirement_extraction").format(
            instruction=instruction,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.1)
            result = self._parse_json_response(response)
            if result and "requirements" in result:
                return result["requirements"]
            return [{"requirement": instruction, "priority": "high"}]
        except Exception as e:
            logger.warning(f"Requirement extraction failed: {e}")
            return [{"requirement": instruction, "priority": "high"}]


def evaluate_tool_examples(
    examples: List[ToolExample],
    llm_config: Optional[Dict[str, Any]] = None,
    min_score: float = 0.7,
    prompts: Optional[Dict[str, str]] = None,
) -> List[ToolExample]:
    """
    Convenience function to filter examples by outcome evaluation.
    
    Args:
        examples: Examples to evaluate
        llm_config: LLM configuration
        min_score: Minimum score to keep
        prompts: Custom prompts
        
    Returns:
        Filtered examples that pass outcome evaluation
    """
    evaluator = OutcomeEvaluator(llm_config=llm_config, prompts=prompts)
    return evaluator.filter_by_outcome(examples, min_score=min_score)
