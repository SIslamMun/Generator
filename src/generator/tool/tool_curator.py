"""
Tool example curation using LLM-as-Judge.

Implements quality filtering based on:
- ToolACE: Dual-layer verification
- APIGen: Format + Execution + Semantic checks
- ToolMind (Nov 2025): Turn-level filtering for step quality
"""

import json
import json5
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .tool_schemas import Tool, ToolExample
from ..clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


class ToolCurator:
    """Curate tool-calling training examples."""
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize curator.
        
        Args:
            llm_config: LLM configuration for rating
            prompts: Prompt templates (loaded from configs/prompts/tool_prompts.yaml)
        """
        self.prompts = prompts or {}
        self.llm: Optional[BaseLLMClient] = None
        
        if llm_config:
            provider = llm_config.pop("provider", "ollama")
            self.llm = get_client(provider, llm_config)
    
    def _get_prompt(self, key: str) -> str:
        """Get prompt template, raising error if not found."""
        if key not in self.prompts:
            raise ValueError(
                f"Missing prompt template '{key}'. "
                f"Add it to configs/prompts/tool_prompts.yaml"
            )
        return self.prompts[key]
    
    def filter_by_execution(
        self,
        examples: List[ToolExample],
        min_success_rate: float = 1.0,
    ) -> List[ToolExample]:
        """
        Filter to successfully executed examples.
        
        Args:
            examples: List of tool examples
            min_success_rate: Minimum success rate (1.0 = all steps must succeed, 0 = skip filter)
            
        Returns:
            Filtered list of examples
        """
        # Skip execution filter if min_success_rate is 0
        if min_success_rate <= 0:
            console.print(
                f"[cyan]Execution filter: skipped (min_success_rate=0)[/cyan]"
            )
            return examples
        
        filtered = []
        
        for example in examples:
            if not example.execution_result:
                continue
            
            if min_success_rate >= 1.0:
                # Strict: all must succeed
                if example.execution_result.success:
                    filtered.append(example)
            else:
                # Partial: check step success rate
                step_results = example.execution_result.step_results or []
                if not step_results:
                    continue
                success_count = sum(1 for s in step_results if s.get("success"))
                rate = success_count / len(step_results)
                if rate >= min_success_rate:
                    filtered.append(example)
        
        console.print(
            f"[cyan]Execution filter: {len(filtered)}/{len(examples)} passed "
            f"(min_success_rate={min_success_rate})[/cyan]"
        )
        return filtered
    
    def filter_by_turn_quality(
        self,
        examples: List[ToolExample],
        min_step_quality: float = 0.7,
    ) -> List[ToolExample]:
        """
        Filter examples by individual turn/step quality (ToolMind approach).
        
        Unlike filter_by_execution which only checks final success,
        this method rates each reasoning step independently to catch
        cases where bad intermediate steps compound errors.
        
        Args:
            examples: List of tool examples
            min_step_quality: Minimum average quality per step (0-1)
            
        Returns:
            Examples where all steps meet quality threshold
        """
        if not self.llm:
            logger.warning("No LLM configured for turn-level rating, skipping")
            return examples
        
        filtered = []
        failed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Turn-level quality check...", 
                total=len(examples)
            )
            
            for example in examples:
                step_ratings = self._rate_individual_steps(example)
                
                if not step_ratings:
                    progress.advance(task)
                    continue
                
                # Check if all steps meet minimum quality
                avg_quality = sum(step_ratings) / len(step_ratings)
                all_pass = all(r >= min_step_quality for r in step_ratings)
                
                if all_pass:
                    # Store step ratings in metadata
                    example.metadata["step_ratings"] = step_ratings
                    example.metadata["avg_step_quality"] = avg_quality
                    filtered.append(example)
                else:
                    # Log which steps failed for analysis
                    failed_steps = [
                        i + 1 for i, r in enumerate(step_ratings) 
                        if r < min_step_quality
                    ]
                    logger.debug(
                        f"Example failed turn filter at steps: {failed_steps}"
                    )
                    failed_count += 1
                
                progress.advance(task)
        
        console.print(
            f"[cyan]Turn-level filter: {len(filtered)}/{len(examples)} passed "
            f"(min_step_quality={min_step_quality})[/cyan]"
        )
        if failed_count > 0:
            console.print(
                f"  [dim]({failed_count} examples had problematic intermediate steps)[/dim]"
            )
        
        return filtered
    
    def _rate_individual_steps(self, example: ToolExample) -> List[float]:
        """
        Rate each reasoning step independently.
        
        Args:
            example: Tool example with reasoning path
            
        Returns:
            List of ratings (0-1) for each step
        """
        if not example.solution.reasoning_path:
            return []
        
        ratings = []
        prompt_template = self._get_prompt("step_quality_rating")
        
        for i, step in enumerate(example.solution.reasoning_path):
            # Build context from previous steps
            previous_context = self._get_previous_context(example, i)
            
            prompt = prompt_template.format(
                instruction=example.instruction,
                step_number=i + 1,
                total_steps=len(example.solution.reasoning_path),
                thought=step.thought,
                tool=step.tool,
                args=json.dumps(step.args, indent=2),
                result=json.dumps(step.actual_result) if step.actual_result else "N/A",
                previous_context=previous_context,
            )
            
            try:
                response = self.llm.generate(prompt, temperature=0.1)
                result = self._parse_json_response(response)
                
                if result and isinstance(result, dict):
                    rating = result.get("rating", 0.5)
                    # Ensure rating is in 0-1 range
                    rating = max(0.0, min(1.0, float(rating)))
                    ratings.append(rating)
                else:
                    ratings.append(0.5)  # Default to neutral if parsing fails
                    
            except Exception as e:
                logger.warning(f"Failed to rate step {i + 1}: {e}")
                ratings.append(0.5)
        
        return ratings
    
    def _get_previous_context(self, example: ToolExample, current_idx: int) -> str:
        """Build context string from previous steps."""
        if current_idx == 0:
            return "This is the first step."
        
        context_parts = []
        for i in range(current_idx):
            step = example.solution.reasoning_path[i]
            context_parts.append(
                f"Step {i + 1}: Used {step.tool}({json.dumps(step.args)}) → "
                f"{json.dumps(step.actual_result) if step.actual_result else 'pending'}"
            )
        
        return "\n".join(context_parts)
    
    def rate_examples(
        self,
        examples: List[ToolExample],
        threshold: float = 7.0,
    ) -> List[ToolExample]:
        """
        Rate examples using LLM-as-Judge and filter by threshold.
        
        Args:
            examples: List of tool examples
            threshold: Minimum rating to keep (1-10 scale)
            
        Returns:
            Filtered and rated examples
        """
        if not self.llm:
            logger.warning("No LLM configured for rating, returning all examples")
            return examples
        
        rated = []
        prompt_template = self._get_prompt("tool_quality_rating")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Rating examples...", total=len(examples))
            
            for example in examples:
                rating_result = self._rate_example(example, prompt_template)
                
                if rating_result:
                    example.rating = rating_result.get("rating", 0)
                    example.rating_breakdown = {
                        "tool_selection": rating_result.get("tool_selection", 0),
                        "parameter_correctness": rating_result.get("parameter_correctness", 0),
                        "reasoning_quality": rating_result.get("reasoning_quality", 0),
                        "result_usefulness": rating_result.get("result_usefulness", 0),
                    }
                    
                    if example.rating >= threshold:
                        rated.append(example)
                
                progress.advance(task)
        
        console.print(
            f"[cyan]Quality filter: {len(rated)}/{len(examples)} passed "
            f"(threshold={threshold})[/cyan]"
        )
        return rated
    
    def _rate_example(
        self,
        example: ToolExample,
        prompt_template: str,
    ) -> Optional[Dict[str, Any]]:
        """Rate a single example."""
        # Build tool calls summary
        tool_calls = []
        for step in example.solution.reasoning_path:
            tool_calls.append({
                "step": step.step,
                "thought": step.thought,
                "tool": step.tool,
                "args": step.args,
                "result": step.actual_result,
            })
        
        # Get execution result
        exec_result = None
        if example.execution_result:
            exec_result = {
                "success": example.execution_result.success,
                "output": example.execution_result.output,
            }
        
        prompt = prompt_template.format(
            instruction=example.instruction,
            tool_calls=json.dumps(tool_calls, indent=2),
            result=json.dumps(exec_result, indent=2) if exec_result else "N/A",
            answer=example.solution.final_answer,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.1)
            return self._parse_json_response(response)
        except Exception as e:
            logger.warning(f"Failed to rate example: {e}")
            return None
    
    def balance_difficulty(
        self,
        examples: List[ToolExample],
        distribution: Optional[Dict[str, float]] = None,
    ) -> List[ToolExample]:
        """
        Balance examples by difficulty distribution.
        
        Args:
            examples: List of tool examples
            distribution: Target distribution {"simple": 0.3, "medium": 0.5, "complex": 0.2}
            
        Returns:
            Balanced list of examples
        """
        if distribution is None:
            distribution = {"simple": 0.3, "medium": 0.5, "complex": 0.2}
        
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for example in examples:
            difficulty = example.metadata.get("difficulty", "medium")
            by_difficulty[difficulty].append(example)
        
        total_target = len(examples)
        balanced = []
        
        for difficulty, target_ratio in distribution.items():
            target_count = int(total_target * target_ratio)
            available = by_difficulty.get(difficulty, [])
            
            # Sort by rating if available
            if any(e.rating for e in available):
                available.sort(key=lambda e: e.rating or 0, reverse=True)
            
            selected = available[:target_count]
            balanced.extend(selected)
        
        console.print(f"[cyan]Balanced to {len(balanced)} examples[/cyan]")
        for diff, exs in by_difficulty.items():
            selected_count = len([e for e in balanced if e.metadata.get("difficulty") == diff])
            console.print(f"  {diff}: {selected_count}/{len(exs)}")
        
        return balanced
    
    def ensure_coverage(
        self,
        examples: List[ToolExample],
        tools: List[Tool],
        min_per_tool: int = 10,
    ) -> List[ToolExample]:
        """
        Ensure minimum examples per tool.
        
        Args:
            examples: List of tool examples
            tools: List of tool definitions
            min_per_tool: Minimum examples per tool
            
        Returns:
            List with coverage ensured (may be same or filtered)
        """
        tool_ids = {t.tool_id for t in tools}
        
        # Count examples per tool
        tool_counts = defaultdict(list)
        for example in examples:
            used_tools = set()
            for step in example.solution.reasoning_path:
                if step.tool in tool_ids or any(step.tool == t.name for t in tools):
                    used_tools.add(step.tool)
            
            for tool in used_tools:
                tool_counts[tool].append(example)
        
        # Check coverage
        under_covered = []
        for tool in tools:
            count = len(tool_counts.get(tool.tool_id, [])) + len(tool_counts.get(tool.name, []))
            if count < min_per_tool:
                under_covered.append((tool, count))
        
        if under_covered:
            console.print("[yellow]Warning: Some tools under minimum coverage:[/yellow]")
            for tool, count in under_covered:
                console.print(f"  {tool.name}: {count}/{min_per_tool}")
        else:
            console.print(f"[green]✓ All tools have at least {min_per_tool} examples[/green]")
        
        return examples
    
    def deduplicate(
        self,
        examples: List[ToolExample],
        similarity_threshold: float = 0.9,
    ) -> List[ToolExample]:
        """
        Remove duplicate or near-duplicate examples.
        
        Args:
            examples: List of tool examples
            similarity_threshold: Threshold for considering duplicates
            
        Returns:
            Deduplicated list
        """
        if not examples:
            return examples
        
        # Simple deduplication by instruction
        seen_instructions = set()
        unique = []
        
        for example in examples:
            # Normalize instruction
            normalized = example.instruction.lower().strip()
            
            if normalized not in seen_instructions:
                seen_instructions.add(normalized)
                unique.append(example)
        
        console.print(
            f"[cyan]Deduplication: {len(unique)}/{len(examples)} unique[/cyan]"
        )
        return unique
    
    def curate(
        self,
        examples: List[ToolExample],
        tools: List[Tool],
        min_success_rate: float = 1.0,
        rating_threshold: float = 7.0,
        balance: bool = True,
        min_per_tool: int = 10,
        deduplicate: bool = True,
        turn_level_filter: bool = False,
        min_step_quality: float = 0.7,
    ) -> List[ToolExample]:
        """
        Full curation pipeline.
        
        Args:
            examples: Raw examples
            tools: Tool definitions
            min_success_rate: Execution success threshold
            rating_threshold: Quality rating threshold
            balance: Whether to balance difficulty
            min_per_tool: Minimum examples per tool
            deduplicate: Whether to remove duplicates
            turn_level_filter: Whether to apply turn-level quality filtering
            min_step_quality: Minimum quality per step (0-1) if turn_level_filter is True
            
        Returns:
            Curated examples
        """
        console.print(f"\n[bold]Starting curation of {len(examples)} examples[/bold]\n")
        
        # Step 1: Filter by execution
        curated = self.filter_by_execution(examples, min_success_rate)
        
        # Step 2: Turn-level quality filter (ToolMind approach)
        if turn_level_filter and self.llm:
            curated = self.filter_by_turn_quality(curated, min_step_quality)
        
        # Step 3: Rate and filter
        if self.llm:
            curated = self.rate_examples(curated, rating_threshold)
        
        # Step 4: Deduplicate
        if deduplicate:
            curated = self.deduplicate(curated)
        
        # Step 5: Balance difficulty
        if balance:
            curated = self.balance_difficulty(curated)
        
        # Step 6: Check coverage
        curated = self.ensure_coverage(curated, tools, min_per_tool)
        
        console.print(f"\n[green]✓ Curation complete: {len(curated)} examples[/green]")
        return curated
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response."""
        text = response.strip()
        
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        try:
            return json5.loads(text)
        except:
            pass
        
        try:
            return json.loads(text)
        except:
            pass
        
        import re
        json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if json_match:
            try:
                return json5.loads(json_match.group())
            except:
                pass
        
        return None


def format_check(example: ToolExample) -> List[str]:
    """
    Rule-based format validation (ToolACE layer 1).
    
    Args:
        example: Tool example to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check instruction
    if not example.instruction or len(example.instruction) < 10:
        errors.append("Instruction too short or empty")
    
    # Check solution
    if not example.solution.reasoning_path:
        errors.append("No reasoning steps")
    
    for step in example.solution.reasoning_path:
        if not step.tool:
            errors.append(f"Step {step.step}: Missing tool name")
        if not isinstance(step.args, dict):
            errors.append(f"Step {step.step}: Args must be a dict")
    
    # Check final answer
    if not example.solution.final_answer:
        errors.append("Missing final answer")
    
    return errors


def execution_check(example: ToolExample) -> bool:
    """
    Check if execution was successful (ToolACE layer 2).
    
    Args:
        example: Tool example to check
        
    Returns:
        True if execution succeeded
    """
    if not example.execution_result:
        return False
    return example.execution_result.success
