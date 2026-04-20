"""
Tool use training data generator.

Unified approach that combines best practices from:
- Toolformer: Single-step tool calls
- Gorilla: API documentation grounding (always included)
- ToolLLM: Multi-step reasoning with chains
- ToolGrad (2025): Chain-first generation (valid chains → synthesize queries)

Two generation approaches:
- query_first (traditional): Generate instructions → annotate solutions
- chain_first (ToolGrad): Generate valid chains → synthesize natural queries

Three modes for solution complexity:
- single: Simple single-tool calls
- multi: Multi-step reasoning with tool chains
- auto (default): Generates balanced mix based on instruction complexity

v7 additions:
- Subset selection: Each example sees N tools (target + distractors) instead of full catalog
- Checkpointing: Intermediate saves every N examples with resume support
"""

import json
import json5
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import numpy as np

from .tool_schemas import Tool, Solution, ReasoningStep, ToolExample, save_examples, load_examples
from ..clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


# =========================================================================
# CHECKPOINT HELPERS
# =========================================================================

def _save_checkpoint(examples: List[ToolExample], path: str) -> None:
    """Save intermediate checkpoint."""
    save_examples(examples, path)
    console.print(f"[dim]  💾 Checkpoint: {len(examples)} examples → {path}[/dim]")


def _load_checkpoint(path: str) -> List[ToolExample]:
    """Load intermediate checkpoint if it exists."""
    p = Path(path)
    if p.exists():
        try:
            examples = load_examples(str(p))
            console.print(f"[yellow]⏩ Resuming from checkpoint: {len(examples)} examples in {p.name}[/yellow]")
            return examples
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
    return []


def _checkpoint_path(output_path: str) -> str:
    """Derive checkpoint path from output path."""
    p = Path(output_path)
    return str(p.parent / f"{p.stem}_intermediate{p.suffix}")


def _is_valid_example(example: ToolExample, valid_tools: set, min_instruction_len: int = 15) -> bool:
    """
    Validate a generated example before accepting it.

    Rejects examples with:
    - Too-short instructions
    - Empty reasoning path
    - Steps with missing/invalid tool names
    """
    if not example.instruction or len(example.instruction) < min_instruction_len:
        return False
    if not example.solution.reasoning_path:
        return False
    for step in example.solution.reasoning_path:
        if not step.tool or step.tool not in valid_tools:
            return False
    return True


# =========================================================================
# SUBSET / DISTRACTOR SELECTION (v7)
# =========================================================================

def _compute_tool_embeddings(tools: List[Tool]) -> np.ndarray:
    """Compute sentence embeddings for tool descriptions (lazy, cached)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        docs = [f"{t.name}: {t.description}" for t in tools]
        return model.encode(docs, show_progress_bar=False)
    except ImportError:
        logger.warning("sentence-transformers not installed; falling back to random distractors")
        return None


def pick_subset(
    all_tools: List[Tool],
    target_tools: List[Tool],
    n: int = 10,
    strategy: str = "mixed",
    tool_embeddings: Optional[np.ndarray] = None,
) -> List[Tool]:
    """
    Pick a subset of tools for a training example (v7 subset selection).

    Args:
        all_tools: Full tool catalog
        target_tools: Tools that MUST be in the subset (used in the solution)
        n: Total subset size (default 10)
        strategy: "semantic" (hard distractors), "random", or "mixed" (60/40)
        tool_embeddings: Pre-computed embeddings for semantic strategy

    Returns:
        Shuffled list of n tools including all targets
    """
    if n >= len(all_tools):
        subset = list(all_tools)
        random.shuffle(subset)
        return subset

    # Pin targets
    subset = list(target_tools)
    target_ids = {t.tool_id for t in target_tools}
    remaining = [t for t in all_tools if t.tool_id not in target_ids]
    need = n - len(subset)

    if need <= 0:
        random.shuffle(subset)
        return subset[:n]

    # Decide strategy for this call
    if strategy == "mixed":
        use_semantic = random.random() < 0.6
    else:
        use_semantic = strategy == "semantic"

    if use_semantic and tool_embeddings is not None:
        # Semantic: pick distractors most similar to target
        tool_id_list = [t.tool_id for t in all_tools]
        target_indices = [tool_id_list.index(t.tool_id) for t in target_tools if t.tool_id in tool_id_list]
        remaining_indices = [tool_id_list.index(t.tool_id) for t in remaining]

        if target_indices and remaining_indices:
            # Average embedding of targets
            target_emb = tool_embeddings[target_indices].mean(axis=0)
            remaining_embs = tool_embeddings[remaining_indices]
            sims = (remaining_embs @ target_emb) / (
                np.linalg.norm(remaining_embs, axis=1) * np.linalg.norm(target_emb) + 1e-8
            )
            # Pick top-N most similar
            top_idx = np.argsort(sims)[-need:][::-1]
            distractors = [remaining[i] for i in top_idx]
        else:
            distractors = random.sample(remaining, min(need, len(remaining)))
    else:
        # Random
        distractors = random.sample(remaining, min(need, len(remaining)))

    subset.extend(distractors)
    random.shuffle(subset)
    return subset


class ToolGenerator:
    """Generate tool-use training data with unified approach."""

    def __init__(self, llm_config: Dict[str, Any], prompts: Dict[str, str],
                 tools_per_example: int = 0, distractor_strategy: str = "mixed"):
        """
        Initialize generator.

        Args:
            llm_config: LLM configuration with provider and settings
            prompts: Prompt templates dict (loaded from configs/prompts/tool_prompts.yaml)
            tools_per_example: Number of tools visible per example (0 = all tools, v7 default: 10)
            distractor_strategy: "semantic", "random", or "mixed" (60% semantic / 40% random)
        """
        self.prompts = prompts
        provider = llm_config.pop("provider", "ollama")
        self.llm = get_client(provider, llm_config)
        self.provider = provider
        self.tools_per_example = tools_per_example
        self.distractor_strategy = distractor_strategy
        self._tool_embeddings: Optional[np.ndarray] = None
    
    def _get_prompt(self, key: str) -> str:
        """Get prompt template, raising error if not found."""
        if key not in self.prompts:
            raise ValueError(
                f"Missing prompt template '{key}'. "
                f"Add it to configs/prompts/tool_prompts.yaml"
            )
        return self.prompts[key]

    def _ensure_tool_embeddings(self, tools: List[Tool]) -> None:
        """Lazily compute and cache tool embeddings for subset selection."""
        if self._tool_embeddings is None and self.tools_per_example > 0:
            console.print("[dim]Computing tool embeddings for subset selection...[/dim]")
            self._tool_embeddings = _compute_tool_embeddings(tools)

    def _get_visible_tools(self, all_tools: List[Tool], target_tools: List[Tool]) -> List[Tool]:
        """Get the tool subset visible for one training example."""
        if self.tools_per_example <= 0 or self.tools_per_example >= len(all_tools):
            return all_tools
        return pick_subset(
            all_tools, target_tools,
            n=self.tools_per_example,
            strategy=self.distractor_strategy,
            tool_embeddings=self._tool_embeddings,
        )
    
    def generate_instructions(
        self,
        tools: List[Tool],
        n_per_tool: int = 10,
        include_multi_tool: bool = True,
        workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse user instructions for tools.
        
        Args:
            tools: List of tool definitions
            n_per_tool: Instructions to generate per tool
            include_multi_tool: Whether to generate multi-tool instructions
            workers: Number of parallel workers (1=sequential)
            
        Returns:
            List of instruction dicts with metadata
        """
        all_instructions = []
        
        prompt_template = self._get_prompt("tool_instruction_generation")
        
        if workers == 1:
            # Sequential processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Generating instructions...", 
                    total=len(tools)
                )
                
                for tool in tools:
                    tool_instructions = self._generate_for_tool(tool, n_per_tool, prompt_template)
                    all_instructions.extend(tool_instructions)
                    progress.advance(task)
        else:
            # Parallel processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Generating instructions...", 
                    total=len(tools)
                )
                
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for tool in tools:
                        future = executor.submit(
                            self._generate_for_tool,
                            tool, n_per_tool, prompt_template
                        )
                        futures[future] = tool.name
                    
                    for future in as_completed(futures):
                        tool_name = futures[future]
                        try:
                            tool_instructions = future.result()
                            all_instructions.extend(tool_instructions)
                            progress.advance(task)
                        except Exception as e:
                            logger.warning(f"Failed for tool {tool_name}: {e}")
                            progress.advance(task)
        
        # Generate multi-tool instructions if enabled
        if include_multi_tool and len(tools) > 1:
            console.print("[cyan]Generating multi-tool instructions...[/cyan]")
            multi_instructions = self._generate_multi_tool_instructions(tools, n_per_tool // 2)
            all_instructions.extend(multi_instructions)
        
        console.print(f"[green]✓ Generated {len(all_instructions)} instructions[/green]")
        return all_instructions
    
    def _generate_for_tool(
        self,
        tool: Tool,
        n: int,
        prompt_template: str,
    ) -> List[Dict[str, Any]]:
        """Generate instructions for a single tool."""
        # Use wiggle room like QA/CoT generators (adaptive based on tool complexity)
        n_min = n
        n_max = min(n * 2, 20)  # Cap at 20 max for tools
        
        prompt = prompt_template.format(
            tool_name=tool.name,
            tool_description=tool.description,
            parameters=json.dumps([p.to_dict() for p in tool.parameters], indent=2),
            examples=json.dumps(tool.examples, indent=2) if tool.examples else "[]",
            n_instructions_min=n_min,
            n_instructions_max=n_max,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.8)
            instructions = self._parse_json_response(response)
            
            # Add metadata
            for inst in instructions:
                inst["required_tools"] = [tool.tool_id]
                inst["multi_tool"] = False
                # Auto-detect if single or multi-step based on difficulty
                inst["mode"] = "single" if inst.get("difficulty") == "simple" else "auto"
            
            return instructions[:n]
        except Exception as e:
            logger.warning(f"Failed to generate for {tool.name}: {e}")
            return []
    
    def _generate_multi_tool_instructions(
        self,
        tools: List[Tool],
        n: int,
    ) -> List[Dict[str, Any]]:
        """Generate instructions requiring multiple tools."""
        # Group tools by category
        categories = {}
        for tool in tools:
            cat = tool.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
        
        instructions = []
        
        # Generate intra-category multi-tool
        for cat, cat_tools in categories.items():
            if len(cat_tools) >= 2:
                for i in range(min(n // len(categories), len(cat_tools) - 1)):
                    tool_subset = cat_tools[i:i+2]
                    inst = self._generate_multi_tool_for_subset(tool_subset)
                    if inst:
                        inst["category"] = cat
                        inst["instruction_type"] = "multi-tool-intra-category"
                        instructions.append(inst)
        
        # Generate cross-category
        if len(categories) >= 2:
            cat_list = list(categories.keys())
            for i in range(min(n // 2, len(cat_list) - 1)):
                tools_subset = [
                    categories[cat_list[i]][0],
                    categories[cat_list[i+1]][0],
                ]
                inst = self._generate_multi_tool_for_subset(tools_subset)
                if inst:
                    inst["instruction_type"] = "multi-tool-cross-category"
                    instructions.append(inst)
        
        return instructions
    
    def _generate_multi_tool_for_subset(self, tools: List[Tool]) -> Optional[Dict[str, Any]]:
        """Generate a multi-tool instruction for a subset of tools."""
        tools_desc = "\n\n".join([
            f"Tool: {t.name}\nDescription: {t.description}\nParameters: {json.dumps([p.to_dict() for p in t.parameters])}"
            for t in tools
        ])
        
        prompt_template = self._get_prompt("multi_tool_instruction_generation")
        prompt = prompt_template.format(tools_documentation=tools_desc)
        
        try:
            response = self.llm.generate(prompt, temperature=0.9)
            result = self._parse_json_response(response)
            if isinstance(result, list):
                result = result[0] if result else None
            if result:
                result["required_tools"] = [t.tool_id for t in tools]
                result["multi_tool"] = True
                result["mode"] = "multi"  # Multi-tool always uses multi-step
            return result
        except Exception as e:
            logger.warning(f"Failed to generate multi-tool instruction: {e}")
            return None
    
    def annotate_solution(
        self,
        instruction: str,
        tools: List[Tool],
        mode: str = "auto",
        max_steps: int = 5,
        all_tools: Optional[List[Tool]] = None,
    ) -> Solution:
        """
        Generate a solution for an instruction.

        Always includes API documentation for better grounding.
        When tools_per_example > 0, selects a subset of tools visible to the model.

        Args:
            instruction: User instruction to solve
            tools: Target tools (used in the solution)
            mode: 'single' (one tool call), 'multi' (chain), or 'auto' (detect)
            max_steps: Maximum reasoning steps for multi mode
            all_tools: Full tool catalog (for subset selection; if None, uses tools)

        Returns:
            Solution object with reasoning path
        """
        # Subset selection (v7): pick visible tools for this example
        if all_tools and self.tools_per_example > 0:
            visible = self._get_visible_tools(all_tools, tools)
        else:
            visible = tools

        # Always include documentation (Gorilla insight)
        docs = "\n\n".join([t.to_documentation() for t in visible])
        tools_json = json.dumps([t.to_schema() for t in visible], indent=2)
        
        # Auto-detect mode based on instruction complexity
        if mode == "auto":
            mode = self._detect_complexity(instruction, tools)
        
        if mode == "single":
            return self._annotate_single(instruction, tools, docs)
        else:
            return self._annotate_multi(instruction, tools, docs, tools_json, max_steps)
    
    def _detect_complexity(self, instruction: str, tools: List[Tool]) -> str:
        """Auto-detect if instruction needs single or multi-step solution."""
        # Simple heuristics
        complexity_indicators = [
            "then", "after", "next", "also", "and then",
            "first", "second", "finally", "both", "combine",
            "use the result", "based on", "followed by"
        ]
        
        instruction_lower = instruction.lower()
        
        # Check for chaining keywords
        for indicator in complexity_indicators:
            if indicator in instruction_lower:
                return "multi"
        
        # Check if multiple tools are mentioned
        tool_names = [t.name.lower() for t in tools]
        matches = sum(1 for name in tool_names if name in instruction_lower)
        if matches >= 2:
            return "multi"
        
        return "single"
    
    def _annotate_single(
        self, 
        instruction: str, 
        tools: List[Tool],
        docs: str,
    ) -> Solution:
        """Single-step annotation with documentation grounding."""
        prompt_template = self._get_prompt("tool_solution_single")
        prompt = prompt_template.format(
            api_documentation=docs,
            instruction=instruction,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            result = self._parse_json_response(response)
            if isinstance(result, list):
                result = result[0]
            
            step = ReasoningStep(
                step=1,
                thought=result.get("thought", ""),
                tool=result.get("tool", ""),
                args=result.get("args", {}),
                expected_result=result.get("expected_result"),
            )
            
            return Solution(
                instruction=instruction,
                reasoning_path=[step],
                final_answer=result.get("final_answer", ""),
                api_documentation=docs,
                method="single",
            )
        except Exception as e:
            logger.error(f"Single-step annotation failed: {e}")
            return Solution(instruction=instruction, reasoning_path=[], final_answer="", method="single")
    
    def _annotate_multi(
        self, 
        instruction: str, 
        tools: List[Tool],
        docs: str,
        tools_json: str,
        max_steps: int,
    ) -> Solution:
        """Multi-step annotation with documentation and chaining."""
        prompt_template = self._get_prompt("tool_solution_multi")
        prompt = prompt_template.format(
            api_documentation=docs,
            instruction=instruction,
            tools_json=tools_json,
            max_steps=max_steps,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.5)
            result = self._parse_json_response(response)
            
            if isinstance(result, list):
                result = {"reasoning_path": result, "final_answer": ""}
            
            steps = []
            for i, step_data in enumerate(result.get("reasoning_path", [])):
                steps.append(ReasoningStep(
                    step=step_data.get("step", i + 1),
                    thought=step_data.get("thought", ""),
                    tool=step_data.get("tool", ""),
                    args=step_data.get("args", {}),
                    expected_result=step_data.get("expected_result"),
                ))
            
            return Solution(
                instruction=instruction,
                reasoning_path=steps,
                final_answer=result.get("final_answer", ""),
                api_documentation=docs,
                method="multi",
            )
        except Exception as e:
            logger.error(f"Multi-step annotation failed: {e}")
            return Solution(instruction=instruction, reasoning_path=[], final_answer="", method="multi")
    
    def generate_examples(
        self,
        tools: List[Tool],
        n_per_tool: int = 10,
        mode: str = "auto",
        max_steps: int = 5,
        _accumulator: Optional[List[ToolExample]] = None,
        _save_every: int = 0,
        _checkpoint_file: str = "",
    ) -> List[ToolExample]:
        """
        Generate complete tool-use examples (instructions + solutions).

        Args:
            tools: Tool definitions
            n_per_tool: Examples per tool
            mode: 'single', 'multi', or 'auto' (balanced mix)
            max_steps: Max reasoning steps for multi-step
            _accumulator: Shared list to append to (for checkpoint support)
            _save_every: Save checkpoint every N examples (0 = disabled)
            _checkpoint_file: Path for checkpoint file

        Returns:
            List of ToolExample objects
        """
        self._ensure_tool_embeddings(tools)

        # First generate instructions
        instructions = self.generate_instructions(tools, n_per_tool)

        # Then annotate solutions
        examples = []
        tool_map = {t.tool_id: t for t in tools}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Annotating solutions...",
                total=len(instructions)
            )

            for inst_data in instructions:
                instruction = inst_data.get("instruction", "")
                required = inst_data.get("required_tools", [])

                # Determine mode for this instruction
                if mode == "auto":
                    inst_mode = inst_data.get("mode", "auto")
                else:
                    inst_mode = mode

                # Get target tools
                relevant_tools = [tool_map[tid] for tid in required if tid in tool_map]
                if not relevant_tools:
                    relevant_tools = tools[:3]  # Fallback

                solution = self.annotate_solution(
                    instruction, relevant_tools, inst_mode, max_steps,
                    all_tools=tools,
                )

                if solution.reasoning_path:  # Only keep non-empty solutions
                    example = ToolExample(
                        instruction=instruction,
                        solution=solution,
                        metadata={
                            "difficulty": inst_data.get("difficulty", "medium"),
                            "scenario": inst_data.get("scenario", ""),
                            "required_tools": required,
                            "multi_tool": inst_data.get("multi_tool", False),
                            "mode": solution.method,
                        }
                    )
                    valid_tool_names = {t.name for t in tools}
                    if _is_valid_example(example, valid_tool_names):
                        examples.append(example)
                        if _accumulator is not None:
                            _accumulator.append(example)
                            if _save_every > 0 and _checkpoint_file and len(_accumulator) % _save_every == 0:
                                _save_checkpoint(_accumulator, _checkpoint_file)

                progress.advance(task)

        console.print(f"[green]✓ Generated {len(examples)} examples[/green]")
        return examples
    
    # =========================================================================
    # CHAIN-FIRST GENERATION (ToolGrad Aug 2025)
    # Generate valid tool chains first, then synthesize natural queries
    # Reduces invalid samples by 40%+ vs query-first approach
    # =========================================================================
    
    def generate_chain_first(
        self,
        tools: List[Tool],
        n_chains: int = 20,
        min_steps: int = 2,
        max_steps: int = 4,
        _accumulator: Optional[List[ToolExample]] = None,
        _save_every: int = 0,
        _checkpoint_file: str = "",
    ) -> List[ToolExample]:
        """
        Chain-first generation: build valid tool chains, then synthesize queries.

        Based on ToolGrad (Aug 2025): https://arxiv.org/abs/2508.04086

        Args:
            tools: List of tool definitions
            n_chains: Number of chains to generate
            min_steps: Minimum tools per chain
            max_steps: Maximum tools per chain
            _accumulator: Shared list for checkpoint support
            _save_every: Save checkpoint every N examples (0 = disabled)
            _checkpoint_file: Path for checkpoint file

        Returns:
            List of ToolExample objects with valid chains
        """
        self._ensure_tool_embeddings(tools)
        console.print(f"\n[bold cyan]Chain-First Generation (ToolGrad)[/bold cyan]")
        console.print(f"[dim]Building {n_chains} valid chains ({min_steps}-{max_steps} steps)...[/dim]\n")

        examples = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Generating chains...", total=n_chains)

            generated = 0
            attempts = 0
            max_attempts = n_chains * 3

            while generated < n_chains and attempts < max_attempts:
                attempts += 1

                # Step 1: Generate a valid tool chain
                chain = self._generate_valid_chain(tools, min_steps, max_steps)
                if not chain or not chain.get("steps"):
                    continue

                # Identify tools used in the chain for subset selection
                chain_tool_names = {s.get("tool", "") for s in chain.get("steps", [])}
                target_tools = [t for t in tools if t.name in chain_tool_names]
                visible = self._get_visible_tools(tools, target_tools) if target_tools else tools
                docs = "\n\n".join([t.to_documentation() for t in visible])

                # Step 2: Synthesize a natural query for this chain
                query = self._synthesize_query_for_chain(chain, docs)
                if not query:
                    continue

                # Step 3: Build the ToolExample
                steps = []
                for i, step_data in enumerate(chain.get("steps", [])):
                    steps.append(ReasoningStep(
                        step=i + 1,
                        thought=step_data.get("thought", ""),
                        tool=step_data.get("tool", ""),
                        args=step_data.get("args", {}),
                        expected_result=step_data.get("expected_result"),
                    ))

                solution = Solution(
                    instruction=query,
                    reasoning_path=steps,
                    final_answer=chain.get("final_answer", "The task is complete."),
                    api_documentation=docs,
                    method="chain_first",
                )

                example = ToolExample(
                    instruction=query,
                    solution=solution,
                    metadata={
                        "generation_method": "chain_first",
                        "chain_length": len(steps),
                        "tools_used": [s.tool for s in steps],
                        "difficulty": "complex" if len(steps) >= 3 else "medium",
                    }
                )

                valid_tool_names = {t.name for t in tools}
                if not _is_valid_example(example, valid_tool_names):
                    continue

                examples.append(example)
                if _accumulator is not None:
                    _accumulator.append(example)
                    if _save_every > 0 and _checkpoint_file and len(_accumulator) % _save_every == 0:
                        _save_checkpoint(_accumulator, _checkpoint_file)
                generated += 1
                progress.advance(task)

        success_rate = generated / max(attempts, 1) * 100
        console.print(f"\n[green]✓ Generated {generated} chain-first examples[/green]")
        console.print(f"[dim]Success rate: {success_rate:.1f}% ({generated}/{attempts})[/dim]")

        return examples
    
    def _generate_valid_chain(
        self,
        tools: List[Tool],
        min_steps: int,
        max_steps: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a valid tool chain with proper data flow.

        Creates chains where:
        - Each step's output can feed into subsequent steps
        - Tools are used in a logical sequence
        - Arguments reference previous results correctly
        """
        # Subset selection: pick a random subset for chain generation
        if self.tools_per_example > 0 and self.tools_per_example < len(tools):
            visible = random.sample(tools, min(self.tools_per_example, len(tools)))
        else:
            visible = tools

        tools_json = json.dumps([{
            "name": t.name,
            "description": t.description,
            "parameters": [p.to_dict() for p in t.parameters],
            "returns": t.returns,
        } for t in visible], indent=2)
        
        prompt_template = self._get_prompt("chain_generation")
        prompt = prompt_template.format(
            tools_json=tools_json,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.7)
            result = self._parse_json_response(response)
            
            if isinstance(result, list):
                result = {"steps": result}
            
            # Validate chain has required structure
            if not result.get("steps") or len(result["steps"]) < min_steps:
                return None
                
            return result
        except Exception as e:
            logger.debug(f"Chain generation failed: {e}")
            return None
    
    def _synthesize_query_for_chain(
        self,
        chain: Dict[str, Any],
        docs: str,
    ) -> Optional[str]:
        """
        Synthesize a natural user query that would require this chain.
        
        Takes a valid chain and creates a natural language request
        that a user might realistically make.
        """
        chain_summary = []
        for i, step in enumerate(chain.get("steps", [])):
            chain_summary.append(f"{i+1}. {step.get('tool', 'unknown')}({step.get('args', {})})")
        
        prompt_template = self._get_prompt("query_synthesis")
        prompt = prompt_template.format(
            chain_steps="\n".join(chain_summary),
            tools_used=", ".join([s.get("tool", "") for s in chain.get("steps", [])]),
            final_result=chain.get("final_answer", "task completed"),
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.6)
            
            # Extract the query from response
            result = self._parse_json_response(response)
            if isinstance(result, dict):
                return result.get("query") or result.get("instruction")
            elif isinstance(result, str):
                return result.strip()
            
            # Fallback: use the raw response if it looks like a query
            if response and len(response) < 500 and "?" in response or "please" in response.lower():
                return response.strip()
                
            return None
        except Exception as e:
            logger.debug(f"Query synthesis failed: {e}")
            return None
    
    # =========================================================================
    # ERROR-RECOVERY GENERATION
    # One-shot full examples from the catalog (no seed file required).
    # =========================================================================

    def generate_error_recovery_examples(
        self,
        tools: List[Tool],
        n: int = 10,
        _accumulator: Optional[List[ToolExample]] = None,
        _save_every: int = 0,
        _checkpoint_file: str = "",
    ) -> List[ToolExample]:
        """
        Generate error-recovery examples where step 1 fails with a realistic
        error and subsequent steps show recovery.
        """
        self._ensure_tool_embeddings(tools)
        console.print(f"\n[bold]Error-Recovery Generation[/bold]")
        console.print(f"[dim]Target: {n} examples[/dim]\n")

        prompt_template = self._get_prompt("tool_error_recovery_full")

        examples: List[ToolExample] = []
        attempts = 0
        max_attempts = n * 3

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Generating error-recovery...", total=n)

            while len(examples) < n and attempts < max_attempts:
                attempts += 1

                # Subset selection: pick a random subset for this error-recovery scenario
                if self.tools_per_example > 0 and self.tools_per_example < len(tools):
                    visible = random.sample(tools, min(self.tools_per_example, len(tools)))
                else:
                    visible = tools

                docs = "\n\n".join([t.to_documentation() for t in visible])
                tools_json = json.dumps([t.to_schema() for t in visible], indent=2)

                prompt = prompt_template.format(
                    api_documentation=docs,
                    tools_json=tools_json,
                )
                try:
                    response = self.llm.generate(prompt, temperature=0.8)
                    result = self._parse_json_response(response)
                except Exception as e:
                    logger.debug(f"error-recovery gen attempt failed: {e}")
                    continue

                if isinstance(result, list):
                    result = result[0] if result else None
                if not result or not result.get("reasoning_path"):
                    continue

                instruction = result.get("instruction", "")
                if not instruction:
                    continue

                steps = []
                for i, sd in enumerate(result["reasoning_path"]):
                    steps.append(ReasoningStep(
                        step=sd.get("step", i + 1),
                        thought=sd.get("thought", ""),
                        tool=sd.get("tool", ""),
                        args=sd.get("args", {}),
                        expected_result=sd.get("expected_result"),
                        actual_result=sd.get("actual_result"),
                        status=sd.get("status", "pending"),
                        error_message=sd.get("error_message"),
                    ))

                if not any(s.status == "failure" for s in steps):
                    continue

                solution = Solution(
                    instruction=instruction,
                    reasoning_path=steps,
                    final_answer=result.get("final_answer", ""),
                    api_documentation=docs,
                    method="error_recovery",
                )

                md = result.get("metadata", {}) or {}
                example = ToolExample(
                    instruction=instruction,
                    solution=solution,
                    metadata={
                        "generation_method": "error_recovery",
                        "error_category": md.get("error_category", "unknown"),
                        "difficulty": md.get("difficulty", "complex"),
                    },
                )
                valid_tool_names = {t.name for t in tools}
                if not _is_valid_example(example, valid_tool_names):
                    continue
                examples.append(example)
                if _accumulator is not None:
                    _accumulator.append(example)
                    if _save_every > 0 and _checkpoint_file and len(_accumulator) % _save_every == 0:
                        _save_checkpoint(_accumulator, _checkpoint_file)
                progress.update(task, completed=len(examples))

        console.print(f"[green]✓ Generated {len(examples)} error-recovery examples[/green]")
        return examples

    # =========================================================================
    # FULL-MIX GENERATION
    # One call → balanced corpus across single / multi / chain / error.
    # Designed for a single long run (e.g. on Delta-AI with a local model).
    # =========================================================================

    def generate_full_mix(
        self,
        tools: List[Tool],
        target_pairs: int = 100,
        ratio_single: float = 0.30,
        ratio_multi: float = 0.30,
        ratio_chain: float = 0.25,
        ratio_error: float = 0.15,
        max_steps: int = 5,
        output_path: str = "",
        save_every: int = 100,
    ) -> List[ToolExample]:
        """
        Generate a balanced mix of single, multi, chain-first, and
        error-recovery examples in one run.

        Supports checkpointing and resume:
        - Saves intermediate results every `save_every` examples
        - On resume, detects existing checkpoint and continues from there
        - Each section's progress is tracked so partial sections resume correctly

        Args:
            tools: Tool definitions
            target_pairs: Total examples to generate
            ratio_*: Category ratios (auto-normalized)
            max_steps: Max steps per multi/chain example
            output_path: Final output file (checkpoint derives from this)
            save_every: Save checkpoint every N examples (0 = disabled)
        """
        # Normalize ratios
        total_ratio = ratio_single + ratio_multi + ratio_chain + ratio_error
        if total_ratio <= 0:
            raise ValueError("Ratios must sum to a positive number")
        ratio_single /= total_ratio
        ratio_multi  /= total_ratio
        ratio_chain  /= total_ratio
        ratio_error  /= total_ratio

        n_single = int(target_pairs * ratio_single)
        n_multi  = int(target_pairs * ratio_multi)
        n_chain  = int(target_pairs * ratio_chain)
        n_error  = int(target_pairs * ratio_error)
        leftover = target_pairs - (n_single + n_multi + n_chain + n_error)
        n_single += max(0, leftover)

        console.print(f"\n[bold]🎯 Full-Mix Tool Generation[/bold]")
        console.print(f"[dim]Target: {target_pairs} examples[/dim]")
        console.print(f"[dim]  single: {n_single}  multi: {n_multi}  chain: {n_chain}  error: {n_error}[/dim]")
        if self.tools_per_example > 0:
            console.print(f"[dim]  tools_per_example: {self.tools_per_example}  distractor_strategy: {self.distractor_strategy}[/dim]")
        console.print()

        # ── Checkpoint / resume logic ───────────────────────────────
        ckpt_file = _checkpoint_path(output_path) if output_path else ""
        all_examples: List[ToolExample] = []

        # Count how many of each category we already have from checkpoint
        done_single = done_multi = done_chain = done_error = 0
        if ckpt_file:
            resumed = _load_checkpoint(ckpt_file)
            if resumed:
                all_examples = resumed
                for ex in resumed:
                    method = ex.solution.method
                    if method == "single":
                        done_single += 1
                    elif method == "multi":
                        done_multi += 1
                    elif method == "chain_first":
                        done_chain += 1
                    elif method == "error_recovery":
                        done_error += 1
                console.print(
                    f"[yellow]  Resumed: single={done_single} multi={done_multi} "
                    f"chain={done_chain} error={done_error} (total={len(all_examples)})[/yellow]\n"
                )

        # Pre-compute tool embeddings once
        self._ensure_tool_embeddings(tools)

        # ── Single-step ─────────────────────────────────────────────
        need_single = n_single - done_single
        if need_single > 0:
            console.print(f"[cyan]━━ Section 1/4: SINGLE-STEP ({need_single} needed, {done_single} done) ━━[/cyan]")
            n_per_tool = max(1, need_single // len(tools))
            single_ex = self.generate_examples(
                tools, n_per_tool=n_per_tool, mode="single", max_steps=max_steps,
                _accumulator=all_examples, _save_every=save_every, _checkpoint_file=ckpt_file,
            )
            # Trim to target (generate_examples may produce more)
            if len(single_ex) > need_single:
                # Remove extras from accumulator too
                excess = len(single_ex) - need_single
                del all_examples[-excess:]
        elif n_single > 0:
            console.print(f"[green]━━ Section 1/4: SINGLE-STEP ━━ DONE ({done_single} already)[/green]")

        # ── Multi-step ──────────────────────────────────────────────
        need_multi = n_multi - done_multi
        if need_multi > 0:
            console.print(f"\n[cyan]━━ Section 2/4: MULTI-STEP ({need_multi} needed, {done_multi} done) ━━[/cyan]")
            n_per_tool = max(1, need_multi // len(tools))
            multi_ex = self.generate_examples(
                tools, n_per_tool=n_per_tool, mode="multi", max_steps=max_steps,
                _accumulator=all_examples, _save_every=save_every, _checkpoint_file=ckpt_file,
            )
            if len(multi_ex) > need_multi:
                excess = len(multi_ex) - need_multi
                del all_examples[-excess:]
        elif n_multi > 0:
            console.print(f"\n[green]━━ Section 2/4: MULTI-STEP ━━ DONE ({done_multi} already)[/green]")

        # ── Chain-first ─────────────────────────────────────────────
        need_chain = n_chain - done_chain
        if need_chain > 0:
            console.print(f"\n[cyan]━━ Section 3/4: CHAIN-FIRST ({need_chain} needed, {done_chain} done) ━━[/cyan]")
            chain_ex = self.generate_chain_first(
                tools, n_chains=need_chain, min_steps=2, max_steps=max_steps,
                _accumulator=all_examples, _save_every=save_every, _checkpoint_file=ckpt_file,
            )
        elif n_chain > 0:
            console.print(f"\n[green]━━ Section 3/4: CHAIN-FIRST ━━ DONE ({done_chain} already)[/green]")

        # ── Error-recovery ──────────────────────────────────────────
        need_error = n_error - done_error
        if need_error > 0:
            console.print(f"\n[cyan]━━ Section 4/4: ERROR-RECOVERY ({need_error} needed, {done_error} done) ━━[/cyan]")
            err_ex = self.generate_error_recovery_examples(
                tools, n=need_error,
                _accumulator=all_examples, _save_every=save_every, _checkpoint_file=ckpt_file,
            )
        elif n_error > 0:
            console.print(f"\n[green]━━ Section 4/4: ERROR-RECOVERY ━━ DONE ({done_error} already)[/green]")

        # Final checkpoint save
        if ckpt_file and all_examples:
            _save_checkpoint(all_examples, ckpt_file)

        console.print(f"\n[bold green]✓ Full-mix complete: {len(all_examples)} examples[/bold green]")
        return all_examples

    def generate_examples_hybrid(
        self,
        tools: List[Tool],
        n_total: int = 50,
        chain_first_ratio: float = 0.4,
        mode: str = "auto",
        max_steps: int = 5,
    ) -> List[ToolExample]:
        """
        Hybrid generation: combine query-first and chain-first approaches.
        
        Recommended for best results. Uses:
        - Chain-first for complex multi-tool examples (better validity)
        - Query-first for simple single-tool examples (better diversity)
        
        Args:
            tools: Tool definitions
            n_total: Total examples to generate
            chain_first_ratio: Portion of examples using chain-first (default 40%)
            mode: Solution mode for query-first ('single', 'multi', 'auto')
            max_steps: Max steps for multi-step solutions
            
        Returns:
            Combined list of ToolExample objects
        """
        console.print(f"\n[bold]Hybrid Generation (Query-First + Chain-First)[/bold]")
        
        n_chain_first = int(n_total * chain_first_ratio)
        n_query_first = n_total - n_chain_first
        
        console.print(f"[dim]Chain-first: {n_chain_first} | Query-first: {n_query_first}[/dim]\n")
        
        all_examples = []
        
        # Chain-first for multi-tool examples
        if n_chain_first > 0:
            chain_examples = self.generate_chain_first(
                tools, 
                n_chains=n_chain_first,
                min_steps=2,
                max_steps=4,
            )
            all_examples.extend(chain_examples)
        
        # Query-first for remaining
        if n_query_first > 0:
            # Calculate per-tool count
            n_per_tool = max(1, n_query_first // len(tools))
            query_examples = self.generate_examples(
                tools,
                n_per_tool=n_per_tool,
                mode=mode,
                max_steps=max_steps,
            )
            all_examples.extend(query_examples[:n_query_first])
        
        console.print(f"\n[bold green]✨ Generated {len(all_examples)} total examples[/bold green]")
        console.print(f"[dim]Chain-first: {len([e for e in all_examples if e.metadata.get('generation_method') == 'chain_first'])}[/dim]")
        console.print(f"[dim]Query-first: {len([e for e in all_examples if e.metadata.get('generation_method') != 'chain_first'])}[/dim]")
        
        return all_examples
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response, handling common issues."""
        # Clean response
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        # Try json5 first (handles trailing commas, etc.)
        try:
            return json5.loads(text)
        except:
            pass
        
        # Try standard json
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON in response
        import re
        json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if json_match:
            try:
                return json5.loads(json_match.group())
            except:
                pass
        
        logger.warning(f"Failed to parse JSON response: {text[:200]}")
        return []
