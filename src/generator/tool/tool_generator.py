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
"""

import json
import json5
import logging
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .tool_schemas import Tool, Solution, ReasoningStep, ToolExample
from ..clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


class ToolGenerator:
    """Generate tool-use training data with unified approach."""
    
    def __init__(self, llm_config: Dict[str, Any], prompts: Dict[str, str]):
        """
        Initialize generator.
        
        Args:
            llm_config: LLM configuration with provider and settings
            prompts: Prompt templates dict (loaded from configs/prompts/tool_prompts.yaml)
        """
        self.prompts = prompts
        provider = llm_config.pop("provider", "ollama")
        self.llm = get_client(provider, llm_config)
        self.provider = provider
    
    def _get_prompt(self, key: str) -> str:
        """Get prompt template, raising error if not found."""
        if key not in self.prompts:
            raise ValueError(
                f"Missing prompt template '{key}'. "
                f"Add it to configs/prompts/tool_prompts.yaml"
            )
        return self.prompts[key]
    
    def generate_instructions(
        self,
        tools: List[Tool],
        n_per_tool: int = 10,
        include_multi_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse user instructions for tools.
        
        Args:
            tools: List of tool definitions
            n_per_tool: Instructions to generate per tool
            include_multi_tool: Whether to generate multi-tool instructions
            
        Returns:
            List of instruction dicts with metadata
        """
        all_instructions = []
        
        prompt_template = self._get_prompt("tool_instruction_generation")
        
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
        prompt = prompt_template.format(
            tool_name=tool.name,
            tool_description=tool.description,
            parameters=json.dumps([p.to_dict() for p in tool.parameters], indent=2),
            examples=json.dumps(tool.examples, indent=2) if tool.examples else "[]",
            n_instructions=n,
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
    ) -> Solution:
        """
        Generate a solution for an instruction.
        
        Always includes API documentation for better grounding.
        
        Args:
            instruction: User instruction to solve
            tools: Available tools
            mode: 'single' (one tool call), 'multi' (chain), or 'auto' (detect)
            max_steps: Maximum reasoning steps for multi mode
            
        Returns:
            Solution object with reasoning path
        """
        # Always include documentation (Gorilla insight)
        docs = "\n\n".join([t.to_documentation() for t in tools])
        tools_json = json.dumps([t.to_schema() for t in tools], indent=2)
        
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
    ) -> List[ToolExample]:
        """
        Generate complete tool-use examples (instructions + solutions).
        
        Args:
            tools: Tool definitions
            n_per_tool: Examples per tool
            mode: 'single', 'multi', or 'auto' (balanced mix)
            max_steps: Max reasoning steps for multi-step
            
        Returns:
            List of ToolExample objects
        """
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
                    # Use instruction's detected mode, or auto-detect
                    inst_mode = inst_data.get("mode", "auto")
                else:
                    inst_mode = mode
                
                # Get relevant tools
                relevant_tools = [tool_map[tid] for tid in required if tid in tool_map]
                if not relevant_tools:
                    relevant_tools = tools[:3]  # Fallback
                
                solution = self.annotate_solution(
                    instruction, relevant_tools, inst_mode, max_steps
                )
                
                if solution.reasoning_path:  # Only keep non-empty solutions
                    examples.append(ToolExample(
                        instruction=instruction,
                        solution=solution,
                        metadata={
                            "difficulty": inst_data.get("difficulty", "medium"),
                            "scenario": inst_data.get("scenario", ""),
                            "required_tools": required,
                            "multi_tool": inst_data.get("multi_tool", False),
                            "mode": solution.method,
                        }
                    ))
                
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
    ) -> List[ToolExample]:
        """
        Chain-first generation: build valid tool chains, then synthesize queries.
        
        Based on ToolGrad (Aug 2025): https://arxiv.org/abs/2508.04086
        
        Approach:
        1. Generate valid tool chains (sequences of compatible calls)
        2. For each chain, synthesize a natural user query that would require it
        3. Validate chain coherence
        
        This reduces invalid samples by ~40% compared to query-first.
        
        Args:
            tools: List of tool definitions
            n_chains: Number of chains to generate
            min_steps: Minimum tools per chain
            max_steps: Maximum tools per chain
            
        Returns:
            List of ToolExample objects with valid chains
        """
        console.print(f"\n[bold cyan]Chain-First Generation (ToolGrad)[/bold cyan]")
        console.print(f"[dim]Building {n_chains} valid chains ({min_steps}-{max_steps} steps)...[/dim]\n")
        
        examples = []
        docs = "\n\n".join([t.to_documentation() for t in tools])
        
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
            max_attempts = n_chains * 3  # Allow some failures
            
            while generated < n_chains and attempts < max_attempts:
                attempts += 1
                
                # Step 1: Generate a valid tool chain
                chain = self._generate_valid_chain(tools, min_steps, max_steps)
                if not chain or not chain.get("steps"):
                    continue
                
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
                
                examples.append(example)
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
        tools_json = json.dumps([{
            "name": t.name,
            "description": t.description,
            "parameters": [p.to_dict() for p in t.parameters],
            "returns": t.returns,
        } for t in tools], indent=2)
        
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
