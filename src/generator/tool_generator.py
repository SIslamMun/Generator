"""
Tool use training data generator.

Unified approach that combines best practices from:
- Toolformer: Single-step tool calls
- Gorilla: API documentation grounding (always included)
- ToolLLM: Multi-step reasoning with chains

Two modes:
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
from .clients import get_client, BaseLLMClient

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
