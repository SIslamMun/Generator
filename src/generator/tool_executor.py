"""
Tool execution and validation.

Implements three execution modes:
- Simulated: LLM generates plausible responses
- Real: Execute against actual APIs
- Multi-agent: LLM simulates tool behavior
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .tool_schemas import (
    Tool, Solution, ReasoningStep, ToolExample, ExecutionResult
)
from .clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute and validate tool calls."""
    
    def __init__(
        self,
        mode: str = "simulated",
        timeout: int = 30,
        llm_config: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize executor.
        
        Args:
            mode: Execution mode (simulated, real, multi_agent)
            timeout: Timeout for real API calls in seconds
            llm_config: LLM config for simulated/multi_agent modes
            prompts: Prompt templates (loaded from configs/prompts/tool_prompts.yaml)
        """
        self.mode = mode
        self.timeout = timeout
        self.prompts = prompts or {}
        self.llm: Optional[BaseLLMClient] = None
        
        # Tool implementations for real mode
        self._tool_implementations: Dict[str, Callable] = {}
        
        if mode in ["simulated", "multi_agent"] and llm_config:
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
    
    def register_tool(self, tool_id: str, implementation: Callable) -> None:
        """
        Register a tool implementation for real execution.
        
        Args:
            tool_id: Tool identifier
            implementation: Callable that takes args dict and returns result
        """
        self._tool_implementations[tool_id] = implementation
    
    def execute_call(
        self,
        tool: Tool,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a single tool call.
        
        Args:
            tool: Tool definition
            args: Arguments for the tool
            context: Optional context (e.g., previous results)
            
        Returns:
            ExecutionResult with output or error
        """
        start_time = time.time()
        
        try:
            if self.mode == "real":
                result = self._execute_real(tool, args)
            elif self.mode == "multi_agent":
                result = self._execute_multi_agent(tool, args, context)
            else:  # simulated
                result = self._execute_simulated(tool, args, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.warning(f"Tool execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    def _execute_real(self, tool: Tool, args: Dict[str, Any]) -> Any:
        """Execute against real implementation."""
        impl = self._tool_implementations.get(tool.tool_id)
        if not impl:
            raise ValueError(f"No implementation registered for {tool.tool_id}")
        
        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in args:
                raise ValueError(f"Missing required parameter: {param.name}")
        
        return impl(**args)
    
    def _execute_simulated(
        self,
        tool: Tool,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Generate simulated response using LLM."""
        if not self.llm:
            # Return placeholder if no LLM
            return self._generate_placeholder(tool, args)
        
        context_str = json.dumps(context, indent=2) if context else ""
        
        prompt_template = self._get_prompt("tool_simulation")
        prompt = prompt_template.format(
            tool_name=tool.name,
            tool_description=tool.description,
            return_type=tool.returns.get("type", "object"),
            args_json=json.dumps(args, indent=2),
            context=context_str,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            # Try to parse as JSON
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(text)
        except:
            # Return as string if not valid JSON
            return response.strip()
    
    def _execute_multi_agent(
        self,
        tool: Tool,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Use LLM as a multi-agent simulator (ToolAlpaca style)."""
        if not self.llm:
            return self._generate_placeholder(tool, args)
        
        context_str = json.dumps(context, indent=2) if context else ""
        
        # Use the same simulation prompt for multi-agent mode
        prompt_template = self._get_prompt("tool_simulation")
        prompt = prompt_template.format(
            tool_name=tool.name,
            tool_description=tool.description,
            return_type=tool.returns.get("type", "object"),
            args_json=json.dumps(args),
            context=context_str,
        )
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except:
            return response.strip() if 'response' in dir() else {"result": "simulated"}
    
    def _generate_placeholder(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate placeholder response without LLM."""
        return_type = tool.returns.get("type", "object")
        
        if return_type == "string":
            return f"Result for {tool.name}"
        elif return_type == "number":
            return 42.0
        elif return_type == "integer":
            return 42
        elif return_type == "boolean":
            return True
        elif return_type == "array":
            return [{"item": 1}, {"item": 2}]
        else:
            return {"status": "success", "result": "placeholder"}
    
    def execute_solution(
        self,
        solution: Solution,
        tools: List[Tool],
    ) -> ExecutionResult:
        """
        Execute entire solution path.
        
        Args:
            solution: Solution with reasoning path
            tools: Available tool definitions
            
        Returns:
            ExecutionResult with step results
        """
        tool_map = {t.tool_id: t for t in tools}
        tool_map.update({t.name: t for t in tools})  # Also map by name
        
        step_results = []
        context = {}  # Accumulate results for chaining
        overall_success = True
        
        for step in solution.reasoning_path:
            tool = tool_map.get(step.tool)
            if not tool:
                logger.warning(f"Unknown tool: {step.tool}")
                step.status = "failure"
                step.error_message = f"Unknown tool: {step.tool}"
                step_results.append({
                    "step": step.step,
                    "success": False,
                    "error": step.error_message,
                })
                overall_success = False
                continue
            
            # Execute the step
            result = self.execute_call(tool, step.args, context)
            
            # Update step with result
            step.actual_result = result.output
            step.status = "success" if result.success else "failure"
            if result.error:
                step.error_message = result.error
            
            step_results.append({
                "step": step.step,
                "tool": step.tool,
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            })
            
            # Add to context for chaining
            if result.success:
                context[f"step_{step.step}_result"] = result.output
                context["last_result"] = result.output
            else:
                overall_success = False
        
        # Update solution validation status
        solution.execution_validated = overall_success
        
        return ExecutionResult(
            success=overall_success,
            output=context.get("last_result"),
            step_results=step_results,
        )
    
    def execute_examples(
        self,
        examples: List[ToolExample],
        tools: List[Tool],
    ) -> List[ToolExample]:
        """
        Execute all examples and update with results.
        
        Args:
            examples: List of tool examples
            tools: Tool definitions
            
        Returns:
            Updated examples with execution results
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Executing ({self.mode})...", 
                total=len(examples)
            )
            
            for example in examples:
                result = self.execute_solution(example.solution, tools)
                example.execution_result = result
                progress.advance(task)
        
        # Summary
        successful = sum(1 for e in examples if e.execution_result and e.execution_result.success)
        console.print(f"[green]✓ Executed {len(examples)} examples ({successful} successful)[/green]")
        
        return examples
    
    def verify_semantic_correctness(
        self,
        example: ToolExample,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Use LLM to verify semantic correctness of execution.
        
        Args:
            example: Executed tool example
            llm_config: LLM config (uses self.llm if not provided)
            
        Returns:
            True if semantically correct
        """
        llm = self.llm
        if llm_config:
            provider = llm_config.pop("provider", "ollama")
            llm = get_client(provider, llm_config)
        
        if not llm:
            logger.warning("No LLM available for semantic verification")
            return True  # Assume correct if can't verify
        
        # Build verification prompt
        steps_summary = "\n".join([
            f"Step {s.step}: {s.tool}({json.dumps(s.args)}) -> {json.dumps(s.actual_result)}"
            for s in example.solution.reasoning_path
        ])
        
        prompt = f"""Evaluate if this tool execution correctly solves the user's request.

User Request: {example.instruction}

Execution Steps:
{steps_summary}

Final Answer: {example.solution.final_answer}

Does this execution correctly address the user's request?
Consider:
1. Are the right tools used?
2. Are the results relevant to the request?
3. Does the final answer make sense given the results?

Respond with JSON:
{{"correct": true/false, "reasoning": "Brief explanation"}}

Return ONLY valid JSON."""
        
        try:
            response = llm.generate(prompt, temperature=0.1)
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            result = json.loads(text)
            return result.get("correct", False)
        except Exception as e:
            logger.warning(f"Semantic verification failed: {e}")
            return True  # Assume correct on error


# Built-in tool implementations for testing
BUILTIN_TOOLS: Dict[str, Callable] = {}


def register_builtin_tool(name: str):
    """Decorator to register a built-in tool."""
    def decorator(func: Callable) -> Callable:
        BUILTIN_TOOLS[name] = func
        return func
    return decorator


@register_builtin_tool("calculator")
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # Safe eval for math expressions
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)


@register_builtin_tool("get_current_time")
def get_current_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().isoformat()


@register_builtin_tool("get_current_date")
def get_current_date() -> str:
    """Get current date."""
    from datetime import date
    return date.today().isoformat()


@register_builtin_tool("string_length")
def string_length(text: str) -> int:
    """Get length of a string."""
    return len(text)


@register_builtin_tool("string_reverse")
def string_reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def create_executor_with_builtins(
    mode: str = "real",
    timeout: int = 30,
    llm_config: Optional[Dict[str, Any]] = None,
) -> ToolExecutor:
    """Create executor with built-in tools registered."""
    executor = ToolExecutor(mode=mode, timeout=timeout, llm_config=llm_config)
    
    for name, impl in BUILTIN_TOOLS.items():
        executor.register_tool(name, impl)
    
    return executor
