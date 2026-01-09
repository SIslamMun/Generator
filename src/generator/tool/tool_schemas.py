"""
Tool use data schemas for Phase 3.

Defines dataclasses for tools, parameters, solutions, and training examples.
Based on Toolformer, Gorilla, and ToolLLM paper formats.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ToolComplexity(str, Enum):
    """Tool complexity levels."""
    SIMPLE = "simple"      # Single call, clear success (calculator, search)
    MEDIUM = "medium"      # Multiple params, some reasoning
    COMPLEX = "complex"    # Multi-step, chaining, error handling


class ExecutionStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class GenerationMethod(str, Enum):
    """Training data generation method."""
    TOOLFORMER = "toolformer"  # Single-step, loss-based filtering
    GORILLA = "gorilla"        # With API documentation context
    TOOLLLM = "toolllm"        # DFSDT multi-step reasoning


@dataclass
class Parameter:
    """Tool parameter definition."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = {"name": self.name, "type": self.type, "description": self.description}
        if not self.required:
            d["required"] = False
        if self.default is not None:
            d["default"] = self.default
        if self.enum is not None:
            d["enum"] = self.enum
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Parameter":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            type=d["type"],
            description=d.get("description", ""),
            required=d.get("required", True),
            default=d.get("default"),
            enum=d.get("enum"),
        )


@dataclass
class Tool:
    """Tool/API definition."""
    tool_id: str
    name: str
    description: str
    parameters: List[Parameter]
    returns: Dict[str, str] = field(default_factory=lambda: {"type": "any", "description": ""})
    examples: List[Dict[str, Any]] = field(default_factory=list)
    category: str = "general"
    complexity: str = "simple"
    documentation_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns,
            "examples": self.examples,
            "category": self.category,
            "complexity": self.complexity,
            "documentation_url": self.documentation_url,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tool":
        """Create from dictionary."""
        return cls(
            tool_id=d["tool_id"],
            name=d["name"],
            description=d["description"],
            parameters=[Parameter.from_dict(p) for p in d.get("parameters", [])],
            returns=d.get("returns", {"type": "any", "description": ""}),
            examples=d.get("examples", []),
            category=d.get("category", "general"),
            complexity=d.get("complexity", "simple"),
            documentation_url=d.get("documentation_url"),
        )
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM function calling."""
        properties = {}
        required = []
        
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_documentation(self) -> str:
        """Generate documentation string for Gorilla-style training."""
        params_str = ", ".join(
            f"{p.name}: {p.type}" + (f" = {p.default}" if p.default is not None else "")
            for p in self.parameters
        )
        
        doc = f"{self.name}({params_str})\n\n{self.description}\n\nParameters:\n"
        for p in self.parameters:
            req = "required" if p.required else "optional"
            doc += f"  - {p.name} ({p.type}, {req}): {p.description}\n"
        
        if self.returns:
            doc += f"\nReturns: {self.returns.get('type', 'any')} - {self.returns.get('description', '')}\n"
        
        return doc


@dataclass
class ToolCall:
    """A single tool invocation."""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "args": self.args}
        if self.call_id:
            d["call_id"] = self.call_id
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(name=d["name"], args=d.get("args", {}), call_id=d.get("call_id"))


@dataclass
class ReasoningStep:
    """A single step in multi-step tool reasoning (ToolLLM style)."""
    step: int
    thought: str
    tool: str
    args: Dict[str, Any]
    expected_result: Optional[Any] = None
    actual_result: Optional[Any] = None
    status: str = "pending"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step": self.step,
            "thought": self.thought,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
        }
        if self.expected_result is not None:
            d["expected_result"] = self.expected_result
        if self.actual_result is not None:
            d["actual_result"] = self.actual_result
        if self.error_message:
            d["error_message"] = self.error_message
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningStep":
        return cls(
            step=d["step"],
            thought=d.get("thought", ""),
            tool=d["tool"],
            args=d.get("args", {}),
            expected_result=d.get("expected_result"),
            actual_result=d.get("actual_result"),
            status=d.get("status", "pending"),
            error_message=d.get("error_message"),
        )


@dataclass
class Solution:
    """Complete solution for a tool-use instruction."""
    instruction: str
    reasoning_path: List[ReasoningStep]
    final_answer: str
    api_documentation: Optional[str] = None  # For Gorilla-style
    execution_validated: bool = False
    method: str = "toolllm"  # Generation method used
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "instruction": self.instruction,
            "reasoning_path": [s.to_dict() for s in self.reasoning_path],
            "final_answer": self.final_answer,
            "execution_validated": self.execution_validated,
            "method": self.method,
        }
        if self.api_documentation:
            d["api_documentation"] = self.api_documentation
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Solution":
        return cls(
            instruction=d["instruction"],
            reasoning_path=[ReasoningStep.from_dict(s) for s in d.get("reasoning_path", [])],
            final_answer=d.get("final_answer", ""),
            api_documentation=d.get("api_documentation"),
            execution_validated=d.get("execution_validated", False),
            method=d.get("method", "toolllm"),
        )
    
    @property
    def tool_calls(self) -> List[ToolCall]:
        """Extract tool calls from reasoning path."""
        return [ToolCall(name=s.tool, args=s.args) for s in self.reasoning_path]
    
    @property
    def is_successful(self) -> bool:
        """Check if all steps succeeded."""
        return all(s.status == "success" for s in self.reasoning_path)


@dataclass
class ExecutionResult:
    """Result of executing a tool call or solution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    step_results: Optional[List[Dict[str, Any]]] = None  # For multi-step
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"success": self.success}
        if self.output is not None:
            d["output"] = self.output
        if self.error:
            d["error"] = self.error
        if self.execution_time_ms:
            d["execution_time_ms"] = self.execution_time_ms
        if self.step_results:
            d["step_results"] = self.step_results
        return d


@dataclass
class ToolExample:
    """Complete training example for tool use."""
    instruction: str
    solution: Solution
    execution_result: Optional[ExecutionResult] = None
    rating: Optional[float] = None
    rating_breakdown: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "instruction": self.instruction,
            "solution": self.solution.to_dict(),
            "metadata": self.metadata,
        }
        if self.execution_result:
            d["execution_result"] = self.execution_result.to_dict()
        if self.rating is not None:
            d["rating"] = self.rating
        if self.rating_breakdown:
            d["rating_breakdown"] = self.rating_breakdown
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolExample":
        execution_result = None
        if "execution_result" in d:
            er = d["execution_result"]
            execution_result = ExecutionResult(
                success=er.get("success", False),
                output=er.get("output"),
                error=er.get("error"),
                execution_time_ms=er.get("execution_time_ms"),
                step_results=er.get("step_results"),
            )
        
        return cls(
            instruction=d["instruction"],
            solution=Solution.from_dict(d["solution"]),
            execution_result=execution_result,
            rating=d.get("rating"),
            rating_breakdown=d.get("rating_breakdown"),
            metadata=d.get("metadata", {}),
        )
    
    def to_training_format(self, format_type: str = "chatml") -> Dict[str, Any]:
        """Convert to training format (ChatML, Alpaca, etc.)."""
        if format_type == "chatml":
            return self._to_chatml()
        elif format_type == "alpaca":
            return self._to_alpaca()
        elif format_type == "sharegpt":
            return self._to_sharegpt()
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _to_chatml(self) -> Dict[str, Any]:
        """Convert to ChatML format with tool calls."""
        messages = [{"role": "user", "content": self.instruction}]
        
        # Add tool calls and responses
        for step in self.solution.reasoning_path:
            messages.append({
                "role": "assistant",
                "content": step.thought,
                "tool_calls": [{"name": step.tool, "arguments": step.args}]
            })
            messages.append({
                "role": "tool",
                "name": step.tool,
                "content": json.dumps(step.actual_result) if step.actual_result else ""
            })
        
        # Final answer
        messages.append({"role": "assistant", "content": self.solution.final_answer})
        
        return {"messages": messages}
    
    def _to_alpaca(self) -> Dict[str, Any]:
        """Convert to Alpaca format."""
        # Build response with reasoning
        response_parts = []
        for step in self.solution.reasoning_path:
            response_parts.append(f"Thought: {step.thought}")
            response_parts.append(f"Action: {step.tool}({json.dumps(step.args)})")
            if step.actual_result:
                response_parts.append(f"Observation: {json.dumps(step.actual_result)}")
        response_parts.append(f"Final Answer: {self.solution.final_answer}")
        
        return {
            "instruction": self.instruction,
            "input": "",
            "output": "\n".join(response_parts)
        }
    
    def _to_sharegpt(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = [{"from": "human", "value": self.instruction}]
        
        # Combine reasoning into assistant response
        response_parts = []
        for step in self.solution.reasoning_path:
            response_parts.append(f"**Step {step.step}:** {step.thought}")
            response_parts.append(f"```\n{step.tool}({json.dumps(step.args, indent=2)})\n```")
            if step.actual_result:
                response_parts.append(f"Result: {json.dumps(step.actual_result)}")
        response_parts.append(f"\n**Answer:** {self.solution.final_answer}")
        
        conversations.append({"from": "gpt", "value": "\n".join(response_parts)})
        
        return {"conversations": conversations}


# Utility functions
def load_tools(path: str) -> List[Tool]:
    """Load tools from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [Tool.from_dict(d) for d in data]
    elif "tools" in data:
        return [Tool.from_dict(d) for d in data["tools"]]
    else:
        raise ValueError("Invalid tools file format")


def save_tools(tools: List[Tool], path: str) -> None:
    """Save tools to JSON file."""
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in tools], f, indent=2)


def load_examples(path: str) -> List[ToolExample]:
    """Load tool examples from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [ToolExample.from_dict(d) for d in data]
    elif "examples" in data:
        return [ToolExample.from_dict(d) for d in data["examples"]]
    else:
        raise ValueError("Invalid examples file format")


def save_examples(examples: List[ToolExample], path: str) -> None:
    """Save tool examples to JSON file."""
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in examples], f, indent=2)
