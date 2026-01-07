"""
Unit tests for Phase 3 Tool Use modules.
"""

import json
import pytest
from pathlib import Path

# Import modules to test
from generator.tool_schemas import (
    Parameter, Tool, ToolCall, ReasoningStep, Solution, 
    ExecutionResult, ToolExample, load_tools, save_tools
)
from generator.tool_parser import ToolParser
from generator.tool_executor import ToolExecutor, BUILTIN_TOOLS, calculator
from generator.tool_curator import format_check


# =============================================================================
# TOOL SCHEMAS TESTS
# =============================================================================

class TestParameter:
    def test_create_simple(self):
        p = Parameter(name="x", type="number", description="A number")
        assert p.name == "x"
        assert p.type == "number"
        assert p.required == True
    
    def test_to_dict(self):
        p = Parameter(name="x", type="number", description="A number", required=False, default=0)
        d = p.to_dict()
        assert d["name"] == "x"
        assert d["required"] == False
        assert d["default"] == 0
    
    def test_from_dict(self):
        d = {"name": "x", "type": "number", "description": "A number", "enum": [1, 2, 3]}
        p = Parameter.from_dict(d)
        assert p.name == "x"
        assert p.enum == [1, 2, 3]


class TestTool:
    def test_create_simple(self):
        t = Tool(
            tool_id="calc",
            name="Calculator",
            description="Do math",
            parameters=[Parameter("expr", "string", "Expression")],
        )
        assert t.tool_id == "calc"
        assert len(t.parameters) == 1
    
    def test_to_schema(self):
        t = Tool(
            tool_id="calc",
            name="calculator",
            description="Calculate",
            parameters=[
                Parameter("a", "number", "First", required=True),
                Parameter("b", "number", "Second", required=False),
            ],
        )
        schema = t.to_schema()
        assert schema["name"] == "calculator"
        assert "a" in schema["parameters"]["properties"]
        assert "a" in schema["parameters"]["required"]
        assert "b" not in schema["parameters"]["required"]
    
    def test_to_documentation(self):
        t = Tool(
            tool_id="calc",
            name="calculator",
            description="Calculate expressions",
            parameters=[Parameter("expr", "string", "Math expression")],
            returns={"type": "number", "description": "Result"}
        )
        doc = t.to_documentation()
        assert "calculator" in doc
        assert "expr: string" in doc
        assert "Result" in doc


class TestSolution:
    def test_create_with_steps(self):
        steps = [
            ReasoningStep(step=1, thought="First", tool="calc", args={"x": 1}),
            ReasoningStep(step=2, thought="Second", tool="calc", args={"x": 2}),
        ]
        s = Solution(
            instruction="Do something",
            reasoning_path=steps,
            final_answer="Done",
        )
        assert len(s.reasoning_path) == 2
        assert len(s.tool_calls) == 2
    
    def test_is_successful(self):
        steps = [
            ReasoningStep(step=1, thought="", tool="calc", args={}, status="success"),
            ReasoningStep(step=2, thought="", tool="calc", args={}, status="success"),
        ]
        s = Solution(instruction="", reasoning_path=steps, final_answer="")
        assert s.is_successful == True
        
        steps[1].status = "failure"
        assert s.is_successful == False


class TestToolExample:
    def test_to_chatml(self):
        steps = [
            ReasoningStep(step=1, thought="Calculate", tool="calc", 
                         args={"expr": "2+2"}, actual_result=4, status="success"),
        ]
        example = ToolExample(
            instruction="What is 2+2?",
            solution=Solution(
                instruction="What is 2+2?",
                reasoning_path=steps,
                final_answer="The answer is 4.",
            ),
        )
        chatml = example.to_training_format("chatml")
        assert "messages" in chatml
        assert chatml["messages"][0]["role"] == "user"
    
    def test_to_alpaca(self):
        steps = [
            ReasoningStep(step=1, thought="Calculate", tool="calc", 
                         args={"expr": "2+2"}, actual_result=4, status="success"),
        ]
        example = ToolExample(
            instruction="What is 2+2?",
            solution=Solution(
                instruction="What is 2+2?",
                reasoning_path=steps,
                final_answer="4",
            ),
        )
        alpaca = example.to_training_format("alpaca")
        assert "instruction" in alpaca
        assert "output" in alpaca


# =============================================================================
# TOOL PARSER TESTS
# =============================================================================

class TestToolParser:
    def test_parse_json_schema(self, tmp_path):
        # Create test JSON
        tools_data = {
            "tools": [
                {
                    "tool_id": "calc",
                    "name": "calculator",
                    "description": "Do math",
                    "parameters": [
                        {"name": "expr", "type": "string", "description": "Expression"}
                    ]
                }
            ]
        }
        json_path = tmp_path / "tools.json"
        with open(json_path, "w") as f:
            json.dump(tools_data, f)
        
        parser = ToolParser()
        tools = parser.parse_json_schema(str(json_path))
        
        assert len(tools) == 1
        assert tools[0].name == "calculator"
    
    def test_validate_tool(self):
        parser = ToolParser()
        
        # Valid tool
        valid = Tool(
            tool_id="calc",
            name="calculator",
            description="Do math",
            parameters=[Parameter("x", "number", "Value")],
        )
        errors = parser.validate_tool(valid)
        assert len(errors) == 0
        
        # Invalid: missing description
        invalid = Tool(
            tool_id="calc",
            name="calculator",
            description="",
            parameters=[],
        )
        errors = parser.validate_tool(invalid)
        assert "description is required" in errors


# =============================================================================
# TOOL EXECUTOR TESTS
# =============================================================================

class TestToolExecutor:
    def test_builtin_calculator(self):
        result = calculator("2 + 2")
        assert result == 4
        
        result = calculator("10 * 5")
        assert result == 50
        
        result = calculator("100 / 4")
        assert result == 25
    
    def test_execute_real_builtin(self):
        executor = ToolExecutor(mode="real")
        executor.register_tool("calculator", calculator)
        
        tool = Tool(
            tool_id="calculator",
            name="calculator",
            description="Calculate",
            parameters=[Parameter("expression", "string", "Math expression")],
        )
        
        result = executor.execute_call(tool, {"expression": "5 + 5"})
        assert result.success == True
        assert result.output == 10
    
    def test_execute_solution(self):
        executor = ToolExecutor(mode="real")
        executor.register_tool("calculator", calculator)
        
        tool = Tool(
            tool_id="calculator",
            name="calculator",
            description="Calculate",
            parameters=[Parameter("expression", "string", "Math expression")],
        )
        
        solution = Solution(
            instruction="Calculate 2+2 then 3*3",
            reasoning_path=[
                ReasoningStep(step=1, thought="First calc", tool="calculator", 
                             args={"expression": "2+2"}),
                ReasoningStep(step=2, thought="Second calc", tool="calculator",
                             args={"expression": "3*3"}),
            ],
            final_answer="Done",
        )
        
        result = executor.execute_solution(solution, [tool])
        
        assert result.success == True
        assert len(result.step_results) == 2
        assert result.step_results[0]["output"] == 4
        assert result.step_results[1]["output"] == 9


# =============================================================================
# TOOL CURATOR TESTS
# =============================================================================

class TestToolCurator:
    def test_format_check_valid(self):
        steps = [
            ReasoningStep(step=1, thought="Do it", tool="calc", args={"x": 1}),
        ]
        example = ToolExample(
            instruction="Calculate something for me please",
            solution=Solution(
                instruction="Calculate something",
                reasoning_path=steps,
                final_answer="The answer is 42.",
            ),
        )
        errors = format_check(example)
        assert len(errors) == 0
    
    def test_format_check_missing_tool(self):
        steps = [
            ReasoningStep(step=1, thought="Do it", tool="", args={}),
        ]
        example = ToolExample(
            instruction="Calculate something for me",
            solution=Solution(
                instruction="Calculate",
                reasoning_path=steps,
                final_answer="Done",
            ),
        )
        errors = format_check(example)
        assert any("Missing tool name" in e for e in errors)
    
    def test_format_check_short_instruction(self):
        example = ToolExample(
            instruction="Hi",  # Too short
            solution=Solution(
                instruction="Hi",
                reasoning_path=[],
                final_answer="",
            ),
        )
        errors = format_check(example)
        assert any("too short" in e.lower() for e in errors)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    def test_load_sample_tools(self):
        """Test loading the sample tools file."""
        sample_path = Path(__file__).parent.parent.parent / "configs" / "sample_tools.json"
        if sample_path.exists():
            tools = load_tools(str(sample_path))
            assert len(tools) > 0
            # Check first tool has required fields
            assert tools[0].tool_id
            assert tools[0].name
            assert tools[0].description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
