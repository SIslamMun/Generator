"""
Test turn-level filtering functionality.

Based on ToolMind (Nov 2025) paper approach.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from generator.tool_curator import ToolCurator
from generator.tool_schemas import (
    Tool, Parameter, ToolExample, Solution, 
    ReasoningStep, ExecutionResult
)


def create_test_example(
    instruction: str,
    steps: list[dict],
    success: bool = True
) -> ToolExample:
    """Helper to create test tool examples."""
    reasoning_path = [
        ReasoningStep(
            step=i + 1,
            thought=s.get("thought", f"Step {i+1} thought"),
            tool=s.get("tool", "test_tool"),
            args=s.get("args", {}),
            actual_result=s.get("result", {"status": "ok"}),
            status="success" if success else "failure"
        )
        for i, s in enumerate(steps)
    ]
    
    solution = Solution(
        instruction=instruction,
        reasoning_path=reasoning_path,
        final_answer="Test answer"
    )
    
    execution_result = ExecutionResult(
        success=success,
        output="Test output"
    )
    
    return ToolExample(
        instruction=instruction,
        solution=solution,
        execution_result=execution_result,
        metadata={}
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_prompts():
    """Load actual prompts from config."""
    prompts_dir = Path(__file__).parent.parent / "configs" / "prompts"
    tool_prompts_path = prompts_dir / "tool_prompts.yaml"
    
    if tool_prompts_path.exists():
        import yaml
        with open(tool_prompts_path) as f:
            return yaml.safe_load(f)
    return {"step_quality_rating": "Test prompt {instruction} {step_number}"}


class TestTurnLevelFiltering:
    """Test the turn-level filtering implementation."""
    
    def test_rate_individual_steps_good_example(self, mock_llm, sample_prompts):
        """Test that a good example gets high ratings."""
        # Mock LLM to return good ratings
        mock_llm.generate.return_value = json.dumps({
            "rating": 0.9,
            "issues": [],
            "is_misleading": False,
            "reasoning": "Step is correct"
        })
        
        curator = ToolCurator(prompts=sample_prompts)
        curator.llm = mock_llm
        
        example = create_test_example(
            instruction="Open the HDF5 file and read temperature data",
            steps=[
                {
                    "thought": "First, I need to open the HDF5 file",
                    "tool": "open_file",
                    "args": {"path": "/data/sim.h5", "mode": "r"},
                    "result": {"file_id": "f1", "status": "opened"}
                },
                {
                    "thought": "Now read the temperature dataset",
                    "tool": "read_dataset",
                    "args": {"path": "/results/temperature"},
                    "result": {"data": [25.0, 26.0, 27.0]}
                }
            ]
        )
        
        ratings = curator._rate_individual_steps(example)
        
        assert len(ratings) == 2
        assert all(r == 0.9 for r in ratings)
        assert mock_llm.generate.call_count == 2
    
    def test_rate_individual_steps_bad_example(self, mock_llm, sample_prompts):
        """Test that a bad example gets low ratings."""
        # Mock LLM to return bad rating for second step
        def generate_side_effect(prompt, temperature=None):
            # Check for step number in format "Step X of Y"
            if "Step 1 of" in prompt:
                return json.dumps({
                    "rating": 0.85,
                    "issues": [],
                    "is_misleading": False,
                    "reasoning": "Good first step"
                })
            else:
                return json.dumps({
                    "rating": 0.3,
                    "issues": ["Wrong tool used", "Args don't make sense"],
                    "is_misleading": True,
                    "reasoning": "This step introduces errors"
                })
        
        mock_llm.generate.side_effect = generate_side_effect
        
        curator = ToolCurator(prompts=sample_prompts)
        curator.llm = mock_llm
        
        example = create_test_example(
            instruction="Calculate statistics for dataset",
            steps=[
                {
                    "thought": "Open the data file",
                    "tool": "open_file",
                    "args": {"path": "/data/test.h5"},
                    "result": {"file_id": "f1"}
                },
                {
                    "thought": "Now close the file without reading",
                    "tool": "close_file",
                    "args": {"file_id": "f1"},
                    "result": {"status": "closed"}
                }
            ]
        )
        
        ratings = curator._rate_individual_steps(example)
        
        assert len(ratings) == 2
        assert ratings[0] == 0.85
        assert ratings[1] == 0.3
    
    def test_filter_by_turn_quality_passes_good(self, mock_llm, sample_prompts):
        """Test that good examples pass the filter."""
        mock_llm.generate.return_value = json.dumps({
            "rating": 0.85,
            "issues": [],
            "is_misleading": False
        })
        
        curator = ToolCurator(prompts=sample_prompts)
        curator.llm = mock_llm
        
        examples = [
            create_test_example(
                f"Task {i}",
                [{"thought": "Good step", "tool": "test_tool"}]
            )
            for i in range(3)
        ]
        
        filtered = curator.filter_by_turn_quality(examples, min_step_quality=0.7)
        
        assert len(filtered) == 3
        assert all("step_ratings" in e.metadata for e in filtered)
    
    def test_filter_by_turn_quality_rejects_bad(self, mock_llm, sample_prompts):
        """Test that bad examples are rejected."""
        # Return low rating
        mock_llm.generate.return_value = json.dumps({
            "rating": 0.4,
            "issues": ["Multiple problems"],
            "is_misleading": True
        })
        
        curator = ToolCurator(prompts=sample_prompts)
        curator.llm = mock_llm
        
        examples = [
            create_test_example(
                "Bad task",
                [{"thought": "Poor reasoning", "tool": "wrong_tool"}]
            )
        ]
        
        filtered = curator.filter_by_turn_quality(examples, min_step_quality=0.7)
        
        assert len(filtered) == 0
    
    def test_get_previous_context_first_step(self, sample_prompts):
        """Test context building for first step."""
        curator = ToolCurator(prompts=sample_prompts)
        
        example = create_test_example(
            "Test instruction",
            [{"tool": "tool1"}, {"tool": "tool2"}]
        )
        
        context = curator._get_previous_context(example, 0)
        assert context == "This is the first step."
    
    def test_get_previous_context_later_step(self, sample_prompts):
        """Test context building for later steps."""
        curator = ToolCurator(prompts=sample_prompts)
        
        example = create_test_example(
            "Test instruction",
            [
                {"tool": "open_file", "args": {"path": "/data/test.h5"}, "result": {"id": "f1"}},
                {"tool": "read_dataset", "args": {"path": "/data"}},
            ]
        )
        
        context = curator._get_previous_context(example, 1)
        
        assert "Step 1:" in context
        assert "open_file" in context
    
    def test_curate_with_turn_level_filter(self, mock_llm, sample_prompts):
        """Test full curation pipeline with turn-level filtering enabled."""
        # Mock good ratings for both turn-level and overall rating
        def generate_side_effect(prompt, temperature=None):
            if "individual reasoning step" in prompt.lower():
                return json.dumps({
                    "rating": 0.85,
                    "issues": [],
                    "is_misleading": False
                })
            else:
                return json.dumps({
                    "rating": 8.0,
                    "tool_selection": 3,
                    "parameter_correctness": 3,
                    "reasoning_quality": 1,
                    "result_usefulness": 1
                })
        
        mock_llm.generate.side_effect = generate_side_effect
        
        curator = ToolCurator(prompts=sample_prompts)
        curator.llm = mock_llm
        
        tools = [
            Tool(
                tool_id="test_tool",
                name="test_tool",
                description="A test tool",
                parameters=[]
            )
        ]
        
        examples = [
            create_test_example(
                f"Task {i}",
                [{"thought": "Good step", "tool": "test_tool"}]
            )
            for i in range(2)
        ]
        
        curated = curator.curate(
            examples,
            tools,
            turn_level_filter=True,
            min_step_quality=0.7,
            balance=False,
            min_per_tool=0
        )
        
        # Should pass both turn-level and overall rating filters
        assert len(curated) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
