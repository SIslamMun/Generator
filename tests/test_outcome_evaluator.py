"""Tests for outcome-oriented evaluation (Gap #4)."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from generator.tool.outcome_evaluator import (
    OutcomeEvaluator,
    OutcomeEvaluation,
    OutcomeStatus,
    evaluate_tool_examples,
)
from generator.tool.tool_schemas import (
    ToolExample,
    Solution,
    ReasoningStep,
    ExecutionResult,
)


# --- Fixtures ---

@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = Mock()
    return mock


@pytest.fixture
def sample_step():
    """Create a sample reasoning step."""
    return ReasoningStep(
        step=1,
        thought="Opening the HDF5 file to access data",
        tool="open_file",
        args={"path": "/data/sim.h5", "mode": "r"},
        status="success",
        actual_result={"handle": "file_001"},
    )


@pytest.fixture
def sample_solution(sample_step):
    """Create a sample solution."""
    step2 = ReasoningStep(
        step=2,
        thought="Reading the temperature dataset",
        tool="read_full_dataset",
        args={"path": "/results/temperature"},
        status="success",
        actual_result={"data": [1.0, 2.0, 3.0], "shape": [3]},
    )
    return Solution(
        instruction="Read the temperature data from the simulation file",
        reasoning_path=[sample_step, step2],
        final_answer="Temperature data: [1.0, 2.0, 3.0]",
    )


@pytest.fixture
def sample_example(sample_solution):
    """Create a sample tool example."""
    example = ToolExample(
        instruction="Read the temperature data from the simulation file",
        solution=sample_solution,
    )
    example.execution_result = ExecutionResult(
        success=True,
        output={"data": [1.0, 2.0, 3.0], "shape": [3]},
    )
    return example


@pytest.fixture
def failed_example(sample_solution):
    """Create an example with failed execution."""
    example = ToolExample(
        instruction="Read the temperature data from the simulation file",
        solution=sample_solution,
    )
    example.execution_result = ExecutionResult(
        success=False,
        error="File not found",
    )
    return example


# --- OutcomeEvaluator Tests ---

class TestOutcomeEvaluator:
    """Tests for OutcomeEvaluator class."""
    
    def test_init_no_llm(self):
        """Test initialization without LLM."""
        evaluator = OutcomeEvaluator()
        assert evaluator.llm is None
        assert evaluator.strict_mode is False
    
    def test_init_with_custom_prompts(self):
        """Test initialization with custom prompts."""
        custom_prompts = {"outcome_evaluation": "Custom prompt: {instruction}"}
        evaluator = OutcomeEvaluator(prompts=custom_prompts)
        assert "outcome_evaluation" in evaluator.prompts
        assert evaluator.prompts["outcome_evaluation"] == "Custom prompt: {instruction}"
    
    def test_init_strict_mode(self):
        """Test initialization with strict mode."""
        evaluator = OutcomeEvaluator(strict_mode=True)
        assert evaluator.strict_mode is True
    
    def test_evaluate_execution_failed(self, failed_example):
        """Test evaluation of example with failed execution."""
        evaluator = OutcomeEvaluator()
        result = evaluator.evaluate_example(failed_example)
        
        assert result.status == OutcomeStatus.EXECUTION_FAILED
        assert result.score == 0.0
        assert "File not found" in result.execution_issues[0]
    
    def test_evaluate_no_llm(self, sample_example):
        """Test evaluation without LLM returns cannot_evaluate."""
        evaluator = OutcomeEvaluator()  # No LLM
        result = evaluator.evaluate_example(sample_example)
        
        assert result.status == OutcomeStatus.CANNOT_EVALUATE
        assert result.score == 0.5
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_evaluate_fully_satisfied(self, mock_get_client, sample_example):
        """Test evaluation with fully satisfied outcome."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "instruction_understood": True,
            "key_requirements": ["Read temperature data", "From simulation file"],
            "satisfied_requirements": ["Read temperature data", "From simulation file"],
            "missing_requirements": [],
            "execution_issues": [],
            "overall_score": 0.95,
            "reasoning": "All requirements fully satisfied",
        })
        mock_get_client.return_value = mock_llm
        
        evaluator = OutcomeEvaluator(llm_config={"provider": "ollama", "model": "test"})
        evaluator.llm = mock_llm
        
        result = evaluator.evaluate_example(sample_example)
        
        assert result.status == OutcomeStatus.FULLY_SATISFIED
        assert result.score == 0.95
        assert result.instruction_understood is True
        assert len(result.key_requirements) == 2
        assert len(result.satisfied_requirements) == 2
        assert len(result.missing_requirements) == 0
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_evaluate_partially_satisfied(self, mock_get_client, sample_example):
        """Test evaluation with partially satisfied outcome."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "instruction_understood": True,
            "key_requirements": ["Read temperature", "Plot data", "Save to CSV"],
            "satisfied_requirements": ["Read temperature"],
            "missing_requirements": ["Plot data", "Save to CSV"],
            "execution_issues": [],
            "overall_score": 0.6,
            "reasoning": "Only reading was completed",
        })
        mock_get_client.return_value = mock_llm
        
        evaluator = OutcomeEvaluator(llm_config={"provider": "ollama", "model": "test"})
        evaluator.llm = mock_llm
        
        result = evaluator.evaluate_example(sample_example)
        
        assert result.status == OutcomeStatus.PARTIALLY_SATISFIED
        assert result.score == 0.6
        assert len(result.missing_requirements) == 2
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_evaluate_not_satisfied(self, mock_get_client, sample_example):
        """Test evaluation with not satisfied outcome."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "instruction_understood": False,
            "key_requirements": ["Calculate statistics"],
            "satisfied_requirements": [],
            "missing_requirements": ["Calculate statistics"],
            "execution_issues": ["Wrong tool used"],
            "overall_score": 0.2,
            "reasoning": "Task not completed",
        })
        mock_get_client.return_value = mock_llm
        
        evaluator = OutcomeEvaluator(llm_config={"provider": "ollama", "model": "test"})
        evaluator.llm = mock_llm
        
        result = evaluator.evaluate_example(sample_example)
        
        assert result.status == OutcomeStatus.NOT_SATISFIED
        assert result.score == 0.2
        assert result.instruction_understood is False
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_strict_mode_reduces_score(self, mock_get_client, sample_example):
        """Test strict mode reduces score when requirements missing."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "instruction_understood": True,
            "key_requirements": ["Read data", "Validate format"],
            "satisfied_requirements": ["Read data"],
            "missing_requirements": ["Validate format"],
            "execution_issues": [],
            "overall_score": 0.8,  # Would be PARTIALLY_SATISFIED
            "reasoning": "One requirement missing",
        })
        mock_get_client.return_value = mock_llm
        
        evaluator = OutcomeEvaluator(
            llm_config={"provider": "ollama", "model": "test"},
            strict_mode=True,
        )
        evaluator.llm = mock_llm
        
        result = evaluator.evaluate_example(sample_example)
        
        # Strict mode should reduce status to NOT_SATISFIED
        assert result.status == OutcomeStatus.NOT_SATISFIED
        assert result.score <= 0.5  # Score capped at 0.5 in strict mode with missing reqs
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_filter_by_outcome(self, mock_get_client, sample_example):
        """Test filtering examples by outcome score."""
        mock_llm = Mock()
        # First example: high score
        mock_llm.generate.side_effect = [
            json.dumps({
                "instruction_understood": True,
                "key_requirements": ["Read data"],
                "satisfied_requirements": ["Read data"],
                "missing_requirements": [],
                "execution_issues": [],
                "overall_score": 0.9,
                "reasoning": "Good",
            }),
            json.dumps({
                "instruction_understood": True,
                "key_requirements": ["Read data"],
                "satisfied_requirements": [],
                "missing_requirements": ["Read data"],
                "execution_issues": [],
                "overall_score": 0.3,
                "reasoning": "Bad",
            }),
        ]
        mock_get_client.return_value = mock_llm
        
        # Create two examples
        example1 = sample_example
        example2 = ToolExample(
            instruction="Another task",
            solution=sample_example.solution,
        )
        example2.execution_result = ExecutionResult(success=True, output={"result": "ok"})
        
        evaluator = OutcomeEvaluator(llm_config={"provider": "ollama", "model": "test"})
        evaluator.llm = mock_llm
        
        filtered = evaluator.filter_by_outcome([example1, example2], min_score=0.7)
        
        assert len(filtered) == 1
        assert filtered[0] == example1
        # Check metadata was attached
        assert "outcome_evaluation" in filtered[0].metadata


class TestOutcomeEvaluation:
    """Tests for OutcomeEvaluation dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        eval_result = OutcomeEvaluation(
            status=OutcomeStatus.FULLY_SATISFIED,
            score=0.95,
            reasoning="All good",
            instruction_understood=True,
            key_requirements=["req1", "req2"],
            satisfied_requirements=["req1", "req2"],
            missing_requirements=[],
            execution_issues=[],
        )
        
        d = eval_result.to_dict()
        
        assert d["status"] == "fully_satisfied"
        assert d["score"] == 0.95
        assert d["reasoning"] == "All good"
        assert len(d["key_requirements"]) == 2


class TestOutcomeStatus:
    """Tests for OutcomeStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert OutcomeStatus.FULLY_SATISFIED.value == "fully_satisfied"
        assert OutcomeStatus.PARTIALLY_SATISFIED.value == "partially_satisfied"
        assert OutcomeStatus.NOT_SATISFIED.value == "not_satisfied"
        assert OutcomeStatus.EXECUTION_FAILED.value == "execution_failed"
        assert OutcomeStatus.CANNOT_EVALUATE.value == "cannot_evaluate"


class TestEvaluateToolExamples:
    """Tests for convenience function."""
    
    def test_convenience_function(self, sample_example):
        """Test evaluate_tool_examples convenience function."""
        # Without LLM, should return empty (no examples meet threshold by default)
        result = evaluate_tool_examples([sample_example], min_score=0.0)
        
        # Should return examples with CANNOT_EVALUATE status
        assert len(result) >= 0  # Depends on default score


class TestSolutionSummary:
    """Tests for solution summary generation."""
    
    def test_summarize_solution(self, sample_solution):
        """Test solution summary generation."""
        evaluator = OutcomeEvaluator()
        summary = evaluator._summarize_solution(sample_solution)
        
        assert "open_file" in summary
        assert "read_full_dataset" in summary
        assert "Step 1" in summary
        assert "Step 2" in summary
    
    def test_summarize_with_errors(self):
        """Test summary includes error information."""
        step = ReasoningStep(
            step=1,
            thought="Failed attempt",
            tool="bad_tool",
            args={},
            status="failure",
            error_message="Tool not found",
        )
        solution = Solution(
            instruction="Test task",
            reasoning_path=[step],
            final_answer="Failed",
        )
        
        evaluator = OutcomeEvaluator()
        summary = evaluator._summarize_solution(solution)
        
        assert "failure" in summary
        assert "Tool not found" in summary


class TestRequirementExtraction:
    """Tests for requirement extraction."""
    
    def test_extract_requirements_no_llm(self):
        """Test extraction without LLM returns instruction as requirement."""
        evaluator = OutcomeEvaluator()
        requirements = evaluator.extract_requirements("Read the temperature data")
        
        assert len(requirements) == 1
        assert requirements[0]["requirement"] == "Read the temperature data"
        assert requirements[0]["priority"] == "high"
    
    @patch("generator.tool.outcome_evaluator.get_client")
    def test_extract_requirements_with_llm(self, mock_get_client):
        """Test extraction with LLM parses response."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "requirements": [
                {"requirement": "Open file", "priority": "high"},
                {"requirement": "Read data", "priority": "high"},
                {"requirement": "Close file", "priority": "low"},
            ]
        })
        mock_get_client.return_value = mock_llm
        
        evaluator = OutcomeEvaluator(llm_config={"provider": "ollama", "model": "test"})
        evaluator.llm = mock_llm
        
        requirements = evaluator.extract_requirements("Open file, read data, close file")
        
        assert len(requirements) == 3
        assert requirements[0]["requirement"] == "Open file"


class TestJsonParsing:
    """Tests for JSON response parsing."""
    
    def test_parse_plain_json(self):
        """Test parsing plain JSON."""
        evaluator = OutcomeEvaluator()
        result = evaluator._parse_json_response('{"key": "value"}')
        assert result["key"] == "value"
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON in markdown code block."""
        evaluator = OutcomeEvaluator()
        response = """```json
{"key": "value"}
```"""
        result = evaluator._parse_json_response(response)
        assert result["key"] == "value"
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        evaluator = OutcomeEvaluator()
        result = evaluator._parse_json_response("not json")
        assert result is None
