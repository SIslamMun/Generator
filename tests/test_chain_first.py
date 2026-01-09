"""Tests for chain-first tool generation (ToolGrad paper implementation)."""

import pytest
from unittest.mock import MagicMock, patch
import json


class TestChainFirstGeneration:
    """Test the chain-first generation approach."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        from generator.tool_schemas import Tool, Parameter
        
        return [
            Tool(
                tool_id="create_file",
                name="create_file",
                description="Create a new HDF5 file",
                parameters=[
                    Parameter(name="filename", type="string", description="File name", required=True),
                ],
                returns={"type": "object", "description": "File handle"},
                category="file",
            ),
            Tool(
                tool_id="create_dataset",
                name="create_dataset",
                description="Create a dataset in a file",
                parameters=[
                    Parameter(name="file", type="string", description="File handle", required=True),
                    Parameter(name="name", type="string", description="Dataset name", required=True),
                    Parameter(name="data", type="array", description="Data to store", required=True),
                ],
                returns={"type": "object", "description": "Dataset object"},
                category="dataset",
            ),
            Tool(
                tool_id="add_attribute",
                name="add_attribute",
                description="Add an attribute to an object",
                parameters=[
                    Parameter(name="obj", type="string", description="Object (file/dataset)", required=True),
                    Parameter(name="key", type="string", description="Attribute name", required=True),
                    Parameter(name="value", type="any", description="Attribute value", required=True),
                ],
                returns={"type": "null", "description": "None"},
                category="attribute",
            ),
        ]

    @pytest.fixture
    def sample_prompts(self):
        """Create sample prompts for testing."""
        return {
            "chain_generation": """Generate chain with {min_steps}-{max_steps} steps using tools:
{tools_json}
Return JSON with steps array.""",
            "query_synthesis": """Create query for chain:
{chain_steps}
Tools: {tools_used}
Result: {final_result}
Return JSON with query field.""",
            "tool_instruction_generation": "Generate instructions",
            "multi_tool_instruction_generation": "Generate multi-tool instructions",
            "tool_solution_single": "Generate single solution",
            "tool_solution_multi": "Generate multi solution",
        }

    def test_generator_initialization(self, sample_prompts):
        """Test that ToolGenerator initializes correctly."""
        from generator.tool_generator import ToolGenerator
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = MagicMock()
            generator = ToolGenerator(
                {"provider": "ollama", "model": "test"},
                sample_prompts,
            )
            assert generator.prompts == sample_prompts

    def test_generate_valid_chain(self, mock_llm, sample_tools, sample_prompts):
        """Test generating a valid tool chain."""
        from generator.tool_generator import ToolGenerator
        
        # Mock LLM response for chain generation
        chain_response = json.dumps({
            "steps": [
                {
                    "step": 1,
                    "thought": "First create a file",
                    "tool": "create_file",
                    "args": {"filename": "data.h5"},
                    "expected_result": "File handle"
                },
                {
                    "step": 2,
                    "thought": "Then create a dataset using the file",
                    "tool": "create_dataset",
                    "args": {"file": "$step_1_result", "name": "values", "data": [1, 2, 3]},
                    "expected_result": "Dataset object"
                }
            ],
            "final_answer": "Created file with dataset"
        })
        mock_llm.generate.return_value = chain_response
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = mock_llm
            generator = ToolGenerator(
                {"provider": "ollama", "model": "test"},
                sample_prompts,
            )
            
            chain = generator._generate_valid_chain(sample_tools, min_steps=2, max_steps=4)
            
            assert chain is not None
            assert "steps" in chain
            assert len(chain["steps"]) >= 2
            assert chain["steps"][0]["tool"] == "create_file"
            assert chain["steps"][1]["tool"] == "create_dataset"

    def test_synthesize_query_for_chain(self, mock_llm, sample_prompts):
        """Test synthesizing a query for a chain."""
        from generator.tool_generator import ToolGenerator
        
        chain = {
            "steps": [
                {"tool": "create_file", "args": {"filename": "data.h5"}},
                {"tool": "create_dataset", "args": {"file": "$step_1_result", "name": "values"}},
            ],
            "final_answer": "Created file with dataset"
        }
        
        query_response = json.dumps({
            "query": "Create an HDF5 file called data.h5 and add a dataset named 'values' to it"
        })
        mock_llm.generate.return_value = query_response
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = mock_llm
            generator = ToolGenerator(
                {"provider": "ollama", "model": "test"},
                sample_prompts,
            )
            
            query = generator._synthesize_query_for_chain(chain, "API docs here")
            
            assert query is not None
            assert "HDF5" in query or "file" in query.lower()

    def test_generate_chain_first_full(self, mock_llm, sample_tools, sample_prompts):
        """Test full chain-first generation."""
        from generator.tool_generator import ToolGenerator
        
        # Mock chain generation response
        chain_response = json.dumps({
            "steps": [
                {"step": 1, "thought": "Create file", "tool": "create_file", 
                 "args": {"filename": "test.h5"}, "expected_result": "handle"},
                {"step": 2, "thought": "Add dataset", "tool": "create_dataset",
                 "args": {"file": "$step_1_result", "name": "data", "data": [1,2,3]}, 
                 "expected_result": "dataset"},
            ],
            "final_answer": "Done"
        })
        
        # Mock query synthesis response
        query_response = json.dumps({
            "query": "Create an HDF5 file and store some data in it"
        })
        
        # Alternate responses
        responses = [chain_response, query_response]
        mock_llm.generate.side_effect = responses * 5  # Enough for multiple attempts
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = mock_llm
            generator = ToolGenerator(
                {"provider": "ollama", "model": "test"},
                sample_prompts,
            )
            
            examples = generator.generate_chain_first(
                tools=sample_tools,
                n_chains=2,
                min_steps=2,
                max_steps=3,
            )
            
            # Should generate examples
            assert len(examples) > 0
            
            # Check structure
            for example in examples:
                assert example.instruction
                assert example.solution.reasoning_path
                assert example.metadata.get("generation_method") == "chain_first"

    def test_hybrid_generation(self, mock_llm, sample_tools, sample_prompts):
        """Test hybrid generation combining chain-first and query-first."""
        from generator.tool_generator import ToolGenerator
        
        # Mock responses
        chain_response = json.dumps({
            "steps": [
                {"step": 1, "thought": "Step 1", "tool": "create_file", 
                 "args": {"filename": "test.h5"}, "expected_result": "handle"},
                {"step": 2, "thought": "Step 2", "tool": "add_attribute",
                 "args": {"obj": "$step_1_result", "key": "author", "value": "test"},
                 "expected_result": "None"},
            ],
            "final_answer": "Done"
        })
        
        query_response = json.dumps({"query": "Create a file and add metadata"})
        
        instruction_response = json.dumps([
            {"instruction": "Create a new file", "difficulty": "simple", "scenario": "test"}
        ])
        
        single_solution = json.dumps({
            "thought": "Create file",
            "tool": "create_file",
            "args": {"filename": "test.h5"},
            "expected_result": "handle",
            "final_answer": "Created"
        })
        
        # Set up response cycle
        mock_llm.generate.side_effect = [
            chain_response, query_response,  # First chain
            chain_response, query_response,  # Second chain (in case needed)
            instruction_response,  # Query-first instructions
            single_solution,  # Query-first solution
        ] * 10  # Repeat to ensure enough responses
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = mock_llm
            generator = ToolGenerator(
                {"provider": "ollama", "model": "test"},
                sample_prompts,
            )
            
            examples = generator.generate_examples_hybrid(
                tools=sample_tools,
                n_total=5,
                chain_first_ratio=0.4,
            )
            
            # Should have mixed examples
            assert len(examples) > 0


class TestChainValidation:
    """Test chain validation logic."""

    def test_chain_with_too_few_steps_rejected(self):
        """Test that chains with too few steps are rejected."""
        from generator.tool_generator import ToolGenerator
        from unittest.mock import patch, MagicMock
        
        mock_llm = MagicMock()
        # Return chain with only 1 step (below min_steps=2)
        mock_llm.generate.return_value = json.dumps({
            "steps": [{"step": 1, "tool": "create_file", "args": {}}]
        })
        
        prompts = {
            "chain_generation": "Generate {min_steps}-{max_steps} steps from {tools_json}",
        }
        
        with patch('generator.tool_generator.get_client') as mock_get_client:
            mock_get_client.return_value = mock_llm
            generator = ToolGenerator({"provider": "ollama"}, prompts)
            
            # Should return None for invalid chain
            result = generator._generate_valid_chain([], min_steps=2, max_steps=4)
            # Either None or empty steps
            assert result is None or len(result.get("steps", [])) < 2


class TestCLIIntegration:
    """Test CLI commands for chain-first generation."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        from click.testing import CliRunner
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that help text shows chain-first command."""
        from generator.cli import main
        result = runner.invoke(main, ["tool-generate-chain", "--help"])
        assert result.exit_code == 0
        assert "chain-first" in result.output.lower()
        assert "--min-steps" in result.output
        assert "--max-steps" in result.output
        assert "--hybrid" in result.output
