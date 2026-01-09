"""Tests for parameter-level dependency graphs (In-N-Out paper implementation)."""

import pytest
from unittest.mock import MagicMock


class TestDependencyGraph:
    """Test the DependencyGraph class."""

    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        from generator.tool.tool_schemas import Tool, Parameter
        
        return [
            Tool(
                tool_id="open_file",
                name="open_file",
                description="Open an HDF5 file",
                parameters=[
                    Parameter(name="path", type="string", description="File path", required=True),
                ],
                returns={"type": "file_handle", "description": "File handle"},
                category="file",
            ),
            Tool(
                tool_id="read_dataset",
                name="read_dataset",
                description="Read a dataset from file",
                parameters=[
                    Parameter(name="file", type="file_handle", description="File handle", required=True),
                    Parameter(name="path", type="string", description="Dataset path", required=True),
                ],
                returns={"type": "array", "description": "Dataset data"},
                category="dataset",
            ),
            Tool(
                tool_id="process_data",
                name="process_data",
                description="Process array data",
                parameters=[
                    Parameter(name="data", type="array", description="Input data", required=True),
                ],
                returns={"type": "array", "description": "Processed data"},
                category="processing",
            ),
            Tool(
                tool_id="close_file",
                name="close_file",
                description="Close an HDF5 file",
                parameters=[
                    Parameter(name="file", type="file_handle", description="File handle", required=True),
                ],
                returns={"type": "null", "description": "None"},
                category="file",
            ),
            Tool(
                tool_id="save_result",
                name="save_result",
                description="Save array to file",
                parameters=[
                    Parameter(name="file", type="file_handle", description="File handle", required=True),
                    Parameter(name="data", type="array", description="Data to save", required=True),
                    Parameter(name="path", type="string", description="Dataset path", required=True),
                ],
                returns={"type": "string", "description": "Success message"},
                category="dataset",
            ),
        ]

    def test_graph_initialization(self):
        """Test that graph initializes correctly."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_build_from_tools(self, sample_tools):
        """Test building graph from tools."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # Should have nodes for each tool
        assert len(graph.nodes) == len(sample_tools)
        assert "open_file" in graph.nodes
        assert "read_dataset" in graph.nodes

    def test_adjacency_created(self, sample_tools):
        """Test that adjacency lists are created."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # open_file output (file_handle) should connect to tools that accept it
        successors = graph.get_successors("open_file")
        assert "read_dataset" in successors  # Takes file_handle
        assert "close_file" in successors    # Takes file_handle
        assert "save_result" in successors   # Takes file_handle

    def test_reverse_adjacency(self, sample_tools):
        """Test reverse adjacency (predecessors)."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # read_dataset can receive from tools that output file_handle
        predecessors = graph.get_predecessors("read_dataset")
        assert "open_file" in predecessors

    def test_validate_valid_chain(self, sample_tools):
        """Test validating a valid tool chain."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # open_file -> read_dataset -> process_data is valid
        chain = ["open_file", "read_dataset", "process_data"]
        is_valid, issues = graph.validate_chain(chain)
        
        assert is_valid
        assert len(issues) == 0

    def test_validate_invalid_chain(self, sample_tools):
        """Test validating an invalid tool chain."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # process_data -> open_file is invalid (array doesn't feed into string input)
        chain = ["process_data", "open_file"]
        is_valid, issues = graph.validate_chain(chain)
        
        # Should be invalid
        assert not is_valid or len(issues) > 0

    def test_get_compatible_chains(self, sample_tools):
        """Test finding compatible chains."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        chains = graph.get_compatible_chains("open_file", max_length=3, min_length=2)
        
        # Should find chains starting from open_file
        assert len(chains) > 0
        
        # All chains should start with open_file
        for chain in chains:
            assert chain[0] == "open_file"
            assert len(chain) >= 2

    def test_find_bridges(self, sample_tools):
        """Test finding bridge tools between incompatible tools."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        
        # Find bridges between open_file and process_data
        # (open_file outputs file_handle, process_data wants array)
        bridges = graph.find_bridges("open_file", "process_data")
        
        # read_dataset should be a bridge (takes file_handle, outputs array)
        bridge_tools = [b[0] if b else None for b in bridges]
        assert "read_dataset" in bridge_tools or bridges == [[]]

    def test_statistics(self, sample_tools):
        """Test graph statistics."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        stats = graph.get_statistics()
        
        assert stats["nodes"] == 5
        assert stats["edges"] > 0
        assert "avg_in_degree" in stats
        assert "avg_out_degree" in stats

    def test_to_dict(self, sample_tools):
        """Test exporting graph to dictionary."""
        from generator.tool.dependency_graph import DependencyGraph
        
        graph = DependencyGraph(sample_tools)
        data = graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert "adjacency" in data
        assert "statistics" in data
        assert len(data["nodes"]) == 5


class TestTypeCompatibility:
    """Test type compatibility checking."""

    def test_exact_match(self):
        """Test exact type matches."""
        from generator.tool.dependency_graph import is_type_compatible
        
        assert is_type_compatible("string", "string")
        assert is_type_compatible("integer", "integer")
        assert is_type_compatible("array", "array")

    def test_any_type(self):
        """Test 'any' type compatibility."""
        from generator.tool.dependency_graph import is_type_compatible
        
        assert is_type_compatible("string", "any")
        assert is_type_compatible("integer", "any")
        assert is_type_compatible("array", "any")

    def test_number_compatibility(self):
        """Test number type compatibility."""
        from generator.tool.dependency_graph import is_type_compatible
        
        assert is_type_compatible("integer", "number")
        assert is_type_compatible("number", "number")

    def test_incompatible_types(self):
        """Test incompatible types."""
        from generator.tool.dependency_graph import is_type_compatible
        
        assert not is_type_compatible("string", "integer")
        assert not is_type_compatible("array", "string")

    def test_compatibility_score(self):
        """Test compatibility scoring."""
        from generator.tool.dependency_graph import compute_compatibility_score
        
        # Exact match = 1.0
        assert compute_compatibility_score("string", "string") == 1.0
        
        # Any accepts = 0.8
        assert compute_compatibility_score("string", "any") == 0.8
        
        # Compatible = 0.6
        assert compute_compatibility_score("integer", "number") == 0.6
        
        # Incompatible = 0.0
        assert compute_compatibility_score("string", "integer") == 0.0


class TestCLIIntegration:
    """Test CLI command for dependency analysis."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        from click.testing import CliRunner
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that help text shows deps command."""
        from generator.cli import main
        result = runner.invoke(main, ["tool-deps", "--help"])
        assert result.exit_code == 0
        assert "dependency" in result.output.lower() or "deps" in result.output.lower()
        assert "--chains" in result.output
        assert "--validate" in result.output

    def test_cli_with_tools_file(self, runner, tmp_path):
        """Test CLI with actual tools file."""
        import json
        
        # Create a minimal tools file
        tools_data = [
            {
                "tool_id": "tool_a",
                "name": "tool_a",
                "description": "Tool A",
                "parameters": [{"name": "input", "type": "string", "description": "Input", "required": True}],
                "returns": {"type": "string", "description": "Output"},
                "category": "test",
            },
            {
                "tool_id": "tool_b",
                "name": "tool_b",
                "description": "Tool B",
                "parameters": [{"name": "input", "type": "string", "description": "Input", "required": True}],
                "returns": {"type": "array", "description": "Output"},
                "category": "test",
            },
        ]
        
        tools_file = tmp_path / "tools.json"
        tools_file.write_text(json.dumps(tools_data))
        
        from generator.cli import main
        result = runner.invoke(main, ["tool-deps", str(tools_file)])
        
        # Should run without error
        assert result.exit_code == 0


class TestToolNodeOperations:
    """Test ToolNode operations."""

    def test_accepts_type(self):
        """Test ToolNode.accepts_type method."""
        from generator.tool.dependency_graph import ToolNode
        
        node = ToolNode(
            tool_id="test",
            name="test",
            description="Test",
            inputs={"file": "file_handle", "path": "string", "data": "array"},
            output_type="string",
        )
        
        # file_handle should match 'file' parameter
        compatible = node.accepts_type("file_handle")
        assert "file" in compatible
        
        # string should match 'path' parameter
        compatible = node.accepts_type("string")
        assert "path" in compatible

    def test_input_types_property(self):
        """Test ToolNode.input_types property."""
        from generator.tool.dependency_graph import ToolNode
        
        node = ToolNode(
            tool_id="test",
            name="test",
            description="Test",
            inputs={"a": "string", "b": "integer", "c": "string"},
            output_type="any",
        )
        
        types = node.input_types
        assert "string" in types
        assert "integer" in types
        assert len(types) == 2  # Only unique types
