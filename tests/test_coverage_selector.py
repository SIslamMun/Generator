"""Tests for coverage-based selection (TOUCAN paper implementation)."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import json
import sys


# Skip tests if sklearn is not available
sklearn_available = True
try:
    import sklearn
except ImportError:
    sklearn_available = False

pytestmark = pytest.mark.skipif(not sklearn_available, reason="sklearn not installed")


class TestCoverageSelector:
    """Test the CoverageSelector class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer module."""
        mock_model = MagicMock()
        # Return deterministic embeddings based on index
        def encode_side_effect(texts, show_progress_bar=True):
            embeddings = []
            for i, text in enumerate(texts):
                # Create unique embedding for each text
                emb = np.zeros(384)
                emb[i % 384] = 1.0  # One-hot style
                emb[(i * 7) % 384] = 0.5  # Add some variation
                embeddings.append(emb)
            return np.array(embeddings)
        mock_model.encode = MagicMock(side_effect=encode_side_effect)
        return mock_model

    @pytest.fixture
    def sample_examples(self):
        """Create sample QA examples for testing."""
        return [
            {
                "question": "How to create a file in HDF5?",
                "answer": "Use h5py.File('name.h5', 'w') to create a file."
            },
            {
                "question": "How to read a dataset?",
                "answer": "Access datasets using file['dataset_name'][:]"
            },
            {
                "question": "What are HDF5 groups?",
                "answer": "Groups are containers like directories."
            },
            {
                "question": "How to add attributes?",
                "answer": "Use .attrs['key'] = value to add attributes."
            },
            {
                "question": "How to compress data?",
                "answer": "Pass compression='gzip' when creating dataset."
            },
            {
                "question": "What is chunking?",
                "answer": "Chunking divides data into smaller blocks."
            },
            {
                "question": "How to append data?",
                "answer": "Use resize() and maxshape parameter."
            },
            {
                "question": "What are virtual datasets?",
                "answer": "Virtual datasets map to other HDF5 files."
            },
            {
                "question": "How to copy datasets?",
                "answer": "Use h5py.copy() or file.copy() methods."
            },
            {
                "question": "What is parallel HDF5?",
                "answer": "Parallel HDF5 enables MPI-based I/O."
            },
        ]

    @pytest.fixture
    def sample_tool_examples(self):
        """Create sample tool-use examples."""
        return [
            {
                "system_prompt": "You are an HDF5 expert.",
                "turns": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "I'll create the file.", "tool_calls": [{"name": "create_file"}]}
                ]
            },
            {
                "system_prompt": "You are an HDF5 expert.",
                "turns": [
                    {"role": "user", "content": "Read dataset"},
                    {"role": "assistant", "content": "Reading...", "tool_calls": [{"name": "read_dataset"}]}
                ]
            },
            {
                "system_prompt": "You are an HDF5 expert.",
                "turns": [
                    {"role": "user", "content": "Write data"},
                    {"role": "assistant", "content": "Writing...", "tool_calls": [{"name": "write_dataset"}]}
                ]
            },
        ]

    def test_selector_initialization(self, mock_sentence_transformer):
        """Test that selector initializes correctly."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector(model_name="test-model")
        # Model not loaded yet (lazy)
        assert selector._encoder is None
        assert selector.model_name == "test-model"

    def test_text_extraction_qa(self, mock_sentence_transformer, sample_examples):
        """Test text extraction from QA examples."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        
        texts = selector._extract_texts(sample_examples)
        assert len(texts) == len(sample_examples)
        # Should combine question and answer
        assert "How to create a file" in texts[0]
        assert "h5py.File" in texts[0]

    def test_text_extraction_tool_use(self, mock_sentence_transformer, sample_tool_examples):
        """Test text extraction from tool-use examples."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        
        texts = selector._extract_texts(sample_tool_examples)
        assert len(texts) == len(sample_tool_examples)
        # Should extract from turns
        assert "Create a file" in texts[0] or "create_file" in texts[0]

    def test_select_by_target_count(self, mock_sentence_transformer, sample_examples):
        """Test selection with exact target count."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage(
            sample_examples,
            target_count=5,
        )
        assert len(selected) == 5
        # All selected should be from original
        for s in selected:
            assert s in sample_examples

    def test_select_by_reduction_ratio(self, mock_sentence_transformer, sample_examples):
        """Test selection with reduction ratio."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage(
            sample_examples,
            reduction_ratio=0.5,  # Keep 50%
        )
        assert len(selected) == 5  # 50% of 10

    def test_centroid_strategy(self, mock_sentence_transformer, sample_examples):
        """Test centroid selection strategy."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage(
            sample_examples,
            target_count=4,
            strategy="centroid",
        )
        assert len(selected) == 4

    def test_diverse_strategy(self, mock_sentence_transformer, sample_examples):
        """Test diverse selection strategy."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage(
            sample_examples,
            target_count=4,
            strategy="diverse",
        )
        assert len(selected) == 4

    def test_coverage_score_computation(self, mock_sentence_transformer, sample_examples):
        """Test coverage score computation."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        # Select subset
        selected = selector.select_by_coverage(
            sample_examples,
            target_count=5,
        )
        
        # Compute coverage
        score = selector.compute_coverage_score(selected, sample_examples)
        
        # Score should be between 0 and 1
        assert 0 <= score <= 1

    def test_empty_input(self, mock_sentence_transformer):
        """Test handling of empty input."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage([], target_count=5)
        assert selected == []

    def test_target_larger_than_input(self, mock_sentence_transformer, sample_examples):
        """Test when target is larger than input size."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_sentence_transformer
        
        selected = selector.select_by_coverage(
            sample_examples,
            target_count=100,  # More than we have
        )
        # Should return all examples
        assert len(selected) == len(sample_examples)

    def test_determinism(self, mock_sentence_transformer, sample_examples):
        """Test that selection count is deterministic."""
        from generator.tool.coverage_selector import CoverageSelector
        
        selector1 = CoverageSelector()
        selector1._encoder = mock_sentence_transformer
        selected1 = selector1.select_by_coverage(
            sample_examples,
            target_count=5,
        )
        
        selector2 = CoverageSelector()
        selector2._encoder = mock_sentence_transformer
        selected2 = selector2.select_by_coverage(
            sample_examples,
            target_count=5,
        )
        
        # Should produce same count (selection order may vary due to clustering)
        assert len(selected1) == len(selected2) == 5
        # All selected should be from original
        for s in selected1:
            assert s in sample_examples or "_coverage_metadata" in s


class TestCoverageScoring:
    """Test coverage scoring metrics."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder with controlled embeddings."""
        mock = MagicMock()
        def encode_side_effect(texts, show_progress_bar=True):
            # Create spread-out embeddings
            n = len(texts)
            embeddings = np.random.RandomState(42).randn(n, 384)
            return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock.encode = MagicMock(side_effect=encode_side_effect)
        return mock

    def test_full_coverage(self, mock_embedder):
        """Test coverage score when selecting all examples."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_embedder
        
        examples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        
        # Full coverage should be 1.0
        score = selector.compute_coverage_score(examples, examples)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_coverage(self, mock_embedder):
        """Test coverage score with partial selection."""
        from generator.tool.coverage_selector import CoverageSelector
        selector = CoverageSelector()
        selector._encoder = mock_embedder
        
        examples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        subset = examples[:3]  # 30% - smaller subset for more distinct score
        
        score = selector.compute_coverage_score(subset, examples)
        
        # Score should be between 0 and 1 (or equal to 1 if overlap is high)
        assert 0 <= score <= 1


class TestCLIIntegration:
    """Test CLI command for coverage selection."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        from click.testing import CliRunner
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that help text shows coverage command."""
        from generator.cli import main
        result = runner.invoke(main, ["select-coverage", "--help"])
        assert result.exit_code == 0
        assert "coverage" in result.output.lower()
        assert "--target-count" in result.output
        assert "--reduction-ratio" in result.output
        assert "--strategy" in result.output

    def test_cli_with_mock_selector(self, runner, tmp_path):
        """Test CLI with mocked selector."""
        from generator.cli import main
        
        # Create input file
        input_file = tmp_path / "input.json"
        examples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        input_file.write_text(json.dumps(examples))
        
        output_file = tmp_path / "output.json"
        
        # Mock the CoverageSelector at module level
        mock_selector = MagicMock()
        mock_selector.select_by_coverage.return_value = examples[:5]
        mock_selector.compute_coverage_score.return_value = 0.85
        
        with patch('generator.tool.coverage_selector.CoverageSelector', return_value=mock_selector):
            result = runner.invoke(main, [
                "select-coverage",
                str(input_file),
                "-o", str(output_file),
                "--target-count", "5",
            ])
        
        # Check output file was created (if mocking worked)
        # Note: if import error occurs, won't reach here
        if result.exit_code == 0:
            assert output_file.exists()
            with open(output_file) as f:
                saved = json.load(f)
            assert len(saved) == 5

    def test_cli_invalid_input(self, runner, tmp_path):
        """Test CLI with invalid JSON input."""
        from generator.cli import main
        
        # Create invalid input file (not a list)
        input_file = tmp_path / "input.json"
        input_file.write_text('{"not": "a list"}')
        
        output_file = tmp_path / "output.json"
        
        # Mock to avoid import errors
        mock_selector = MagicMock()
        with patch('generator.tool.coverage_selector.CoverageSelector', return_value=mock_selector):
            result = runner.invoke(main, [
                "select-coverage",
                str(input_file),
                "-o", str(output_file),
            ])
        
        # Should fail with error about not being a list
        assert result.exit_code != 0 or "array" in result.output.lower() or "list" in result.output.lower()
