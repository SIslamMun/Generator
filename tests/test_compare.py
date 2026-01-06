"""Unit tests for compare module."""

import pytest
import json
from pathlib import Path
from src.generator.compare import DatasetComparator


class TestCompare:
    """Test dataset comparison functionality."""

    @pytest.fixture
    def comparator(self):
        """Create a DatasetComparator instance for testing."""
        llm_config = {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "api_key": "test-key"
        }
        return DatasetComparator(llm_config)

    @pytest.fixture
    def sample_datasets(self, tmp_path):
        """Create sample dataset files for testing."""
        dataset1 = tmp_path / "dataset1.json"
        dataset2 = tmp_path / "dataset2.json"
        
        data1 = [
            {"question": "What is X?", "answer": "X is...", "rating": 8},
            {"question": "What is Y?", "answer": "Y is...", "rating": 7},
        ]
        data2 = [
            {"question": "How does Z work?", "answer": "Z works by...", "rating": 9},
            {"question": "Why use Z?", "answer": "Because...", "rating": 8},
        ]
        
        dataset1.write_text(json.dumps(data1))
        dataset2.write_text(json.dumps(data2))
        
        return [dataset1, dataset2]

    def test_load_datasets(self, comparator, sample_datasets):
        """Test loading multiple datasets."""
        datasets = comparator.load_datasets(sample_datasets)

        assert len(datasets) == 2
        assert "dataset1" in datasets
        assert "dataset2" in datasets
        assert len(datasets["dataset1"]) == 2

    def test_load_datasets_empty_file(self, comparator, tmp_path):
        """Test handling of empty dataset file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]")

        datasets = comparator.load_datasets([empty_file])

        assert len(datasets) == 1
        assert len(datasets["empty"]) == 0

    def test_compute_metrics_basic(self, comparator):
        """Test computing basic dataset metrics."""
        dataset = [
            {"question": "What is HDF5?", "answer": "HDF5 is a file format."},
            {"question": "How does it work?", "answer": "It works by..."},
        ]

        metrics = comparator.compute_metrics(dataset)

        assert metrics["count"] == 2
        assert "avg_question_length" in metrics
        assert "avg_answer_length" in metrics
        assert metrics["avg_question_length"] > 0

    def test_compute_metrics_with_ratings(self, comparator):
        """Test metrics computation with rated pairs."""
        dataset = [
            {"question": "Q1", "answer": "A1", "rating": 8},
            {"question": "Q2", "answer": "A2", "rating": 7},
            {"question": "Q3", "answer": "A3", "rating": 9},
        ]

        metrics = comparator.compute_metrics(dataset)

        assert "avg_rating" in metrics
        assert metrics["avg_rating"] == 8.0
        assert "min_rating" in metrics
        assert metrics["min_rating"] == 7
        assert "max_rating" in metrics
        assert metrics["max_rating"] == 9

    def test_compute_metrics_source_diversity(self, comparator):
        """Test source diversity calculation."""
        dataset = [
            {"question": "Q1", "answer": "A1", "source": "paper1"},
            {"question": "Q2", "answer": "A2", "source": "paper1"},
            {"question": "Q3", "answer": "A3", "source": "paper2"},
        ]

        metrics = comparator.compute_metrics(dataset)

        assert "unique_sources" in metrics
        assert metrics["unique_sources"] == 2
        assert "source_distribution" in metrics

    def test_compute_metrics_question_types(self, comparator):
        """Test question type diversity calculation."""
        dataset = [
            {"question": "What is X?", "answer": "A1"},
            {"question": "What is Y?", "answer": "A2"},
            {"question": "How does Z work?", "answer": "A3"},
            {"question": "Why use Z?", "answer": "A4"},
        ]

        metrics = comparator.compute_metrics(dataset)

        assert "question_type_diversity" in metrics
        assert "top_question_types" in metrics
        assert metrics["question_type_diversity"] >= 2  # At least "what", "how", "why"

    def test_sample_pairs_exact_count(self, comparator):
        """Test sampling when dataset size equals sample size."""
        dataset = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)
        ]

        sampled = comparator.sample_pairs(dataset, 5)

        assert len(sampled) == 5

    def test_sample_pairs_undersample(self, comparator):
        """Test sampling fewer pairs than available."""
        dataset = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)
        ]

        sampled = comparator.sample_pairs(dataset, 5)

        assert len(sampled) == 5
        assert all("question" in pair for pair in sampled)

    def test_sample_pairs_oversample(self, comparator):
        """Test requesting more samples than available."""
        dataset = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        sampled = comparator.sample_pairs(dataset, 10)

        # Should return all available pairs
        assert len(sampled) == 2

    def test_compute_metrics_empty_dataset(self, comparator):
        """Test metrics computation on empty dataset."""
        metrics = comparator.compute_metrics([])

        assert metrics["count"] == 0
        assert "error" in metrics

    def test_metrics_calculation_accuracy(self, comparator):
        """Test accuracy of length calculations."""
        dataset = [
            {"question": "12345", "answer": "123"},  # 5 and 3
            {"question": "123", "answer": "12345"},  # 3 and 5
        ]

        metrics = comparator.compute_metrics(dataset)

        assert metrics["avg_question_length"] == 4.0  # (5+3)/2
        assert metrics["avg_answer_length"] == 4.0    # (3+5)/2

    def test_sample_pairs_preserves_structure(self, comparator):
        """Test that sampling preserves pair structure."""
        dataset = [
            {"question": "Q1", "answer": "A1", "rating": 8, "chunk_id": "c1"},
            {"question": "Q2", "answer": "A2", "rating": 7, "chunk_id": "c2"},
        ]

        sampled = comparator.sample_pairs(dataset, 2)

        # Should preserve all fields
        assert "rating" in sampled[0]
        assert "chunk_id" in sampled[0]
        assert all(k in sampled[0] for k in ["question", "answer"])

    def test_rating_distribution(self, comparator):
        """Test rating distribution calculation."""
        dataset = [
            {"question": "Q1", "answer": "A1", "rating": 8},
            {"question": "Q2", "answer": "A2", "rating": 8},
            {"question": "Q3", "answer": "A3", "rating": 7},
            {"question": "Q4", "answer": "A4", "rating": 9},
        ]

        metrics = comparator.compute_metrics(dataset)

        assert "rating_distribution" in metrics
        assert metrics["rating_distribution"][8] == 2
        assert metrics["rating_distribution"][7] == 1
        assert metrics["rating_distribution"][9] == 1
