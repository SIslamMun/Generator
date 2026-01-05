"""Pytest configuration and shared fixtures for Generator tests."""

import pytest
from pathlib import Path
from typing import Dict, Any, List


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_qa_pairs() -> List[Dict[str, Any]]:
    """Sample QA pairs for testing."""
    return [
        {
            "question": "What is HDF5?",
            "answer": "HDF5 is a file format and library for storing scientific data.",
            "chunk_id": "test_chunk_1",
            "source_file": "test.md",
            "generated_at": "2026-01-05T12:00:00",
            "model": "gemini-2.0-flash-exp"
        },
        {
            "question": "What is parallel I/O?",
            "answer": "Parallel I/O allows multiple processes to read/write data simultaneously.",
            "chunk_id": "test_chunk_2",
            "source_file": "test.md",
            "generated_at": "2026-01-05T12:00:01",
            "model": "gemini-2.0-flash-exp"
        },
    ]


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "llm": {
            "provider": "ollama",
            "model": "mistral:latest",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "generation": {
            "n_pairs_per_chunk": 3,
            "batch_size": 50,
            "max_retries": 3,
        },
    }


@pytest.fixture
def sample_prompts() -> Dict[str, str]:
    """Sample prompts for testing."""
    return {
        "qa_generation": """Given the following text, generate {n_pairs} diverse question-answer pairs.

Text: {text}

Return a JSON array of objects with "question" and "answer" fields.""",
        "qa_rating": """Rate the following QA pair on a scale of 1-10.

Question: {question}
Answer: {answer}

Return only a number between 1 and 10.""",
    }


@pytest.fixture
def mock_lancedb_chunk() -> Dict[str, Any]:
    """Mock LanceDB chunk for testing."""
    return {
        "id": "test_chunk_123",
        "content": "HDF5 is a file format designed for storing large amounts of numerical data. "
                   "It provides a flexible and efficient I/O library with support for parallel I/O.",
        "source_file": "papers/hdf5_intro.md",
        "token_count": 150,
    }
