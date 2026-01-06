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
def sample_cot_pairs() -> List[Dict[str, Any]]:
    """Sample CoT pairs for testing."""
    return [
        {
            "question": "How does HDF5 handle parallel I/O?",
            "reasoning": "Step 1: HDF5 uses MPI-IO for parallel operations.\nStep 2: Multiple processes can access the same file.\nStep 3: Collective operations ensure data consistency.",
            "answer": "HDF5 handles parallel I/O through MPI-IO, enabling multiple processes to access shared files with coordinated operations.",
            "chunk_id": "test_chunk_1",
            "source": "test.md"
        },
        {
            "question": "What are the benefits of chunking in HDF5?",
            "reasoning": "Step 1: Chunking divides datasets into fixed-size blocks.\nStep 2: This enables efficient partial I/O operations.\nStep 3: Compression works better on chunks.",
            "answer": "Chunking in HDF5 improves performance by enabling partial I/O and better compression.",
            "chunk_id": "test_chunk_2",
            "source": "test.md"
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
        "cot_generation": """Create {n_pairs} complex reasoning examples from this text.

Text: {text}

Return JSON with "question", "reasoning", and "answer" fields.""",
        "cot_enhancement": """Add step-by-step reasoning to these conversations.

Conversations: {conversations}

Return enhanced conversations with reasoning.""",
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
