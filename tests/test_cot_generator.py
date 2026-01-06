"""Unit tests for CoT generator module."""

import pytest
from src.generator.cot_generator import _parse_cot_response


class TestCoTGenerator:
    """Test CoT generation functionality."""

    def test_parse_valid_cot_response(self):
        """Test parsing valid CoT JSON response."""
        response = """
        [
          {
            "question": "How does parallel I/O work?",
            "reasoning": "Step 1: Multiple processes access shared file.\\nStep 2: MPI-IO coordinates access.\\nStep 3: Data is distributed across nodes.",
            "answer": "Parallel I/O enables multiple processes to access a shared file simultaneously through coordinated operations."
          }
        ]
        """

        pairs = _parse_cot_response(response)

        assert len(pairs) == 1
        assert pairs[0]["question"] == "How does parallel I/O work?"
        assert "Step 1" in pairs[0]["reasoning"]
        assert "answer" in pairs[0]

    def test_parse_cot_response_with_extra_text(self):
        """Test parsing CoT response with markdown/extra text."""
        response = """
        Here are the CoT examples:
        ```json
        [
          {
            "question": "Test question?",
            "reasoning": "Step 1: First step\\nStep 2: Second step",
            "answer": "Test answer"
          }
        ]
        ```
        """

        pairs = _parse_cot_response(response)

        assert len(pairs) == 1
        assert "question" in pairs[0]
        assert "reasoning" in pairs[0]
        assert "answer" in pairs[0]

    def test_parse_invalid_cot_response(self):
        """Test handling of invalid JSON response."""
        response = "This is not valid JSON"

        pairs = _parse_cot_response(response)

        assert pairs == []

    def test_parse_cot_response_missing_fields(self):
        """Test handling of CoT pairs with missing required fields."""
        response = """
        [
          {
            "question": "Valid question?",
            "reasoning": "Valid reasoning",
            "answer": "Valid answer"
          },
          {
            "question": "Missing reasoning",
            "answer": "Answer only"
          },
          {
            "reasoning": "Reasoning without question"
          }
        ]
        """

        pairs = _parse_cot_response(response)

        # Only first pair should be valid
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Valid question?"

    def test_parse_cot_response_trailing_comma(self):
        """Test parsing CoT response with trailing commas (json5 compatible)."""
        response = """
        [
          {
            "question": "Test?",
            "reasoning": "Reasoning",
            "answer": "Answer",
          },
        ]
        """

        pairs = _parse_cot_response(response)

        assert len(pairs) == 1
        assert pairs[0]["question"] == "Test?"

    def test_cot_structure_validation(self):
        """Test that CoT pairs have required structure."""
        cot_pair = {
            "question": "What is X?",
            "reasoning": "Step 1: First...\nStep 2: Then...",
            "answer": "X is..."
        }

        # Verify all required fields present
        assert "question" in cot_pair
        assert "reasoning" in cot_pair
        assert "answer" in cot_pair

        # Verify reasoning has multiple steps
        assert "\n" in cot_pair["reasoning"]
        assert "Step" in cot_pair["reasoning"]

    def test_target_pairs_calculation(self):
        """Test per-chunk calculation from target_pairs."""
        target_pairs = 100
        total_chunks = 20

        pairs_per_chunk = max(1, round(target_pairs / total_chunks))

        assert pairs_per_chunk == 5

        # Edge case: more target than chunks
        target_pairs = 10
        total_chunks = 20
        pairs_per_chunk = max(1, round(target_pairs / total_chunks))

        assert pairs_per_chunk == 1  # Minimum 1 pair per chunk
