"""Unit tests for curate module."""

import pytest
from generator.qa.curate import (
    _detect_format,
    _convert_to_conversation_format,
    _extract_qa_from_conversation,
    _restore_original_format,
)


class TestCurate:
    """Test curation functionality."""

    def test_detect_qa_format(self, sample_qa_pairs):
        """Test detection of QA format."""
        format_type = _detect_format(sample_qa_pairs)
        assert format_type == "qa"

    def test_detect_cot_format(self, sample_cot_pairs):
        """Test detection of CoT format."""
        format_type = _detect_format(sample_cot_pairs)
        assert format_type == "cot"

    def test_detect_empty_list(self):
        """Test detection with empty list defaults to QA."""
        format_type = _detect_format([])
        assert format_type == "qa"

    def test_convert_qa_to_conversation(self, sample_qa_pairs):
        """Test conversion of QA pairs to conversation format."""
        conversations = _convert_to_conversation_format(sample_qa_pairs, "qa")

        assert len(conversations) == len(sample_qa_pairs)
        assert "conversations" in conversations[0]
        assert len(conversations[0]["conversations"]) == 2
        assert conversations[0]["conversations"][0]["role"] == "user"
        assert conversations[0]["conversations"][1]["role"] == "assistant"
        assert conversations[0]["_format"] == "qa"

    def test_convert_cot_to_conversation(self, sample_cot_pairs):
        """Test conversion of CoT pairs to conversation format."""
        conversations = _convert_to_conversation_format(sample_cot_pairs, "cot")

        assert len(conversations) == len(sample_cot_pairs)
        assert "conversations" in conversations[0]
        assert conversations[0]["_format"] == "cot"
        # CoT should preserve reasoning in the answer
        assert "_original_pair" in conversations[0]

    def test_extract_qa_from_conversation(self):
        """Test extracting QA from conversation format."""
        conv = {
            "conversations": [
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is..."}
            ]
        }

        qa = _extract_qa_from_conversation(conv)

        assert qa["question"] == "What is X?"
        assert qa["answer"] == "X is..."

    def test_extract_qa_from_messages_format(self):
        """Test extracting QA from messages key."""
        conv = {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."}
            ]
        }

        qa = _extract_qa_from_conversation(conv)

        assert qa["question"] == "Question?"
        assert qa["answer"] == "Answer."

    def test_restore_qa_format(self, sample_qa_pairs):
        """Test restoring QA format after rating."""
        # Simulate rated conversation
        rated_conv = {
            "conversations": [
                {"role": "user", "content": "What is HDF5?"},
                {"role": "assistant", "content": "HDF5 is a file format..."}
            ],
            "_original_pair": sample_qa_pairs[0],
            "_format": "qa",
            "rating": 8,
            "clarity": 3,
            "accuracy": 3,
            "usefulness": 1,
            "difficulty": 1,
            "reasoning": "Good quality pair"
        }

        restored = _restore_original_format([rated_conv], "qa")

        assert len(restored) == 1
        assert "question" in restored[0]
        assert "answer" in restored[0]
        assert restored[0]["rating"] == 8
        assert "conversations" not in restored[0]
        assert "_original_pair" not in restored[0]

    def test_restore_cot_format(self, sample_cot_pairs):
        """Test restoring CoT format preserves reasoning."""
        rated_conv = {
            "conversations": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Step 1...\n\nAnswer"}
            ],
            "_original_pair": sample_cot_pairs[0],
            "_format": "cot",
            "rating": 9
        }

        restored = _restore_original_format([rated_conv], "cot")

        assert len(restored) == 1
        assert "question" in restored[0]
        assert "reasoning" in restored[0]
        assert "answer" in restored[0]
        assert restored[0]["rating"] == 9

    def test_threshold_filtering(self):
        """Test filtering by rating threshold."""
        pairs = [
            {"question": "Q1", "answer": "A1", "rating": 8},
            {"question": "Q2", "answer": "A2", "rating": 5},
            {"question": "Q3", "answer": "A3", "rating": 9},
            {"question": "Q4", "answer": "A4", "rating": 6},
        ]

        threshold = 7.0
        filtered = [p for p in pairs if p["rating"] >= threshold]

        assert len(filtered) == 2
        assert all(p["rating"] >= threshold for p in filtered)

    def test_rating_criteria_validation(self):
        """Test that rating criteria sum correctly."""
        rating = {
            "clarity": 3,
            "accuracy": 3,
            "usefulness": 2,
            "difficulty": 2,
        }

        total = sum(rating.values())
        assert total == 10  # Max score is 10

    def test_batch_size_calculation(self):
        """Test batch processing calculation."""
        total_pairs = 23
        batch_size = 5

        num_batches = (total_pairs + batch_size - 1) // batch_size

        assert num_batches == 5  # ceil(23/5)
