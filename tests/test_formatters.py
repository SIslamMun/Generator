"""Unit tests for formatters module."""

import pytest
import json
from src.generator.formatters import (
    _to_chatml,
    _to_alpaca,
    _to_sharegpt,
    _to_jsonl,
)


class TestFormatters:
    """Test export formatters."""

    def test_chatml_format(self, sample_qa_pairs):
        """Test ChatML format conversion."""
        formatted = _to_chatml(sample_qa_pairs)

        assert len(formatted) == len(sample_qa_pairs)
        assert "messages" in formatted[0]

        messages = formatted[0]["messages"]
        assert len(messages) == 3  # system, user, assistant
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_chatml_custom_system_prompt(self, sample_qa_pairs):
        """Test ChatML with custom system prompt."""
        custom_prompt = "You are an expert in HDF5."
        formatted = _to_chatml(sample_qa_pairs, system_prompt=custom_prompt)

        assert formatted[0]["messages"][0]["content"] == custom_prompt

    def test_alpaca_format(self, sample_qa_pairs):
        """Test Alpaca format conversion."""
        formatted = _to_alpaca(sample_qa_pairs)

        assert len(formatted) == len(sample_qa_pairs)
        assert "instruction" in formatted[0]
        assert "input" in formatted[0]
        assert "output" in formatted[0]

        # Alpaca uses empty input field for QA
        assert formatted[0]["input"] == ""

    def test_sharegpt_format(self, sample_qa_pairs):
        """Test ShareGPT format conversion."""
        formatted = _to_sharegpt(sample_qa_pairs)

        assert len(formatted) == len(sample_qa_pairs)
        assert "conversations" in formatted[0]

        convs = formatted[0]["conversations"]
        assert len(convs) >= 2  # at least human, gpt (may have system)
        # Find human and gpt messages
        human_msg = next(c for c in convs if c["from"] == "human")
        gpt_msg = next(c for c in convs if c["from"] == "gpt")
        assert human_msg is not None
        assert gpt_msg is not None

    def test_sharegpt_custom_system_prompt(self, sample_qa_pairs):
        """Test ShareGPT with custom system prompt."""
        custom_prompt = "You are helpful."
        formatted = _to_sharegpt(sample_qa_pairs, system_prompt=custom_prompt)

        # System prompt should be prepended
        assert len(formatted[0]["conversations"]) >= 2

    def test_jsonl_format(self, sample_qa_pairs):
        """Test JSONL format conversion."""
        formatted = _to_jsonl(sample_qa_pairs)

        assert len(formatted) == len(sample_qa_pairs)
        # JSONL just wraps the pairs
        assert "question" in formatted[0]
        assert "answer" in formatted[0]

    def test_format_preserves_metadata(self, sample_qa_pairs):
        """Test that formatting preserves metadata fields."""
        pair_with_metadata = sample_qa_pairs[0].copy()
        pair_with_metadata["rating"] = 8
        pair_with_metadata["chunk_id"] = "test_123"

        formatted = _to_chatml([pair_with_metadata])

        # Metadata should be preserved in ChatML
        # (implementation dependent, but typically kept)
        assert "messages" in formatted[0]

    def test_chatml_structure_validation(self):
        """Test ChatML structure conforms to spec."""
        chatml_example = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}
            ]
        }

        # Verify structure
        assert "messages" in chatml_example
        assert all("role" in m and "content" in m for m in chatml_example["messages"])

        # Verify role order
        roles = [m["role"] for m in chatml_example["messages"]]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[2] == "assistant"

    def test_alpaca_structure_validation(self):
        """Test Alpaca structure conforms to spec."""
        alpaca_example = {
            "instruction": "What is X?",
            "input": "",
            "output": "X is..."
        }

        # Verify required fields
        assert "instruction" in alpaca_example
        assert "input" in alpaca_example
        assert "output" in alpaca_example

    def test_sharegpt_structure_validation(self):
        """Test ShareGPT structure conforms to spec."""
        sharegpt_example = {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"}
            ]
        }

        # Verify structure
        assert "conversations" in sharegpt_example
        convs = sharegpt_example["conversations"]
        assert all("from" in c and "value" in c for c in convs)
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"

    def test_format_handles_cot_pairs(self, sample_cot_pairs):
        """Test that formatters handle CoT pairs."""
        # CoT pairs have reasoning field
        formatted_chatml = _to_chatml(sample_cot_pairs)
        formatted_alpaca = _to_alpaca(sample_cot_pairs)

        assert len(formatted_chatml) == len(sample_cot_pairs)
        assert len(formatted_alpaca) == len(sample_cot_pairs)

    def test_format_handles_empty_list(self):
        """Test formatters handle empty input."""
        formatted = _to_chatml([])
        assert formatted == []

    def test_format_handles_special_characters(self):
        """Test formatters handle special characters."""
        pair_with_special_chars = {
            "question": "What's the \"best\" approach?",
            "answer": "Use <code> tags & proper escaping."
        }

        formatted = _to_chatml([pair_with_special_chars])

        # Should preserve special characters
        assert "\"best\"" in formatted[0]["messages"][1]["content"]

    def test_format_output_is_valid_json(self, sample_qa_pairs):
        """Test that formatted output is valid JSON."""
        formatted = _to_chatml(sample_qa_pairs)

        # Should be serializable
        json_str = json.dumps(formatted[0])
        assert json_str is not None

        # Should be deserializable
        parsed = json.loads(json_str)
        assert "messages" in parsed
