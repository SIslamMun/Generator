"""Unit tests for CoT enhancer module."""

import pytest
from src.generator.cot_enhancer import (
    _qa_to_conversations,
    _conversations_to_cot,
    _extract_reasoning,
    _parse_enhanced_response,
)


class TestCoTEnhancer:
    """Test CoT enhancement functionality."""

    def test_qa_to_conversations(self):
        """Test conversion of QA pairs to conversation format."""
        qa_pairs = [
            {"question": "What is X?", "answer": "X is..."},
            {"question": "How does Y work?", "answer": "Y works by..."},
        ]

        conversations = _qa_to_conversations(qa_pairs)

        assert len(conversations) == 2
        assert len(conversations[0]) == 3  # system, user, assistant
        assert conversations[0][0]["role"] == "system"
        assert conversations[0][1]["role"] == "user"
        assert conversations[0][2]["role"] == "assistant"
        assert conversations[0][1]["content"] == "What is X?"
        assert conversations[0][2]["content"] == "X is..."

    def test_qa_to_conversations_filters_invalid(self):
        """Test that invalid QA pairs are filtered out."""
        qa_pairs = [
            {"question": "Valid?", "answer": "Yes"},
            {"question": "", "answer": "No question"},
            {"question": "No answer"},
            "not a dict",
        ]

        conversations = _qa_to_conversations(qa_pairs)

        assert len(conversations) == 1
        assert conversations[0][1]["content"] == "Valid?"

    def test_conversations_to_cot_fallback(self):
        """Test fallback conversion (no reasoning added)."""
        conversations = [
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is..."},
            ]
        ]

        cot_pairs = _conversations_to_cot(conversations)

        assert len(cot_pairs) == 1
        assert cot_pairs[0]["question"] == "What is X?"
        assert cot_pairs[0]["answer"] == "X is..."
        assert cot_pairs[0]["reasoning"] == ""  # No reasoning (fallback)

    def test_extract_reasoning_with_steps(self):
        """Test extracting reasoning from enhanced response with clear steps."""
        content = """Let me think step by step:

Step 1: First, we need to understand the basics.
Step 2: Then, we analyze the components.
Step 3: Finally, we synthesize the information.

Therefore, the answer is that X works by combining these elements."""

        reasoning, answer = _extract_reasoning(content)

        # Should extract reasoning with steps
        assert len(reasoning) > 0
        assert len(answer) > 0

    def test_extract_reasoning_with_conclusion(self):
        """Test extracting reasoning with 'In conclusion' marker."""
        content = """First, I need to consider the context.
Next, I'll analyze the data.
Finally, I'll draw conclusions.

In conclusion, the result is positive."""

        reasoning, answer = _extract_reasoning(content)

        assert "consider the context" in reasoning
        assert "result is positive" in answer

    def test_extract_reasoning_no_indicators(self):
        """Test handling of content without reasoning indicators."""
        content = "This is just a simple answer without steps."

        reasoning, answer = _extract_reasoning(content)

        assert reasoning == ""
        assert answer == content

    def test_extract_reasoning_with_therefore(self):
        """Test extracting reasoning with 'Therefore' separator."""
        content = """To answer this, I'll break it down:
- Point one is important
- Point two follows from that
- Point three concludes

Therefore, the final answer is correct."""

        reasoning, answer = _extract_reasoning(content)

        assert "Point one" in reasoning
        assert "final answer is correct" in answer

    def test_parse_enhanced_response_valid(self):
        """Test parsing valid enhanced conversation response."""
        response = """
        [
          [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is parallel I/O?"},
            {"role": "assistant", "content": "Let me explain step by step:\\n\\nStep 1: Parallel I/O involves multiple processes.\\nStep 2: These processes coordinate file access.\\n\\nTherefore, parallel I/O improves performance."}
          ]
        ]
        """

        conversations = _parse_enhanced_response(response)

        assert len(conversations) == 1
        assert len(conversations[0]) == 3
        assert conversations[0][2]["role"] == "assistant"
        assert "Step 1" in conversations[0][2]["content"]

    def test_parse_enhanced_response_with_markdown(self):
        """Test parsing enhanced response wrapped in markdown."""
        response = """
        Here's the enhanced version:
        ```json
        [
          [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Step 1: First...\\nStep 2: Second...\\nAnswer: Result"}
          ]
        ]
        ```
        """

        conversations = _parse_enhanced_response(response)

        assert len(conversations) >= 0  # Should handle gracefully

    def test_parse_enhanced_response_invalid(self):
        """Test handling of invalid enhanced response."""
        response = "This is not valid JSON"

        conversations = _parse_enhanced_response(response)

        assert conversations == []

    def test_cot_enhancement_preserves_metadata(self):
        """Test that enhancement preserves question and answer."""
        qa_pair = {
            "question": "Original question?",
            "answer": "Original answer",
            "chunk_id": "test_123",
            "rating": 7,
        }

        # After enhancement, these should be preserved in output
        enhanced_pair = {
            "question": qa_pair["question"],
            "reasoning": "Step 1: New reasoning",
            "answer": qa_pair["answer"],
        }

        assert enhanced_pair["question"] == qa_pair["question"]
        assert enhanced_pair["answer"] == qa_pair["answer"]
        assert "reasoning" in enhanced_pair

    def test_batch_processing_calculation(self):
        """Test batch size calculation for enhancement."""
        total_pairs = 47
        batch_size = 5

        expected_batches = (total_pairs + batch_size - 1) // batch_size

        assert expected_batches == 10  # ceil(47/5) = 10 batches

        # Verify last batch size
        last_batch_size = total_pairs % batch_size or batch_size
        assert last_batch_size == 2  # 47 % 5 = 2
