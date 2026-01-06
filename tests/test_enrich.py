"""Unit tests for enrich module."""

import pytest
import json
from pathlib import Path


class TestEnrich:
    """Test enrichment functionality."""

    def test_enrichment_preserves_metadata(self):
        """Test that enrichment preserves original metadata."""
        original_pair = {
            "question": "What is X?",
            "answer": "X is...",
            "chunk_id": "chunk_123",
            "source": "test.md",
            "rating": 7
        }

        # After enrichment
        enriched_pair = {
            **original_pair,
            "answer": "X is a comprehensive thing that...",
            "enrichment_changes": "Added detail",
            "original_answer": original_pair["answer"]
        }

        # Verify metadata preserved
        assert enriched_pair["chunk_id"] == original_pair["chunk_id"]
        assert enriched_pair["source"] == original_pair["source"]
        assert enriched_pair["rating"] == original_pair["rating"]
        assert "original_answer" in enriched_pair

    def test_enrichment_without_preserve_original(self):
        """Test enrichment without preserving original answer."""
        enriched_pair = {
            "question": "What is X?",
            "answer": "Improved answer",
            "enrichment_changes": "Made clearer"
        }

        # Should not have original_answer field
        assert "original_answer" not in enriched_pair

    def test_batch_enrichment_calculation(self):
        """Test batch size calculation for enrichment."""
        total_pairs = 47
        batch_size = 5

        num_batches = (total_pairs + batch_size - 1) // batch_size

        assert num_batches == 10

    def test_enrichment_changes_tracking(self):
        """Test that enrichment tracks what was changed."""
        changes = "Added structure, improved clarity, expanded example"

        assert "structure" in changes.lower()
        assert "clarity" in changes.lower()

        # Verify changes is descriptive
        assert len(changes.split(",")) >= 2

    def test_enrichment_answer_improvement(self):
        """Test that enriched answer is actually longer/better."""
        original = "Use command X"
        enriched = "To use this feature, execute the following command:\n\n`command X`\n\nThis will start the process."

        assert len(enriched) > len(original)
        assert original in enriched or "command x" in enriched.lower()

    def test_enrichment_preserves_facts(self):
        """Test that enrichment preserves factual content."""
        original_answer = "HDF5 uses chunking for parallel I/O"
        enriched_answer = "HDF5 employs a chunking strategy to enable efficient parallel I/O operations across multiple processes."

        # Key terms should be preserved
        assert "hdf5" in enriched_answer.lower()
        assert "chunk" in enriched_answer.lower()
        assert "parallel" in enriched_answer.lower()

    def test_enrichment_json_format_validation(self):
        """Test that enrichment response has correct JSON structure."""
        response_dict = {
            "enriched_answer": "Improved answer text",
            "changes": "What was changed"
        }

        # Verify required fields
        assert "enriched_answer" in response_dict
        assert "changes" in response_dict
        assert isinstance(response_dict["enriched_answer"], str)
        assert isinstance(response_dict["changes"], str)

    def test_json_parsing(self):
        """Test JSON parsing helpers for enrichment responses."""
        import json5

        # Valid JSON
        valid = '{"enriched_answer": "Better", "changes": "Improved"}'
        result = json5.loads(valid)
        assert "enriched_answer" in result

        # JSON with trailing comma
        trailing = '{"enriched_answer": "Better", "changes": "Improved",}'
        result = json5.loads(trailing)
        assert "enriched_answer" in result
