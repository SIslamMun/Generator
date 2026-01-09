"""Integration tests for enrich module."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from generator.qa.enrich import enrich_qa_pairs


class TestEnrichIntegration:
    """Integration tests for QA enrichment."""

    @pytest.fixture
    def sample_qa_data(self):
        """Create sample QA data for testing."""
        return [
            {
                "question": "What is HDF5?",
                "answer": "HDF5 is a file format.",
                "chunk_id": "chunk_1",
                "rating": 8
            },
            {
                "question": "How does it work?",
                "answer": "It uses hierarchical structure.",
                "chunk_id": "chunk_2",
                "rating": 7
            }
        ]

    @pytest.fixture
    def llm_config(self):
        """Create LLM config for testing."""
        return {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "api_key": "test-key"
        }

    @pytest.fixture
    def prompts_dir(self, tmp_path):
        """Create a mock prompts directory."""
        # Create prompts directory structure
        prompts_d = tmp_path / "prompts"
        prompts_d.mkdir()
        
        # Create qa_enrichment.yaml
        qa_enrichment = prompts_d / "qa_enrichment.yaml"
        qa_enrichment.write_text("prompt: 'Enrich this answer: {answer}'")
        
        return tmp_path  # Return parent directory

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = Mock()
        # Mock returns response for a single pair (not a batch list)
        mock_client.generate = Mock(return_value=json.dumps({
            "enriched_answer": "HDF5 (Hierarchical Data Format version 5) is a versatile file format.",
            "changes": "Added full name"
        }))
        return mock_client

    def test_enrich_preserves_metadata(self, sample_qa_data, llm_config, mock_llm_client, prompts_dir):
        """Test that enrichment preserves all metadata."""
        
        with patch('generator.qa.enrich.get_client', return_value=mock_llm_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.3
            )
        
        # Verify metadata preserved
        assert enriched[0]["chunk_id"] == "chunk_1"
        assert enriched[0]["rating"] == 8
        assert "question" in enriched[0]

    def test_enrich_batch_processing(self, sample_qa_data, llm_config, mock_llm_client, prompts_dir):
        """Test batch processing in enrichment."""
        
        with patch('generator.qa.enrich.get_client', return_value=mock_llm_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                batch_size=1,  # Process one at a time
                temperature=0.3
            )
        
        assert len(enriched) == 2
        # Should have called generate for each pair (batch_size=1)
        assert mock_llm_client.generate.call_count == 2

    def test_enrich_with_preserve_original(self, sample_qa_data, llm_config, mock_llm_client, prompts_dir):
        """Test enrichment with preserve_original flag."""
        
        with patch('generator.qa.enrich.get_client', return_value=mock_llm_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                preserve_original=True,
                temperature=0.3
            )
        
        # Should have original_answer field
        assert "original_answer" in enriched[0]
        assert enriched[0]["original_answer"] == "HDF5 is a file format."

    def test_enrich_without_preserve_original(self, sample_qa_data, llm_config, mock_llm_client, prompts_dir):
        """Test enrichment without preserving original."""
        
        with patch('generator.qa.enrich.get_client', return_value=mock_llm_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                preserve_original=False,
                temperature=0.3
            )
        
        # Should NOT have original_answer field
        assert "original_answer" not in enriched[0]

    def test_enrich_handles_malformed_response(self, sample_qa_data, llm_config, prompts_dir):
        """Test handling of malformed LLM response."""
        
        mock_client = Mock()
        mock_client.generate = Mock(return_value="Not valid JSON")
        
        with patch('generator.qa.enrich.get_client', return_value=mock_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.3
            )
        
        # Should fallback to original pairs when LLM fails
        assert len(enriched) == 2
        assert enriched[0]["answer"] == "HDF5 is a file format."

    def test_enrich_temperature_parameter(self, sample_qa_data, llm_config, mock_llm_client, prompts_dir):
        """Test that temperature parameter is used."""
        
        with patch('generator.qa.enrich.get_client', return_value=mock_llm_client):
            enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.5
            )
        
        # Verify generate was called with temperature
        assert mock_llm_client.generate.called
        call_kwargs = mock_llm_client.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.5

    def test_enrich_adds_enrichment_changes(self, sample_qa_data, llm_config, prompts_dir):
        """Test that enrichment_changes field is added."""
        
        mock_client = Mock()
        mock_client.generate = Mock(return_value=json.dumps({
            "enriched_answer": "Improved answer",
            "changes": "Made it better"
        }))
        
        with patch('generator.qa.enrich.get_client', return_value=mock_client):
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.3
            )
        
        # Should have enrichment_changes field
        assert "enrichment_changes" in enriched[0]

    def test_enrich_empty_input(self, llm_config, prompts_dir):
        """Test enrichment with empty input."""
        
        mock_client = Mock()
        
        with patch('generator.qa.enrich.get_client', return_value=mock_client):
            enriched = enrich_qa_pairs(
                qa_pairs=[],
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.3
            )
        
        assert len(enriched) == 0
        # Should not have called LLM for empty input
        assert not mock_client.generate.called
