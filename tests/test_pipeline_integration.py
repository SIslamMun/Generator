"""Integration tests for full pipeline workflows."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    @pytest.fixture
    def sample_qa_data(self):
        """Sample QA data."""
        return [
            {
                "question": "What is HDF5?",
                "answer": "HDF5 is a file format for storing large datasets.",
                "chunk_id": "chunk_1"
            },
            {
                "question": "What is parallel I/O?",
                "answer": "Parallel I/O allows concurrent file access.",
                "chunk_id": "chunk_2"
            }
        ]

    @pytest.fixture
    def llm_config(self):
        """Sample LLM config."""
        return {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "api_key": "test-key"
        }

    @pytest.fixture
    def prompts_dir(self, tmp_path):
        """Create prompts directory."""
        prompts_d = tmp_path / "prompts"
        prompts_d.mkdir()
        
        # Create necessary prompt files
        (prompts_d / "qa_enrichment.yaml").write_text("prompt: 'Enrich: {answer}'")
        (prompts_d / "curate.yaml").write_text("prompt: 'Rate this QA pair'")
        (prompts_d / "cot_enhancement.yaml").write_text("prompt: 'Add reasoning'")
        
        return tmp_path

    def test_curate_to_enrich_pipeline(self, sample_qa_data, llm_config, prompts_dir, tmp_path):
        """Test pipeline from curation to enrichment."""
        from generator.qa.enrich import enrich_qa_pairs
        
        # Add rating to simulate curated data
        curated_data = [
            {**item, "rating": 8}
            for item in sample_qa_data
        ]
        
        mock_client = Mock()
        mock_client.generate.return_value = json.dumps({
            "enriched_answer": "HDF5 (Hierarchical Data Format version 5) is a file format.",
            "changes": "Added full name"
        })
        
        with patch('generator.qa.enrich.get_client', return_value=mock_client):
            enriched = enrich_qa_pairs(
                qa_pairs=curated_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                temperature=0.3
            )
        
        # Verify enrichment preserved curation data
        assert len(enriched) == 2
        assert enriched[0]["rating"] == 8
        assert "chunk_id" in enriched[0]

    def test_export_pipeline(self, sample_qa_data, tmp_path):
        """Test exporting to different formats."""
        from generator.formatters import export_to_format
        
        # Add ratings for export
        qa_with_ratings = [
            {**item, "rating": 8}
            for item in sample_qa_data
        ]
        
        # Save to temp file first
        input_file = tmp_path / "qa_data.json"
        input_file.write_text(json.dumps(qa_with_ratings))
        
        # Test ChatML format
        chatml_file = tmp_path / "train_chatml.jsonl"
        count = export_to_format(str(input_file), str(chatml_file), "chatml")
        
        assert count == 2
        assert chatml_file.exists()
        
        # Verify ChatML structure
        with open(chatml_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) >= 2  # user + assistant

    @patch('generator.qa.compare.load_prompts', return_value={})
    def test_compare_datasets_pipeline(self, mock_prompts, sample_qa_data, tmp_path, llm_config):
        """Test comparing multiple datasets."""
        from generator.qa.compare import DatasetComparator
        
        # Create two dataset files
        dataset1 = tmp_path / "dataset1.json"
        dataset2 = tmp_path / "dataset2.json"
        
        data1 = [
            {**item, "rating": 8}
            for item in sample_qa_data
        ]
        data2 = [
            {**item, "rating": 9, "answer": item["answer"] + " Improved!"}
            for item in sample_qa_data
        ]
        
        dataset1.write_text(json.dumps(data1))
        dataset2.write_text(json.dumps(data2))
        
        # Compare datasets (DatasetComparator needs llm_config)
        comparator = DatasetComparator(llm_config)
        datasets = comparator.load_datasets([dataset1, dataset2])
        
        # Compute metrics for each dataset
        metrics = {}
        for name, data in datasets.items():
            metrics[name] = comparator.compute_metrics(data)
        
        # Verify metrics computed
        assert len(metrics) == 2
        assert metrics["dataset1"]["count"] == 2
        assert metrics["dataset2"]["avg_rating"] == 9.0

    def test_enrichment_workflow(self, sample_qa_data, llm_config, prompts_dir):
        """Test complete enrichment workflow with different parameters."""
        from generator.qa.enrich import enrich_qa_pairs
        
        mock_client = Mock()
        mock_client.generate.return_value = json.dumps({
            "enriched_answer": "Enhanced answer",
            "changes": "Improved clarity"
        })
        
        with patch('generator.qa.enrich.get_client', return_value=mock_client):
            # Test with preserve_original=True
            enriched = enrich_qa_pairs(
                qa_pairs=sample_qa_data,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                preserve_original=True,
                batch_size=2,
                temperature=0.3
            )
        
        assert len(enriched) == 2
        assert "original_answer" in enriched[0]
        assert "enrichment_changes" in enriched[0]
        
    def test_export_multiple_formats(self, sample_qa_data, tmp_path):
        """Test exporting to multiple formats."""
        from generator.formatters import export_to_format
        
        qa_with_ratings = [
            {**item, "rating": 8}
            for item in sample_qa_data
        ]
        
        input_file = tmp_path / "qa_data.json"
        input_file.write_text(json.dumps(qa_with_ratings))
        
        # Export to different formats
        formats = {
            "chatml": tmp_path / "train_chatml.jsonl",
            "alpaca": tmp_path / "train_alpaca.jsonl",
            "jsonl": tmp_path / "train.jsonl"
        }
        
        for fmt, output_file in formats.items():
            count = export_to_format(str(input_file), str(output_file), fmt)
            assert count == 2
            assert output_file.exists()
            
            # Verify file has content
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) == 2
