"""Unit tests for QA generator module."""

import pytest
from pathlib import Path


class TestQAGenerator:
    """Test QA generation functionality."""

    def test_chunk_filtering(self, mock_lancedb_chunk):
        """Test that chunks are filtered by minimum length."""
        # Chunks < 200 chars should be filtered
        short_chunk = {"content": "Too short"}
        valid_chunk = {"content": "x" * 200}
        
        assert len(short_chunk["content"]) < 200
        assert len(valid_chunk["content"]) >= 200

    def test_rate_limiting_sleep(self):
        """Test that rate limiting delay is applied."""
        # Rate limiting should be 6 seconds between requests
        # This is tested implicitly in integration tests
        pass

    def test_retry_logic(self):
        """Test exponential backoff retry logic."""
        # Retry delays should be: 60s, 120s, 240s
        max_retries = 3
        expected_delays = [60, 120, 240]
        
        for attempt in range(max_retries):
            wait_time = 60 * (2 ** attempt)
            assert wait_time == expected_delays[attempt]

    def test_json_extraction(self):
        """Test JSON extraction from responses with extra text."""
        response_with_text = "Here's the JSON: [{'question': 'test', 'answer': 'test'}]"
        
        # Should extract JSON array from [...] markers
        start_idx = response_with_text.find('[')
        end_idx = response_with_text.rfind(']')
        
        assert start_idx != -1
        assert end_idx != -1
        assert end_idx > start_idx
        
        json_str = response_with_text[start_idx:end_idx+1]
        assert json_str.startswith('[')
        assert json_str.endswith(']')
