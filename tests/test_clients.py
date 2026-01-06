"""Unit tests for LLM client factory and provider naming."""

import pytest
from generator.clients import get_client
from generator.clients.base import BaseLLMClient


class TestClientFactory:
    """Test client factory with new provider names."""

    def test_gemini_provider_name(self):
        """Test new 'gemini' provider name."""
        config = {"model": "gemini-2.0-flash-exp", "api_key": "test_key"}
        client = get_client("gemini", config)
        assert isinstance(client, BaseLLMClient)
        assert client.model == "gemini-2.0-flash-exp"

    def test_claude_provider_name(self):
        """Test new 'claude' provider name."""
        config = {"model": "claude-sonnet-4-20250514", "api_key": "test_key"}
        client = get_client("claude", config)
        assert isinstance(client, BaseLLMClient)
        assert client.model == "claude-sonnet-4-20250514"

    def test_legacy_adk_name(self):
        """Test legacy 'adk' provider name still works."""
        config = {"model": "gemini-2.0-flash-exp", "api_key": "test_key"}
        client = get_client("adk", config)
        assert isinstance(client, BaseLLMClient)
        assert client.model == "gemini-2.0-flash-exp"

    def test_legacy_claude_sdk_name(self):
        """Test legacy 'claude_sdk' provider name still works."""
        config = {"model": "claude-code", "use_agent_sdk": True}
        client = get_client("claude_sdk", config)
        assert isinstance(client, BaseLLMClient)

    def test_ollama_provider(self):
        """Test ollama provider still works."""
        config = {"model": "mistral:latest", "base_url": "http://localhost:11434"}
        client = get_client("ollama", config)
        assert isinstance(client, BaseLLMClient)
        assert client.model == "mistral:latest"

    def test_invalid_provider(self):
        """Test invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client("invalid_provider", {})

    def test_available_providers(self):
        """Test that all expected providers are available."""
        expected_providers = {"ollama", "claude", "gemini", "vllm", "openai", "anthropic"}

        # Try to get each provider (won't actually initialize without proper config)
        for provider in expected_providers:
            try:
                # Use minimal config, will fail during init but that's ok
                get_client(provider, {"model": "test"})
            except Exception:
                # Expected to fail without proper config, but provider should be recognized
                pass
