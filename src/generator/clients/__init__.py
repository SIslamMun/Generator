"""LLM client factory - instantiate the correct client based on provider."""

from typing import Dict, Any

from .base import BaseLLMClient
from .ollama import OllamaClient
from .claude import ClaudeSDKClient
from .google_adk import GoogleADKClient
from .vllm import VLLMClient
from .openai import OpenAIClient
from .anthropic import AnthropicClient


def get_client(provider: str, config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: Provider name (ollama, claude_sdk, adk, vllm, openai, anthropic)
        config: Configuration dictionary for the provider

    Returns:
        Initialized LLM client instance

    Raises:
        ValueError: If provider is unknown
    """
    clients = {
        "ollama": OllamaClient,
        "claude_sdk": ClaudeSDKClient,
        "adk": GoogleADKClient,
        "vllm": VLLMClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }

    client_class = clients.get(provider)
    if not client_class:
        available = ", ".join(clients.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    return client_class(config)


__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "ClaudeSDKClient",
    "GoogleADKClient",
    "VLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "get_client",
]
