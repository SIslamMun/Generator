"""Anthropic API client."""

import os
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic API client (paid, requires API key)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic client.

        Args:
            config: Configuration dict with:
                - model: Model name (default: claude-sonnet-4-20250514)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 4096)
                - api_key: Anthropic API key (required, can use ${VAR} syntax)
        """
        super().__init__(config)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Install with: pip install anthropic")

        # Get API key from config
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Anthropic requires api_key in config")

        # Expand environment variables
        api_key = os.path.expandvars(api_key)

        self.client = Anthropic(api_key=api_key)
        self.model = config.get("model", "claude-sonnet-4-20250514")

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using Anthropic API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        return response_text
