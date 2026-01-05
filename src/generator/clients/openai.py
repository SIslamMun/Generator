"""OpenAI API client."""

import os
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (paid, requires API key)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client.

        Args:
            config: Configuration dict with:
                - model: Model name (default: gpt-4o)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 4096)
                - api_key: OpenAI API key (required, can use ${VAR} syntax)
        """
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        # Get API key from config
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("OpenAI requires api_key in config")

        # Expand environment variables
        api_key = os.path.expandvars(api_key)

        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o")

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using OpenAI API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        return response.choices[0].message.content or ""
