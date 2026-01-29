"""Ollama LLM client."""

import httpx
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama client.

        Args:
            config: Configuration dict with:
                - base_url: Ollama server URL (default: http://localhost:11434)
                - model: Model name (default: qwen2.5:72b-instruct)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 24576)
                - timeout: Request timeout in seconds (default: 600)
                - max_connections: Max concurrent HTTP connections (default: 100)
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:72b-instruct")

        # Configurable timeout (default 10 min for complex generations)
        timeout_seconds = config.get("timeout", 600)
        self.timeout = httpx.Timeout(timeout_seconds, connect=30.0)

        # Connection pool for parallel processing
        max_connections = config.get("max_connections", 100)
        limits = httpx.Limits(
            max_keepalive_connections=max_connections,
            max_connections=max_connections,
        )

        # Persistent client with connection pooling
        self._client = httpx.Client(timeout=self.timeout, limits=limits)

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using Ollama API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "stream": False,
        }
        
        # Add num_predict to prevent JSON truncation
        # Ollama uses 'num_predict' instead of 'max_tokens'
        token_limit = max_tokens or self.max_tokens
        # Set context window to 24K to balance performance and capacity
        # Model supports 131K context, 24K allows ~8K input + 16K output
        payload["options"] = {
            "num_predict": token_limit,
            "num_ctx": 24576,  # 24K context window (input + output)
        }

        # Reuse persistent client with connection pooling
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        return result["response"]  # type: ignore[no-any-return]

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            self._client.close()
