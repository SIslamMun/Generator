"""Ollama LLM client."""

import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

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
                - max_tokens: Maximum tokens to generate (default: 4096)
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:72b-instruct")
        self.client = httpx.Client(timeout=120.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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
            "options": {
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        response = self.client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        return result["response"]

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
