"""vLLM client for OpenAI-compatible servers (local vLLM or remote services like ALCF)."""

import os
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """vLLM client (OpenAI-compatible server)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vLLM client.

        Args:
            config: Configuration dict with:
                - model: Model name
                - base_url: vLLM server URL (default: http://localhost:8000/v1)
                - api_key: Bearer token (default: "EMPTY"; supports ${VAR} expansion)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 24576)
        """
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = os.path.expandvars(str(config.get("api_key") or "EMPTY"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = config.get("model", "meta-llama/Llama-3.1-8B-Instruct")

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using vLLM.

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

        msg = response.choices[0].message
        # gpt-oss models emit reasoning into a separate field; if content
        # is empty (e.g. tokens budget consumed by reasoning), fall back.
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            content = (getattr(msg, "reasoning", None) or "").strip()
        return content
