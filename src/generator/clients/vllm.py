"""vLLM client for local OpenAI-compatible servers."""

from typing import Optional, Dict, Any

from .base import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """vLLM client (local OpenAI-compatible server)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vLLM client.

        Args:
            config: Configuration dict with:
                - model: Model name
                - base_url: vLLM server URL (default: http://localhost:8000/v1)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 4096)
        """
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        base_url = config.get("base_url", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
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

        return response.choices[0].message.content or ""
