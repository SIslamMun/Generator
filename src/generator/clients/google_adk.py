"""Google Gemini API client."""

import os
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class GoogleADKClient(BaseLLMClient):
    """Google Gemini API client (free tier: 10 requests/minute)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google Gemini ADK client.

        Args:
            config: Configuration dict with:
                - model: Model name (default: gemini-2.0-flash-exp)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 24576)
                - api_key: Google API key (required, can use ${VAR} syntax)
        """
        super().__init__(config)

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai not installed. Install with: pip install google-genai>=0.2.2"
            )

        # Get API key from config
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Google Gemini requires api_key in config")

        # Expand environment variables (e.g., ${GOOGLE_API_KEY})
        api_key = os.path.expandvars(api_key)

        # Initialize client
        self.client = genai.Client(api_key=api_key)
        self.model = config.get("model", "gemini-2.0-flash-exp")
        self.types = types

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using Google Gemini API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        response = self.client.models.generate_content(
            model=self.model,  # type: ignore[arg-type]
            contents=prompt,
            config=self.types.GenerateContentConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
            ),
        )

        return response.text or ""
