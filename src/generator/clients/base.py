"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client.

        Args:
            config: Configuration dictionary with provider-specific settings
        """
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.model = config.get("model")

    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)

        Returns:
            Generated text response
        """
        pass
