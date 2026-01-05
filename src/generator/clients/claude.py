"""Claude API client using Agent SDK."""

import os
import asyncio
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class ClaudeSDKClient(BaseLLMClient):
    """Claude API client (uses Agent SDK for free access, or API with key)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Claude Agent SDK client.

        Args:
            config: Configuration dict with:
                - model: Model name (default: claude-code)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 4096)
        """
        super().__init__(config)
        
        try:
            import claude_agent_sdk  # noqa: F401
            self.model = config.get("model", "claude-code")
        except ImportError:
            raise ImportError(
                "claude-agent-sdk not installed. Install with: "
                "pip install claude-agent-sdk>=0.1.18"
            )

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using Claude Agent SDK.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (not used for Claude SDK)
            max_tokens: Maximum tokens (not used for Claude SDK)

        Returns:
            Generated text response
        """
        try:
            return self._generate_impl(prompt)
        except Exception as e:
            error_msg = str(e)

            # Check for rate limit or auth issues
            if "exit code 1" in error_msg.lower() or "rate limit" in error_msg.lower():
                print("\n⚠️  Claude CLI rate limit or auth issue.")
                print("    Check with: claude auth status")

            # Try fallback to Anthropic API if key available
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                print("Falling back to Anthropic API...")
                return self._generate_anthropic_fallback(prompt, temperature, max_tokens, api_key)
            else:
                raise RuntimeError(
                    f"Claude Agent SDK failed: {e}\n\n"
                    "Possible causes:\n"
                    "  1. Claude CLI rate limit (check: claude auth status)\n"
                    "  2. Not authenticated (run: claude auth login)\n"
                    "  3. Network issues\n\n"
                    "To use API fallback, set ANTHROPIC_API_KEY."
                ) from e

    def _generate_impl(self, prompt: str) -> str:
        """Actual Claude SDK generation."""
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

        options = ClaudeAgentOptions(max_turns=1)

        async def _async_query():
            response_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
            return response_text

        # Run the async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_async_query())  # type: ignore[no-any-return]

    def _generate_anthropic_fallback(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Fallback to Anthropic API."""
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self.model or "claude-sonnet-4-20250514",
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        return response_text
