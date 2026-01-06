"""Claude API client using Agent SDK."""

import os
import anyio
from typing import Optional, Dict, Any

from .base import BaseLLMClient


class ClaudeSDKClient(BaseLLMClient):
    """Claude Agent SDK client (uses free local Claude Code CLI)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Claude Agent SDK client.

        Args:
            config: Configuration dict with:
                - model: Not used (Agent SDK uses configured model)
                - temperature: Not used (Agent SDK manages this)
                - max_tokens: Not used (Agent SDK manages this)
        
        Note:
            Requires Claude Code CLI installed and authenticated.
            Install: curl -fsSL https://claude.ai/install.sh | bash
            Authenticate: claude auth login
        """
        super().__init__(config)

        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            raise ImportError(
                "claude-agent-sdk not installed. Install with:\n"
                "  pip install claude-agent-sdk>=0.1.18\n\n"
                "Also requires Claude Code CLI:\n"
                "  curl -fsSL https://claude.ai/install.sh | bash\n"
                "  claude auth login"
            )

    def generate(
        self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate using Claude Agent SDK.

        Args:
            prompt: Input prompt
            temperature: Not used (Agent SDK manages temperature)
            max_tokens: Not used (Agent SDK manages max_tokens)

        Returns:
            Generated text response
        
        Raises:
            RuntimeError: If CLI is not authenticated or encounters errors
        """
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

            # Configure options for one-shot query
            options = ClaudeAgentOptions(
                max_turns=1,  # Single response
                permission_mode='acceptEdits'  # Auto-accept (non-interactive)
            )

            async def _async_query():
                """Run async query and collect response."""
                response_text = ""
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text
                
                if not response_text:
                    raise RuntimeError("No response received from Claude Agent SDK")
                
                return response_text

            # Run async function using anyio
            return anyio.run(_async_query)

        except Exception as e:
            error_msg = str(e)

            # Check for common errors
            if "CLI not found" in error_msg or "CLINotFoundError" in error_msg:
                raise RuntimeError(
                    "Claude Code CLI not installed.\n\n"
                    "Install with:\n"
                    "  curl -fsSL https://claude.ai/install.sh | bash\n\n"
                    "Then authenticate:\n"
                    "  claude auth login\n\n"
                    "Or use 'provider: anthropic' for paid API access."
                ) from e
            
            if "not authenticated" in error_msg.lower() or "auth" in error_msg.lower():
                raise RuntimeError(
                    "Claude Code CLI not authenticated.\n\n"
                    "Run: claude auth login\n\n"
                    "Or use 'provider: anthropic' for paid API access."
                ) from e

            # Try fallback to Anthropic API if key available
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                print("\n⚠️  Claude Agent SDK failed, falling back to Anthropic API...")
                return self._generate_anthropic_fallback(prompt, temperature, max_tokens, api_key)
            
            raise RuntimeError(
                f"Claude Agent SDK error: {e}\n\n"
                "Possible causes:\n"
                "  1. Claude CLI not installed (install: curl -fsSL https://claude.ai/install.sh | bash)\n"
                "  2. Not authenticated (run: claude auth login)\n"
                "  3. Rate limit reached\n\n"
                "Alternative: Use 'provider: anthropic' for paid API access."
            ) from e

    def _generate_anthropic_fallback(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Fallback to Anthropic API if SDK fails."""
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        return response_text
