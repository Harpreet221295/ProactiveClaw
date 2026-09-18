from __future__ import annotations

from .base import LLMClient, ChatResponse, ToolCall, Usage
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient


def get_llm_client(model: str) -> LLMClient:
    """Return the right LLM client based on model name prefix."""
    if model.startswith("claude-"):
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                f"Model '{model}' requires ANTHROPIC_API_KEY which is not set. "
                "Use an OpenAI model or set the key."
            )
        return AnthropicClient(model)
    return OpenAIClient(model)


__all__ = [
    "LLMClient", "ChatResponse", "ToolCall", "Usage",
    "OpenAIClient", "AnthropicClient", "get_llm_client",
]
