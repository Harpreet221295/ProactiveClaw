from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class ChatResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Complete a conversation.

        Messages are always in OpenAI internal format (dicts with role/content/tool_calls).
        Implementations convert to/from provider-specific formats as needed.
        """
        ...
