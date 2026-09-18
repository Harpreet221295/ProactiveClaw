from __future__ import annotations

from openai import OpenAI

from .base import LLMClient, ChatResponse, ToolCall, Usage


class OpenAIClient(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = OpenAI()

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        usage = response.usage
        return ChatResponse(
            text=msg.content,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            ),
            stop_reason=response.choices[0].finish_reason or "end_turn",
        )
