from __future__ import annotations

import json

import anthropic

from .base import LLMClient, ChatResponse, ToolCall, Usage


class AnthropicClient(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Schema conversion: OpenAI tools → Anthropic tools
    # ------------------------------------------------------------------

    def _convert_tools(self, openai_tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-tool schema to Anthropic tool schema."""
        result = []
        for tool in openai_tools:
            fn = tool.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    # ------------------------------------------------------------------
    # Message conversion: OpenAI format → Anthropic format
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format.

        Returns (system_prompt, anthropic_messages).

        Key differences handled:
        - system messages → extracted as separate `system` param
        - assistant tool_calls → content array with tool_use blocks
        - tool results (role="tool") → grouped into a single user message
          with tool_result blocks (Anthropic requires this)
        - image_url blocks → Anthropic base64 image blocks
        """
        system = ""
        result: list[dict] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            # ---- system ----
            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    system = " ".join(
                        c.get("text", "") for c in content if c.get("type") == "text"
                    )
                else:
                    system = content or ""
                i += 1
                continue

            # ---- tool results: group consecutive into one user message ----
            if role == "tool":
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    t = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": t.get("tool_call_id", ""),
                        "content": t.get("content", ""),
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})
                continue

            # ---- user ----
            if role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    result.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    anthropic_content = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            anthropic_content.append({"type": "text", "text": block["text"]})
                        elif btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:") and ";base64," in url:
                                media_type, data = url.split(";base64,", 1)
                                media_type = media_type.replace("data:", "")
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                })
                    if anthropic_content:
                        result.append({"role": "user", "content": anthropic_content})
                i += 1
                continue

            # ---- assistant ----
            if role == "assistant":
                content = msg.get("content")
                tool_calls = msg.get("tool_calls") or []

                anthropic_content = []
                if content:
                    anthropic_content.append({"type": "text", "text": content})

                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_str = fn.get("arguments", "{}")
                        tc_id = tc.get("id", "")
                    else:
                        # OpenAI ChatCompletionMessageToolCall object
                        name = tc.function.name
                        args_str = tc.function.arguments
                        tc_id = tc.id

                    try:
                        input_data = json.loads(args_str)
                    except Exception:
                        input_data = {}

                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": name,
                        "input": input_data,
                    })

                if anthropic_content:
                    result.append({"role": "assistant", "content": anthropic_content})
                i += 1
                continue

            i += 1  # skip unknown roles

        return system, result

    # ------------------------------------------------------------------
    # Main completion
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        system, anthropic_messages = self._convert_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or 8192,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self._client.messages.create(**kwargs)

        text = next((b.text for b in response.content if b.type == "text"), None)
        tool_calls = [
            ToolCall(
                id=b.id,
                name=b.name,
                arguments=json.dumps(b.input),
            )
            for b in response.content if b.type == "tool_use"
        ]

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason or "end_turn",
        )
