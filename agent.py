import base64
import json
import os
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from tools import TOOLS_SCHEMA, dispatch_tool_call
from prompts import SYSTEM_PROMPT, PRE_EXIT_PROMPT, SUMMARY_PROMPT

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")

DEFAULT_MODEL = "gpt-5-mini"


class Agent:
    def __init__(self, model: str = DEFAULT_MODEL, session_id: str | None = None,
                 chat_summary: str | None = None):
        self.client = OpenAI()
        self.model = model
        self.session_id = session_id
        self.chat_summary = chat_summary
        now_local = datetime.now().astimezone()
        system_prompt = SYSTEM_PROMPT.format(current_time=now_local.isoformat())

        if chat_summary:
            system_prompt += f"\n\n{chat_summary}"

        self.messages = [{"role": "system", "content": system_prompt}]

        if session_id:
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            if not chat_summary:
                self._load_session()

    def _session_path(self) -> str:
        return os.path.join(SESSIONS_DIR, f"{self.session_id}.json")

    def _load_session(self) -> None:
        path = self._session_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            self.messages = data.get("messages", self.messages)

    def _serialize_messages(self) -> list[dict]:
        serialized = []
        for msg in self.messages:
            if isinstance(msg, dict):
                serialized.append(msg)
            else:
                # OpenAI ChatCompletionMessage object
                d = {"role": msg.role, "content": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "refusal") and msg.refusal:
                    d["refusal"] = msg.refusal
                serialized.append(d)
        return serialized

    def save_session(self) -> None:
        if not self.session_id:
            return
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        with open(self._session_path(), "w") as f:
            json.dump({"messages": self._serialize_messages()}, f, indent=2)

    def run(self, user_message: str, images: list[bytes] | None = None) -> str:
        if images:
            content = [{"type": "text", "text": user_message}]
            for img_bytes in images:
                mime = "image/png"
                # Try to detect from magic bytes
                if img_bytes[:3] == b"\xff\xd8\xff":
                    mime = "image/jpeg"
                elif img_bytes[:4] == b"\x89PNG":
                    mime = "image/png"
                elif img_bytes[:4] == b"GIF8":
                    mime = "image/gif"
                elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
                    mime = "image/webp"
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "user", "content": user_message})

        max_rounds = 15
        for _round in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._serialize_messages(),
                tools=TOOLS_SCHEMA,
            )
            message = response.choices[0].message
            self.messages.append(message)

            if not message.tool_calls:
                self.save_session()
                return message.content

            for tool_call in message.tool_calls:
                print(f"  [tool] {tool_call.function.name}({tool_call.function.arguments})")
                result = dispatch_tool_call(tool_call.function.name, tool_call.function.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # If we exhausted all rounds, force a final response without tools
        print(f"  [warning] Hit {max_rounds} tool rounds — forcing final response")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._serialize_messages(),
        )
        message = response.choices[0].message
        self.messages.append(message)
        self.save_session()
        return message.content or "Sorry, I got stuck processing that. Could you rephrase?"

    def format_conversation_for_memory(self) -> list[dict]:
        """Convert conversation to SimpleMem dialogue format.

        Filters out system/tool messages and strips any previous-session
        summary so only the current conversation is stored in long-term memory.
        """
        dialogues = []
        for msg in self._serialize_messages():
            role = msg.get("role", "")
            content = msg.get("content")
            if role in ("system", "tool") or content is None:
                continue
            if isinstance(content, list):
                text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
            else:
                text = content
            if not text:
                continue
            # Skip messages that are the previous-session summary itself
            if self.chat_summary and self.chat_summary.strip() in text:
                continue
            dialogues.append({"role": role, "content": text})
        return dialogues

    def run_pre_exit(self, bandit_recommendations: str = "", existing_reminders: str = "", previous_invocations: str = "") -> str:
        now_local = datetime.now().astimezone()
        current_time = now_local.isoformat()
        prompt = PRE_EXIT_PROMPT.format(
            current_time=current_time,
            bandit_recommendations=bandit_recommendations,
            existing_reminders=existing_reminders,
            previous_invocations=previous_invocations,
        )
        print("\n[timeout] Session idle — running pre-exit flow...")
        answer = self.run(prompt)
        print(f"\nAgent: {answer}")
        return answer

    # ------------------------------------------------------------------
    # Conversation summary generation
    # ------------------------------------------------------------------

    def _format_conversation_for_summary(self) -> str:
        """Convert conversation messages to a readable transcript for summarization."""
        lines = []
        for msg in self._serialize_messages():
            role = msg.get("role", "")
            content = msg.get("content")
            if role == "system" or role == "tool" or content is None:
                continue
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                text = " ".join(text_parts)
                if text:
                    lines.append(f"{role.upper()}: {text}")
            else:
                lines.append(f"{role.upper()}: {content}")
        return "\n\n".join(lines)

    def generate_summary(self) -> str:
        """Generate a summary of the current conversation via a separate LLM call."""
        conversation_text = self._format_conversation_for_summary()

        messages = [
            {"role": "user", "content": f"{conversation_text}\n\n{SUMMARY_PROMPT}"},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Summary persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_path(session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, f"{session_id}_summary.json")

    @staticmethod
    def load_summary(session_id: str) -> str:
        """Load the most recent chat summary from disk. Returns empty string if none."""
        path = Agent._summary_path(session_id)
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("summary", "")
        except (json.JSONDecodeError, IOError):
            return ""

    @staticmethod
    def save_summary(session_id: str, summary: str) -> None:
        """Save chat summary to disk."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = Agent._summary_path(session_id)
        with open(path, "w") as f:
            json.dump({
                "summary": summary,
                "generated_at": datetime.now().astimezone().isoformat(),
            }, f, indent=2)

    def archive_session(self) -> None:
        """Archive the current session file with a timestamp suffix."""
        if not self.session_id:
            return
        path = self._session_path()
        if not os.path.exists(path):
            return
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(SESSIONS_DIR, f"{self.session_id}_{now_str}.json")
        os.rename(path, archive_path)
        print(f"[archive] Session archived: {archive_path}")
