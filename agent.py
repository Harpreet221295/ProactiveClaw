import json
import os
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from tools import TOOLS_SCHEMA, dispatch_tool_call
from prompts import SYSTEM_PROMPT, PRE_EXIT_PROMPT

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")

DEFAULT_MODEL = "gpt-5-mini"


class Agent:
    def __init__(self, model: str = DEFAULT_MODEL, session_id: str | None = None):
        self.client = OpenAI()
        self.model = model
        self.session_id = session_id
        now_local = datetime.now().astimezone()
        system_prompt = SYSTEM_PROMPT.format(current_time=now_local.isoformat())
        self.messages = [{"role": "system", "content": system_prompt}]

        if session_id:
            os.makedirs(SESSIONS_DIR, exist_ok=True)
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

    def run(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
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

    def run_pre_exit(self) -> None:
        now_local = datetime.now().astimezone()
        current_time = now_local.isoformat()
        prompt = PRE_EXIT_PROMPT.format(current_time=current_time)
        print("\n[timeout] Session idle — running pre-exit flow...")
        answer = self.run(prompt)
        print(f"\nAgent: {answer}")
