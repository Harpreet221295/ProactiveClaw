from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# Resolve src/ on the path so `llms` is importable from here
_SRC = os.path.join(os.path.dirname(__file__), "..", "..")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from llms import get_llm_client, ChatResponse
from core import config as _config
from core.paths import SESSIONS_DIR as _SESSIONS_DIR_P
from .tools import active_tools_schema, dispatch_tool_call
from .prompts.prompts import build_system_prompt, build_pre_exit_prompt, SUMMARY_PROMPT

SESSIONS_DIR = str(_SESSIONS_DIR_P)

DEFAULT_MAX_ROUNDS = 25


def _default_model() -> str:
    return _config.default_model()


class _DefaultModelUnused(str):
    """Backwards-compatible module attribute that always reflects the config."""
    def __new__(cls):
        return str.__new__(cls, _default_model())


DEFAULT_MODEL = _default_model()

# Context compaction constants
BROWSER_TOOL_NAMES = {
    "browser_snapshot", "browser_navigate", "browser_click",
    "browser_type", "browser_screenshot", "browser_get_page_info",
    "browser_scroll",
}
BROWSER_SNAPSHOT_TOOLS = {"browser_snapshot", "browser_get_page_info"}
MAX_KEPT_SNAPSHOTS = 2
TOKEN_COMPACT_THRESHOLD = 100_000
KEEP_RECENT_MESSAGES = 6


class Agent:
    def __init__(self, model: str | None = None, session_id: str | None = None,
                 chat_summary: str | None = None, tool_logger=None, max_rounds: int | None = None,
                 event_callback=None, system_prompt: str | None = None):
        self.model = model or _default_model()
        self.llm = get_llm_client(self.model)
        self.session_id = session_id
        self.chat_summary = chat_summary
        self.tool_logger = tool_logger  # Optional callback: (name, args, result) -> None
        self.event_callback = event_callback  # Optional callback: (event: str, payload: dict) -> None
        self.max_rounds = max_rounds or DEFAULT_MAX_ROUNDS
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        now_local = datetime.now().astimezone()
        system_prompt = system_prompt or build_system_prompt(current_time=now_local.isoformat())

        if chat_summary:
            system_prompt += f"\n\n{chat_summary}"

        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

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
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return
            loaded = data.get("messages")
            if loaded:
                # Always refresh the system prompt so config changes take effect mid-session
                if loaded and loaded[0].get("role") == "system":
                    loaded[0] = self.messages[0]
                self.messages = loaded

    def _emit(self, event: str, payload: dict) -> None:
        if self.event_callback:
            try:
                self.event_callback(event, payload)
            except Exception:
                pass

    @staticmethod
    def session_exists(session_id: str) -> bool:
        return os.path.exists(os.path.join(SESSIONS_DIR, f"{session_id}.json"))

    def _serialize_messages(self) -> list[dict]:
        """Return messages as plain dicts (for JSON serialization and LLM calls).

        Messages are always stored as dicts now, but we keep a fallback for
        any lingering OpenAI message objects from old sessions.
        """
        serialized = []
        for msg in self.messages:
            if isinstance(msg, dict):
                serialized.append(msg)
            else:
                # Fallback: OpenAI ChatCompletionMessage object
                d: dict = {"role": msg.role, "content": msg.content}
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
                serialized.append(d)
        return serialized

    def save_session(self) -> None:
        if not self.session_id:
            return
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        with open(self._session_path(), "w") as f:
            json.dump({"messages": self._serialize_messages()}, f, indent=2)

    # ------------------------------------------------------------------
    # Mid-conversation context compaction
    # ------------------------------------------------------------------

    def _log_usage(self, response: ChatResponse, label: str = "llm") -> None:
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        tot = inp + out
        self.token_usage["input"] += inp
        self.token_usage["output"] += out
        self.token_usage["total"] += tot
        print(f"  [tokens:{label}] in={inp} out={out} total={tot} | cumulative: in={self.token_usage['input']} out={self.token_usage['output']} total={self.token_usage['total']}")

    def _estimate_tokens(self) -> int:
        """Rough token estimate: chars / 4 across all messages."""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
            total += len(str(content))
        return total // 4

    def _find_tool_call_name(self, tool_call_id: str) -> str | None:
        """Walk backwards to find the function name for a given tool_call_id."""
        for msg in reversed(self.messages):
            tc_list = None
            if isinstance(msg, dict):
                tc_list = msg.get("tool_calls")
            elif hasattr(msg, "tool_calls"):
                tc_list = msg.tool_calls
            if not tc_list:
                continue
            for tc in tc_list:
                tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                if tc_id == tool_call_id:
                    if isinstance(tc, dict):
                        return tc.get("function", {}).get("name")
                    return tc.function.name
        return None

    @staticmethod
    def _summarize_snapshot(content: str) -> str:
        """Build a rich one-line summary of a browser snapshot."""
        lines = content.split("\n")

        url = ""
        for line in lines[:10]:
            stripped = line.strip()
            if stripped.startswith("Page info:"):
                try:
                    import json as _json
                    info = _json.loads(stripped.split("Page info:", 1)[1].strip())
                    url = info.get("url", "")
                except Exception:
                    pass
                break
            if stripped.startswith("http://") or stripped.startswith("https://"):
                url = stripped[:200]
                break
            if "url" in stripped.lower() and "http" in stripped:
                for part in stripped.split():
                    if part.startswith("http"):
                        url = part[:200]
                        break
                if url:
                    break

        title = ""
        for line in lines[:15]:
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                title = stripped.split(":", 1)[1].strip()[:120]
                break
            if "title" in stripped and "{" in stripped:
                try:
                    import json as _json
                    info = _json.loads(stripped.split("Page info:", 1)[-1].strip())
                    title = info.get("title", "")[:120]
                except Exception:
                    pass

        preview_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 10:
                continue
            if stripped.startswith("[ref:") or stripped.startswith("http"):
                continue
            preview_lines.append(stripped[:100])
            if len(preview_lines) >= 3:
                break
        content_preview = " | ".join(preview_lines) if preview_lines else "(no text content)"

        ref_count = content.count("[ref:")

        parts = ["[Compacted browser snapshot]"]
        if url:
            parts.append(f"URL: {url}  (use browser_navigate to return)")
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"Content preview: {content_preview}")
        if ref_count:
            parts.append(f"Interactive elements: {ref_count} refs")
        parts.append(f"Original size: {len(content)} chars")
        return "\n".join(parts)

    def _compact_messages(self) -> None:
        """Compact context to stay under token limits."""
        serialized = self._serialize_messages()

        # --- Phase 1: Browser snapshot compaction ---
        snapshot_indices = []
        for i, msg in enumerate(serialized):
            if msg.get("role") != "tool":
                continue
            tool_name = self._find_tool_call_name(msg.get("tool_call_id", ""))
            if tool_name in BROWSER_SNAPSHOT_TOOLS:
                snapshot_indices.append(i)

        if len(snapshot_indices) > MAX_KEPT_SNAPSHOTS:
            to_compact = snapshot_indices[:-MAX_KEPT_SNAPSHOTS]
            for idx in to_compact:
                orig = self.messages[idx]
                content = str(orig.get("content", "") if isinstance(orig, dict) else (orig.content or ""))
                summary = self._summarize_snapshot(content)
                if isinstance(self.messages[idx], dict):
                    self.messages[idx]["content"] = summary
                else:
                    self.messages[idx] = {
                        "role": "tool",
                        "tool_call_id": orig.tool_call_id if hasattr(orig, "tool_call_id") else orig.get("tool_call_id"),
                        "content": summary,
                    }

        # --- Phase 2: Token-threshold summarization ---
        if self._estimate_tokens() <= TOKEN_COMPACT_THRESHOLD:
            return

        total = len(self.messages)
        if total <= KEEP_RECENT_MESSAGES + 1:
            return

        middle_start = 1
        middle_end = total - KEEP_RECENT_MESSAGES
        if middle_end <= middle_start:
            return

        middle = self._serialize_messages()[middle_start:middle_end]

        lines = []
        for msg in middle:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role.upper()}: {str(content)[:500]}")
        middle_text = "\n".join(lines)

        try:
            resp = self.llm.complete(
                messages=[
                    {"role": "system", "content": "You are a concise summarizer. Summarize the following conversation excerpt, preserving key facts, decisions, and any URLs or data the user/assistant referenced. Be brief."},
                    {"role": "user", "content": middle_text},
                ],
                max_tokens=1024,
            )
            self._log_usage(resp, "compact")
            summary_text = resp.text or "[summary unavailable]"
        except Exception as e:
            print(f"  [compact] Summarization failed: {e}")
            return

        summary_msg = {"role": "assistant", "content": f"[Context summary of earlier conversation]\n{summary_text}"}

        recent = self.messages[middle_end:]
        self.messages = [self.messages[0], summary_msg] + list(recent)

        # Remove orphaned tool messages
        valid_tc_ids: set[str] = set()
        for msg in self.messages:
            tc_list = None
            if isinstance(msg, dict):
                tc_list = msg.get("tool_calls")
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                tc_list = msg.tool_calls
            if tc_list:
                for tc in tc_list:
                    tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                    valid_tc_ids.add(tc_id)

        self.messages = [
            msg for msg in self.messages
            if not (
                isinstance(msg, dict) and msg.get("role") == "tool"
                and msg.get("tool_call_id") not in valid_tc_ids
            )
        ]

        print(f"  [compact] Context compacted — {len(self.messages)} messages, ~{self._estimate_tokens()} tokens")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, user_message: str, images: list[bytes] | None = None) -> str:
        if images:
            content: list[dict] = [{"type": "text", "text": user_message}]
            for img_bytes in images:
                mime = "image/png"
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

        max_rounds = self.max_rounds
        tools = active_tools_schema()
        for _round in range(max_rounds):
            self._compact_messages()
            response = self.llm.complete(
                self._serialize_messages(),
                tools=tools,
            )
            self._log_usage(response, "run")

            # Build canonical assistant message dict and store it
            assistant_msg: dict = {"role": "assistant", "content": response.text}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not response.tool_calls:
                self.save_session()
                return response.text or ""

            for tc in response.tool_calls:
                print(f"  [tool] {tc.name}({tc.arguments[:300]})")
                self._emit("tool_start", {"name": tc.name, "args": tc.arguments[:500]})
                result = dispatch_tool_call(tc.name, tc.arguments)
                self._emit("tool_end", {"name": tc.name, "result": (result or "")[:500]})
                if self.tool_logger:
                    try:
                        self.tool_logger(tc.name, tc.arguments, result)
                    except Exception:
                        pass
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Exhausted rounds — force final response without tools
        print(f"  [warning] Hit {max_rounds} tool rounds — forcing final response")
        response = self.llm.complete(self._serialize_messages())
        self._log_usage(response, "run-final")
        assistant_msg = {"role": "assistant", "content": response.text}
        self.messages.append(assistant_msg)
        self.save_session()
        return response.text or "Sorry, I got stuck processing that. Could you rephrase?"

    # ------------------------------------------------------------------
    # Memory / conversation formatting
    # ------------------------------------------------------------------

    def format_conversation_for_memory(self) -> list[dict]:
        """Convert conversation to SimpleMem dialogue format."""
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
            if self.chat_summary and self.chat_summary.strip() in text:
                continue
            dialogues.append({"role": role, "content": text})
        return dialogues

    def run_pre_exit(self, bandit_recommendations: str = "", existing_reminders: str = "",
                     previous_invocations: str = "", scheduled_nudges: str = "", care_digest: str = "") -> str:
        now_local = datetime.now().astimezone()
        if not care_digest:
            try:
                from care.brief import registry_digest
                care_digest = registry_digest()
            except Exception:
                care_digest = ""
        if not scheduled_nudges:
            try:
                from .tools.scheduling import list_scheduled_nudges
                scheduled_nudges = "Nudges already queued:\n" + list_scheduled_nudges()
            except Exception:
                scheduled_nudges = ""
        prompt = build_pre_exit_prompt(
            current_time=now_local.isoformat(),
            bandit_recommendations=bandit_recommendations,
            existing_reminders=existing_reminders,
            previous_invocations=previous_invocations,
            scheduled_nudges=scheduled_nudges,
            care_digest=care_digest,
        )
        print("\n[timeout] Session idle — running pre-exit flow...")
        answer = self.run(prompt)
        print(f"\nAgent: {answer}")
        return answer

    # ------------------------------------------------------------------
    # Conversation summary generation
    # ------------------------------------------------------------------

    def _format_conversation_for_summary(self) -> str:
        lines = []
        for msg in self._serialize_messages():
            role = msg.get("role", "")
            content = msg.get("content")
            if role in ("system", "tool") or content is None:
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
        response = self.llm.complete(
            messages=[{"role": "user", "content": f"{conversation_text}\n\n{SUMMARY_PROMPT}"}],
        )
        self._log_usage(response, "summary")
        return response.text or ""

    # ------------------------------------------------------------------
    # Summary persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_path(session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, f"{session_id}_summary.json")

    @staticmethod
    def load_summary(session_id: str) -> str:
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
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = Agent._summary_path(session_id)
        with open(path, "w") as f:
            json.dump({
                "summary": summary,
                "generated_at": datetime.now().astimezone().isoformat(),
            }, f, indent=2)

    def archive_session(self) -> None:
        if not self.session_id:
            return
        path = self._session_path()
        if not os.path.exists(path):
            return
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(SESSIONS_DIR, f"{self.session_id}_{now_str}.json")
        os.rename(path, archive_path)
        print(f"[archive] Session archived: {archive_path}")
