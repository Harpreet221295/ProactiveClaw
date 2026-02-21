import base64
import json
import os
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from tools import TOOLS_SCHEMA, dispatch_tool_call
from prompts import SYSTEM_PROMPT, PRE_EXIT_PROMPT, SUMMARY_PROMPT

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")

DEFAULT_MODEL = "gpt-5-mini"

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
    def __init__(self, model: str = DEFAULT_MODEL, session_id: str | None = None,
                 chat_summary: str | None = None):
        self.client = OpenAI()
        self.model = model
        self.session_id = session_id
        self.chat_summary = chat_summary
        self.token_usage = {"input": 0, "output": 0, "total": 0}
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

    # ------------------------------------------------------------------
    # Mid-conversation context compaction
    # ------------------------------------------------------------------

    def _log_usage(self, response, label: str = "llm") -> None:
        """Extract token usage from an OpenAI response and log it."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        inp = usage.prompt_tokens
        out = usage.completion_tokens
        tot = usage.total_tokens
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
        """Build a rich one-line summary of a browser snapshot.

        Preserves: page URL (so agent can browser_navigate back), page title,
        a content preview of key visible elements, and interactive-element count.
        """
        lines = content.split("\n")

        # --- Extract URL ---
        url = ""
        for line in lines[:10]:
            stripped = line.strip()
            # browser_get_page_info returns JSON with "url" key
            if stripped.startswith("Page info:"):
                try:
                    import json as _json
                    info = _json.loads(stripped.split("Page info:", 1)[1].strip())
                    url = info.get("url", "")
                except Exception:
                    pass
                break
            # snapshot content may contain the URL in the first few lines
            if stripped.startswith("http://") or stripped.startswith("https://"):
                url = stripped[:200]
                break
            if "url" in stripped.lower() and "http" in stripped:
                # Try to pull URL from lines like 'URL: https://...'
                for part in stripped.split():
                    if part.startswith("http"):
                        url = part[:200]
                        break
                if url:
                    break

        # --- Extract page title ---
        title = ""
        for line in lines[:15]:
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                title = stripped.split(":", 1)[1].strip()[:120]
                break
            # Also check for JSON title from page_info
            if "title" in stripped and "{" in stripped:
                try:
                    import json as _json
                    info = _json.loads(stripped.split("Page info:", 1)[-1].strip())
                    title = info.get("title", "")[:120]
                except Exception:
                    pass

        # --- Content preview: grab first meaningful non-empty lines ---
        preview_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip very short lines, ref-only lines, and structural markers
            if len(stripped) < 10:
                continue
            if stripped.startswith("[ref:") or stripped.startswith("http"):
                continue
            preview_lines.append(stripped[:100])
            if len(preview_lines) >= 3:
                break
        content_preview = " | ".join(preview_lines) if preview_lines else "(no text content)"

        # --- Count interactive elements ---
        ref_count = content.count("[ref:")

        # --- Build summary ---
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
        """Compact context to stay under token limits.

        1. Replace old browser snapshot results with one-line summaries.
        2. If estimated tokens still exceed threshold, summarize the middle
           section of the conversation.
        """
        serialized = self._serialize_messages()

        # --- Phase 1: Browser snapshot compaction ---
        # Find indices of tool messages that are browser snapshot results
        snapshot_indices = []
        for i, msg in enumerate(serialized):
            if msg.get("role") != "tool":
                continue
            tool_name = self._find_tool_call_name(msg.get("tool_call_id", ""))
            if tool_name in BROWSER_SNAPSHOT_TOOLS:
                snapshot_indices.append(i)

        # Replace all but the last MAX_KEPT_SNAPSHOTS with a rich summary
        if len(snapshot_indices) > MAX_KEPT_SNAPSHOTS:
            to_compact = snapshot_indices[:-MAX_KEPT_SNAPSHOTS]
            for idx in to_compact:
                orig = self.messages[idx]
                content = str(orig.get("content", "") if isinstance(orig, dict) else (orig.content or ""))

                summary = self._summarize_snapshot(content)

                # Replace in-place
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
            return  # Nothing to summarize

        # Boundaries: keep system (0), keep last KEEP_RECENT_MESSAGES
        middle_start = 1
        middle_end = total - KEEP_RECENT_MESSAGES  # exclusive
        if middle_end <= middle_start:
            return

        middle = self._serialize_messages()[middle_start:middle_end]

        # Format middle section for summarization
        lines = []
        for msg in middle:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role.upper()}: {str(content)[:500]}")
        middle_text = "\n".join(lines)

        # LLM summarization call
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise summarizer. Summarize the following conversation excerpt, preserving key facts, decisions, and any URLs or data the user/assistant referenced. Be brief."},
                    {"role": "user", "content": middle_text},
                ],
                max_tokens=1024,
            )
            self._log_usage(resp, "compact")
            summary_text = resp.choices[0].message.content or "[summary unavailable]"
        except Exception as e:
            print(f"  [compact] Summarization failed: {e}")
            return

        # Build replacement message
        summary_msg = {"role": "assistant", "content": f"[Context summary of earlier conversation]\n{summary_text}"}

        # Rebuild self.messages: system + summary + recent
        recent = self.messages[middle_end:]
        self.messages = [self.messages[0], summary_msg] + list(recent)

        # Remove orphaned tool messages whose tool_call_id doesn't match any assistant tool_call
        valid_tc_ids = set()
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
                (isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") not in valid_tc_ids)
            )
        ]

        print(f"  [compact] Context compacted — {len(self.messages)} messages, ~{self._estimate_tokens()} tokens")

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
            self._compact_messages()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._serialize_messages(),
                tools=TOOLS_SCHEMA,
            )
            self._log_usage(response, "run")
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
        self._log_usage(response, "run-final")
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
        self._log_usage(response, "summary")
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
