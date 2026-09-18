"""Delivery channels.

The web UI (WebSocket) is always on. Slack is an optional extra outbound
channel, active only when SLACK_BOT_TOKEN / SLACK_USER_ID are set. Every
user-visible message is also appended to data/transcript.jsonl so the UI can
reload history (including nudges and reminders, which never live in the
agent's session file).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import DATA_DIR

TRANSCRIPT_FILE = DATA_DIR / "transcript.jsonl"
_MAX_TRANSCRIPT_LINES = 2000

_clients: set[Any] = set()
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

# Slack (optional)
_slack_client = None
_slack_channel: str | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def register(ws: Any) -> None:
    with _clients_lock:
        _clients.add(ws)


def unregister(ws: Any) -> None:
    with _clients_lock:
        _clients.discard(ws)


def client_count() -> int:
    with _clients_lock:
        return len(_clients)


# ── Transcript ──────────────────────────────────────────────────────────────

_transcript_lock = threading.Lock()


def _append_transcript(entry: dict[str, Any]) -> None:
    with _transcript_lock:
        TRANSCRIPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_transcript(limit: int = 200) -> list[dict[str, Any]]:
    if not TRANSCRIPT_FILE.exists():
        return []
    try:
        lines = TRANSCRIPT_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear_transcript() -> None:
    with _transcript_lock:
        if TRANSCRIPT_FILE.exists():
            TRANSCRIPT_FILE.unlink()


# ── Broadcasting ────────────────────────────────────────────────────────────

async def broadcast(event: dict[str, Any]) -> None:
    """Send an event to every connected web client."""
    with _clients_lock:
        targets = list(_clients)
    dead = []
    for ws in targets:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister(ws)


def emit_threadsafe(event: dict[str, Any]) -> None:
    """Broadcast from a worker thread (e.g. agent tool callbacks)."""
    if _loop is None or _loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast(event), _loop)
    except RuntimeError:
        pass


async def post_message(role: str, text: str, *, source: str = "", files: list[str] | None = None,
                       meta: dict[str, Any] | None = None, slack: bool = True) -> dict[str, Any]:
    """Deliver a user-visible message to every channel and persist it."""
    entry = {
        "type": "message",
        "role": role,                      # user | assistant | nudge | reminder | morning_review | system
        "text": text or "",
        "source": source,
        "files": files or [],
        "meta": meta or {},
        "at": datetime.now().astimezone().isoformat(),
    }
    _append_transcript(entry)
    await broadcast(entry)
    if slack and role != "user":
        _slack_post(role, text, files or [])
    return entry


async def set_status(state: str, detail: str = "") -> None:
    await broadcast({"type": "status", "state": state, "detail": detail,
                     "at": datetime.now().astimezone().isoformat()})


def set_status_threadsafe(state: str, detail: str = "") -> None:
    emit_threadsafe({"type": "status", "state": state, "detail": detail,
                     "at": datetime.now().astimezone().isoformat()})


# ── Slack (optional outbound) ───────────────────────────────────────────────

def slack_enabled() -> bool:
    return bool(os.getenv("SLACK_BOT_TOKEN") and os.getenv("SLACK_USER_ID"))


def init_slack() -> bool:
    global _slack_client, _slack_channel
    if not slack_enabled():
        return False
    try:
        from slack_sdk import WebClient
        _slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        resp = _slack_client.conversations_open(users=[os.environ["SLACK_USER_ID"]])
        _slack_channel = resp["channel"]["id"]
        print(f"[slack] Optional Slack channel active (DM {_slack_channel})")
        return True
    except Exception as e:
        print(f"[slack] Disabled — could not open DM channel: {e}")
        _slack_client = None
        return False


def _md_to_mrkdwn(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^#+\s*(.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\((https?://[^)]+)\)", r"<\2|\1>", text)
    return text


def _slack_post(role: str, text: str, files: list[str]) -> None:
    if _slack_client is None or not _slack_channel or not text:
        return
    prefix = {"nudge": ":loudspeaker: ", "reminder": ":bell: *Reminder:* ", "morning_review": ":sunrise: *Morning review*\n"}.get(role, "")
    body = prefix + _md_to_mrkdwn(text)
    try:
        for i in range(0, len(body), 3900):
            _slack_client.chat_postMessage(channel=_slack_channel, text=body[i:i + 3900])
        for path in files:
            if os.path.isfile(path):
                _slack_client.files_upload_v2(channel=_slack_channel, file=path, title=os.path.basename(path))
    except Exception as e:
        print(f"[slack] post failed: {e}")
