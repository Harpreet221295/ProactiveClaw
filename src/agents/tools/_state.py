from __future__ import annotations

import asyncio
from pathlib import Path

from core import paths as _paths

# Project root: kept as module attributes because tests monkeypatch them.
_PROJECT_ROOT = _paths.PROJECT_ROOT
_ENGAGEMENT_DATA_DIR = _paths.ENGAGEMENT_DATA_DIR

QUEUE_FILE = str(_paths.QUEUE_FILE)
REMINDERS_FILE = str(_paths.REMINDERS_FILE)
CRON_JOBS_FILE = str(_paths.CRON_JOBS_FILE)
REENGAGEMENT_FILE = str(_paths.REENGAGEMENT_FILE)

# Agent file system base directory
_AGENT_FS_BASE: Path = _paths.AGENT_FS_DIR

_current_session_id: str | None = None

# Event loop reference — set by the server lifespan, used by browser tool's sync-to-async bridge
_event_loop: asyncio.AbstractEventLoop | None = None


def set_current_session_id(session_id: str) -> None:
    global _current_session_id
    _current_session_id = session_id


def get_current_session_id() -> str | None:
    return _current_session_id
