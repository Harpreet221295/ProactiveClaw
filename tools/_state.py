import os
from pathlib import Path

QUEUE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "queue.json")
REMINDERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reminders.json")
CRON_JOBS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cron_jobs.json")
REENGAGEMENT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reengagement.json")

# Agent file system base directory
_AGENT_FS_BASE = Path(__file__).parent.parent / "agent_file_system"
_AGENT_FS_BASE.mkdir(exist_ok=True)

_current_session_id: str | None = None


def set_current_session_id(session_id: str) -> None:
    global _current_session_id
    _current_session_id = session_id


def get_current_session_id() -> str | None:
    return _current_session_id
