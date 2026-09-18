"""Single source of truth for every on-disk location ProactiveClaw uses.

Everything is anchored to the project root (the directory containing `src/`).
Runtime data never lives inside `src/` so a clean checkout stays clean.
Override individual locations with environment variables when needed.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# ── User-facing files ─────────────────────────────────────────────────────────
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = Path(os.getenv("PROACTIVECLAW_CONFIG", CONFIG_DIR / "config.json"))
CONFIG_EXAMPLE_FILE = CONFIG_DIR / "config.example.json"

# ── Google OAuth files (Gmail + Calendar) ─────────────────────────────────────
GOOGLE_CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", PROJECT_ROOT / "credentials.json"))
GOOGLE_TOKEN_FILE = Path(os.getenv("GOOGLE_TOKEN_FILE", PROJECT_ROOT / "token.json"))

# ── Runtime state (all gitignored) ────────────────────────────────────────────
ENGAGEMENT_DATA_DIR = PROJECT_ROOT / "engagement_data"
QUEUE_FILE = ENGAGEMENT_DATA_DIR / "queue.json"
REMINDERS_FILE = ENGAGEMENT_DATA_DIR / "reminders.json"
CRON_JOBS_FILE = ENGAGEMENT_DATA_DIR / "cron_jobs.json"
REENGAGEMENT_FILE = ENGAGEMENT_DATA_DIR / "reengagement.json"
CARE_REGISTRY_FILE = ENGAGEMENT_DATA_DIR / "care_registry.json"
CARE_PATTERNS_FILE = ENGAGEMENT_DATA_DIR / "care_patterns.json"
NUDGE_LOG_FILE = ENGAGEMENT_DATA_DIR / "nudge_log.json"

AGENT_FS_DIR = PROJECT_ROOT / "agent_file_system"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

DATA_DIR = PROJECT_ROOT / "data"
MEM0_DIR = DATA_DIR / "mem0"
KUZU_GRAPH_DIR = DATA_DIR / "kuzu_graph"
BANDIT_STATE_FILE = DATA_DIR / "bandit_state.json"


def ensure_runtime_dirs() -> None:
    """Create every runtime directory that must exist before the app starts."""
    for d in (ENGAGEMENT_DATA_DIR, AGENT_FS_DIR, SESSIONS_DIR, DATA_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()
