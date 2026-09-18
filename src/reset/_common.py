"""Shared helpers for all reset scripts."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_SRC          = Path(__file__).parents[1]          # src/
_ROOT         = Path(__file__).parents[2]          # project root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from core import paths as _paths

ENGAGEMENT    = _paths.ENGAGEMENT_DATA_DIR
AGENT_FS      = _paths.AGENT_FS_DIR
MEM0_DATA     = _paths.MEM0_DIR
KUZU_GRAPH    = _paths.KUZU_GRAPH_DIR
SESSIONS_DIR  = _paths.SESSIONS_DIR
DATA_DIR      = _paths.DATA_DIR

# individual files
QUEUE_FILE        = _paths.QUEUE_FILE
REMINDERS_FILE    = _paths.REMINDERS_FILE
CRON_JOBS_FILE    = _paths.CRON_JOBS_FILE
REENGAGEMENT_FILE = _paths.REENGAGEMENT_FILE
CARE_REGISTRY_FILE = _paths.CARE_REGISTRY_FILE
CARE_PATTERNS_FILE = _paths.CARE_PATTERNS_FILE
NUDGE_LOG_FILE    = _paths.NUDGE_LOG_FILE
TRANSCRIPT_FILE   = DATA_DIR / "transcript.jsonl"
BANDIT_STATE_FILE = _paths.BANDIT_STATE_FILE


# ── Rich (optional, graceful fallback) ────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markup import escape
    console = Console()

    def ok(msg: str)   -> None: console.print(f"[green]✓[/green] {msg}")
    def warn(msg: str) -> None: console.print(f"[yellow]⚠[/yellow] {msg}")
    def err(msg: str)  -> None: console.print(f"[red]✗[/red] {msg}")
    def info(msg: str) -> None: console.print(f"[dim]  {msg}[/dim]")
    def header(msg: str) -> None:
        from rich.rule import Rule
        console.print(Rule(f"[bold cyan]{msg}[/bold cyan]"))

except ImportError:
    def ok(msg):   print(f"✓ {msg}")
    def warn(msg): print(f"⚠ {msg}")
    def err(msg):  print(f"✗ {msg}")
    def info(msg): print(f"  {msg}")
    def header(msg): print(f"\n── {msg} ──")


# ── Helpers ───────────────────────────────────────────────────────────────────

def confirm(prompt: str, force: bool) -> bool:
    """Return True if user confirms or --force is passed."""
    if force:
        return True
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def write_empty_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def remove_tree(path: Path, label: str) -> None:
    if path.exists():
        shutil.rmtree(path)
        ok(f"Removed {label}  [dim]({path})[/dim]")
    else:
        info(f"{label} not found — skipping")


def remove_file(path: Path, label: str) -> None:
    if path.exists():
        path.unlink()
        ok(f"Removed {label}  [dim]({path})[/dim]")
    else:
        info(f"{label} not found — skipping")
