#!/usr/bin/env python3
"""
Clear notification and invocation queues.

Resets:
  • queue.json          — scheduled invocations (agent-generated nudges)
  • reminders.json      — user-requested reminders
  • reengagement.json   — re-engagement DM tracking

Usage:
    python src/reset/clear_queues.py [--force] [--queue] [--reminders] [--reengagement]

    With no flags, clears all three.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from reset._common import (
    QUEUE_FILE, REMINDERS_FILE, REENGAGEMENT_FILE,
    confirm, ok, info, header, write_empty_json,
)


def clear_queue(force: bool) -> None:
    header("Invocation Queue  (queue.json)")
    if QUEUE_FILE.exists():
        import json
        data = json.loads(QUEUE_FILE.read_text())
        count = len(data) if isinstance(data, list) else 0
        info(f"Contains {count} pending invocation(s)")
    else:
        count = 0
        info("queue.json not found — will create empty")

    if count == 0 or confirm(f"Clear {count} pending invocation(s)?", force):
        write_empty_json(QUEUE_FILE, [])
        ok("queue.json reset to []")
    else:
        info("Skipped queue.json")


def clear_reminders(force: bool) -> None:
    header("Reminders  (reminders.json)")
    if REMINDERS_FILE.exists():
        import json
        data = json.loads(REMINDERS_FILE.read_text())
        count = len(data) if isinstance(data, list) else 0
        info(f"Contains {count} reminder(s)")
    else:
        count = 0
        info("reminders.json not found — will create empty")

    if count == 0 or confirm(f"Clear {count} reminder(s)?", force):
        write_empty_json(REMINDERS_FILE, [])
        ok("reminders.json reset to []")
    else:
        info("Skipped reminders.json")


def clear_reengagement(force: bool) -> None:
    header("Re-engagement tracker  (reengagement.json)")
    if REENGAGEMENT_FILE.exists():
        import json
        data = json.loads(REENGAGEMENT_FILE.read_text())
        info(f"last_dm: {data.get('last_dm', 'none')}  interval_index: {data.get('interval_index', 0)}")
    else:
        info("reengagement.json not found — will create empty")

    if confirm("Reset re-engagement tracker?", force):
        write_empty_json(REENGAGEMENT_FILE, {})
        ok("reengagement.json reset to {}")
    else:
        info("Skipped reengagement.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear notification and invocation queues")
    parser.add_argument("--force",        action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--queue",        action="store_true", help="Clear invocation queue only")
    parser.add_argument("--reminders",    action="store_true", help="Clear reminders only")
    parser.add_argument("--reengagement", action="store_true", help="Clear re-engagement tracker only")
    args = parser.parse_args()

    # If no specific flag given, clear all
    all_targets = not (args.queue or args.reminders or args.reengagement)

    if all_targets or args.queue:
        clear_queue(args.force)
    if all_targets or args.reminders:
        clear_reminders(args.force)
    if all_targets or args.reengagement:
        clear_reengagement(args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
