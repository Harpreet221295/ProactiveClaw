#!/usr/bin/env python3
"""
Clear cron jobs.

Options:
  --all        Remove ALL cron jobs including protected ones (e.g. cron_deep_think)
  --force      Skip confirmation prompts
  --list       Just list current jobs without deleting

By default (no --all), protected jobs (id starts with "cron_deep_think" or
has "protected": true) are preserved.

Usage:
    python src/reset/clear_cron_jobs.py [--force] [--all] [--list]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from reset._common import (
    CRON_JOBS_FILE, confirm, ok, warn, info, header, write_empty_json,
)

try:
    from rich.table import Table
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    _console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _load_jobs() -> list[dict]:
    if not CRON_JOBS_FILE.exists():
        return []
    try:
        return json.loads(CRON_JOBS_FILE.read_text()) or []
    except Exception:
        return []


def _is_protected(job: dict) -> bool:
    return job.get("protected", False) or job.get("id", "").startswith("cron_deep_think")


def _print_jobs(jobs: list[dict]) -> None:
    if not jobs:
        info("No cron jobs found.")
        return

    if HAS_RICH:
        table = Table(
            title=f"Cron Jobs ({len(jobs)} total)",
            box=box.ROUNDED,
            border_style="cyan",
            show_lines=True,
        )
        table.add_column("ID",        style="cyan",   max_width=20)
        table.add_column("Time",      style="yellow", width=6)
        table.add_column("Days",      style="dim",    max_width=30)
        table.add_column("Protected", style="red",    width=9)
        table.add_column("Last Run",  style="dim",    width=22)
        table.add_column("Task",      style="white",  max_width=50)

        for j in jobs:
            days = ", ".join(d[:3].capitalize() for d in j.get("days", []))
            table.add_row(
                escape(j.get("id", "")),
                j.get("time", ""),
                escape(days),
                "🔒 yes" if _is_protected(j) else "",
                j.get("last_run", "never")[:19] if j.get("last_run") else "never",
                escape(j.get("task", "")[:80]) + ("…" if len(j.get("task","")) > 80 else ""),
            )
        _console.print(table)
    else:
        for j in jobs:
            prot = " [PROTECTED]" if _is_protected(j) else ""
            print(f"  {j.get('id','')}  {j.get('time','')}  {prot}")
            print(f"    {j.get('task','')[:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear cron jobs")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--all",   action="store_true", help="Also delete protected jobs")
    parser.add_argument("--list",  action="store_true", help="List jobs without deleting")
    args = parser.parse_args()

    header("Cron Jobs  (cron_jobs.json)")

    jobs = _load_jobs()
    _print_jobs(jobs)

    if args.list or not jobs:
        return 0

    protected = [j for j in jobs if _is_protected(j)]
    deletable = [j for j in jobs if not _is_protected(j)]

    if args.all:
        to_delete = jobs
        warn(f"--all flag set: this will also delete {len(protected)} protected job(s)")
    else:
        to_delete = deletable
        if protected:
            info(f"Keeping {len(protected)} protected job(s): "
                 + ", ".join(j.get("id","") for j in protected))

    if not to_delete:
        ok("Nothing to delete.")
        return 0

    if not confirm(f"Delete {len(to_delete)} cron job(s)?", args.force):
        info("Aborted.")
        return 0

    remaining = [] if args.all else protected
    write_empty_json(CRON_JOBS_FILE, remaining)

    ok(f"Deleted {len(to_delete)} cron job(s)"
       + (f", kept {len(remaining)} protected" if remaining else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
