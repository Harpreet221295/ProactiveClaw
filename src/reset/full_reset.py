#!/usr/bin/env python3
"""
Full system reset — wipes everything back to a clean slate.

What gets cleared:
  ✦ Long-term memory (Mem0 / Qdrant)
  ✦ Invocation queue  (queue.json)
  ✦ Reminders         (reminders.json)
  ✦ Re-engagement     (reengagement.json)
  ✦ Cron jobs         (cron_jobs.json)  — protected jobs kept unless --nuke
  ✦ Session history   (sessions/*.json)
  ✦ Agent file system (agent_file_system/) — keeps the directory, clears contents
      Preserves: user_preferences.json  (unless --nuke)
  ✦ Subagent registry (currently_running_subagents.json)
  ✦ Daily brief       (daily_brief.json)
  ✦ Last session summary (last_session_summary.txt)

Options:
  --force   Skip all confirmation prompts
  --nuke    Also delete protected cron jobs and user_preferences.json
  --dry-run Show what would be deleted without actually doing it

Usage:
    python src/reset/full_reset.py [--force] [--nuke] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from reset._common import (
    QUEUE_FILE, REMINDERS_FILE, CRON_JOBS_FILE, REENGAGEMENT_FILE,
    CARE_REGISTRY_FILE, CARE_PATTERNS_FILE, NUDGE_LOG_FILE, TRANSCRIPT_FILE, BANDIT_STATE_FILE,
    AGENT_FS, MEM0_DATA, KUZU_GRAPH, SESSIONS_DIR,
    confirm, ok, warn, err, info, header,
    write_empty_json, remove_tree, remove_file,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


_MEM0_CONFIG = {
    "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini", "temperature": 0.1}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memory",
            "path": str(MEM0_DATA),
            "on_disk": True,
            "embedding_model_dims": 1536,
        },
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": str(MEM0_DATA.parent / "kuzu_graph"),
        },
    },
}


def _clear_memory_api() -> bool:
    import os
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        from mem0 import Memory
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            mem = Memory.from_config(_MEM0_CONFIG)
        result = mem.get_all(user_id="default")
        memories = result.get("results", []) if isinstance(result, dict) else (result or [])
        for m in memories:
            mem.delete(memory_id=m["id"])
        ok(f"Deleted {len(memories)} memories via Mem0 API")
        return True
    except Exception as e:
        warn(f"Mem0 API delete failed ({e}) — falling back to file wipe")
        return False


def _clear_memory_files() -> None:
    remove_tree(MEM0_DATA, "mem0_data/")
    MEM0_DATA.mkdir(parents=True, exist_ok=True)
    ok("Recreated empty mem0_data/")
    if KUZU_GRAPH.exists():
        remove_tree(KUZU_GRAPH, "kuzu_graph/")
        ok("Deleted kuzu_graph/")


def _clear_sessions() -> None:
    if not SESSIONS_DIR.exists():
        info("sessions/ not found — skipping")
        return
    files = list(SESSIONS_DIR.glob("*.json"))
    if not files:
        info("No session files found")
        return
    for f in files:
        f.unlink()
    ok(f"Deleted {len(files)} session file(s)")


def _clear_cron_jobs(nuke: bool) -> None:
    if not CRON_JOBS_FILE.exists():
        info("cron_jobs.json not found — skipping")
        return
    jobs = json.loads(CRON_JOBS_FILE.read_text()) or []
    if nuke:
        write_empty_json(CRON_JOBS_FILE, [])
        ok(f"Deleted all {len(jobs)} cron job(s) (--nuke)")
    else:
        protected = [j for j in jobs if j.get("protected") or j.get("id","").startswith("cron_deep_think")]
        write_empty_json(CRON_JOBS_FILE, protected)
        deleted = len(jobs) - len(protected)
        ok(f"Deleted {deleted} cron job(s), kept {len(protected)} protected")


def _clear_agent_fs(nuke: bool, dry_run: bool) -> None:
    """Clear agent_file_system contents, preserving user_preferences.json unless --nuke."""
    if not AGENT_FS.exists():
        info("agent_file_system/ not found — skipping")
        return

    preserve = set()
    if not nuke:
        pref = AGENT_FS / "user_preferences.json"
        if pref.exists():
            preserve.add(pref)
            info("Preserving user_preferences.json (use --nuke to delete)")

    deleted = 0
    for item in AGENT_FS.iterdir():
        if item in preserve:
            continue
        if dry_run:
            info(f"[dry-run] would delete {item.name}")
            deleted += 1
            continue
        if item.is_dir():
            import shutil
            shutil.rmtree(item)
        else:
            item.unlink()
        deleted += 1

    if not dry_run:
        # Recreate subagents directory (expected by agent)
        (AGENT_FS / "subagents").mkdir(exist_ok=True)
        ok(f"Cleared {deleted} item(s) from agent_file_system/")
    else:
        info(f"[dry-run] would clear {deleted} item(s) from agent_file_system/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full system reset")
    parser.add_argument("--force",   action="store_true", help="Skip all prompts")
    parser.add_argument("--nuke",    action="store_true",
                        help="Also wipe protected cron jobs and user_preferences.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without deleting anything")
    args = parser.parse_args()

    if HAS_RICH:
        nuke_note = "  [red bold]+ protected cron jobs + user_preferences.json (--nuke)[/red bold]" if args.nuke else ""
        dry_note  = "  [yellow][DRY RUN — nothing will be deleted][/yellow]" if args.dry_run else ""
        _console.print(Panel(
            "[bold red]Full System Reset[/bold red]\n\n"
            "This will permanently delete:\n"
            "  • All long-term memories\n"
            "  • Invocation queue, reminders, re-engagement tracker\n"
            "  • Cron jobs (non-protected)\n"
            "  • Session history files\n"
            "  • Agent file system contents\n"
            "  • Daily brief and last session summary\n"
            "  • Subagent registry"
            + nuke_note + dry_note,
            border_style="red",
            padding=(1, 2),
        ))
    else:
        print("=== Full System Reset ===")
        if args.nuke:
            print("  WARNING: --nuke flag set")
        if args.dry_run:
            print("  DRY RUN — nothing will be deleted")

    if not args.dry_run:
        if not confirm("Proceed with full reset?", args.force):
            info("Aborted.")
            return 0

    # ── 1. Memory ─────────────────────────────────────────────────────────────
    header("1. Long-term Memory")
    if args.dry_run:
        info("[dry-run] would clear all Mem0 memories")
    else:
        if not _clear_memory_api():
            _clear_memory_files()

    # ── 2. Queues ─────────────────────────────────────────────────────────────
    header("2. Queues & Reminders")
    if args.dry_run:
        for f in [QUEUE_FILE, REMINDERS_FILE, REENGAGEMENT_FILE]:
            info(f"[dry-run] would reset {f.name}")
    else:
        write_empty_json(QUEUE_FILE, []);        ok("queue.json → []")
        write_empty_json(REMINDERS_FILE, []);    ok("reminders.json → []")
        write_empty_json(REENGAGEMENT_FILE, {}); ok("reengagement.json → {}")
        remove_file(NUDGE_LOG_FILE, "nudge_log.json")

    # ── 2b. Care registry & patterns ──────────────────────────────────────────
    header("2b. Care registry, patterns, transcript")
    if args.dry_run:
        for f in [CARE_REGISTRY_FILE, CARE_PATTERNS_FILE, TRANSCRIPT_FILE]:
            info(f"[dry-run] would remove {f.name}")
        if args.nuke:
            info("[dry-run] would remove bandit_state.json")
    else:
        remove_file(CARE_REGISTRY_FILE, "care_registry.json")
        remove_file(CARE_PATTERNS_FILE, "care_patterns.json")
        remove_file(TRANSCRIPT_FILE, "transcript.jsonl")
        if args.nuke:
            remove_file(BANDIT_STATE_FILE, "bandit_state.json")

    # ── 3. Cron jobs ──────────────────────────────────────────────────────────
    header("3. Cron Jobs")
    if args.dry_run:
        info(f"[dry-run] would clear cron_jobs.json" + (" (all, --nuke)" if args.nuke else " (non-protected)"))
    else:
        _clear_cron_jobs(args.nuke)

    # ── 4. Sessions ───────────────────────────────────────────────────────────
    header("4. Session History")
    if args.dry_run:
        count = len(list(SESSIONS_DIR.glob("*.json"))) if SESSIONS_DIR.exists() else 0
        info(f"[dry-run] would delete {count} session file(s)")
    else:
        _clear_sessions()

    # ── 5. Agent file system ──────────────────────────────────────────────────
    header("5. Agent File System")
    _clear_agent_fs(args.nuke, args.dry_run)

    # ── 6. Subagent registry ──────────────────────────────────────────────────
    header("6. Subagent Registry")
    registry = AGENT_FS / "currently_running_subagents.json"
    if args.dry_run:
        info(f"[dry-run] would reset currently_running_subagents.json")
    else:
        write_empty_json(registry, [])
        ok("currently_running_subagents.json → []")

    # ── Done ──────────────────────────────────────────────────────────────────
    if HAS_RICH:
        from rich.rule import Rule
        _console.print()
        _console.print(Rule("[bold green]Reset complete[/bold green]" if not args.dry_run
                            else "[bold yellow]Dry run complete — nothing was deleted[/bold yellow]"))
    else:
        print("\n✓ Reset complete" if not args.dry_run else "\n[dry-run] Done — nothing deleted")

    if not args.dry_run:
        try:
            import os as _os
            _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
