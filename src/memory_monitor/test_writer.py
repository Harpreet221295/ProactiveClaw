#!/usr/bin/env python3
"""
Memory Test Writer
──────────────────
Run this on Terminal 1 to add test memories one by one with a delay.
Watch them appear live in the memory monitor on Terminal 2:

    Terminal 1:  ./memory_test.sh
    Terminal 2:  ./memory_monitor.sh --watch

Options:
    --delay N    Seconds between each add (default: 4)
    --batch      Add all at once without pausing
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SRC  = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.markup import escape
except ImportError:
    print("rich not installed. Run: pip install rich")
    sys.exit(1)

console = Console()

# ── test memories ──────────────────────────────────────────────────────────────
# These simulate the kind of things the agent would extract from conversations.
TEST_MEMORIES = [
    "User's name is Harpreet Singh and he works at a tech startup in San Francisco.",
    "Harpreet is building an AI-powered personal assistant called ProactiveClaw.",
    "Harpreet prefers concise Slack messages — no walls of text, short and scannable.",
    "Harpreet is interested in Physical AI, robotics, and embodied intelligence.",
    "ProactiveClaw uses mem0 for long-term memory and Qdrant as the vector store.",
    "Harpreet likes to start his mornings with a daily brief from the agent.",
    "The agent uses OpenAI gpt-4.1-mini for LLM and text-embedding-3-small for embeddings.",
    "Harpreet dislikes being asked for information the agent can look up itself.",
    "ProactiveClaw integrates with Google Calendar, Gmail, Notion, and Slack.",
    "Harpreet focuses on deep work in the mornings and prefers no meetings before 10am.",
    "The TasksDB in Notion has ID 2eb5d319-2f3e-8061-a2af-f0691b0e6740.",
    "Harpreet is learning about bandit algorithms for notification scheduling.",
]

from core.paths import MEM0_DIR as _MEM0_DIR_P, KUZU_GRAPH_DIR as _KUZU_DIR_P
_MEMORY_DIR = str(_MEM0_DIR_P)
_GRAPH_DIR  = str(_KUZU_DIR_P)
_USER_ID    = "default"

_MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4.1-mini", "temperature": 0.1},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memory",
            "path": _MEMORY_DIR,
            "on_disk": True,
            "embedding_model_dims": 1536,
        },
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": _GRAPH_DIR,
        },
    },
}


def _init_memory():
    import os
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]✗ OPENAI_API_KEY not set[/red]")
        sys.exit(1)
    try:
        from mem0 import Memory
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = Memory.from_config(_MEM0_CONFIG)
        return m
    except Exception as e:
        console.print(f"[red]✗ Mem0 init failed: {e}[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Add test memories to Mem0 one by one")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="Seconds between each add (default: 4)")
    parser.add_argument("--batch", action="store_true",
                        help="Add all memories at once without pausing")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]Memory Test Writer[/bold cyan]\n"
        "[dim]Adds test memories one by one so you can watch them appear in the monitor[/dim]\n\n"
        f"[dim]Terminal 2:[/dim]  [cyan]./memory_monitor.sh --watch[/cyan]",
        border_style="cyan",
        padding=(1, 4),
    ))

    console.print("[cyan]Connecting to Mem0…[/cyan]")
    mem = _init_memory()
    console.print("[green]✓ Connected[/green]\n")

    total = len(TEST_MEMORIES)
    console.print(f"[dim]Will add {total} test memories"
                  + (" one by one" if not args.batch else " all at once")
                  + (f" with {args.delay}s delay" if not args.batch else "")
                  + "[/dim]\n")

    if not args.batch:
        console.print("[dim]Press Ctrl+C at any time to stop.[/dim]\n")

    added_count = 0

    try:
        for i, text in enumerate(TEST_MEMORIES, 1):
            console.print(Rule(f"[dim]Memory {i}/{total}[/dim]"))
            console.print(f"[yellow]Adding:[/yellow] {escape(text)}")

            try:
                result = mem.add(text, user_id=_USER_ID)
                items = result.get("results", []) if isinstance(result, dict) else []
                if items:
                    for item in items:
                        event = item.get("event", "ADD")
                        color = {"ADD": "green", "UPDATE": "yellow", "NONE": "dim"}.get(event, "white")
                        mem_text = item.get("memory", text)
                        short_id = (item.get("id") or "")[:8]
                        console.print(
                            f"  [{color}]{event}[/{color}]  "
                            f"[dim]{short_id}[/dim]  "
                            f"{escape(mem_text[:80])}{'…' if len(mem_text) > 80 else ''}"
                        )
                    added_count += len(items)
                else:
                    console.print("  [green]✓ Stored[/green]")
                    added_count += 1
            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")

            if not args.batch and i < total:
                console.print(f"[dim]  Waiting {args.delay}s… (Ctrl+C to stop)[/dim]")
                time.sleep(args.delay)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped early.[/dim]")

    console.print()
    console.print(Rule())
    console.print(f"[green]Done — added {added_count} memory item(s) across {i} inputs.[/green]")

    try:
        import os as _os
        _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
    except OSError:
        pass


if __name__ == "__main__":
    main()
