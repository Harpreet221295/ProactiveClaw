#!/usr/bin/env python3
"""
Memory Conversation Simulator
──────────────────────────────
Test memory storage and graph extraction without running the full agent.

Two modes:

  Interactive (default):
    You type user messages, optionally type the agent response, and each
    turn is stored via store_dialogues() — exactly how the real agent does it.
    After each store, shows what memory items were extracted and what new
    graph edges appeared.

  Scripted (--script):
    Feeds pre-written conversation pairs automatically with a delay,
    so you can focus on watching the graph update on another terminal.

Terminal layout:
  Terminal 1:  ./bash_scripts/memory_simulator.sh
  Terminal 2:  ./memory_monitor.sh --watch-graph

Usage:
  ./bash_scripts/memory_simulator.sh                  # interactive
  ./bash_scripts/memory_simulator.sh --script         # scripted pairs
  ./bash_scripts/memory_simulator.sh --script --delay 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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
    from rich.table import Table
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    print("rich not installed — run: pip install rich")
    sys.exit(1)

# ── Scripted conversation pairs ────────────────────────────────────────────────
# These simulate realistic agent conversations so mem0's LLM extracts
# meaningful entities and relationships into the graph.
SCRIPTED_PAIRS = [
    (
        "Hey, I'm Harpreet. I work at a tech startup called NovaBridge in San Francisco.",
        "Nice to meet you Harpreet! I'll remember that you work at NovaBridge in San Francisco.",
    ),
    (
        "I'm the CTO there. We're building an AI product called ProactiveClaw.",
        "Got it — you're the CTO at NovaBridge and you're building ProactiveClaw.",
    ),
    (
        "My co-founder is Sarah. She handles the product side.",
        "Noted — Sarah is your co-founder and handles product at NovaBridge.",
    ),
    (
        "I prefer working in Python and we use OpenAI APIs for our LLM calls.",
        "Understood — Python is your main language and you use OpenAI APIs.",
    ),
    (
        "I have a meeting with Sarah tomorrow at 10am to review the Q2 roadmap.",
        "I'll note that you have a meeting with Sarah tomorrow at 10am for Q2 roadmap review.",
    ),
    (
        "Sarah is based in New York. She flies to SF every two weeks.",
        "Got it — Sarah is in New York and visits San Francisco every two weeks.",
    ),
    (
        "ProactiveClaw uses mem0 for long-term memory and Qdrant for vector storage.",
        "Noted — ProactiveClaw uses mem0 and Qdrant as its memory stack.",
    ),
    (
        "I really dislike context switching. I prefer deep work blocks of 3-4 hours.",
        "Understood — you prefer long uninterrupted deep work sessions over context switching.",
    ),
    (
        "We're planning to raise a Series A round next quarter. Our lead investor is Maya from Sequoia.",
        "Got it — you're raising a Series A next quarter and Maya from Sequoia is your lead investor.",
    ),
    (
        "I went to IIT Delhi for undergrad and then did my Masters at Carnegie Mellon.",
        "Noted — you studied at IIT Delhi and Carnegie Mellon.",
    ),
]


# ── Memory helpers ─────────────────────────────────────────────────────────────

def _init_store():
    """Import and return store_dialogues. Exits if unavailable."""
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]✗ OPENAI_API_KEY not set[/red]")
        sys.exit(1)
    try:
        from memory import store_dialogues, _get_memory
        # warm up the mem instance so first turn is fast
        mem = _get_memory()
        if mem is None:
            console.print("[red]✗ Memory init failed — check OPENAI_API_KEY and dependencies[/red]")
            sys.exit(1)
        return store_dialogues
    except Exception as e:
        console.print(f"[red]✗ Memory init failed: {e}[/red]")
        sys.exit(1)


def _graph_snapshot() -> list[dict]:
    """Read current Kuzu graph edges."""
    try:
        from memory_monitor.monitor import _kuzu_graph_data
        _, relations = _kuzu_graph_data()
        return relations
    except Exception:
        return []


def _diff_edges(before: list[dict], after: list[dict]) -> list[dict]:
    """Return edges that are in after but not in before."""
    before_keys = {
        (r["source"].lower(), r["relationship"].lower(), r["destination"].lower())
        for r in before
    }
    return [
        r for r in after
        if (r["source"].lower(), r["relationship"].lower(), r["destination"].lower())
        not in before_keys
    ]


def _print_turn_result(store_result: str, new_edges: list[dict], turn: int) -> None:
    """Print what was stored and what new graph edges appeared."""
    # Memory items
    items = [
        line.lstrip("- ").strip()
        for line in store_result.splitlines()
        if line.strip() and not line.startswith("*") and not line.startswith("Stored")
    ]
    count_match = [l for l in store_result.splitlines() if "Stored" in l]
    count_str = count_match[0] if count_match else store_result

    console.print(f"  [dim]{count_str}[/dim]")

    if new_edges:
        console.print(f"  [bold green]+{len(new_edges)} new graph edge(s):[/bold green]")
        for e in new_edges:
            console.print(
                f"    [cyan]{escape(e['source'])}[/cyan]"
                f"  [dim]—[{escape(e['relationship'])}]→[/dim]"
                f"  [green]{escape(e['destination'])}[/green]"
            )
    else:
        console.print("  [dim]No new graph edges this turn.[/dim]")


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_interactive(store_dialogues) -> None:
    console.print(Panel(
        "[bold cyan]Interactive Mode[/bold cyan]\n\n"
        "Type messages as if you're talking to the agent.\n"
        "Each turn is stored via [cyan]store_dialogues()[/cyan] — same path as the real agent.\n\n"
        "[dim]Terminal 2:[/dim]  [cyan]./memory_monitor.sh --watch-graph[/cyan]\n\n"
        "[dim]Commands:[/dim]  [yellow]quit[/yellow] or [yellow]exit[/yellow] to stop  •  "
        "[yellow]graph[/yellow] to print current graph  •  [yellow]skip[/yellow] to skip agent response",
        border_style="cyan",
        padding=(1, 2),
    ))

    turn = 0
    while True:
        console.print()
        console.print(Rule(f"[dim]Turn {turn + 1}[/dim]"))

        # ── User message ──────────────────────────────────────────────────────
        try:
            user_msg = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "q"):
            break
        if user_msg.lower() == "graph":
            from memory_monitor.monitor import _kuzu_graph_data, _render_relations_tree
            entities, relations = _kuzu_graph_data()
            console.print(f"[dim]Graph: {len(entities)} nodes, {len(relations)} edges[/dim]")
            _render_relations_tree(relations, title="Current graph")
            continue

        # ── Agent response ────────────────────────────────────────────────────
        try:
            agent_resp = console.input(
                "[bold yellow]Agent (Enter for default):[/bold yellow] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if agent_resp.lower() == "skip":
            console.print("[dim]  Skipped — not storing this turn.[/dim]")
            continue
        if not agent_resp:
            agent_resp = "Got it, noted."

        # ── Store + diff graph ────────────────────────────────────────────────
        before = _graph_snapshot()
        result = store_dialogues([
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": agent_resp},
        ])
        after = _graph_snapshot()
        new_edges = _diff_edges(before, after)

        _print_turn_result(result, new_edges, turn)
        turn += 1

    console.print()
    console.print(Rule("[dim]Session ended[/dim]"))
    _print_final_summary()


def run_scripted(store_dialogues, delay: float) -> None:
    console.print(Panel(
        "[bold cyan]Scripted Mode[/bold cyan]\n\n"
        f"Feeding {len(SCRIPTED_PAIRS)} pre-written conversation pairs with {delay}s delay.\n"
        "Watch the graph update on another terminal:\n\n"
        "[dim]Terminal 2:[/dim]  [cyan]./memory_monitor.sh --watch-graph[/cyan]\n\n"
        "[dim]Press Ctrl+C to stop early.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    total = len(SCRIPTED_PAIRS)
    try:
        for i, (user_msg, agent_resp) in enumerate(SCRIPTED_PAIRS, 1):
            console.print()
            console.print(Rule(f"[dim]Turn {i}/{total}[/dim]"))
            console.print(f"[bold cyan]User:[/bold cyan]  {escape(user_msg)}")
            console.print(f"[bold yellow]Agent:[/bold yellow] {escape(agent_resp)}")

            before = _graph_snapshot()
            result = store_dialogues([
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": agent_resp},
            ])
            after  = _graph_snapshot()
            new_edges = _diff_edges(before, after)

            _print_turn_result(result, new_edges, i)

            if i < total:
                console.print(f"[dim]  Waiting {delay}s…  (Ctrl+C to stop)[/dim]")
                time.sleep(delay)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped early.[/dim]")

    console.print()
    console.print(Rule("[dim]Done[/dim]"))
    _print_final_summary()


def _print_final_summary() -> None:
    from memory_monitor.monitor import _kuzu_graph_data, _fetch_all, _init_memory
    entities, relations = _kuzu_graph_data()

    t = Table(title=f"Final Graph State — {len(entities)} nodes, {len(relations)} edges",
              box=box.ROUNDED, border_style="cyan", show_lines=False)
    t.add_column("Source",       style="cyan",   max_width=25)
    t.add_column("Relationship", style="yellow", max_width=20, no_wrap=True)
    t.add_column("Destination",  style="green",  max_width=30)
    for r in relations:
        t.add_row(escape(r["source"]), escape(r["relationship"]), escape(r["destination"]))
    console.print(t)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Memory Conversation Simulator")
    parser.add_argument("--script", action="store_true",
                        help="Use pre-written scripted conversation pairs instead of interactive input")
    parser.add_argument("--delay",  type=float, default=5.0,
                        help="Seconds between turns in scripted mode (default: 5)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]ProactiveClaw — Memory Conversation Simulator[/bold cyan]\n"
        "[dim]Test graph-backed memory without running the full agent[/dim]",
        border_style="cyan", padding=(1, 4),
    ))

    console.print("[cyan]Initialising memory store (vector + graph)…[/cyan]")
    store_dialogues = _init_store()
    console.print("[green]✓ Ready[/green]\n")

    if args.script:
        run_scripted(store_dialogues, args.delay)
    else:
        run_interactive(store_dialogues)

    try:
        import os as _os
        _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
