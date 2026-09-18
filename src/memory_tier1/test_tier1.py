#!/usr/bin/env python3
"""
Tier 1 Memory Retrieval — Test Runner

Two modes:
  Scripted (default): runs a predefined set of test cases and shows results
  Interactive:        you type messages and see what Tier 1 injects

Run:
    python src/memory_tier1/test_tier1.py             # scripted
    python src/memory_tier1/test_tier1.py --interactive
"""

from __future__ import annotations

import argparse
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
    from rich.table import Table
    from rich.markup import escape
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FakeConsole:
        def print(self, *a, **kw): print(*a)
        def input(self, prompt=""): return input(prompt)
    console = _FakeConsole()

# ── test cases ─────────────────────────────────────────────────────────────────
# Each entry: (message, what_we_expect_to_see_description)
TEST_CASES = [
    # ── Original cases ───────────────────────────────────────────────────────
    (
        "when is my meeting with Sarah?",
        "Expect: harpreet→sarah edge (co-founder / weekly_1:1)",
    ),
    (
        "I want to call my co-founder",
        "Expect: harpreet→sarah via co_founder relationship",
    ),
    (
        "what do I know about my lead investor?",
        "Expect: maya→harpreet via investor_of",
    ),
    (
        "tell me about ProactiveClaw's tech stack",
        "Expect: proactiveclaw→kuzu, proactiveclaw→qdrant, proactiveclaw→openai edges",
    ),
    (
        "Maya told me the term sheet is ready",
        "Expect: maya→harpreet (no-rel fallback)",
    ),
    (
        "I need to prep for my IIT Delhi reunion",
        "Expect: harpreet→iit_delhi edges",
    ),
    (
        "what's the latest with NovaBridge's Anthropic partnership?",
        "Expect: novabridge→anthropic partnership edge",
    ),
    (
        "Jake is joining the design review tomorrow",
        "Expect: jake→sarah works_with edge",
    ),
    (
        "my advisor mentioned something interesting",
        "Expect: harpreet→dr_chen via advisor",
    ),
    # ── New cases (batch 2 data) ─────────────────────────────────────────────
    (
        "can you remind me what Riya's background is?",
        "Expect: riya→google (previously_worked_at) edge",
    ),
    (
        "I need to prep for the board meeting",
        "Expect: board_meeting→maya or harpreet→maya edges",
    ),
    (
        "Arjun mentioned a new AI fund at a16z",
        "Expect: arjun→andreessen_horowitz edge",
    ),
    (
        "what is DataBridge about?",
        "Expect: databridge→novabridge or enterprise_data_pipelines edge",
    ),
    (
        "I want to call my mom this weekend",
        "Expect: harpreet→gurpreet edges",
    ),
    (
        "how is Priya's experiment going?",
        "Expect: priya→recommendation_model via experiments edge",
    ),
    (
        "when are we closing the Series A?",
        "Expect: series_a→q2 or maya→series_a edges",
    ),
    (
        "we're demoing at YC next month",
        "Expect: proactiveclaw→yc's_ai_summit edge",
    ),
    # ── Noise — should return nothing ────────────────────────────────────────
    (
        "ok",
        "Expect: nothing (too short)",
    ),
    (
        "sounds good",
        "Expect: nothing (no entities)",
    ),
    (
        "what time is it?",
        "Expect: nothing (no named entities)",
    ),
    # ── Fuzzy matching ───────────────────────────────────────────────────────
    (
        "Harpreet needs to talk to Sara about the roadmap",
        "Expect: sara→sarah fuzzy, sarah→harpreet co-founder edge",
    ),
    (
        "any updates from Ria on the engineering side?",
        "Expect: riya matched via fuzzy (Ria→riya), riya→google edge",
    ),
]


def _run_test_case(message: str, expected: str, session_id: str) -> dict:
    from memory_tier1.extractor import extract_entities
    from memory_tier1.retriever import retrieve
    from memory_tier1.injector import build_tier1_context

    t0 = time.time()
    extracted = extract_entities(message)
    t1 = time.time()
    triples = retrieve(extracted)
    t2 = time.time()
    context = build_tier1_context.__wrapped__(message, session_id) if hasattr(build_tier1_context, '__wrapped__') else ""

    # Build context directly (bypass session dedup for testing)
    from memory_tier1.injector import _format_triple
    context_lines = [_format_triple(t) for t in triples]

    return {
        "message":    message,
        "expected":   expected,
        "extracted":  extracted,
        "triples":    triples,
        "context":    context_lines,
        "t_extract":  round((t1 - t0) * 1000, 1),
        "t_retrieve": round((t2 - t1) * 1000, 1),
    }


def run_scripted() -> None:
    console.print(Panel(
        "[bold cyan]Tier 1 Memory Retrieval — Scripted Test[/bold cyan]\n"
        "[dim]Running predefined test cases against the KG[/dim]",
        border_style="cyan", padding=(1, 4),
    ))

    import memory_tier1.matcher as _m
    _m._load_entity_cache()
    console.print(f"[dim]KG entity cache: {len(_m._entity_names)} entities loaded[/dim]\n")

    for i, (message, expected) in enumerate(TEST_CASES, 1):
        console.print(Rule(f"[dim]Test {i}/{len(TEST_CASES)}[/dim]"))
        result = _run_test_case(message, expected, session_id=f"test_{i}")

        console.print(f"[bold]Message:[/bold] {escape(message)}")
        console.print(f"[dim]Expected: {escape(expected)}[/dim]")

        # Extraction results
        if result["extracted"]:
            if HAS_RICH:
                t = Table(box=box.SIMPLE, show_header=True, border_style="blue")
                t.add_column("Entity", style="cyan")
                t.add_column("Rels extracted", style="yellow")
                for item in result["extracted"]:
                    t.add_row(
                        escape(item["entity"]),
                        escape(", ".join(item["rels"]) or "—"),
                    )
                console.print(t)
            else:
                for item in result["extracted"]:
                    print(f"  entity={item['entity']}  rels={item['rels']}")
        else:
            console.print("  [dim](no entities extracted)[/dim]")

        # Retrieved triples
        if result["triples"]:
            console.print(f"  [green]✓ {len(result['triples'])} triple(s) retrieved:[/green]")
            for line in result["context"]:
                console.print(f"  [white]{escape(line)}[/white]")
        else:
            console.print("  [dim]∅ no triples retrieved[/dim]")

        console.print(
            f"  [dim]timing: extract={result['t_extract']}ms  retrieve={result['t_retrieve']}ms[/dim]"
        )
        console.print()


def run_interactive() -> None:
    console.print(Panel(
        "[bold cyan]Tier 1 Memory Retrieval — Interactive Mode[/bold cyan]\n"
        "Type a message to see what Tier 1 injects.\n"
        "[dim]Commands: 'graph' to show KG entities, 'quit' to exit[/dim]",
        border_style="cyan", padding=(1, 2),
    ))

    import memory_tier1.matcher as _m
    _m._load_entity_cache()
    console.print(f"[dim]KG entity cache: {len(_m._entity_names)} entities loaded[/dim]\n")

    session_id = "interactive_test"
    turn = 0

    while True:
        console.print(Rule(f"[dim]Turn {turn + 1}[/dim]"))
        try:
            if HAS_RICH:
                message = console.input("[bold cyan]Message:[/bold cyan] ").strip()
            else:
                message = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            break
        if message.lower() == "graph":
            import memory_tier1.matcher as _m
            console.print(f"[dim]KG entities ({len(_m._entity_names)}): {', '.join(_m._entity_names[:30])}{'…' if len(_m._entity_names) > 30 else ''}[/dim]")
            continue

        result = _run_test_case(message, "", session_id)

        if result["extracted"]:
            console.print("[dim]Extracted:[/dim]")
            for item in result["extracted"]:
                rels = ", ".join(item["rels"]) or "—"
                console.print(f"  [cyan]{escape(item['entity'])}[/cyan]  rels=[yellow]{escape(rels)}[/yellow]")
        else:
            console.print("[dim](no entities extracted)[/dim]")

        if result["triples"]:
            console.print(f"[green]✓ Injecting {len(result['triples'])} triple(s):[/green]")
            for line in result["context"]:
                console.print(f"  {escape(line)}")
        else:
            console.print("[dim]∅ nothing to inject[/dim]")

        console.print(f"[dim]extract={result['t_extract']}ms  retrieve={result['t_retrieve']}ms[/dim]")
        turn += 1

    console.print(Rule("[dim]Session ended[/dim]"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 1 Memory Retrieval Test")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — type messages manually")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_scripted()
    return 0


if __name__ == "__main__":
    sys.exit(main())
