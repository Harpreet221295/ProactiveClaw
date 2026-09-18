#!/usr/bin/env python3
"""
ProactiveClaw Memory Monitor
─────────────────────────────
Interactive terminal viewer for the Mem0 knowledge base.

Lets you browse, search, inspect history, visualise entities/relationships,
and export everything — all without touching the live agent.

Usage:
    python src/memory_monitor/monitor.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]          # project root
_SRC  = Path(__file__).parent.parent       # src/

sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# ── rich ──────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.tree import Tree
    from rich.markup import escape
    from rich.live import Live
    from rich.layout import Layout
except ImportError:
    print("rich is not installed. Run: pip install rich")
    sys.exit(1)

console = Console()

# ── Mem0 config (mirrors src/memory.py exactly) ───────────────────────────────
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
    """Return an initialised Memory instance, or None on failure."""
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]✗ OPENAI_API_KEY not set — cannot initialise Mem0.[/red]")
        return None
    try:
        from mem0 import Memory  # noqa: PLC0415
        # suppress Mem0's startup print
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = Memory.from_config(_MEM0_CONFIG)
        return m
    except Exception as e:
        console.print(f"[red]✗ Mem0 init failed: {e}[/red]")
        return None


def _fetch_all(mem) -> list[dict]:
    result = mem.get_all(user_id=_USER_ID)
    return result.get("results", []) if isinstance(result, dict) else (result or [])


# ── Entity extraction (lightweight, no external NLP) ─────────────────────────

_SKIP_WORDS = {
    "The", "A", "An", "He", "She", "It", "They", "His", "Her", "Their",
    "This", "That", "These", "Those", "Is", "Are", "Was", "Were",
    "And", "But", "Or", "If", "So", "As", "In", "At", "By", "On",
    "For", "To", "Of", "With", "From", "About", "Also", "Has", "Have",
}

_RELATIONSHIP_PATTERNS = [
    (r"(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+)",     "IS_A"),
    (r"(.+?)\s+(?:loves?|likes?|enjoys?|prefers?)\s+(.+)",       "LIKES"),
    (r"(.+?)\s+(?:hates?|dislikes?|avoids?)\s+(.+)",             "DISLIKES"),
    (r"(.+?)\s+(?:works?(?:\s+at)?|works?(?:\s+for)?)\s+(.+)",   "WORKS_AT"),
    (r"(.+?)\s+(?:uses?|uses?)\s+(.+)",                          "USES"),
    (r"(.+?)\s+(?:builds?|building|created?|creates?)\s+(.+)",   "BUILDS"),
    (r"(.+?)\s+(?:lives?(?:\s+in)?|located?\s+in)\s+(.+)",       "LOCATED_IN"),
    (r"(.+?)\s+(?:has|have|owns?)\s+(.+)",                       "HAS"),
    (r"(.+?)\s+(?:called|named)\s+(.+)",                         "NAMED"),
    (r"(.+?)\s+(?:focuses? on|focused on|specializes? in)\s+(.+)","FOCUSES_ON"),
]


def extract_entities(memories: list[dict]) -> tuple[set[str], list[dict]]:
    """Return (entities, relations) extracted heuristically from memory texts."""
    entities: set[str] = set()
    relations: list[dict] = []

    for mem in memories:
        text = mem.get("memory", "")

        # collect capitalized proper nouns
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for c in caps:
            if c not in _SKIP_WORDS and len(c) > 2:
                entities.add(c)

        # match relationship patterns on each sentence
        for sentence in re.split(r'[.!?]', text):
            sentence = sentence.strip()
            for pattern, rel_type in _RELATIONSHIP_PATTERNS:
                m = re.match(pattern, sentence, re.IGNORECASE)
                if m:
                    subj = m.group(1).strip().rstrip(",")
                    obj  = m.group(2).strip().rstrip(".,")
                    # keep only reasonably short phrases
                    if len(subj) < 60 and len(obj) < 60:
                        relations.append({
                            "source": subj,
                            "relationship": rel_type,
                            "destination": obj,
                            "memory_id": mem.get("id", ""),
                        })

    return entities, relations


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16] if ts else "—"


def _age(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        d = delta.days
        h = delta.seconds // 3600
        m = (delta.seconds % 3600) // 60
        if d > 0:
            return f"{d}d ago"
        if h > 0:
            return f"{h}h ago"
        return f"{m}m ago"
    except Exception:
        return "—"


# ── Views ─────────────────────────────────────────────────────────────────────

def view_dashboard(mem) -> None:
    memories = _fetch_all(mem)

    if not memories:
        console.print(Panel(
            "[yellow]No memories stored yet.[/yellow]\n\n"
            "Use [cyan]Add memory[/cyan] to store your first memory,\n"
            "or start the agent and have a conversation.",
            title="Memory Dashboard", border_style="blue"
        ))
        return

    # Stats
    total = len(memories)
    now_utc = datetime.now(timezone.utc)
    today = sum(
        1 for m in memories
        if m.get("created_at") and
           (now_utc - datetime.fromisoformat(m["created_at"]).astimezone(timezone.utc)).days < 1
    )
    with_meta = sum(1 for m in memories if m.get("metadata"))
    _, relations = extract_entities(memories)

    # stat cards
    stats = [
        Panel(f"[bold cyan]{total}[/bold cyan]\ntotal", border_style="cyan",   padding=(0,2)),
        Panel(f"[bold green]{today}[/bold green]\ntoday",  border_style="green", padding=(0,2)),
        Panel(f"[bold yellow]{len(relations)}[/bold yellow]\nrelations", border_style="yellow", padding=(0,2)),
        Panel(f"[bold magenta]{with_meta}[/bold magenta]\nwith metadata", border_style="magenta", padding=(0,2)),
    ]
    console.print(Columns(stats, equal=True))
    console.print()

    # Recent memories table
    table = Table(
        title="Recent Memories (latest 10)",
        box=box.ROUNDED,
        border_style="blue",
        show_lines=True,
        expand=True,
    )
    table.add_column("#",          style="dim",    width=3,  no_wrap=True)
    table.add_column("Memory",     style="white",  ratio=3)
    table.add_column("Created",    style="cyan",   width=16, no_wrap=True)
    table.add_column("Age",        style="yellow", width=8,  no_wrap=True)
    table.add_column("ID (short)", style="dim",    width=10, no_wrap=True)

    sorted_mems = sorted(memories, key=lambda m: m.get("created_at") or "", reverse=True)
    for i, m in enumerate(sorted_mems[:10], 1):
        text = m.get("memory", "")
        short_id = m.get("id", "")[:8]
        table.add_row(
            str(i),
            escape(text[:120]) + ("…" if len(text) > 120 else ""),
            _fmt_ts(m.get("created_at")),
            _age(m.get("created_at")),
            short_id,
        )

    console.print(table)


def view_all_memories(mem) -> None:
    memories = _fetch_all(mem)
    if not memories:
        console.print("[yellow]No memories stored yet.[/yellow]")
        return

    sorted_mems = sorted(memories, key=lambda m: m.get("created_at") or "", reverse=True)
    table = Table(
        title=f"All Memories ({len(memories)} total)",
        box=box.SIMPLE_HEAD,
        border_style="blue",
        show_lines=True,
        expand=True,
    )
    table.add_column("#",          style="dim",    width=4,  no_wrap=True)
    table.add_column("Memory",     style="white",  ratio=3)
    table.add_column("Created",    style="cyan",   width=16, no_wrap=True)
    table.add_column("Updated",    style="dim",    width=16, no_wrap=True)
    table.add_column("Metadata",   style="green",  width=18, no_wrap=False)
    table.add_column("ID (short)", style="dim",    width=10, no_wrap=True)

    for i, m in enumerate(sorted_mems, 1):
        meta = m.get("metadata") or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "—"
        table.add_row(
            str(i),
            escape(m.get("memory", "")),
            _fmt_ts(m.get("created_at")),
            _fmt_ts(m.get("updated_at")),
            escape(meta_str[:40]),
            (m.get("id") or "")[:8],
        )

    console.print(table)


def view_search(mem) -> None:
    query = Prompt.ask("[cyan]Search query[/cyan]")
    if not query.strip():
        return

    with Progress(SpinnerColumn(), TextColumn("[cyan]Searching…"), transient=True) as p:
        p.add_task("search")
        results_raw = mem.search(query, user_id=_USER_ID, limit=10)

    results = results_raw.get("results", []) if isinstance(results_raw, dict) else (results_raw or [])
    relations = results_raw.get("relations", []) if isinstance(results_raw, dict) else []

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(
        title=f"Search results for: \"{escape(query)}\"  ({len(results)} hits)",
        box=box.ROUNDED,
        border_style="green",
        show_lines=True,
        expand=True,
    )
    table.add_column("Score", style="yellow", width=7, no_wrap=True)
    table.add_column("Memory", style="white", ratio=3)
    table.add_column("Created", style="cyan", width=16, no_wrap=True)
    table.add_column("ID (short)", style="dim", width=10, no_wrap=True)

    for r in results:
        score = r.get("score")
        score_str = f"{score:.3f}" if score is not None else "—"
        table.add_row(
            score_str,
            escape(r.get("memory", "")),
            _fmt_ts(r.get("created_at")),
            (r.get("id") or "")[:8],
        )
    console.print(table)

    if relations:
        rel_table = Table(title="Related graph edges", box=box.SIMPLE_HEAD, border_style="yellow")
        rel_table.add_column("Source",       style="cyan")
        rel_table.add_column("Relationship", style="yellow")
        rel_table.add_column("Destination",  style="green")
        for rel in relations:
            rel_table.add_row(
                escape(str(rel.get("source", ""))),
                escape(str(rel.get("relationship", ""))),
                escape(str(rel.get("destination", ""))),
            )
        console.print(rel_table)


def view_memory_detail(mem) -> None:
    memory_id = Prompt.ask("[cyan]Memory ID (full or short prefix)[/cyan]").strip()
    if not memory_id:
        return

    memories = _fetch_all(mem)
    # support short prefix match
    matches = [m for m in memories if m.get("id", "").startswith(memory_id)]
    if not matches:
        console.print(f"[red]No memory found with ID starting with '{memory_id}'[/red]")
        return
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous prefix — {len(matches)} matches. Use more characters.[/yellow]")
        return

    m = matches[0]
    full_id = m["id"]

    # Memory card
    meta = m.get("metadata") or {}
    meta_str = json.dumps(meta, indent=2) if meta else "none"
    console.print(Panel(
        f"[bold white]{escape(m.get('memory', ''))}[/bold white]\n\n"
        f"[dim]ID:[/dim]       {full_id}\n"
        f"[dim]Created:[/dim]  {_fmt_ts(m.get('created_at'))} ({_age(m.get('created_at'))})\n"
        f"[dim]Updated:[/dim]  {_fmt_ts(m.get('updated_at'))}\n"
        f"[dim]User:[/dim]     {m.get('user_id', '—')}\n"
        f"[dim]Hash:[/dim]     {m.get('hash', '—')}\n"
        f"[dim]Metadata:[/dim] {escape(meta_str)}",
        title="Memory Detail",
        border_style="cyan",
    ))

    # History
    with Progress(SpinnerColumn(), TextColumn("[cyan]Loading history…"), transient=True) as p:
        p.add_task("h")
        history = mem.history(memory_id=full_id)

    if not history:
        console.print("[dim]No history records found.[/dim]")
        return

    h_table = Table(
        title=f"Edit History ({len(history)} event(s))",
        box=box.SIMPLE_HEAD,
        border_style="yellow",
        show_lines=True,
        expand=True,
    )
    h_table.add_column("Event",      style="bold", width=8)
    h_table.add_column("When",       style="cyan", width=16)
    h_table.add_column("Old memory", style="dim",  ratio=1)
    h_table.add_column("New memory", style="white", ratio=1)

    event_colors = {"ADD": "green", "UPDATE": "yellow", "DELETE": "red"}
    for h in history:
        event = h.get("event", "?")
        color = event_colors.get(event, "white")
        h_table.add_row(
            f"[{color}]{event}[/{color}]",
            _fmt_ts(h.get("created_at")),
            escape(str(h.get("old_memory") or "—")[:80]),
            escape(str(h.get("new_memory") or "—")[:80]),
        )
    console.print(h_table)


def _kuzu_graph_data() -> tuple[list[dict], list[dict]]:
    """Query Kuzu directly using mem0's schema:
      Node table : Entity  (id, user_id, name, mentions, created, embedding)
      Edge table : CONNECTED_TO  (name, mentions, created, updated)

    Returns (entities, relations) — both as lists of dicts.
    Falls back to empty lists if Kuzu is not available or graph is empty.
    """
    try:
        import kuzu
        # read_only=True allows concurrent access alongside the Memory instance's writer
        db   = kuzu.Database(_GRAPH_DIR, read_only=True)
        conn = kuzu.Connection(db)

        # ── Entity nodes ──────────────────────────────────────────────────────
        entities = []
        try:
            rows = conn.execute(
                "MATCH (n:Entity) RETURN n.name, n.mentions, n.created, n.user_id"
            )
            for row in rows:
                entities.append({
                    "name":       str(row[0] or ""),
                    "mentions":   int(row[1] or 0),
                    "created_at": str(row[2] or ""),
                    "user_id":    str(row[3] or ""),
                })
        except Exception:
            pass

        # ── Relationships ─────────────────────────────────────────────────────
        relations = []
        try:
            rows = conn.execute(
                "MATCH (a:Entity)-[r:CONNECTED_TO]->(b:Entity) "
                "RETURN a.name, r.name, b.name, r.created, r.updated, r.mentions"
            )
            for row in rows:
                relations.append({
                    "source":       str(row[0] or ""),
                    "relationship": str(row[1] or "").upper(),
                    "destination":  str(row[2] or ""),
                    "created_at":   str(row[3] or ""),
                    "updated_at":   str(row[4] or ""),
                    "mentions":     int(row[5] or 0),
                    "valid":        True,   # mem0-kuzu soft-deletes by removal, not flag
                })
        except Exception:
            pass

        return entities, relations

    except Exception:
        return [], []


def view_entities(mem) -> None:
    """Show the real Kuzu knowledge graph — entity nodes + typed relationships."""
    console.print(Rule("[bold cyan]Knowledge Graph  (Kuzu)[/bold cyan]"))

    with Progress(SpinnerColumn(), TextColumn("[cyan]Reading Kuzu graph…"), transient=True) as p:
        p.add_task("g")
        entities, relations = _kuzu_graph_data()

    # ── Fallback: if Kuzu has no data yet, try search-based relations ─────────
    if not entities and not relations:
        console.print("[yellow]Kuzu graph is empty — no entities extracted yet.[/yellow]")
        console.print("[dim]Memories need to be added with graph store enabled to populate the graph.[/dim]")
        console.print()
        # Show regex fallback so the view isn't totally empty
        memories = _fetch_all(mem)
        if memories:
            console.print("[dim]Falling back to heuristic extraction from memory text:[/dim]")
            _, heuristic_rels = extract_entities(memories)
            if heuristic_rels:
                _render_relations_tree(heuristic_rels, title="Heuristic relations (not from graph)")
        return

    # ── Entity nodes ─────────────────────────────────────────────────────────
    # Count how many relations each entity participates in
    rel_count: Counter = Counter()
    for r in relations:
        rel_count[r["source"].lower()]      += 1
        rel_count[r["destination"].lower()] += 1

    e_table = Table(
        title=f"Entity Nodes ({len(entities)} total)",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=False,
    )
    e_table.add_column("Entity",       style="bold cyan",   max_width=30)
    e_table.add_column("Type",         style="dim",         width=12, no_wrap=True)
    e_table.add_column("Relations",    style="yellow",      width=9,  justify="right")
    e_table.add_column("Created",      style="dim",         width=16, no_wrap=True)

    for e in sorted(entities, key=lambda x: rel_count.get(x["name"].lower(), 0), reverse=True):
        e_table.add_row(
            escape(e["name"]),
            escape(e.get("type", "") or ""),
            str(rel_count.get(e["name"].lower(), 0)),
            _fmt_ts(e.get("created_at")),
        )
    console.print(e_table)
    console.print()

    if not relations:
        console.print("[yellow]No relationships in graph yet.[/yellow]")
        return

    # ── Relationship tree by type ─────────────────────────────────────────────
    valid_rels   = [r for r in relations if r.get("valid", True)]
    invalid_rels = [r for r in relations if not r.get("valid", True)]

    _render_relations_tree(valid_rels,
                           title=f"Active Relationships ({len(valid_rels)})")

    # ── Flat table ────────────────────────────────────────────────────────────
    r_table = Table(
        title=f"All Relationships ({len(valid_rels)} active"
              + (f", {len(invalid_rels)} soft-deleted)" if invalid_rels else ")"),
        box=box.SIMPLE_HEAD,
        border_style="yellow",
        show_lines=False,
    )
    r_table.add_column("Source",       style="cyan",   max_width=25)
    r_table.add_column("Relationship", style="yellow", max_width=20, no_wrap=True)
    r_table.add_column("Destination",  style="green",  max_width=30)
    r_table.add_column("Status",       style="dim",    width=8,  no_wrap=True)
    r_table.add_column("Created",      style="dim",    width=16, no_wrap=True)

    for r in relations:
        status = "" if r.get("valid", True) else "[red]deleted[/red]"
        r_table.add_row(
            escape(r["source"]),
            escape(r["relationship"]),
            escape(r["destination"]),
            status,
            _fmt_ts(r.get("created_at")),
        )
    console.print(r_table)

    if invalid_rels:
        console.print(f"[dim]  {len(invalid_rels)} soft-deleted relation(s) shown above — "
                      "these are outdated facts preserved for temporal reasoning.[/dim]")


def _render_relations_tree(relations: list[dict], title: str = "Relationships") -> None:
    """Render relations grouped by type as a Rich Tree."""
    if not relations:
        return
    by_type: dict[str, list] = defaultdict(list)
    for r in relations:
        by_type[r["relationship"]].append(r)

    tree = Tree(f"[bold yellow]{title}[/bold yellow]")
    for rel_type in sorted(by_type):
        rels = by_type[rel_type]
        branch = tree.add(f"[cyan]{rel_type}[/cyan]  [dim]({len(rels)})[/dim]")
        for r in rels[:10]:
            branch.add(
                f"[white]{escape(r['source'][:28])}[/white]"
                f"  [dim]→[/dim]  "
                f"[green]{escape(r['destination'][:35])}[/green]"
            )
        if len(rels) > 10:
            branch.add(f"[dim]… and {len(rels) - 10} more[/dim]")
    console.print(tree)
    console.print()


def view_stats(mem) -> None:
    memories = _fetch_all(mem)
    if not memories:
        console.print("[yellow]No memories yet.[/yellow]")
        return

    _, relations = extract_entities(memories)

    # ── Word frequency ────────────────────────────────────────────────────────
    stop_words = {
        "the","a","an","is","are","was","were","and","but","or","in","at","by",
        "on","for","to","of","with","from","he","she","it","they","his","her",
        "their","this","that","has","have","had","be","been","being","will",
        "can","do","does","did","i","my","we","our","its","not","no","also",
        "which","who","how","what","when","where","about","more","any","all",
    }
    word_freq: Counter = Counter()
    for m in memories:
        words = re.findall(r'\b[a-z]{3,}\b', m.get("memory","").lower())
        word_freq.update(w for w in words if w not in stop_words)

    # ── Timeline (by day) ────────────────────────────────────────────────────
    by_day: Counter = Counter()
    for m in memories:
        ts = m.get("created_at")
        if ts:
            try:
                day = datetime.fromisoformat(ts).strftime("%Y-%m-%d")
                by_day[day] += 1
            except Exception:
                pass

    console.print(Rule("[bold]Memory Statistics[/bold]"))

    # Summary cards
    cards = [
        Panel(f"[bold cyan]{len(memories)}[/bold cyan]\ntotal memories",    border_style="cyan",    padding=(0,3)),
        Panel(f"[bold yellow]{len(relations)}[/bold yellow]\nrelationships",border_style="yellow",  padding=(0,3)),
        Panel(f"[bold green]{len(by_day)}[/bold green]\ndays active",       border_style="green",   padding=(0,3)),
        Panel(f"[bold magenta]{len(word_freq)}[/bold magenta]\nunique words",border_style="magenta", padding=(0,3)),
    ]
    console.print(Columns(cards, equal=True))
    console.print()

    # Top keywords
    kw_table = Table(title="Top 20 Keywords", box=box.SIMPLE_HEAD, border_style="magenta")
    kw_table.add_column("Keyword", style="white")
    kw_table.add_column("Count",   style="yellow", justify="right")
    for word, cnt in word_freq.most_common(20):
        kw_table.add_row(word, str(cnt))
    console.print(kw_table)
    console.print()

    # Timeline
    if by_day:
        tl_table = Table(title="Memories by Day", box=box.SIMPLE_HEAD, border_style="cyan")
        tl_table.add_column("Date",  style="cyan")
        tl_table.add_column("Count", style="yellow", justify="right")
        tl_table.add_column("Bar",   style="green")
        max_count = max(by_day.values())
        for day in sorted(by_day.keys(), reverse=True)[:14]:
            cnt = by_day[day]
            bar = "█" * int(cnt / max_count * 30)
            tl_table.add_row(day, str(cnt), bar)
        console.print(tl_table)

    # Relationship type breakdown
    if relations:
        console.print()
        rel_type_counts: Counter = Counter(r["relationship"] for r in relations)
        rt_table = Table(title="Relationship Types", box=box.SIMPLE_HEAD, border_style="yellow")
        rt_table.add_column("Type",  style="yellow")
        rt_table.add_column("Count", style="cyan", justify="right")
        for rtype, cnt in rel_type_counts.most_common():
            rt_table.add_row(rtype, str(cnt))
        console.print(rt_table)


def view_export(mem) -> None:
    memories = _fetch_all(mem)
    if not memories:
        console.print("[yellow]No memories to export.[/yellow]")
        return

    entities, relations = extract_entities(memories)

    export_path = _ROOT / "agent_file_system" / "memory_export.json"
    data = {
        "exported_at": datetime.now().isoformat(),
        "total_memories": len(memories),
        "memories": memories,
        "entities": sorted(entities),
        "relations": relations,
    }
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(data, indent=2, default=str))
    console.print(f"[green]✓ Exported {len(memories)} memories → {export_path}[/green]")


def view_add(mem) -> None:
    console.print("[dim]Enter the memory text to store (simulates what the agent would extract).[/dim]")
    text = Prompt.ask("[cyan]Memory text[/cyan]").strip()
    if not text:
        return
    with Progress(SpinnerColumn(), TextColumn("[cyan]Storing…"), transient=True) as p:
        p.add_task("add")
        result = mem.add(text, user_id=_USER_ID)
    added = result.get("results", []) if isinstance(result, dict) else []
    if added:
        console.print(f"[green]✓ Stored {len(added)} memory item(s).[/green]")
        for item in added:
            console.print(f"  [dim]{item.get('id','')[:8]}[/dim] {escape(item.get('memory',''))}")
    else:
        console.print("[yellow]Stored (no result detail returned).[/yellow]")


def view_delete(mem) -> None:
    memory_id = Prompt.ask("[cyan]Memory ID to delete (full or short prefix)[/cyan]").strip()
    if not memory_id:
        return
    memories = _fetch_all(mem)
    matches = [m for m in memories if m.get("id","").startswith(memory_id)]
    if not matches:
        console.print(f"[red]No memory found with ID '{memory_id}'[/red]")
        return
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous — {len(matches)} matches. Use more characters.[/yellow]")
        return
    m = matches[0]
    console.print(f"[bold]Will delete:[/bold] {escape(m.get('memory',''))}")
    if Confirm.ask("[red]Confirm delete?[/red]", default=False):
        mem.delete(memory_id=m["id"])
        console.print("[green]✓ Deleted.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


# ── Watch mode (live refresh) ─────────────────────────────────────────────────

def _build_watch_table(memories: list[dict], prev_ids: set[str]) -> Table:
    """Build the live-updating table, highlighting newly added rows."""
    sorted_mems = sorted(memories, key=lambda m: m.get("created_at") or "", reverse=True)

    table = Table(
        title=f"Live Memory Store — {len(memories)} total  [dim](refreshes every 3s, Ctrl+C to exit)[/dim]",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("#",          style="dim",    width=4,  no_wrap=True)
    table.add_column("Memory",     style="white",  ratio=4)
    table.add_column("Created",    style="cyan",   width=16, no_wrap=True)
    table.add_column("Age",        style="yellow", width=8,  no_wrap=True)
    table.add_column("ID (short)", style="dim",    width=10, no_wrap=True)

    for i, m in enumerate(sorted_mems, 1):
        mid = m.get("id", "")
        text = m.get("memory", "")
        is_new = mid not in prev_ids
        row_style = "bold green" if is_new else ""
        new_badge = " [bold green]NEW[/bold green]" if is_new else ""
        table.add_row(
            str(i),
            escape(text[:140]) + ("…" if len(text) > 140 else "") + new_badge,
            _fmt_ts(m.get("created_at")),
            _age(m.get("created_at")),
            mid[:8],
            style=row_style,
        )

    return table


def run_graph_watch(interval: float = 3.0) -> None:
    """Live-refresh view of the Kuzu graph — shows new edges as they appear."""
    console.print(Panel(
        "[bold cyan]Graph Watch Mode[/bold cyan]  [dim]Auto-refreshes every 3s[/dim]\n"
        "[dim]Run the test writer on another terminal:[/dim]  [cyan]./memory_test.sh[/cyan]\n"
        "[dim]Press Ctrl+C to exit[/dim]",
        border_style="yellow",
        padding=(0, 2),
    ))

    prev_edge_count = 0

    def _build_graph_renderable(entities, relations, prev_count):
        from rich.console import Group as RGroup
        diff = len(relations) - prev_count
        diff_str = (
            f"  [bold green]+{diff} new edge(s)[/bold green]" if diff > 0
            else f"  [bold red]{diff} removed[/bold red]" if diff < 0
            else "  [dim]no change[/dim]"
        )
        header = Panel(
            f"[bold]Nodes:[/bold] {len(entities)}   "
            f"[bold]Edges:[/bold] {len(relations)}{diff_str}   "
            f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
            border_style="dim", padding=(0, 1),
        )

        r_table = Table(
            box=box.SIMPLE_HEAD, border_style="yellow",
            show_lines=False, expand=True,
            title=f"All Relationships ({len(relations)})",
        )
        r_table.add_column("Source",       style="cyan",   max_width=25)
        r_table.add_column("Relationship", style="yellow", max_width=20, no_wrap=True)
        r_table.add_column("Destination",  style="green",  max_width=30)
        r_table.add_column("Created",      style="dim",    width=16, no_wrap=True)

        sorted_rels = sorted(relations, key=lambda r: r.get("created_at") or "", reverse=True)
        for r in sorted_rels:
            r_table.add_row(
                escape(r["source"]),
                escape(r["relationship"]),
                escape(r["destination"]),
                _fmt_ts(r.get("created_at")),
            )
        return RGroup(header, r_table)

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        try:
            while True:
                entities, relations = _kuzu_graph_data()
                live.update(_build_graph_renderable(entities, relations, prev_edge_count))
                time.sleep(interval)
                prev_edge_count = len(relations)
        except KeyboardInterrupt:
            pass

    console.print("\n[dim]Graph watch exited.[/dim]")


def run_watch_mode(mem, interval: float = 3.0) -> None:
    """Live-refresh view — shows all memories, highlights new ones as they appear."""
    console.print(Panel(
        "[bold cyan]Watch Mode[/bold cyan]  [dim]Auto-refreshes every 3s[/dim]\n"
        "[dim]Run the test writer on another terminal:[/dim]  [cyan]./memory_test.sh[/cyan]\n"
        "[dim]Press Ctrl+C to exit[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

    known_ids: set[str] = set()
    memories = _fetch_all(mem)
    known_ids = {m.get("id", "") for m in memories}

    prev_count = len(memories)

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        try:
            while True:
                memories = _fetch_all(mem)
                new_count = len(memories)

                table = _build_watch_table(memories, known_ids)

                # stats header
                diff = new_count - prev_count
                diff_str = (
                    f"  [bold green]+{diff} new[/bold green]" if diff > 0
                    else f"  [bold red]{diff} removed[/bold red]" if diff < 0
                    else "  [dim]no change[/dim]"
                )
                header = Panel(
                    f"[bold]Total:[/bold] {new_count}{diff_str}   "
                    f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                )

                from rich.console import Group
                live.update(Group(header, table))

                # after first render, update known_ids so next cycle shows new ones correctly
                # wait interval, then snapshot again
                time.sleep(interval)
                prev_count = new_count
                known_ids = {m.get("id", "") for m in memories}

        except KeyboardInterrupt:
            pass

    console.print("\n[dim]Watch mode exited.[/dim]")


# ── Main menu ─────────────────────────────────────────────────────────────────

_MENU = [
    ("1", "Dashboard",                    view_dashboard),
    ("2", "Browse all memories",          view_all_memories),
    ("3", "Search",                       view_search),
    ("4", "Memory detail + history",      view_memory_detail),
    ("5", "Knowledge graph  (Kuzu)",      view_entities),
    ("6", "Statistics",                   view_stats),
    ("7", "Export to JSON",               view_export),
    ("8", "Add memory",                   view_add),
    ("9", "Delete memory",                view_delete),
    ("w", "Watch memories  (live)",       "_watch_memories"),
    ("g", "Watch graph     (live)",       "_watch_graph"),
    ("q", "Quit",                         None),
]


def print_menu() -> None:
    console.print()
    console.print(Rule("[bold cyan]ProactiveClaw — Memory Monitor[/bold cyan]"))
    for key, label, _ in _MENU:
        style = "red" if key == "q" else "cyan"
        console.print(f"  [{style}]{key}[/{style}]  {label}")
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ProactiveClaw Memory Monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Live auto-refreshing memory view")
    parser.add_argument("--watch-graph", action="store_true",
                        help="Live auto-refreshing Kuzu graph view")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Refresh interval in seconds for watch modes (default: 3)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]ProactiveClaw Memory Monitor[/bold cyan]\n"
        "[dim]Inspect, search, and visualise the Mem0 knowledge base[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # --watch-graph reads only Kuzu — skip Qdrant init to avoid lock conflicts
    if args.watch_graph:
        run_graph_watch(interval=args.interval)
        try:
            import os as _os
            _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
        except OSError:
            pass
        sys.exit(0)

    with Progress(SpinnerColumn(), TextColumn("[cyan]Connecting to Mem0…"), transient=True) as p:
        p.add_task("init")
        mem = _init_memory()

    if mem is None:
        console.print("[red]Cannot start memory monitor — Mem0 failed to initialise.[/red]")
        sys.exit(1)

    console.print("[green]✓ Connected to Mem0[/green]  "
                  f"[dim]storage: {_MEMORY_DIR}[/dim]")

    if args.watch:
        run_watch_mode(mem, interval=args.interval)
        try:
            import os as _os
            _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
        except OSError:
            pass
        sys.exit(0)

    while True:
        print_menu()
        choice = Prompt.ask("[bold]Choose[/bold]", default="1").strip().lower()

        match_found = False
        for key, label, fn in _MENU:
            if choice == key:
                match_found = True
                if fn is None:  # quit
                    console.print("[dim]Goodbye.[/dim]")
                    try:
                        import os as _os
                        _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
                    except OSError:
                        pass
                    sys.exit(0)
                console.print()
                console.print(Rule(f"[bold]{label}[/bold]"))
                try:
                    if fn == "_watch_memories":
                        run_watch_mode(mem)
                    elif fn == "_watch_graph":
                        run_graph_watch()
                    else:
                        fn(mem)
                except KeyboardInterrupt:
                    console.print("\n[dim]Interrupted.[/dim]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                break

        if not match_found:
            console.print("[yellow]Unknown choice.[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
        try:
            import os as _os
            _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
        except OSError:
            pass
