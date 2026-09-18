"""
mem0 Logging Instrumentation
─────────────────────────────
Monkey-patches the key mem0 internals to emit rich, colour-coded logs for
every step of the memory pipeline.  Import and call ``enable_mem0_logging()``
*before* ``Memory.from_config()`` to activate.

Logged events
─────────────
  [GRAPH] Entity extraction   — which entities the LLM found
  [GRAPH] Relation extraction — what triples were returned
  [GRAPH] Similarity search   — existing nodes that matched
  [GRAPH] Deletion decision   — which old edges get removed
  [GRAPH] Graph write         — nodes / edges actually written to Kuzu
  [VEC]   Fact extraction     — facts the LLM pulled from the conversation
  [VEC]   Update decisions    — ADD / UPDATE / DELETE / NONE per fact
  [VEC]   Vector write        — final operations committed to Qdrant
"""

from __future__ import annotations

import logging
import functools
import threading
from typing import Any

_PATCHED = False
_print_lock = threading.Lock()   # serialise output across concurrent threads

# ── stdlib logging for mem0's own logger ──────────────────────────────────────
def _configure_mem0_stdlib_logging(level: int = logging.DEBUG) -> None:
    """Make mem0's internal logger.debug() calls visible."""
    log = logging.getLogger("mem0")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[mem0] %(levelname)s  %(message)s"))
        log.addHandler(h)
    log.setLevel(level)


# ── rich helpers ───────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markup import escape
    from rich import box as rbox
    _console = Console(stderr=False)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _print(msg: str) -> None:
    if HAS_RICH:
        _console.print(msg)
    else:
        print(msg)


def _section(title: str, style: str = "cyan") -> None:
    if HAS_RICH:
        from rich.rule import Rule
        _console.print(Rule(f"[{style}]{title}[/{style}]", style=style))
    else:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")


def _entity_table(entity_type_map: dict) -> None:
    if not entity_type_map:
        _print("  [dim]  (no entities extracted)[/dim]")
        return
    if HAS_RICH:
        t = Table(box=rbox.SIMPLE, border_style="cyan", show_lines=False, padding=(0, 1))
        t.add_column("Entity",      style="bold cyan",  max_width=35)
        t.add_column("Type",        style="yellow",     max_width=25)
        for ent, typ in entity_type_map.items():
            t.add_row(escape(ent), escape(typ))
        _console.print(t)
    else:
        for ent, typ in entity_type_map.items():
            print(f"    {ent!r:30s}  {typ}")


def _relation_table(entities: list[dict]) -> None:
    if not entities:
        _print("  [dim]  (no relations extracted)[/dim]")
        return
    if HAS_RICH:
        t = Table(box=rbox.SIMPLE, border_style="green", show_lines=False, padding=(0, 1))
        t.add_column("Source",       style="cyan",   max_width=25)
        t.add_column("Relationship", style="yellow", max_width=22, no_wrap=True)
        t.add_column("Destination",  style="green",  max_width=25)
        for e in entities:
            t.add_row(escape(str(e.get("source", ""))),
                      escape(str(e.get("relationship", ""))),
                      escape(str(e.get("destination", ""))))
        _console.print(t)
    else:
        for e in entities:
            print(f"    {e.get('source')}  --[{e.get('relationship')}]-->  {e.get('destination')}")


def _facts_list(facts: list[str], style: str = "white") -> None:
    if not facts:
        _print("  [dim]  (no facts)[/dim]")
        return
    for f in facts:
        _print(f"  [dim]•[/dim] [{style}]{escape(str(f))}[/{style}]")


# ── Patch: KuzuMemory ──────────────────────────────────────────────────────────

def _patch_kuzu_memory() -> None:
    try:
        from mem0.memory.kuzu_memory import MemoryGraph
    except ImportError:
        return

    # ── _retrieve_nodes_from_data ─────────────────────────────────────────────
    _orig_retrieve = MemoryGraph._retrieve_nodes_from_data

    @functools.wraps(_orig_retrieve)
    def _patched_retrieve(self, data, filters):
        _section("GRAPH  ▶  Entity extraction", "cyan")
        _print(f"  [dim]Input text:[/dim] [white]{escape(str(data)[:200])}[/white]")
        entity_type_map = _orig_retrieve(self, data, filters)
        _print(f"  [bold cyan]Extracted {len(entity_type_map)} entit{'y' if len(entity_type_map)==1 else 'ies'}:[/bold cyan]")
        _entity_table(entity_type_map)
        return entity_type_map

    MemoryGraph._retrieve_nodes_from_data = _patched_retrieve

    # ── _establish_nodes_relations_from_data ─────────────────────────────────
    _orig_establish = MemoryGraph._establish_nodes_relations_from_data

    @functools.wraps(_orig_establish)
    def _patched_establish(self, data, filters, entity_type_map):
        _section("GRAPH  ▶  Relation extraction", "green")
        entities = _orig_establish(self, data, filters, entity_type_map)
        # Normalise "user_id:_X" → "X" — the LLM sometimes emits the composite
        # user_identity string instead of just the user_id value.
        user_id = filters.get("user_id", "")
        prefix = f"user_id:_{user_id}"
        for e in entities:
            for key in ("source", "destination"):
                if e.get(key) == prefix or e.get(key, "").startswith("user_id:_"):
                    e[key] = e[key].split("user_id:_", 1)[-1]
        _print(f"  [bold green]Extracted {len(entities)} relation(s):[/bold green]")
        _relation_table(entities)
        return entities

    MemoryGraph._establish_nodes_relations_from_data = _patched_establish

    # ── _search_graph_db ──────────────────────────────────────────────────────
    _orig_search = MemoryGraph._search_graph_db

    @functools.wraps(_orig_search)
    def _patched_search(self, node_list, filters, limit=100, threshold=None):
        result = _orig_search(self, node_list, filters, limit=limit, threshold=threshold)
        _section("GRAPH  ▶  Similarity search (existing nodes)", "yellow")
        _print(f"  [dim]Query nodes:[/dim] {escape(str(node_list))}")
        if result:
            _print(f"  [bold yellow]{len(result)} matching neighbour(s) found:[/bold yellow]")
            _relation_table(result)
        else:
            _print("  [dim]  (no similar existing nodes found — everything is new)[/dim]")
        return result

    MemoryGraph._search_graph_db = _patched_search

    # ── _get_delete_entities_from_search_output ───────────────────────────────
    _orig_delete_decision = MemoryGraph._get_delete_entities_from_search_output

    @functools.wraps(_orig_delete_decision)
    def _patched_delete_decision(self, search_output, data, filters):
        to_delete = _orig_delete_decision(self, search_output, data, filters)
        _section("GRAPH  ▶  Deletion decision", "red")
        if to_delete:
            _print(f"  [bold red]{len(to_delete)} edge(s) marked for deletion:[/bold red]")
            _relation_table(to_delete)
        else:
            _print("  [dim]  (no existing edges to delete)[/dim]")
        return to_delete

    MemoryGraph._get_delete_entities_from_search_output = _patched_delete_decision

    # ── add (top-level, shows final graph write result) ───────────────────────
    _orig_add = MemoryGraph.add

    @functools.wraps(_orig_add)
    def _patched_add(self, data, filters):
        with _print_lock:
            result = _orig_add(self, data, filters)
        deleted = result.get("deleted_entities", [])
        added   = result.get("added_entities", [])
        with _print_lock:
          _section("GRAPH  ▶  Graph write complete", "bold green")
          _print(f"  [green]✓ Added:[/green]   {len(added)} edge(s) written to Kuzu")
          _print(f"  [red]✗ Deleted:[/red] {len(deleted)} edge(s) removed from Kuzu")
        if added:
            _print("  [dim]New edges:[/dim]")
            # _add_entities returns list-of-lists; each inner list is rows from kuzu_execute
            # Each row dict has keys: source, relationship, target
            normalised = []
            for item in added:
                rows = item if isinstance(item, list) else [item]
                for row in rows:
                    if isinstance(row, dict):
                        normalised.append({
                            "source":       row.get("source", "?"),
                            "relationship": row.get("relationship", "?"),
                            "destination":  row.get("destination", row.get("target", "?")),
                        })
            _relation_table(normalised)
        return result

    MemoryGraph.add = _patched_add


# ── Patch: Memory (vector store path) ─────────────────────────────────────────

def _patch_memory_main() -> None:
    try:
        from mem0.memory.main import Memory
    except ImportError:
        return

    _orig_vector = Memory._add_to_vector_store

    @functools.wraps(_orig_vector)
    def _patched_vector(self, messages, metadata, filters, infer):
        result = _orig_vector(self, messages, metadata, filters, infer)

        with _print_lock:
            _section("VEC  ▶  Vector store ingestion", "magenta")

            if not result:
                _print("  [dim]  (no new vector memories — facts already stored or nothing extractable)[/dim]")
                return result

            by_event: dict[str, list] = {}
            for r in result:
                ev = r.get("event", "?")
                by_event.setdefault(ev, []).append(r)

            summary_parts = []
            for ev, items in sorted(by_event.items()):
                colour = {"ADD": "green", "UPDATE": "yellow", "DELETE": "red"}.get(ev, "white")
                summary_parts.append(f"[{colour}]{ev}: {len(items)}[/{colour}]")
            _print("  " + "  ".join(summary_parts))

            if HAS_RICH:
                t = Table(box=rbox.SIMPLE, border_style="magenta", show_lines=False, padding=(0, 1))
                t.add_column("Event",  style="bold",  width=8,  no_wrap=True)
                t.add_column("Memory", style="white", max_width=70)
                for r in result:
                    ev = r.get("event", "?")
                    colour = {"ADD": "green", "UPDATE": "yellow", "DELETE": "red"}.get(ev, "white")
                    t.add_row(f"[{colour}]{escape(ev)}[/{colour}]",
                              escape(str(r.get("memory", r.get("data", "")))))
                _console.print(t)
            else:
                for r in result:
                    print(f"    [{r.get('event','?')}] {r.get('memory','')}")

        return result

    Memory._add_to_vector_store = _patched_vector

    # ── _add_to_graph wrapper (shows raw input to graph) ─────────────────────
    _orig_graph_call = Memory._add_to_graph

    @functools.wraps(_orig_graph_call)
    def _patched_graph_call(self, messages, filters):
        # Run graph ingestion, then print the full summary under the lock
        result = _orig_graph_call(self, messages, filters)
        return result

    Memory._add_to_graph = _patched_graph_call


# ── Public API ────────────────────────────────────────────────────────────────

def enable_mem0_logging(stdlib_level: int = logging.WARNING) -> None:
    """
    Activate comprehensive mem0 logging.

    Call this once, before ``Memory.from_config()``.

    stdlib_level controls mem0's own ``logging.getLogger("mem0")`` verbosity.
    Defaults to WARNING (suppress mem0's own noisy debug lines); set to
    ``logging.DEBUG`` to also see mem0's internal debug output.
    """
    global _PATCHED
    if _PATCHED:
        return
    _configure_mem0_stdlib_logging(stdlib_level)
    _patch_kuzu_memory()
    _patch_memory_main()
    _PATCHED = True
    if HAS_RICH:
        _console.print("[bold green]✓ mem0 logging enabled[/bold green]  "
                       "[dim](entity extraction, relation extraction, vector ops)[/dim]")
    else:
        print("✓ mem0 logging enabled")
