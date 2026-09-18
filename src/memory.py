"""Long-term agent memory backed by Mem0 with local embedded storage.

Mem0 extracts facts from conversations and stores them in two complementary
stores that are searched in parallel on every query:

  • Qdrant (vector)  — semantic similarity search over memory facts
  • Kuzu  (graph)    — entity nodes + typed relationships, temporal anchoring,
                       soft-delete / fact versioning, 1-hop neighbor traversal

Graph relations are returned alongside vector hits, giving the agent richer
context — not just "what facts match" but "how entities relate to each other."

Requires: pip install mem0ai kuzu
Requires: OPENAI_API_KEY in env (used for LLM extraction + embeddings)
"""

import os

# Instrument mem0 internals before first import of Memory.
# This is a no-op if mem0 is not installed.
try:
    from mem0_logging import enable_mem0_logging
    enable_mem0_logging()
except Exception:
    pass

from core.paths import MEM0_DIR as _MEM0_DIR_P, KUZU_GRAPH_DIR as _KUZU_DIR_P
from core.config import user_slug as _user_slug

_MEMORY_DIR = str(_MEM0_DIR_P)
_GRAPH_DIR  = str(_KUZU_DIR_P)


def _uid() -> str:
    return _user_slug()

_mem: "Memory | None" = None
_init_failed = False


def _get_memory():
    """Lazy-initialize the Mem0 Memory instance (vector + graph). Returns None on failure."""
    global _mem, _init_failed
    if _mem is not None:
        return _mem
    if _init_failed:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        _init_failed = True
        return None

    try:
        from mem0 import Memory

        os.makedirs(_MEMORY_DIR, exist_ok=True)
        # Do NOT pre-create _GRAPH_DIR — Kuzu creates it on first init.
        # If an empty directory exists from a previous failed init, remove it.
        import shutil
        if os.path.isdir(_GRAPH_DIR) and not os.listdir(_GRAPH_DIR):
            shutil.rmtree(_GRAPH_DIR)

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4.1-mini",
                    "temperature": 0.1,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                },
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

        _mem = Memory.from_config(config)
        print(f"[memory] Mem0 initialized — vector: {_MEMORY_DIR}  graph: {_GRAPH_DIR}")
        return _mem
    except Exception as e:
        print(f"[warning] Mem0 initialization failed: {e}")
        _init_failed = True
        return None


def _format_relations(relations: list[dict]) -> str:
    """Format graph relations into a concise readable block."""
    if not relations:
        return ""
    lines = []
    seen = set()
    for r in relations:
        src  = r.get("source", r.get("subject", ""))
        rel  = r.get("relationship", r.get("relation", r.get("predicate", "")))
        dest = r.get("destination", r.get("target", r.get("object", "")))
        if not (src and rel and dest):
            continue
        key = (str(src).lower(), str(rel).lower(), str(dest).lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {src} —[{rel}]→ {dest}")
    return "\n".join(lines)


def store_dialogues(dialogues: list[dict]) -> str:
    """Store conversation dialogues in long-term memory (vector + graph).

    dialogues: list of {"role": "user"|"assistant", "content": "..."}
    """
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        result = mem.add(dialogues, user_id=_uid())
        count = len(result.get("results", []))
        return f"Stored {count} memory entries."
    except Exception as e:
        return f"Error storing memories: {e}"


def query_memory(query: str) -> str:
    """Semantic + graph search over long-term memory.

    Returns vector-matched memory facts followed by any graph relations
    (entity connections) discovered during graph traversal.
    """
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        results   = mem.search(query, user_id=_uid(), limit=10)
        memories  = results.get("results", [])
        relations = results.get("relations", [])

        if not memories and not relations:
            return "No relevant memories found."

        parts = []

        if memories:
            parts.append("*Memory facts:*")
            parts.extend(f"- {m['memory']}" for m in memories if m.get("memory"))

        rel_block = _format_relations(relations)
        if rel_block:
            parts.append("\n*Graph relations:*")
            parts.append(rel_block)

        return "\n".join(parts)
    except Exception as e:
        return f"Error querying memory: {e}"


def retrieve_memory(query: str) -> str:
    """Retrieve specific memories by topic — focused (limit=5) with graph relations."""
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        results   = mem.search(query, user_id=_uid(), limit=5)
        memories  = results.get("results", [])
        relations = results.get("relations", [])

        if not memories and not relations:
            return "No relevant memories found."

        parts = []

        if memories:
            parts.extend(f"- {m['memory']}" for m in memories if m.get("memory"))

        rel_block = _format_relations(relations)
        if rel_block:
            parts.append("\n*Related graph connections:*")
            parts.append(rel_block)

        return "\n".join(parts)
    except Exception as e:
        return f"Error retrieving memory: {e}"
