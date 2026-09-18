#!/usr/bin/env python3
"""
Clear all Mem0 memories.

Wipes the Qdrant vector store collection and the SQLite history database
that back the agent's long-term memory.

Usage:
    python src/reset/clear_memory.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from reset._common import (
    MEM0_DATA, confirm, ok, warn, err, info, header,
    remove_tree, remove_file,
)

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


def clear_via_mem0_api() -> bool:
    """Delete all memories via the Mem0 API (clean, preserves store structure)."""
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

        if not memories:
            ok("Memory store is already empty")
            return True

        info(f"Found {len(memories)} memories — deleting…")
        for m in memories:
            mem.delete(memory_id=m["id"])

        ok(f"Deleted {len(memories)} memories via Mem0 API")
        return True
    except Exception as e:
        warn(f"Mem0 API delete failed ({e}) — falling back to file wipe")
        return False


def clear_via_file_wipe() -> None:
    """Nuclear option: delete the entire mem0_data directory and recreate it empty."""
    remove_tree(MEM0_DATA, "mem0_data directory")
    MEM0_DATA.mkdir(parents=True, exist_ok=True)
    ok("Recreated empty mem0_data directory")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear all Mem0 long-term memories")
    parser.add_argument("--force",    action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--wipe",     action="store_true",
                        help="Delete mem0_data files directly instead of using the API")
    args = parser.parse_args()

    header("Clear Long-Term Memory")

    if not confirm("This will permanently delete ALL agent memories. Continue?", args.force):
        info("Aborted.")
        return 0

    if args.wipe:
        clear_via_file_wipe()
    else:
        # Try clean API delete first; fall back to file wipe if env not set
        if not clear_via_mem0_api():
            clear_via_file_wipe()

    try:
        import os as _os
        _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 2)
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
