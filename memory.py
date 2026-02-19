"""Long-term agent memory backed by Mem0 with local embedded storage.

Mem0 extracts facts from conversations, stores them in an embedded Qdrant
vector DB on disk, and retrieves them via semantic search.  All data stays
local — no cloud services, no MCP server, just a Python library.

Requires: pip install mem0ai
Requires: OPENAI_API_KEY in env (used for LLM extraction + embeddings)
"""

import os

_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "mem0_data")
_USER_ID = "default"

_mem: "Memory | None" = None
_init_failed = False


def _get_memory():
    """Lazy-initialize the Mem0 Memory instance. Returns None on failure."""
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
        }

        _mem = Memory.from_config(config)
        print(f"[memory] Mem0 initialized — storage at {_MEMORY_DIR}")
        return _mem
    except Exception as e:
        print(f"[warning] Mem0 initialization failed: {e}")
        _init_failed = True
        return None


def store_dialogues(dialogues: list[dict]) -> str:
    """Store conversation dialogues in long-term memory.

    dialogues: list of {"role": "user"|"assistant", "content": "..."}
    """
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        result = mem.add(dialogues, user_id=_USER_ID)
        count = len(result.get("results", []))
        return f"Stored {count} memory entries."
    except Exception as e:
        return f"Error storing memories: {e}"


def query_memory(query: str) -> str:
    """Semantic search over long-term memory."""
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        results = mem.search(query, user_id=_USER_ID, limit=10)
        memories = results.get("results", [])
        if not memories:
            return "No relevant memories found."
        lines = [m.get("memory", "") for m in memories if m.get("memory")]
        return "\n".join(f"- {line}" for line in lines)
    except Exception as e:
        return f"Error querying memory: {e}"


def retrieve_memory(query: str) -> str:
    """Retrieve memories by topic — same as query but with fewer results."""
    mem = _get_memory()
    if mem is None:
        return "Long-term memory is not available."
    try:
        results = mem.search(query, user_id=_USER_ID, limit=5)
        memories = results.get("results", [])
        if not memories:
            return "No relevant memories found."
        lines = [m.get("memory", "") for m in memories if m.get("memory")]
        return "\n".join(f"- {line}" for line in lines)
    except Exception as e:
        return f"Error retrieving memory: {e}"
