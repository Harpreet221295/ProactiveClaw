"""
Context block formatter and session deduplication for Tier 1 memory.

build_tier1_context(message, session_id) → context string or ""
clear_session_cache(session_id)          → clears dedup state for session
"""

from __future__ import annotations

import threading

# session_id → set of (source, relationship, target) already injected
_session_cache: dict[str, set[tuple]] = {}
_cache_lock = threading.Lock()


def clear_session_cache(session_id: str) -> None:
    with _cache_lock:
        _session_cache.pop(session_id, None)


def _get_session_seen(session_id: str) -> set[tuple]:
    with _cache_lock:
        if session_id not in _session_cache:
            _session_cache[session_id] = set()
        return _session_cache[session_id]


def _format_triple(t: dict) -> str:
    src = t.get("source", "?")
    rel = t.get("relationship", "?").replace("_", " ")
    tgt = t.get("target", t.get("destination", "?"))
    return f"  • {src} —[{rel}]→ {tgt}"


def build_tier1_context(message: str, session_id: str = "default") -> str:
    """
    Full Tier 1 pipeline: extract → match → retrieve → deduplicate → format.

    Returns a <tier1_memory_context> block string, or "" if nothing found.
    """
    try:
        from .extractor import extract_entities
        from .retriever import retrieve

        extracted = extract_entities(message)

        if not extracted:
            print(f"[tier1 memory] no entities extracted")
            return ""

        entities_str = ", ".join(
            f"{e['entity']}{'[' + ','.join(e['rels']) + ']' if e['rels'] else ''}"
            for e in extracted
        )
        print(f"[tier1 memory] extracted: {entities_str}")

        triples = retrieve(extracted)

        if not triples:
            print(f"[tier1 memory] no triples retrieved")
            return ""

        # Session deduplication — only inject triples not seen this session
        seen = _get_session_seen(session_id)
        new_triples = []
        for t in triples:
            key = (t["source"], t["relationship"], t.get("target", t.get("destination", "")))
            if key not in seen:
                seen.add(key)
                new_triples.append(t)

        if not new_triples:
            print(f"[tier1 memory] all {len(triples)} triple(s) already seen this session — skipping")
            return ""

        for t in new_triples:
            print(f"[tier1 memory] injecting: {_format_triple(t).strip()}")

        lines = [
            "<tier1_memory_context>",
            "Relevant knowledge graph context retrieved automatically:",
            "",
        ]
        for t in new_triples:
            lines.append(_format_triple(t))
        lines.append("</tier1_memory_context>")

        return "\n".join(lines)

    except Exception as e:
        print(f"[tier1 memory] pipeline error: {e}")
        return ""
