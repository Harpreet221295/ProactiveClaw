"""
Fuzzy matching utilities for Tier 1 memory retrieval.

  match_entity(text)         → best matching KG entity name or None
  match_rels(rels, triples)  → triples whose relationship fuzzy-matches any rel
  normalize(text)            → lowercase, replace underscores/hyphens with spaces
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import sys as _sys
_SRC = Path(__file__).parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from core.paths import KUZU_GRAPH_DIR as _KUZU_DIR_P
_GRAPH_DIR = str(_KUZU_DIR_P)

# ── thresholds ─────────────────────────────────────────────────────────────────
ENTITY_THRESHOLD = 0.82    # Jaro-Winkler minimum for entity name match
REL_THRESHOLD    = 0.78    # slightly looser for relationship names
CACHE_TTL        = 300     # seconds between entity cache refreshes

# ── entity name cache ──────────────────────────────────────────────────────────
_cache_lock      = threading.Lock()
_entity_names:   list[str] = []
_cache_loaded_at: float    = 0.0


def normalize(text: str) -> str:
    """Lowercase and replace underscores/hyphens with spaces."""
    return text.lower().replace("_", " ").replace("-", " ").strip()


def _jw(a: str, b: str) -> float:
    """Jaro-Winkler similarity, falls back to prefix-overlap ratio."""
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.similarity(a, b)
    except ImportError:
        if a == b:
            return 1.0
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        common = sum(1 for x, y in zip(shorter, longer) if x == y)
        return common / max(len(longer), 1)


def _load_entity_cache() -> None:
    global _entity_names, _cache_loaded_at
    if not os.path.exists(_GRAPH_DIR):
        return
    try:
        import kuzu
        db   = kuzu.Database(_GRAPH_DIR, read_only=True)
        conn = kuzu.Connection(db)
        rows = conn.execute("MATCH (n:Entity) RETURN n.name")
        _entity_names    = [str(r[0]) for r in rows if r[0]]
        _cache_loaded_at = time.time()
    except Exception as e:
        print(f"[tier1/matcher] cache load failed: {e}")


def _ensure_fresh() -> None:
    with _cache_lock:
        age = time.time() - _cache_loaded_at
        if not _entity_names or age > CACHE_TTL:
            _load_entity_cache()


def invalidate_cache() -> None:
    global _cache_loaded_at
    with _cache_lock:
        _cache_loaded_at = 0.0


def match_entity(text: str) -> tuple[str, float] | None:
    """
    Find the best matching KG entity for *text*.
    Returns (kg_name, score) or None if nothing clears ENTITY_THRESHOLD.
    """
    _ensure_fresh()
    if not _entity_names:
        return None

    query = normalize(text)
    best_name:  str   = ""
    best_score: float = 0.0

    for kg_name in _entity_names:
        score = _jw(query, normalize(kg_name))
        if score > best_score:
            best_score = score
            best_name  = kg_name

    if best_score >= ENTITY_THRESHOLD:
        return (best_name, round(best_score, 2))
    return None


def match_rels(
    rels: list[str],
    triples: list[dict],
) -> list[dict]:
    """
    From *triples* (each {"source", "relationship", "target"}), return those
    whose relationship name fuzzy-matches any rel in *rels* above REL_THRESHOLD.

    Sorted by best match score descending.
    """
    if not rels or not triples:
        return []

    norm_rels = [normalize(r) for r in rels]
    scored: list[tuple[float, dict]] = []

    for triple in triples:
        rel_norm = normalize(triple.get("relationship", ""))
        best = max(_jw(nr, rel_norm) for nr in norm_rels)
        if best >= REL_THRESHOLD:
            scored.append((best, triple))

    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored]
