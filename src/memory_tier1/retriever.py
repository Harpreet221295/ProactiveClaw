"""
KG retrieval with decision tree for Tier 1 memory.

retrieve(extracted, session_kg_nodes) → list of triples to inject

Decision tree per (entity, rels) pair:
  1. Fuzzy match entity in KG → not found → skip
  2. Found, rels extracted    → fuzzy match rels against entity's edges → inject matches
  3. Found, no rels, entity IS the user → skip user-user, go to cross-entity
  4. Found, no rels, non-user:
       a. edges to the user   → inject
       b. edges to any other session KG node → inject
       c. fallback: entity name + any 3 of its edges
"""

from __future__ import annotations

import os
from pathlib import Path

import sys as _sys
_SRC       = Path(__file__).parent.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from core.paths import KUZU_GRAPH_DIR as _KUZU_DIR_P
from core.config import user_slug as _user_slug
_GRAPH_DIR = str(_KUZU_DIR_P)


def _self_node() -> str:
    return _user_slug()

MAX_FALLBACK_EDGES = 3


def _get_conn():
    import kuzu
    db   = kuzu.Database(_GRAPH_DIR, read_only=True)
    conn = kuzu.Connection(db)
    return conn


def _fetch_all_edges(conn, kg_name: str) -> list[dict]:
    """Fetch all edges where kg_name is source OR destination, including created timestamp."""
    triples: list[dict] = []
    seen: set[tuple] = set()

    for query in [
        "MATCH (a:Entity)-[r:CONNECTED_TO]->(b:Entity) WHERE a.name = $n RETURN a.name, r.name, b.name, r.created",
        "MATCH (a:Entity)-[r:CONNECTED_TO]->(b:Entity) WHERE b.name = $n RETURN a.name, r.name, b.name, r.created",
    ]:
        try:
            rows = conn.execute(query, {"n": kg_name})
            for row in rows:
                src, rel, tgt = str(row[0] or ""), str(row[1] or ""), str(row[2] or "")
                created = row[3]
                key = (src, rel, tgt)
                if key not in seen:
                    seen.add(key)
                    triples.append({"source": src, "relationship": rel, "target": tgt, "created": created})
        except Exception:
            pass
    return triples


def _edges_between(conn, node_a: str, node_b: str) -> list[dict]:
    """Fetch all edges between node_a and node_b in either direction, with timestamps."""
    triples: list[dict] = []
    seen: set[tuple] = set()

    for query in [
        "MATCH (a:Entity)-[r:CONNECTED_TO]->(b:Entity) WHERE a.name = $a AND b.name = $b RETURN a.name, r.name, b.name, r.created",
        "MATCH (a:Entity)-[r:CONNECTED_TO]->(b:Entity) WHERE a.name = $b AND b.name = $a RETURN a.name, r.name, b.name, r.created",
    ]:
        try:
            rows = conn.execute(query, {"a": node_a, "b": node_b})
            for row in rows:
                src, rel, tgt = str(row[0] or ""), str(row[1] or ""), str(row[2] or "")
                key = (src, rel, tgt)
                if key not in seen:
                    seen.add(key)
                    triples.append({"source": src, "relationship": rel, "target": tgt, "created": row[3]})
        except Exception:
            pass
    return triples


def _temporal_pair_edges(conn, kg_nodes: list[str], n: int = 2) -> list[dict]:
    """
    For each pair of KG-matched nodes, fetch edges between them and return
    earliest n + latest n by timestamp.
    """
    from itertools import combinations
    result: list[dict] = []
    seen: set[tuple] = set()

    for node_a, node_b in combinations(kg_nodes, 2):
        edges = _edges_between(conn, node_a, node_b)
        if not edges:
            continue

        sortable = [e for e in edges if e.get("created") is not None]
        if sortable:
            sortable.sort(key=lambda e: e["created"])
            candidates = sortable[:n] + sortable[-n:]
        else:
            candidates = edges[:n * 2]

        for e in candidates:
            key = (e["source"], e["relationship"], e["target"])
            if key not in seen:
                seen.add(key)
                result.append(e)

    return result


def _edges_to_node(all_edges: list[dict], target_name: str) -> list[dict]:
    """Filter edges that connect to/from target_name."""
    t = target_name.lower()
    return [
        e for e in all_edges
        if e["source"].lower() == t or e["target"].lower() == t
    ]


def retrieve(
    extracted: list[dict],
    user_id: str | None = None,
) -> list[dict]:
    """
    Run the full decision tree for all extracted (entity, rels) pairs.

    extracted: [{"entity": str, "rels": [str]}, ...]
    Returns a deduplicated list of triples: [{"source", "relationship", "target"}, ...]
    """
    from .matcher import match_entity, match_rels

    if user_id is None:
        user_id = _self_node()
    if not extracted or not os.path.exists(_GRAPH_DIR):
        return []

    try:
        conn = _get_conn()
    except Exception:
        return []

    # Step 1: match all entities first so we have the full session KG node set
    # for cross-entity fallback
    kg_map: dict[str, str] = {}   # extracted_entity → kg_node_name
    for item in extracted:
        entity = item["entity"]
        match = match_entity(entity)
        if match:
            kg_map[entity] = match[0]

    session_kg_nodes = set(kg_map.values())

    result: list[dict] = []
    seen_triples: set[tuple] = set()

    def _add(triples: list[dict]) -> None:
        for t in triples:
            key = (t["source"], t["relationship"], t["target"])
            if key not in seen_triples:
                seen_triples.add(key)
                result.append(t)

    for item in extracted:
        entity = item["entity"]
        rels   = item["rels"]

        kg_name = kg_map.get(entity)
        if not kg_name:
            continue   # not found in KG — skip

        all_edges = _fetch_all_edges(conn, kg_name)

        if rels:
            # Branch: rels extracted → fuzzy match rels against edges
            matched = match_rels(rels, all_edges)
            if matched:
                _add(matched)
                continue   # matched — don't fall through
            # No matches above threshold — fall through to no-rel branch

        if True:
            # Branch: no rels extracted
            if kg_name.lower() == user_id:
                # Skip user→user, go straight to cross-entity
                # If no other session nodes, skip — the user node has too many edges
                # to return anything meaningful without a rel filter
                other_nodes = session_kg_nodes - {kg_name}
                cross = []
                for other in other_nodes:
                    cross.extend(_edges_to_node(all_edges, other))
                if cross:
                    _add(cross)
            else:
                # a. edges to the user
                to_user = _edges_to_node(all_edges, user_id)
                if to_user:
                    _add(to_user)
                else:
                    # b. edges to any other session KG node
                    other_nodes = session_kg_nodes - {kg_name}
                    cross = []
                    for other in other_nodes:
                        cross.extend(_edges_to_node(all_edges, other))
                    if cross:
                        _add(cross)
                    else:
                        # c. fallback: name + any 3 edges
                        _add(all_edges[:MAX_FALLBACK_EDGES])

    # Cross-entity temporal edges — earliest 2 + latest 2 per matched pair
    kg_nodes = list(session_kg_nodes)
    if len(kg_nodes) >= 2:
        _add(_temporal_pair_edges(conn, kg_nodes))

    return result
