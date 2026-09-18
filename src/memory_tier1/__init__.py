"""
memory_tier1 — Passive spreading-activation memory retrieval.

Fires on every incoming user message. Uses an LLM to extract (entity, rels)
pairs, fuzzy-matches them against the Kuzu knowledge graph, and returns
a context block of relevant triples for injection into the user message.

Usage:
    from memory_tier1 import build_tier1_context, clear_session_cache

    ctx = build_tier1_context(user_message, session_id="slack_U123")
    if ctx:
        augmented_message = ctx + "\\n\\n" + user_message
"""

from __future__ import annotations

from .injector import build_tier1_context, clear_session_cache

__all__ = ["build_tier1_context", "clear_session_cache"]
