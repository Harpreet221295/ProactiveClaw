"""
LLM-based entity + relationship extractor for Tier 1 memory retrieval.

Given a user message, extracts a list of (entity, [rels]) pairs where:
  - entity: the named entity or concept mentioned
  - rels:   relationship types connecting the user to this entity,
            or [] if none are implied

Self-references (I, me, my, myself, we) always resolve to the configured user slug.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SRC  = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

_EXTRACTION_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT_TEMPLATE = """You are an entity and relationship extractor for a personal assistant's memory system.

Given a user message, extract ALL named entities or concepts mentioned. For each entity, also extract the relationship type implied by the message if any.

Rules:
- Self-references (I, me, my, myself, mine, our, we) → always use "{self_node}" as the entity
- Extract EVERY named entity in the message — people, organizations, projects, places, products. Do not skip any.
- For each entity, extract relationship types if the message implies one
- If no relationship is implied for an entity, leave rels as an empty list — but still include the entity
- Normalize entity names to lowercase with underscores (e.g. "ProactiveClaw" → "proactiveclaw", "San Francisco" → "san_francisco", "Oakridge University" → "oakridge_university")
- Normalize relationship types to lowercase with underscores
- Do NOT extract generic words, verbs, adjectives, or role descriptions as standalone entities (e.g. "investor", "advisor", "designer" are roles/relationships, NOT entities — but "my advisor" or "my lead investor" should still produce {self_node} with the role as the rel)
- If the message is purely conversational with no named entities (e.g. "ok thanks", "sounds good"), return an empty list

Output a JSON array. Each element: {{"entity": "...", "rels": ["...", ...]}}

Examples:

Message: "when is my meeting with Sarah?"
Output: [{{"entity": "{self_node}", "rels": ["meeting_with"]}}, {{"entity": "sarah", "rels": ["meeting_with"]}}]

Message: "my co-founder Sarah told me about the new investor"
Output: [{{"entity": "{self_node}", "rels": ["co_founder"]}}, {{"entity": "sarah", "rels": ["co_founder_of"]}}]

Message: "what do I know about my lead investor?"
Output: [{{"entity": "{self_node}", "rels": ["investor"]}}]

Message: "my advisor mentioned something interesting"
Output: [{{"entity": "{self_node}", "rels": ["advisor"]}}]

Message: "I work at NovaBridge in San Francisco"
Output: [{{"entity": "{self_node}", "rels": ["works_at"]}}, {{"entity": "novabridge", "rels": ["workplace_of"]}}, {{"entity": "san_francisco", "rels": ["located_in"]}}]

Message: "ProactiveClaw uses mem0 for memory"
Output: [{{"entity": "proactiveclaw", "rels": ["uses"]}}, {{"entity": "mem0", "rels": ["used_by"]}}]

Message: "I need to prep for my Oakridge University reunion"
Output: [{{"entity": "{self_node}", "rels": ["attended"]}}, {{"entity": "oakridge_university", "rels": ["attended_by"]}}]

Message: "what's the latest with NovaBridge's Anthropic partnership?"
Output: [{{"entity": "novabridge", "rels": ["partnership_with"]}}, {{"entity": "anthropic", "rels": ["partner_of"]}}]

Message: "Maya told me the term sheet is ready"
Output: [{{"entity": "maya", "rels": []}}, {{"entity": "{self_node}", "rels": []}}]

Message: "Jake is joining the design review tomorrow"
Output: [{{"entity": "jake", "rels": ["joining"]}}]

Message: "when is my meeting with Maya from Sequoia?"
Output: [{{"entity": "{self_node}", "rels": ["meeting_with"]}}, {{"entity": "maya", "rels": ["meeting_with"]}}, {{"entity": "sequoia", "rels": []}}]

Message: "what did we discuss last time?"
Output: [{{"entity": "{self_node}", "rels": []}}]

Message: "what is DataBridge about?"
Output: [{{"entity": "databridge", "rels": []}}]

Message: "when are we closing the Series A?"
Output: [{{"entity": "{self_node}", "rels": ["raising"]}}, {{"entity": "series_a", "rels": []}}]

Message: "ok sounds good"
Output: []"""


def _system_prompt() -> str:
    from core.config import user_slug
    return _SYSTEM_PROMPT_TEMPLATE.format(self_node=user_slug())


def extract_entities(message: str) -> list[dict]:
    """
    Extract (entity, rels) pairs from a user message.

    Returns a list of dicts: [{"entity": str, "rels": [str]}, ...]
    Returns [] on any error or if nothing meaningful found.
    """
    if not message or not message.strip():
        return []

    stripped = message.strip()
    if len(stripped) < 6:
        return []

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": stripped},
            ],
            max_tokens=400,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Extract JSON array from response (may have markdown fences)
        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return []
        items = parsed

        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entity = item.get("entity", "").strip().lower().replace(" ", "_")
            rels = [r.strip().lower().replace(" ", "_") for r in item.get("rels", []) if r]
            if entity:
                result.append({"entity": entity, "rels": rels})
        return result

    except Exception as e:
        print(f"[tier1] entity extraction failed: {e}")
        return []
