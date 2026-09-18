# Tier 1 Memory Retrieval — Spreading Activation Plan

## What it is

Tier 1 is **passive, automatic, zero-cost associative memory** — analogous to how hearing a name instantly surfaces related concepts in human memory without conscious deliberate recall.

It is NOT a search. No embedding call, no LLM, no API cost. It fires on every incoming user message and injects ambient context into the system prompt silently.

Contrast with **Tier 2** (what exists today): the agent explicitly calls `query_long_term_memory`, which does a semantic vector + graph search. That's deliberate recall. Tier 1 is automatic spreading activation.

---

## Pipeline

```
User message arrives
        │
        ▼
1. Entity Extraction  ──────────────────────────────────────────────
   Extract proper nouns / named entities from the message.
   Recommended: spaCy NER (en_core_web_sm, ~12MB, ~2ms per message)
   Output: list of (entity_text, entity_type) e.g. [("Pete", "PERSON"), ("ProactiveClaw", "ORG")]
   Fallback: POS tagging (NNP/NNPS tags) if spaCy not available
        │
        ▼
2. Edit Distance Matching against KG entities  ──────────────────────
   For each extracted entity, compute similarity against ALL entity
   names currently in the Kuzu graph.

   Algorithm: Jaro-Winkler (better than Levenshtein for proper names)
   - Weights prefix matches more heavily (first chars carry more signal for names)
   - Handles transpositions naturally
   - Returns a 0.0–1.0 similarity score

   Data structure: rapidfuzz over flat cached entity list
   - At current scale (<1000 entities): O(n) is <1ms, no tree needed
   - When scale grows (10K+ entities): migrate to BK-Tree (pybktree)
     which supports O(log n) range queries via triangle inequality of edit distance

   Threshold: retain matches with Jaro-Winkler similarity >= 0.80
   Output: for each input entity → top-K KG matches with scores
        │
        ▼
3. 1-Hop Neighbor Retrieval from Kuzu  ──────────────────────────────
   For each matched KG entity above threshold:
   - Fetch the entity node (type, summary if stored)
   - Fetch all 1-hop neighbors (direct relationships in either direction)
   - Include relationship type (WORKS_AT, IS_A, BUILDS, etc.)

   Cap: max 5 matched entities, max 8 neighbors per entity
   (prevents context block explosion)
        │
        ▼
4. Confidence-annotated context injection  ──────────────────────────
   Format results as a <tier1_memory_context> block and prepend
   to the user message before the agent sees it.

   The similarity score is surfaced explicitly — this tells the LLM
   how confident the match is, allowing it to reason about ambiguity.

   See format below.
        │
        ▼
5. Session deduplication + accumulation  ────────────────────────────
   Track which entity IDs have already been injected this session.
   On subsequent messages, only inject NEW entities not yet in context.
   The context block grows throughout the session but never duplicates.
```

---

## Context Block Format

```
<tier1_memory_context>
Note: These associations were retrieved automatically based on entities
detected in your message. Confidence scores indicate match certainty.

"Pete"  →  Peter  [confidence: 0.93]
  • Peter IS_A  person
  • Peter WORKS_AT  tech startup, San Francisco
  • Peter HAS  ongoing Q2 budget discussion with Sarah
  • Peter CO_OCCURS_WITH  Sarah, Ada, ProactiveClaw

"Pete"  →  Pits  [confidence: 0.75]  ← surfaced for disambiguation
  • Pits IS_A  ...
  • Pits LOCATED_IN  ...

"ProactiveClaw"  →  ProactiveClaw  [confidence: 1.00]
  • ProactiveClaw BUILDS_ON  mem0, Qdrant, Slack
  • ProactiveClaw USES  gpt-4.1-mini, text-embedding-3-small
  • ProactiveClaw HAS  daily brief system, cron jobs, sub-agents
</tier1_memory_context>
```

Multiple KG candidates for the same input entity are included when
confidence is ambiguous (e.g. 0.75–0.90) — the LLM can use the neighbor
context to self-disambiguate, or know to ask.

---

## The Pete/Peter/Pits insight

When a user writes "Pete", the system doesn't know if they mean Peter (a
known contact) or Pits or someone else. Rather than silently picking one:

- Surface Peter at confidence 0.93 with its neighbors
- Surface Pits at confidence 0.75 with its (different) neighbors

The LLM sees both, reads the neighbor contexts, and can usually tell which
is meant from the conversation. If not, it knows to ask. This mirrors how
human memory handles ambiguous name recognition — the brain surfaces
multiple candidates with varying activation strength, and conscious
reasoning disambiguates.

---

## Multi-word entity handling

For multi-word entities ("San Francisco", "Q2 budget"), character-level
edit distance is suboptimal. Use token-level edit distance (edit distance
over word tokens) for spans with 2+ words. spaCy's NER already chunks
multi-word spans correctly — detect span length and choose algorithm
accordingly.

---

## Where it runs

`slack_server.py` — intercepts every incoming user message BEFORE passing
to the agent. Runs synchronously but should complete in <10ms:
- spaCy NER: ~2ms
- rapidfuzz over cached entity list: <1ms
- Kuzu 1-hop query: ~2-5ms
- Formatting: <1ms

Entity cache (flat list of all KG entity names) is refreshed lazily —
on server start, then every N minutes or on a background timer.

---

## Prerequisites

1. Kuzu graph store must be configured in `_MEM0_CONFIG` (done in Tier 2 implementation)
2. Existing memories must be re-processed through mem0 with graph enabled to populate Kuzu
3. spaCy small model: `pip install spacy && python -m spacy download en_core_web_sm`
4. rapidfuzz: `pip install rapidfuzz`

---

## Files to create (when implementing)

- `src/memory_tier1/extractor.py` — spaCy NER wrapper, entity extraction
- `src/memory_tier1/matcher.py` — Jaro-Winkler matching + BK-Tree, entity cache
- `src/memory_tier1/retriever.py` — Kuzu 1-hop neighbor fetch
- `src/memory_tier1/injector.py` — context block formatter, session deduplication
- `src/memory_tier1/__init__.py`
- Hook into `slack_server.py`: call `build_tier1_context(user_message, session_id)` and prepend result to message

---

## Open questions to resolve at implementation time

1. Should Tier 1 context go in the system prompt (grows per session, busts prompt cache)
   or prepended to each user message (keeps system prompt stable, better for caching)?
   → Recommendation: prepend to user message to preserve prompt cache.

2. What is the right Jaro-Winkler threshold? Start at 0.80, tune empirically.

3. How many neighbors per entity? Start at 8, reduce if context gets too long.

4. Should Tier 1 fire on every message, or skip if message is short/purely conversational?
   → Simple heuristic: skip if no capitalized words / NER finds no entities.

5. Entity cache refresh strategy — lazy (on first miss) vs scheduled (every 5 min)?
