#!/usr/bin/env python3
"""
Insert fake test data into the KG via store_dialogues so the Tier 1 pipeline
has something to work with during testing.

Run from project root:
    python src/memory_tier1/insert_test_data.py

Requires: OPENAI_API_KEY set in .env
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SRC  = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# Fake conversation pairs that will seed the KG with rich entities + relationships
FAKE_CONVERSATIONS = [
    # ── Batch 2: new people, projects, context ──────────────────────────────
    (
        "We just hired Riya as our Head of Engineering. She previously worked at Google.",
        "Got it — Riya is your new Head of Engineering, previously at Google.",
    ),
    (
        "My friend Arjun is a VC at Andreessen Horowitz. He introduced me to Maya.",
        "Noted — Arjun is your friend and a VC at Andreessen Horowitz who connected you with Maya.",
    ),
    (
        "NovaBridge is also building a product called DataBridge for enterprise data pipelines.",
        "Understood — DataBridge is NovaBridge's enterprise data pipeline product.",
    ),
    (
        "I have a board meeting with Maya and two other board members every quarter.",
        "Got it — you have quarterly board meetings with Maya and the other board members.",
    ),
    (
        "My mom's name is Meera. She lives in Pune and calls me every Sunday.",
        "Noted — your mom Meera lives in Pune and you speak every Sunday.",
    ),
    (
        "Jake is working on the new design system for DataBridge. He reports to Sarah.",
        "Got it — Jake is leading the DataBridge design system and reports to Sarah.",
    ),
    (
        "I'm going to YC's AI Summit in San Francisco next month to demo ProactiveClaw.",
        "Noted — you're demoing ProactiveClaw at YC's AI Summit in San Francisco.",
    ),
    (
        "Priya is our data scientist. She's been running experiments on the recommendation model.",
        "Got it — Priya is your data scientist, running experiments on the recommendation model.",
    ),
    (
        "I'm planning to close the Series A by end of Q2. Maya is leading the round.",
        "Understood — you're targeting Series A close by end of Q2 with Maya leading.",
    ),
    (
        "ProactiveClaw uses Kuzu for its knowledge graph and Qdrant for vector search.",
        "Got it — ProactiveClaw's memory uses Kuzu for KG and Qdrant for vector search.",
    ),
    # ── Original batch ───────────────────────────────────────────────────────
    (
        "I'm Ada and I'm the CTO at NovaBridge, a tech startup in San Francisco.",
        "Got it — you're the CTO at NovaBridge based in San Francisco.",
    ),
    (
        "My co-founder is Sarah. She handles the product side and is based in New York.",
        "Noted — Sarah is your co-founder and handles product at NovaBridge, based in New York.",
    ),
    (
        "We're building ProactiveClaw, an AI personal assistant powered by mem0 and Qdrant.",
        "Got it — ProactiveClaw is your AI product built with mem0 and Qdrant.",
    ),
    (
        "My lead investor is Maya from Sequoia Capital. We're raising a Series A next quarter.",
        "Noted — Maya from Sequoia Capital is your lead investor for the Series A round.",
    ),
    (
        "I went to Oakridge University for undergrad and Westbrook Tech for my Masters.",
        "Understood — you studied at Oakridge University for undergrad and Westbrook Tech for your Masters.",
    ),
    (
        "I have a weekly 1:1 with Sarah on Mondays at 10am to review the roadmap.",
        "Got it — you have a weekly Monday 10am 1:1 with Sarah for roadmap review.",
    ),
    (
        "ProactiveClaw uses OpenAI APIs for LLM calls and Slack for user interaction.",
        "Noted — ProactiveClaw uses OpenAI for LLMs and Slack as the user interface.",
    ),
    (
        "My advisor is Dr. Chen from Westbrook Tech. He mentors me on AI research.",
        "Understood — Dr. Chen from Westbrook Tech is your advisor and AI research mentor.",
    ),
    (
        "Sarah is also working with our designer Jake who is based in London.",
        "Got it — Jake is your designer based in London, working with Sarah.",
    ),
    (
        "NovaBridge has a partnership with Anthropic for enterprise AI features.",
        "Noted — NovaBridge has a partnership with Anthropic for enterprise AI.",
    ),
]


def main() -> int:
    try:
        from memory import store_dialogues, _get_memory
    except ImportError as e:
        print(f"✗ Could not import memory module: {e}")
        return 1

    print("Initializing memory store (vector + graph)…")
    mem = _get_memory()
    if mem is None:
        print("✗ Memory init failed — check OPENAI_API_KEY")
        return 1
    print("✓ Memory ready\n")

    total = len(FAKE_CONVERSATIONS)
    for i, (user_msg, agent_resp) in enumerate(FAKE_CONVERSATIONS, 1):
        print(f"[{i}/{total}] Storing: {user_msg[:70]}…")
        result = store_dialogues([
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": agent_resp},
        ])
        print(f"         → {result}")
        if i < total:
            time.sleep(1)   # small pause to avoid rate limits

    print("\n✓ All test data inserted.")
    print("Run python src/memory_tier1/test_tier1.py to test Tier 1 retrieval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
