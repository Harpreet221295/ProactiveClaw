# ProactiveClaw — portfolio & resume brief

> Source of truth for writing about this project (resume bullets, portfolio page, LinkedIn). Everything here is verifiable in the repo. Written 2026-09-19.

- **Repo:** https://github.com/Harpreet221295/ProactiveClaw (MIT, default branch `main`)
- **Author:** Harpreet Singh, solo project. Design, architecture, implementation, tests, docs.
- **Timeline:** Feb 2026 (Slack bot + bandit scheduling) → Sep 2026 (care registry, proactiveness levels, web UI, public release).
- **Screenshot:** `docs/screenshot.png` (demo data, safe to publish). README has two Mermaid diagrams (architecture, day-in-the-life).

## One-liner

A local-first AI personal assistant that *reaches out to you*: it keeps a registry of everything that matters (emails needing action, tasks, promises you made in conversation), reviews it every morning, nudges you at the right time, learns what you care about, and lets you set exactly how proactive it is with one dial.

## Elevator pitch (3 sentences)

Every AI assistant waits to be asked; the things that actually slip are the ones nobody asks about. ProactiveClaw inverts the loop: a care registry with a real item lifecycle, a deterministic morning review, code-enforced nudge budgets and quiet hours, pattern learning from how you react, and re-engagement timed by a bandit that learns when you reply. Users choose one of five proactiveness levels, and optional connectors (Gmail, Calendar, Notion, web search, Slack) are invisible to the model unless enabled.

## What makes it technically interesting (pick the ones that fit the audience)

1. **Care registry as source of truth.** Items carry lifecycle state, the user's own words about them (`user_intent`), deferral counts, nudge counts and deadlines. A deterministic `tend()` pass escalates repeated deferrals, expires snoozes, flags broken "I'll do it tonight" promises and archives stale items, all before any LLM call.
2. **Proactiveness as configuration, enforced in code.** Five levels × five situational "care modes" resolve into effective settings (`defaults ← level ← mode ← overrides`). Daily nudge caps, quiet hours, 30-minute spacing and morning-review headroom are enforced in the scheduling tool, so the guarantee holds regardless of which model is used. In live testing gpt-4o-mini ignored the prompt-level budget; the code caught it.
3. **Commitment capture and intent inference from natural conversation.** "I have to call Riya tonight" becomes a tracked item with a deadline; "I just called her" resolves it. No commands.
4. **Pattern learning with decay.** Per-sender/topic/category observations (acted / dismissed / deferred) with exponential decay and a minimum-observation threshold, plus explicit mute/boost rules from things the user says. Feeds urgency scoring and filtering of new items.
5. **Contextual multi-armed bandit for timing.** 168 arms (day × hour), time-decayed Q-values, kernel-smoothed reward propagation over a periodic (torus) day/hour grid with a weekday/weekend two-cluster kernel. Rewards from reply latency; penalties for ignored nudges. Used for nudge timing and re-engagement.
6. **Two-tier memory.** mem0 vector store + Kuzu knowledge graph for explicit recall; a passive "tier-1" layer that extracts entities from each message, fuzzy-matches them (Jaro-Winkler) against the graph, and injects 1-hop context silently.
7. **Background sub-agents** as separate processes with tool *profiles*, hard timeouts with a wrap-up phase, a watchdog, and a queue-based completion protocol back to the main agent.
8. **Connector gating.** A connector is active only if enabled in config *and* credentialed; its tools are removed from the model's tool list and from sub-agents, and the system prompt is rebuilt from config so the model never hallucinates access.
9. **Provider-agnostic LLM layer** (OpenAI + Anthropic behind one interface with tool-call and image conversion) and mid-conversation context compaction (browser-snapshot summarisation, token-threshold summarisation with orphan-tool-call cleanup).
10. **Clone-and-go release engineering:** setup wizard, health check that adapts to enabled connectors, reset tooling, centralised paths, a dependency-free web UI (vanilla JS + WebSocket, no build step), and a git-history scrub of personal data before going public.

## Numbers (as of 2026-09-19, all from the repo)

| Metric | Value |
|---|---|
| Python source | ~15,500 lines across 117 tracked files |
| Web UI | ~530 lines (HTML/CSS/JS, no framework, no build step) |
| Agent tools | 64 (care, scheduling, Gmail, Calendar, Notion, web, filesystem, data, charts, memory, browser, sub-agents) |
| Automated tests | 194 (pytest, no API calls; care, config, scheduling, prompts, sub-agents) |
| Proactiveness levels / care modes | 5 / 5 |
| Optional connectors | 6 (Gmail, Calendar, Notion, web search, browser extension, Slack) |
| Background loops | 5 (nudge delivery, idle→pre-exit, cron/morning review, re-engagement, sub-agent watchdog) |
| License | MIT |

## Tech stack

Python 3.11+, FastAPI + WebSockets, OpenAI / Anthropic SDKs, mem0 (Qdrant) + Kuzu graph DB, RapidFuzz, Google Gmail/Calendar APIs (OAuth), Notion API, Tavily, Slack SDK, multiprocessing sub-agents, vanilla JS UI (marked + DOMPurify), pytest, Chrome extension (Manifest V3, CDP over WebSocket).

## Resume bullets (choose length)

**Short (1 line)**
- Built ProactiveClaw, an open-source local AI assistant that proactively tracks commitments, emails and tasks, with user-controlled proactiveness levels, code-enforced nudge budgets, pattern learning and a bandit-timed re-engagement loop (Python, FastAPI, OpenAI/Anthropic, mem0/Kuzu; 194 tests).

**Medium (2–3 bullets)**
- Designed and shipped ProactiveClaw (MIT), a local-first proactive AI assistant: a care registry with item lifecycle and deterministic escalation, a daily morning review, and commitment capture from natural conversation, so the assistant reaches out with the right thing at the right time.
- Made proactiveness a first-class, user-controlled dial: 5 levels × 5 situational modes resolved into effective settings, with daily caps, quiet hours and spacing enforced in code rather than prompts; optional connectors (Gmail, Calendar, Notion, search, Slack) are gated so the model never assumes access it doesn't have.
- Built the supporting systems end to end: a 168-arm contextual bandit with kernel-smoothed rewards for nudge timing, two-tier memory (vector + knowledge graph with passive entity recall), process-isolated sub-agents with tool profiles and watchdog, a provider-agnostic LLM layer, a dependency-free web UI, and a clone-and-go setup with wizard, health checks and 194 tests.

**Impact-flavoured (if you want outcomes)**
- Replaced a Slack-only prototype with a self-contained web app and one-command setup; a new user needs one API key and no connectors to get value on day one.
- Live-tested against real Gmail/Calendar/Notion data: morning review, nudge scheduling and pre-exit follow-ups run end to end; code-level budget enforcement caught the model over-scheduling in the first run.

## Portfolio page copy (long form)

**ProactiveClaw** · Open source · Python, FastAPI, LLM agents · [GitHub](https://github.com/Harpreet221295/ProactiveClaw)

*The assistant that comes to you.*

Most assistants are reactive; the things that slip are the ones nobody asks about: the email you meant to answer tonight, the "I'll call her before Thursday" you said out loud, the task deferred three mornings in a row. ProactiveClaw keeps a **care registry** of those things and tends it: every morning it expires snoozes, escalates repeated deferrals, flags broken promises, pulls in only what's new from your sources, writes a brief, and schedules a handful of specific, actionable nudges. During the day it captures commitments from ordinary conversation and closes items when you mention you've done them. When you go quiet it waits, then comes back with one concrete thing, at an hour it has learned you actually reply.

The part I care most about is that **proactiveness is the user's decision**. Five levels from *off* to *max* and situational modes like *focus* or *fundraising* resolve into concrete settings, and the limits (daily cap, quiet hours, spacing) are enforced in the scheduling code, not left to the model's discretion. Integrations are optional and invisible to the model unless enabled.

Under the hood: a deterministic registry engine, pattern learning with decay, a kernel-smoothed contextual bandit for timing, two-tier memory (vector + knowledge graph with passive recall), process-isolated sub-agents, a provider-agnostic LLM layer, and a dependency-free web UI. Everything runs locally with your own key; ~15k lines of Python, 194 tests, one-command setup.

## Talking points / lessons (good for interviews)

- **Prompts are guidance, code is the guarantee.** The model ignored "leave headroom in the budget"; moving limits into the tool made behaviour reliable across models.
- **State beats snapshots.** Replacing a daily-brief JSON with a registry that persists lifecycle and user intent is what made continuity across days possible.
- **Optional by construction.** Gating tools and rebuilding the system prompt from config was cheaper and more robust than teaching the model which integrations exist.
- **Release hygiene is real work.** Central paths, wizard, connector-aware health check, history scrub of personal data, demo screenshot with fictional data.

## Things NOT to claim

- No user base or adoption metrics yet (just released).
- No Docker image, no Windows-native support (WSL only).
- Bandit and memory are implemented and used but not yet evaluated quantitatively.
