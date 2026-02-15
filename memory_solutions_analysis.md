# Long-Term Memory Solutions for AI Agents — Research & Analysis

This document compares long-term memory solutions for AI agents and explains why SimpleMem + MCP was chosen for ProactiveClaw.

---

## Solution Comparison

### 1. SimpleMem (Our Choice)

**What it is:** An efficient lifelong memory system for LLM agents that applies semantic lossless compression to filter redundant interaction content, reformulating raw dialogue streams into compact memory units — self-contained facts with resolved coreferences and absolute timestamps.

**How it works:** Uses an implicit semantic density gating mechanism integrated into the LLM generation process. Stores compressed memories in a LanceDB vector store and retrieves them via a three-stage hybrid retrieval pipeline: semantic search + lexical matching + symbolic lookup.

**Protocol:** Model Context Protocol (MCP) via JSON-RPC 2.0 over Streamable HTTP (MCP 2025-03-26 spec).

**Self-hosted vs Cloud:** Both. Cloud-hosted at `mcp.simplemem.cloud` with multi-tenant user isolation. Self-hosted uses LanceDB as an embedded file-based backend — no external database server required.

**Key features:**
- Semantic lossless compression of dialogues
- Three-stage hybrid retrieval (semantic + lexical + symbolic)
- Automatic relationship discovery via semantic backlinks
- Flexible tag system for organizing memories
- Vector embeddings powered by Voyage AI
- DuckDB/LanceDB backend for fast vector similarity search
- 43.24% F1 score on LoCoMo-10 benchmark (64% performance boost over Claude-Mem)

**Pros:**
- Native MCP support — seamless integration with Claude, Cursor, and any MCP-compatible client
- Lightweight — no heavy infrastructure required for self-hosting
- Semantic compression reduces storage bloat while preserving meaning
- Strong benchmark performance
- Simple API surface (`memory_add_batch`, `memory_query`, `memory_retrieve`)

**Cons:**
- Relatively newer project with a smaller community compared to Mem0 or Zep
- LanceDB/DuckDB backend may not scale to enterprise workloads as easily as dedicated vector databases
- Limited ecosystem integrations compared to more established tools

**Pricing:** Open-source self-hosted option is free. Cloud-hosted service available with token-based authentication.

---

### 2. Mem0 (formerly EmbedChain)

**What it is:** A universal memory layer for AI agents that dynamically extracts, consolidates, and retrieves salient information from ongoing conversations. Uses priority scoring and contextual tagging to decide what gets stored.

**How it works:** A memory orchestration layer that sits between AI agents and storage systems. Extracts information from agent interactions, applies priority scoring and contextual tagging, and stores/retrieves memories efficiently. An enhanced variant (Mem0g) adds a graph-based store for richer multi-session relationships.

**Protocol:** Python SDK and REST API. Works with OpenAI, LangGraph, CrewAI, and more.

**Self-hosted vs Cloud:** Both. Open-source self-hosted runs on Kubernetes, air-gapped servers, or private clouds. Managed cloud service with SOC 2 and HIPAA compliance.

**Key features:**
- Unified APIs for episodic, semantic, procedural, and associative memories
- Memory scopes: user-level, session-level, and agent-level
- Dynamic forgetting (decaying low-relevance entries over time)
- Graph memory layer for entity relationships (optional)
- 91% lower p95 latency vs OpenAI baseline
- 90%+ token cost savings

**Pros:**
- Largest community and mindshare (raised $24M in Oct 2025)
- Multiple memory scopes provide flexible granularity
- Dynamic forgetting prevents memory bloat organically
- Graph memory for complex relationship tracking
- Works with any LLM provider
- Strong enterprise compliance (SOC 2, HIPAA)

**Cons:**
- More opinionated architecture — may be overkill for simpler use cases
- Cloud pricing can add up for high-volume applications
- Graph memory adds latency and complexity
- Not MCP-native (requires SDK integration)

**Pricing:** Open-source core is free. Managed cloud is usage-based.

---

### 3. Zep

**What it is:** A context engineering and agent memory platform that automatically extracts entities, relationships, and facts to build a temporal knowledge graph that evolves with every interaction.

**How it works:** Organizes data into three hierarchical subgraph tiers: an episode subgraph (raw interaction data), a semantic entity subgraph (extracted entities and relationships), and a community subgraph (higher-level clusters). Uses a dual-timestamp model tracking both event time and ingestion time.

**Protocol:** REST API with Python and TypeScript SDKs. Integrations with LangChain, OpenAI, Anthropic, LlamaIndex.

**Self-hosted vs Cloud:** Core temporal knowledge graph engine is open-sourced as **Graphiti** (uses Neo4j). Zep Cloud is the managed enterprise platform with <200ms latency guarantees.

**Key features:**
- Temporal knowledge graph with dual timestamps (event time + ingestion time)
- Three-tier subgraph hierarchy (episode, semantic entity, community)
- Sub-200ms retrieval latency (optimized for voice AI)
- Automatic entity and relationship extraction
- SOC 2 Type II certified, HIPAA BAA available

**Pros:**
- Best-in-class temporal reasoning — understands how facts change over time
- Knowledge graph captures relationships that vector-only approaches miss
- Very low latency retrieval (<200ms)
- Graphiti open-source provides a path to self-hosting

**Cons:**
- Neo4j dependency for self-hosted adds operational complexity
- Steeper learning curve due to graph concepts
- Cloud pricing uses credit-based system which can be hard to predict
- Not MCP-native

**Pricing:** Free tier available. Paid tiers use credit-based billing. Enterprise plans offer flexible deployment (managed, BYOK, BYOM, BYOC).

---

### 4. LangChain Memory Modules

**What it is:** A suite of memory abstractions built into the LangChain framework, providing different strategies for maintaining conversational context.

**Memory types:**

| Module | Strategy | Best For |
|--------|----------|----------|
| ConversationBufferMemory | Stores every message verbatim | Short conversations, full fidelity |
| ConversationBufferWindowMemory | Keeps last K messages | Medium conversations, bounded context |
| ConversationSummaryMemory | LLM-summarizes history | Long conversations, space efficiency |
| ConversationSummaryBufferMemory | Recent messages + summary of older ones | Balance of fidelity and compression |
| VectorStoreRetrieverMemory | Embeds messages, retrieves by similarity | Long-term recall by topic |
| ConversationEntityMemory | Tracks entities mentioned in conversation | Entity-centric applications |
| ConversationKGMemory | Builds knowledge graph from conversation | Relationship tracking |

**Protocol:** Python SDK (part of LangChain). No standalone API — tightly coupled to the framework.

**Self-hosted vs Cloud:** Library-based, entirely self-hosted. Pluggable storage backends (in-memory, Redis, SQL, vector stores).

**Pros:**
- Zero additional dependencies if already using LangChain
- Wide variety of strategies for different needs
- Well-documented with large community
- Composable — can combine multiple memory types
- Free and open-source

**Cons:**
- Tightly coupled to LangChain — not usable outside the framework
- No standalone deployment or API
- ConversationBufferMemory scales poorly (linear token growth)
- Many modules are now deprecated in favor of LangGraph memory patterns
- No built-in temporal reasoning

**Pricing:** Free and open-source. Costs from underlying LLM calls and storage backends.

---

### 5. Motorhead (by Metal)

**What it is:** An open-source memory and information retrieval server for LLMs, built in Rust. Manages conversation context windows with automatic incremental summarization.

**How it works:** Uses Redis to store conversation messages with a configurable `MAX_WINDOW_SIZE`. When the window is exceeded, it processes messages into a summary. Long-term retrieval uses Redis Vector Similarity Search (VSS).

**Protocol:** REST API with 4 endpoints: `GET/POST/DELETE /sessions/:id/memory` and `POST /sessions/:id/retrieval`.

**Self-hosted vs Cloud:** Self-hosted via Docker + Redis. Project appears largely unmaintained since 2023.

**Key features:**
- Built in Rust for performance
- Automatic incremental summarization
- Redis-backed with VSS for long-term retrieval
- Session-based memory management

**Pros:**
- Extremely simple API (4 endpoints)
- Fast (Rust + Redis)
- Lightweight and easy to deploy

**Cons:**
- Largely unmaintained (limited activity since 2023)
- Redis dependency for all storage
- No knowledge graph, entity extraction, or temporal reasoning
- No MCP support

**Pricing:** Free and open-source. Costs are Redis infrastructure only.

---

### 6. Vector Databases as Memory Backends

These are **storage backends**, not memory systems. They handle embedding storage and retrieval but do not handle memory extraction, summarization, temporal reasoning, or consolidation. You need application-level logic on top.

#### ChromaDB
- **Type:** Open-source, lightweight embedding database
- **Hosting:** In-process, local persistent, or Chroma Cloud (serverless)
- **Performance:** ~20ms median search at 100K vectors
- **Pros:** Simplest onboarding, great for prototyping, full data portability
- **Cons:** May struggle at >1M vectors, limited filtering, newer cloud offering
- **Pricing:** Open-source core free. Cloud has free credits to start.

#### Pinecone
- **Type:** Fully managed, cloud-native vector database
- **Hosting:** Cloud-only (no self-hosting)
- **Performance:** Sub-50ms latency at billion-scale
- **Pros:** Zero operational overhead, consistent performance at any scale
- **Cons:** Vendor lock-in, limited data export, no self-hosting
- **Pricing:** Usage-based (storage + read/write units). Free tier available.

#### Weaviate
- **Type:** Open-source vector database with knowledge graph focus
- **Hosting:** Self-hosted (Docker/K8s) or Weaviate Cloud Service
- **Performance:** GraphQL API, hybrid search (vector + keyword)
- **Pros:** Complex queries via GraphQL, strong self-hosting, object-level relationships
- **Cons:** More complex setup, higher learning curve
- **Pricing:** Open-source core free. WCS is usage-based.

---

### 7. LlamaIndex Memory

**What it is:** Memory components within LlamaIndex for agentic workflows, supporting both short-term and long-term memory via composable memory blocks.

**Memory blocks:**
- **ChatMemoryBuffer** (deprecated): Simple FIFO queue
- **Memory class**: Flexible configuration with pluggable blocks
- **FactExtractionMemoryBlock**: Extracts facts from chat history using LLM
- **VectorMemoryBlock**: Vector-based storage and retrieval

**Protocol:** Python SDK, part of the LlamaIndex framework.

**Pros:** Composable architecture, fact extraction via LLM, free and open-source.

**Cons:** Tightly coupled to LlamaIndex, still evolving, no standalone deployment or temporal reasoning.

**Pricing:** Free and open-source. Costs from LLM calls and storage.

---

### 8. Other Notable Solutions

#### MemGPT / Letta
An "LLM Operating System" for memory management. Treats the context window as constrained RAM with three memory tiers:
- **Core memory** (in-context, like RAM): persona + user info blocks the agent actively reads/writes
- **Archival memory** (like disk): long-term vector storage for overflow
- **Recall memory**: searchable conversation history

Agents self-manage memory via built-in tools (`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`). Features git-based memory versioning (Context Repositories) and shared memory across parallel sessions (Conversations API).

**Pros:** Most sophisticated memory management model, active development, open-source.
**Cons:** Heavier framework, higher learning curve, agent-managed memory can be unpredictable, not MCP-native.

#### Cognee
An open-source knowledge engine using an ECL (Extract, Cognify, Load) pipeline to build knowledge graphs on-the-fly from retrieved data. Combines vector search with graph-aware embeddings.

**Pros:** Reduces hallucinations through relationship understanding, graph-aware embeddings.
**Cons:** More complex setup, graph construction adds latency, smaller ecosystem.

---

## Summary Comparison Table

| Solution | Type | Protocol | Self-Hosted | Temporal | Graph | MCP Native | Cost |
|----------|------|----------|-------------|----------|-------|------------|------|
| **SimpleMem** | Memory service | MCP/JSON-RPC | Yes | Timestamps | Backlinks | **Yes** | Free OSS + Cloud |
| **Mem0** | Memory orchestration | SDK/REST | Yes | Decay scoring | Optional | No | Free OSS + Paid |
| **Zep** | Memory platform | SDK/REST | Via Graphiti | **Best** | **Yes** (temporal KG) | No | Free tier + Credits |
| **LangChain** | Framework module | Python SDK | Yes (library) | No | ConversationKG only | No | Free OSS |
| **Motorhead** | Memory server | REST | Yes (Redis) | No | No | No | Free OSS |
| **ChromaDB** | Vector DB | Python/REST | Yes | No | No | No | Free OSS + Cloud |
| **Pinecone** | Vector DB | REST/SDK | No | No | No | No | Usage-based |
| **Weaviate** | Vector DB | GraphQL/REST | Yes | No | Cross-refs | No | Free OSS + Cloud |
| **LlamaIndex** | Framework module | Python SDK | Yes (library) | No | No | No | Free OSS |
| **Letta/MemGPT** | Agent framework | REST/SDK | Yes | Via versioning | No | No | Free OSS + Cloud |
| **Cognee** | Knowledge engine | Python SDK | Yes | Time-graph | **Yes** | No | Free OSS |

---

## Architectural Patterns & Best Practices

### The Three Types of Long-Term Memory

1. **Episodic Memory:** Records specific past experiences. *"Last Tuesday, when we tried approach X with client Y, it failed because Z."* Useful for learning from past interactions and avoiding repeated mistakes.

2. **Semantic Memory:** Stores generalized knowledge and facts. *"Approach X generally works best when conditions A and B are present."* Often integrates with RAG systems to pull domain-specific knowledge.

3. **Procedural Memory:** Encodes how to do things. Stores successful workflows, tool usage patterns, and multi-step procedures. Enables agents to improve execution over time.

### Common Architectural Patterns

**RAG-Based Memory:**
The most common pattern. Store memories as embeddings, retrieve relevant ones via semantic similarity, inject into context. Works well for factual QA but struggles with preference inference and temporal reasoning.

**Memory Consolidation:**
Analogous to how human memory consolidates during sleep. New events are stored in short-term memory, then an asynchronous extraction process identifies meaningful information and stores it long-term. Extraction typically completes in 20-40 seconds; retrieval returns in ~200ms.

**Tiered Memory Architecture:**
Combine multiple memory strategies in layers:
- Layer 1: Recent messages (buffer/window) — full fidelity, limited history
- Layer 2: Summarized history — compressed, broader context
- Layer 3: Vector-indexed facts — semantic retrieval across all time
- Layer 4: Knowledge graph — entity relationships and temporal evolution

**Memory Lifecycle Management:**
- **Extraction:** LLM-based analysis of conversations to identify facts, preferences, entities
- **Consolidation:** Merging, deduplicating, and updating existing memories with new information
- **Decay:** Gradually reducing relevance scores of unused memories (dynamic forgetting)
- **Retrieval:** Multi-stage pipeline combining semantic, lexical, and symbolic search

---

## Why SimpleMem + MCP for ProactiveClaw

### 1. Protocol-Level Interoperability via MCP
SimpleMem speaks MCP natively. Our client (`memory.py`) is a thin ~120-line wrapper using standard JSON-RPC 2.0 over HTTP. No heavy SDK dependency — just the `requests` library. Any MCP-compatible client can swap in without code changes. Future-proof as MCP becomes the standard protocol for AI tool integration.

### 2. Lightweight Self-Hosting
ProactiveClaw is a self-hosted personal assistant. We need full data sovereignty (personal conversations, calendar data, credentials), no vendor lock-in, and low operational overhead. SimpleMem's LanceDB backend is embedded and file-based — no external Neo4j (like Zep), no Redis (like Motorhead), no Kubernetes (like Mem0 enterprise).

### 3. Right-Sized Complexity
- **Mem0** is a full orchestration layer with graph memory, priority scoring, and dynamic forgetting — more than a personal assistant needs
- **Zep** requires Neo4j and targets enterprise multi-tenant scenarios
- **Letta/MemGPT** is an entire agent framework — we already have our own (`agent.py`, `bandit.py`)
- **LangChain/LlamaIndex memory** would require adopting those frameworks
- SimpleMem provides exactly what we need: store dialogues, query semantically, retrieve by topic

### 4. Semantic Compression
SimpleMem converts raw dialogues into compact memory units with resolved coreferences and absolute timestamps. For a personal assistant accumulating conversations over months/years, this prevents memory bloat while preserving meaning — critical for a self-hosted system with finite storage.

### 5. Three Clean API Operations
Our integration uses exactly three operations:
- `memory_add_batch` — store conversation dialogues at pre-exit
- `memory_query` — semantic search over memory (agent tool)
- `memory_retrieve` — direct retrieval by topic/keyword (agent tool)

This maps perfectly to our memory needs: save what happened, find relevant context for new interactions, and recall specific facts when asked.

### 6. Graceful Degradation
If SimpleMem is down or unconfigured, everything else works fine. Tools return "not configured" strings. No crashes, no blocked startup, no dependency on an external service for core functionality.

---

## Sources

- [SimpleMem GitHub](https://github.com/aiming-lab/SimpleMem)
- [SimpleMem MCP Server](https://mcp.simplemem.cloud/)
- [Mem0 Official Site](https://mem0.ai/)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 Research Paper (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)
- [Zep Official Site](https://www.getzep.com/)
- [Zep Temporal KG Paper (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Graphiti GitHub (Zep OSS)](https://github.com/getzep/graphiti)
- [LangChain Conversational Memory Guide](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
- [Motorhead GitHub](https://github.com/getmetal/motorhead)
- [Vector Database Comparison 2025](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [LlamaIndex Memory Docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)
- [Letta/MemGPT Docs](https://docs.letta.com/concepts/memgpt/)
- [Letta GitHub](https://github.com/letta-ai/letta)
- [Cognee GitHub](https://github.com/topoteretes/cognee)
- [3 Types of Long-Term Memory for AI Agents (ML Mastery)](https://machinelearningmastery.com/beyond-short-term-memory-the-3-types-of-long-term-memory-ai-agents-need/)
- [AWS AgentCore Long-Term Memory Deep Dive](https://aws.amazon.com/blogs/machine-learning/building-smarter-ai-agents-agentcore-long-term-memory-deep-dive/)
