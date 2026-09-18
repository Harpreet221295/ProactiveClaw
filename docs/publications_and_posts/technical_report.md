# ProactiveClaw: A Long-Horizon Persistent Agent with Adaptive Notification Scheduling

## Abstract

We present ProactiveClaw, a persistent conversational AI agent that maintains long-running sessions, integrates with the user's productivity stack (Google Calendar, Gmail, Notion), and proactively re-engages users through scheduled notifications. The agent implements a ReAct-style tool-use loop powered by large language models, with session persistence across restarts, multimodal input support (text + images), and a modular 29-tool architecture spanning web search, scheduling, email, calendar, task management, file system operations, structured data processing, and chart generation.

The system's central contribution is its notification scheduling layer. Rather than relying on fixed time horizons ("remind me in 2 hours"), ProactiveClaw learns optimal notification timing through a multi-armed bandit (MAB) framework with 168 arms (7 days x 24 hours). Time-decayed reward signals capture when users actually respond, and these recommendations feed into the LLM alongside calendar data, email context, and task urgency — the LLM makes the final scheduling decision using all available context. This hybrid architecture combines statistical learning with contextual reasoning, enabling personalized proactive behavior without explicit user configuration.

## 1. Introduction

### 1.1 The Problem with Reactive Agents

Most conversational AI agents are purely reactive: they respond when spoken to and go silent when the user leaves. This creates a gap between the agent's capabilities and the user's needs. A user researching a topic may forget to follow up. A task discussed on Monday may slip through the cracks by Wednesday. An important email mentioned in passing may go unread.

Proactive agents — those that initiate contact — address this gap, but introduce a new problem: **when** should they reach out? Too often, and the user disables notifications. Too rarely, and the follow-ups lose relevance. At the wrong time, and the message goes unseen.

### 1.2 Our Approach

ProactiveClaw addresses this with a three-layer architecture:

1. **Persistent conversational agent** — maintains full conversation history across sessions, uses 29 tools to interact with the user's productivity stack, and supports multimodal input
2. **LLM-driven pre-exit flow** — when the user goes idle, the agent autonomously reviews the conversation, checks calendar/email/tasks for context, and decides whether and when to schedule follow-up notifications
3. **Adaptive scheduling via multi-armed bandit** — learns from implicit user feedback (response times, organic engagement patterns) to recommend optimal notification times, which the LLM incorporates alongside urgency, calendar conflicts, and conversational context

The key insight is that neither statistical learning nor LLM reasoning alone produces good scheduling. The MAB can track that "Wednesday 2 PM has an 85% response rate" but cannot reason about a deadline at 3 PM. The LLM can reason about deadlines but has no memory across sessions about when the user typically responds. Together, they cover each other's blind spots.

## 2. System Architecture

### 2.1 Overview

ProactiveClaw operates in two modes:

**CLI Mode**: The user runs a terminal-based chat interface (`main.py`). A separate scheduler process (`scheduler.py`) polls a notification queue and delivers due reminders to the terminal.

**Slack Mode**: A FastAPI server (`slack_server.py`) receives messages via the Slack Events API, runs the agent, and responds via DM. The same server handles notification queue polling and delivery as Slack DMs. An idle monitor tracks inactivity and triggers the pre-exit flow.

```
┌──────────────┐     ┌───────────────────────────────────────────────────┐
│  User (Slack │     │  slack_server.py                                 │
│  or CLI)     │────▶│                                                   │
└──────────────┘     │  ┌─────────┐   ┌──────────┐   ┌───────────────┐ │
                     │  │ Event   │──▶│ Agent    │──▶│ Tool dispatch │ │
                     │  │ handler │   │ (ReAct)  │   │ (29 tools)    │ │
                     │  └─────────┘   └────┬─────┘   └───────────────┘ │
                     │                     │ idle timeout               │
                     │                     ▼                            │
                     │  ┌──────────────────────────────────┐            │
                     │  │ Pre-exit flow                    │            │
                     │  │  1. bandit.get_recommendations() │            │
                     │  │  2. LLM checks calendar/email    │            │
                     │  │  3. LLM schedules notifications  │            │
                     │  └──────────────┬───────────────────┘            │
                     │                 ▼                                │
                     │  ┌──────────────────┐   ┌────────────────────┐  │
                     │  │ queue.json       │──▶│ poll_queue()       │  │
                     │  │ (notification Q) │   │ delivers as DMs    │  │
                     │  └──────────────────┘   └────────────────────┘  │
                     │                                                   │
                     │  ┌──────────────────┐                            │
                     │  │ bandit_state.json │ ◀── reward signals        │
                     │  │ (MAB Q-values)   │                            │
                     │  └──────────────────┘                            │
                     └───────────────────────────────────────────────────┘
```

### 2.2 Agent Core

The `Agent` class wraps an OpenAI-compatible LLM (currently GPT-5-mini) in a ReAct loop:

1. User message is appended to conversation history
2. LLM generates a response, potentially including tool calls
3. Tool calls are dispatched and results appended to history
4. Loop repeats until the LLM produces a text-only response
5. Conversation is serialized and saved to disk (`sessions/<id>.json`)

The agent supports multimodal input — images are base64-encoded and passed inline using the vision API format. MIME types are auto-detected from magic bytes.

Session persistence is straightforward JSON serialization. OpenAI message objects are converted to plain dicts, preserving tool call metadata. On resume, the full conversation history is restored, giving the agent complete context of prior interactions.

### 2.3 Tool Architecture

The agent has access to 29 tools organized into 10 modules:

| Module | Tools | Purpose |
|--------|-------|---------|
| `web.py` | 1 | Tavily web search |
| `scheduling.py` | 2 | Notification scheduling + current datetime |
| `calendar.py` | 4 | Google Calendar CRUD (list, create, update, delete events) |
| `gmail.py` | 4 | Gmail operations (list, read, send, reply) |
| `notion.py` | 6 | Notion workspace (search, read, create, update pages; query databases; create entries) |
| `filesystem.py` | 7 | Sandboxed file system (list, read, write, mkdir, delete, move, search) |
| `data.py` | 4 | Structured data — JSON and CSV read/write with pagination |
| `charts.py` | 1 | Matplotlib chart generation |

All tools are defined as OpenAI function-calling schemas in a `TOOLS_SCHEMA` list and dispatched via a unified `dispatch_tool_call()` function. The modular design allows adding new tool modules without modifying the agent core.

The Google integrations (Calendar, Gmail) share an OAuth flow (`_google_auth.py`) that handles token refresh and credential management. The file system tools are sandboxed to an `agent_file_system/` directory, preventing the agent from accessing files outside its designated space.

The scheduling tool (`schedule_notifications`) writes to a JSON queue file (`queue.json`) that a background poller reads from. Each notification entry includes an ISO 8601 timestamp, a message, and a session ID. The poller checks every 30 seconds and delivers due notifications via the appropriate channel (terminal or Slack DM).

### 2.4 Slack Integration

The Slack server handles several concerns:

- **Signature verification**: Every incoming request is validated against Slack's signing secret using HMAC-SHA256
- **Event deduplication**: Slack may retry event deliveries; a set of seen event IDs prevents duplicate processing
- **Concurrency control**: An asyncio lock serializes agent calls to prevent session file races
- **File handling**: Image attachments are downloaded via Slack's API and passed to the agent as multimodal input
- **Long message splitting**: Responses exceeding Slack's ~4000 character limit are split into chunks
- **File uploads**: When the agent creates files (charts, CSVs, etc.), they're uploaded to the Slack conversation

### 2.5 Idle Timeout and Pre-Exit Flow

The idle monitor runs as a background asyncio task. It tracks the timestamp of the last user message and triggers the pre-exit flow when the idle duration exceeds `AGENT_TIMEOUT` (configurable, default 300 seconds).

The pre-exit flow is where proactive behavior happens:

1. The bandit generates top-K time slot recommendations based on historical engagement
2. These recommendations are formatted and injected into the LLM's pre-exit prompt
3. The prompt instructs the LLM to:
   - Review the conversation for follow-up-worthy topics
   - **Check Google Calendar** for upcoming events, meetings, and busy times
   - **Check Gmail** for relevant recent emails that may inform urgency
   - **Check Notion** for tasks, notes, or deadlines tied to what was discussed
   - Use all of this context — plus bandit recommendations, task urgency, and its own judgment — to decide whether and when to schedule notifications
4. The LLM calls `schedule_notifications` with 0-5 notifications (or decides none are warranted)
5. The agent enters a "sleeping" state — subsequent user messages wake it up and clear pending notifications

The LLM is explicitly instructed that bandit recommendations are **one signal among many**. It should override them for urgent deadlines, avoid scheduling during calendar-blocked time, and not schedule at unreasonable hours regardless of what the scores suggest. When no bandit data exists (cold start), the LLM falls back to general time horizon guidelines.

## 3. Adaptive Scheduling via Multi-Armed Bandit

### 3.1 Problem Formulation

We model the "when to notify" problem as a 168-arm bandit. Each arm corresponds to a (day_of_week, hour) tuple — Monday 7 AM, Friday 10 PM, etc. The agent's goal is to learn which arms yield the highest expected reward, defined as user responsiveness.

### 3.2 Reward Signals

Two implicit feedback signals drive learning:

- **Notification response (reward = 1.0)**: The user responds to a delivered notification within a 10-minute window. This is the strongest signal — the notification arrived at a time when the user was available and chose to engage.

- **Organic engagement (reward = 0.5)**: The user initiates a conversation without being prompted. A weaker but still informative signal — the user was available at this time, even if a notification didn't cause it.

The asymmetric rewards reflect signal strength. A notification response directly validates the time slot. Organic engagement is confounded — the user may have been available but not particularly receptive to proactive outreach.

Both signals are captured in the Slack server's message handler. When a message arrives while the agent is sleeping and a notification was sent within the last 10 minutes, the notification's time slot gets reward 1.0. When a message arrives while the agent is awake (organic chat), the current time slot gets reward 0.5.

### 3.3 Time-Decayed Update Rule

User schedules are non-stationary. Someone who responds at Wednesday 2 PM reliably for three months may start a new job and shift to Thursday mornings. A standard sample-mean estimator would be slow to adapt because all historical data carries equal weight.

We use exponential time decay:

```
time_since = (now - last_updated[arm]) in days
decay = exp(-lambda * time_since)
Q[arm] = decay * Q_old + (1 - decay) * reward
```

Where lambda is a tunable decay parameter (default: 0.1).

**Decay properties at lambda = 0.1:**

| Days since last update | Decay factor | Effect |
|------------------------|-------------|--------|
| 1 | 0.90 | Old value retains 90% weight |
| 7 | 0.50 | Roughly equal weighting |
| 10 | 0.37 | New reward dominates |
| 23 | 0.10 | Old value nearly erased |

This ensures stale arms fade naturally toward 0 without explicit pruning or windowing.

### 3.4 Recommendation Generation

At decision time (pre-exit), we compute each arm's effective Q-value by applying decay from its last update to the current moment:

```
effective_Q[arm] = Q[arm] * exp(-lambda * (now - last_updated[arm]))
```

Arms below 0.01 are filtered out. The rest are sorted by effective Q-value and the top-K (default 10) are returned. Each recommendation includes day name, hour in 12-hour format, and score.

### 3.5 LLM Integration

Recommendations are injected into the pre-exit prompt as structured text:

```
## User Availability Predictions (based on past engagement)
The following time slots have the highest probability of user response.
Schedule notifications at or near these times when possible.
Only schedule forward chronologically from the current time.

1. Wednesday 2:00 PM (score: 0.85)
2. Thursday 10:00 AM (score: 0.68)
3. Friday 9:00 AM (score: 0.50)

These are recommendations — adjust based on task urgency and calendar availability.
For urgent deadlines, schedule more aggressively regardless of scores.
For casual/habit tasks, use only the top 1-2 slots.
```

The LLM then uses these alongside:
- **Google Calendar data** (is the user in a meeting? when is their next free slot?)
- **Gmail context** (is there an urgent email related to the conversation?)
- **Notion tasks** (are there deadlines or priorities relevant to the discussion?)
- **Conversation context** (was this a deep research session or a casual question?)
- **Common sense** (don't notify at 3 AM, cluster before deadlines, etc.)

This is the key architectural decision: the bandit provides **data**, the LLM provides **judgment**. Neither alone produces good scheduling.

### 3.6 Cold Start

When `bandit_state.json` doesn't exist, `get_recommendations()` returns an empty list. The `format_recommendations()` function returns an empty string, the prompt placeholder resolves to nothing, and the LLM falls back to general time horizon guidelines baked into the prompt:

1. Short-term (15min-2hrs): resume nudge or topic follow-up
2. Medium-term (3-6hrs): check for updates or new information
3. Long-term (8-24hrs): end-of-day recap or next-morning reminder

As the user interacts over days, the bandit accumulates signal and recommendations start appearing in the prompt. The transition is seamless and requires zero configuration.

### 3.7 State Persistence

Bandit state is stored as a single JSON file (`bandit_state.json`, gitignored):

```json
{
  "q_values": {"1_14": 0.85, "3_10": 0.68},
  "last_updated": {"1_14": "2026-02-14T14:30:00-08:00", "3_10": "2026-02-13T10:15:00-08:00"},
  "lambda": 0.1
}
```

Keys follow the format `"{day}_{hour}"` (e.g., `"1_14"` = Tuesday 2 PM). State is loaded on demand and survives server restarts. The entire module is ~120 lines of Python with no external dependencies beyond the standard library.

## 4. Design Decisions

### 4.1 Why a Persistent ReAct Agent?

Stateless agents (single request-response) can't build context over a conversation. But a conversation about "research X, schedule a meeting about it, draft an email to the team" requires multi-turn context. Session persistence via JSON serialization gives the agent full history across reconnects without requiring a database.

The ReAct pattern (Reason + Act in a loop) is critical for the pre-exit flow: the agent needs to call multiple tools (check calendar, check email, schedule notifications) in a sequence that depends on intermediate results. A single-shot prompt can't do this.

### 4.2 Why 29 Tools?

The tool count reflects the agent's proactive needs. Scheduling a good notification requires context: Is the user free at 3 PM? Did they get an email about the topic? Is there a Notion task with a deadline? Without calendar/email/task integration, the agent is scheduling blind.

The file system and data tools serve a different purpose — they give the agent persistent memory beyond conversation history. The agent can write notes, store research, generate charts, and reference them in later sessions.

### 4.3 Why Multi-Armed Bandit for Scheduling?

We considered several alternatives:

- **Collaborative filtering**: Requires multiple users; ProactiveClaw is single-user
- **Time series forecasting**: Requires significant history and is prone to overfitting on sparse data
- **Full RL (MDP)**: Overly complex — notification scheduling is essentially stateless (the optimal time doesn't depend on which notifications were sent previously)
- **Simple heuristics**: Cannot adapt to individual patterns
- **Just let the LLM decide**: LLMs have no memory across sessions; they can't track that "Wednesday 2 PM has an 85% response rate over 3 weeks"

The MAB provides the right balance: learns from sparse feedback, handles non-stationarity via decay, has negligible computational overhead, and requires no external dependencies.

### 4.4 Why Time Decay Instead of Sliding Window?

A sliding window (e.g., "only count last 30 days") creates a hard boundary where all older data is discarded at once. Exponential decay provides a smooth transition. This matters with sparse data — if a user chats 3 times per week, a 14-day window contains only 6 data points. Decay preserves all of them with appropriate weighting.

### 4.5 Why LLM-in-the-Loop?

Pure MAB scheduling would ignore context entirely. A user with a report due at 3 PM needs a reminder before 3 PM — regardless of whether Thursday 3 PM historically has a low response rate. Calendar conflicts, task urgency, email context, and conversational nuance all matter. The LLM can reason about all of these while using MAB scores as empirical grounding.

The prompt explicitly instructs the LLM to use tools (list calendar events, check emails, query Notion) before scheduling. This makes the pre-exit flow a genuine multi-signal decision, not a blind schedule.

## 5. Session Lifecycle

A complete session lifecycle in Slack mode:

1. **User sends a message** → Slack Events API delivers it to `/slack/events`
2. **Organic reward recorded** → `bandit.update_arm(weekday, hour, 0.5)` captures the engagement timestamp
3. **Agent processes message** → ReAct loop with tool calls, response posted as DM
4. **Conversation continues** → each message resets the idle timer and records organic rewards
5. **User goes idle** → after `AGENT_TIMEOUT` seconds:
   a. Bandit generates top-K recommendations
   b. LLM checks calendar, email, Notion for context
   c. LLM schedules 0-5 notifications using all signals
   d. Agent enters sleeping state
6. **Notification fires** → `poll_queue()` delivers the DM, records the delivery time and arm
7. **User responds within 10 min** → `bandit.update_arm(arm, 1.0)` records the notification response reward
8. **Agent wakes up** → pending notifications cleared, conversation resumes with full history

If the user responds after 10 minutes (or doesn't respond at all), no notification response reward is recorded — only the organic engagement reward when they eventually return.

## 6. Limitations and Future Work

### Agent Core
- **Single LLM provider**: Currently hardcoded to OpenAI. An abstraction layer would enable provider switching.
- **No streaming**: Responses are generated in full before delivery. Streaming would improve perceived latency for long responses.
- **Session bloat**: Conversation history grows unbounded. A summarization or context-window management strategy would help for very long-running sessions.

### Tool System
- **No parallel tool execution**: Tool calls within a single ReAct step are executed sequentially. Parallel execution would speed up multi-tool steps.
- **No tool-use feedback**: The agent doesn't learn which tools are most useful. Tool-use patterns could inform future behavior.

### Notification Scheduling
- **Single reward signal per notification**: We only track whether the user responded within 10 minutes. Response quality, conversation length, or engagement depth could provide richer signals.
- **No exploration strategy**: We rely on organic engagement for exploration of new time slots. An epsilon-greedy or UCB strategy could actively probe undersampled arms.
- **Hour-level granularity**: Some users have finer patterns (e.g., 12:00-12:30 lunch break). Sub-hour arms would require more data but capture these.
- **No contextual bandits**: Arm selection doesn't condition on features like "day after holiday" or "user's calendar is empty." Contextual bandits could capture these richer patterns.
- **CLI mode doesn't track rewards**: The bandit integration is currently Slack-only. Extending to CLI mode would require a similar message handler.

## 7. Conclusion

ProactiveClaw demonstrates that proactive AI agents don't need complex infrastructure. Session persistence is JSON files. Notification scheduling is a queue polled every 30 seconds. Adaptive timing is a 120-line bandit with a JSON state file. The heavy lifting — contextual reasoning about urgency, calendar conflicts, and conversational relevance — is delegated to the LLM through carefully designed prompts.

The hybrid MAB + LLM scheduling architecture is the system's core contribution. By separating empirical learning (when does the user respond?) from contextual reasoning (should we notify now given the deadline/calendar/conversation?), we get the best of both worlds: data-driven timing that adapts to changing routines, and context-aware judgment that handles edge cases the data can't predict.

The system is fully backward-compatible: a fresh install with no bandit data behaves identically to a version without the bandit. As data accumulates, recommendations gradually appear and improve scheduling quality — all without requiring any user configuration or explicit feedback.
