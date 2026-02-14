# ProactiveClaw

A long-horizon persistent ReACT agent that remembers conversations, times out on idle, and proactively schedules follow-up notifications to re-engage the user.

## How It Works

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   main.py   │──save──▶│  sessions/   │◀──load──│   main.py   │
│  (agent CLI)│         │  <id>.json   │         │  (resumed)  │
└──────┬──────┘         └──────────────┘         └─────────────┘
       │ timeout
       ▼
┌──────────────┐        ┌──────────────┐
│  pre-exit    │──write─▶│  queue.json  │
│  flow (LLM)  │        └──────┬───────┘
└──────────────┘               │ poll every 30s
                         ┌─────▼───────┐
                         │ scheduler.py │
                         │  (FastAPI)   │
                         └─────────────┘
```

1. **User starts `main.py`** — picks an existing session or creates a new one
2. **User chats** — the agent uses web search (Tavily) to answer questions; conversation is saved to disk after every turn
3. **User goes idle** — after `AGENT_TIMEOUT` seconds of no input, the agent runs a **pre-exit flow** where the LLM generates 5 scheduled notifications at different time horizons
4. **Scheduler fires notifications** — `scheduler.py` polls `queue.json` every 30s and prints notifications when their timestamps are due
5. **User returns** — runs `main.py` again, resumes the session with full conversation history, and pending queue entries for that session are cleared

## Project Structure

```
ProactiveClaw/
├── .env                # API keys + AGENT_TIMEOUT setting
├── requirements.txt    # Python dependencies
├── prompts.py          # System prompt + pre-exit prompt
├── tools.py            # Tool definitions (web search, schedule notifications)
├── agent.py            # Agent class with persistence + pre-exit flow
├── main.py             # CLI entrypoint with timeout + session management
├── scheduler.py        # FastAPI server + background queue poller
├── sessions/           # Created at runtime — stores conversation JSON files
└── queue.json          # Created at runtime — notification queue
```

### File Breakdown

| File | Purpose |
|------|---------|
| **`prompts.py`** | Contains `SYSTEM_PROMPT` (agent behavior rules) and `PRE_EXIT_PROMPT` (instructs the LLM to generate 5 timezone-aware notifications across different time horizons) |
| **`tools.py`** | Defines two tools the LLM can call: `tavily_search` for web lookups and `schedule_notifications` for writing entries to `queue.json`. Also provides `set_current_session_id()` so notifications are tagged with the correct session |
| **`agent.py`** | The `Agent` class — manages conversation history, serializes OpenAI message objects to JSON for persistence, saves/loads sessions from disk, and runs the pre-exit flow on timeout |
| **`main.py`** | CLI entrypoint — uses `select.select()` for input with timeout (macOS/Linux), manages session selection, clears stale queue entries on resume |
| **`scheduler.py`** | FastAPI app that runs independently — polls `queue.json` every 30s and prints due notifications. Also exposes `POST /notify` for future integrations (Slack, WhatsApp, etc.) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
AGENT_TIMEOUT=300
```

- `OPENAI_API_KEY` — your OpenAI API key (uses GPT-4o)
- `TAVILY_API_KEY` — your [Tavily](https://tavily.com/) API key for web search
- `AGENT_TIMEOUT` — seconds of idle time before the agent auto-exits (default: 300 = 5 minutes)

## Running

You need **two terminals**:

### Terminal 1 — Start the scheduler (keep running)

```bash
python scheduler.py
```

This starts a FastAPI server on port 8000 and begins polling `queue.json` every 30 seconds.

### Terminal 2 — Start the agent CLI

```bash
python main.py
```

On first run, a new session is created automatically. On subsequent runs, you'll see a menu:

```
Existing sessions:
  1. a3f8b2c1
  2. 7e9d4f06
  3. Start new session

Pick a session number:
```

### Chatting

```
You: What's the latest news about SpaceX?

  [tool] tavily_search({"query": "latest SpaceX news"})

Agent: SpaceX successfully launched...
```

### Timeout & Notifications

If you stop typing for `AGENT_TIMEOUT` seconds:

```
[timeout] Session idle — running pre-exit flow...
  [tool] schedule_notifications({"notifications": [...]})

Agent: I've scheduled some follow-ups. See you later!

[session saved — exiting due to inactivity]
```

Over in Terminal 1 (scheduler), you'll see notifications fire at their scheduled times:

```
[notification] (session a3f8b2c1) Hey! We were looking into SpaceX — ready to pick back up?
[notification] (session a3f8b2c1) The launch results you asked about may have updated by now.
```

### Resuming a Session

Just run `python main.py` again, pick the session, and the full conversation history is restored. Any pending queue entries for that session are cleared so you don't get stale notifications.

## Notification Time Horizons

When the agent times out, the LLM schedules exactly 5 notifications:

| # | Window | Purpose |
|---|--------|---------|
| 1 | 15–30 min | "Welcome back" nudge summarizing where you left off |
| 2 | 1–2 hours | Follow-up on the most recent topic |
| 3 | 3–6 hours | Check for updates on something you researched |
| 4 | 8–12 hours | End-of-day or next-morning recap |
| 5 | 18–24 hours | Next-day reminder or general re-engagement |

All timestamps are generated with your local timezone offset (e.g. `-08:00` for PST) so they fire at the correct time regardless of timezone.

## API Endpoint

The scheduler also exposes a REST endpoint for external integrations:

```bash
curl -X POST http://localhost:8000/notify \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-06-15T14:00:00-08:00",
    "message": "Check on your research results",
    "session_id": "a3f8b2c1"
  }'
```

This can be used to build Slack, WhatsApp, or email notification integrations in the future.

## Quick Test (low timeout)

To test the full flow quickly, set a short timeout:

```bash
# In .env, set:
AGENT_TIMEOUT=30

# Terminal 1:
python scheduler.py

# Terminal 2:
python main.py
# Ask a question, then wait 30 seconds
# Watch the pre-exit flow trigger and notifications appear in Terminal 1
```
