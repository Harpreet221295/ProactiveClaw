# ProactiveClaw

A long-horizon persistent ReACT agent that remembers conversations, times out on idle, and proactively schedules follow-up notifications to re-engage the user.

## How It Works

**CLI mode:**
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
                         │  (terminal)  │
                         └─────────────┘
```

**Slack mode:**
```
┌─────────────────────────────────────┐
│  slack_server.py (always running)   │
│                                     │
│  POST /slack/events                 │
│    → receives user DM               │
│    → runs agent.run(msg)            │
│    → posts response back via DM     │
│                                     │
│  Background: poll_queue()           │
│    → polls queue.json every 30s     │
│    → sends due notifications as DMs │
│                                     │
│  Exposed via ngrok                  │
└─────────────────────────────────────┘
```

1. **User starts `main.py`** — picks an existing session or creates a new one
2. **User chats** — the agent uses web search (Tavily) to answer questions; conversation is saved to disk after every turn
3. **User goes idle** — after `AGENT_TIMEOUT` seconds of no input, the agent runs a **pre-exit flow** where the LLM generates 5 scheduled notifications at different time horizons
4. **Scheduler fires notifications** — `scheduler.py` polls `queue.json` every 30s and prints notifications when their timestamps are due
5. **User returns** — runs `main.py` again, resumes the session with full conversation history, and pending queue entries for that session are cleared

## Project Structure

```
ProactiveClaw/
├── .env                # API keys + AGENT_TIMEOUT + Slack credentials
├── requirements.txt    # Python dependencies
├── prompts.py          # System prompt + pre-exit prompt
├── tools/              # Tool definitions — modular package
│   ├── __init__.py     # Aggregates all tools, re-exports public interface
│   ├── _state.py       # Shared state (session ID, queue file path, FS base)
│   ├── _google_auth.py # Google OAuth (shared by calendar + gmail)
│   ├── web.py          # tavily_search (1 tool)
│   ├── scheduling.py   # schedule_notifications, get_current_datetime (2 tools)
│   ├── calendar.py     # Google Calendar CRUD (4 tools)
│   ├── gmail.py        # Gmail read/send/reply (4 tools)
│   ├── notion.py       # Notion search/read/create/update/query (6 tools)
│   ├── filesystem.py   # File system ops + search (7 tools)
│   ├── data.py         # Structured data — JSON & CSV read/write (4 tools)
│   └── charts.py       # generate_chart via matplotlib (1 tool)
├── agent.py            # Agent class with persistence + pre-exit flow
├── main.py             # CLI entrypoint with timeout + session management
├── scheduler.py        # FastAPI server + background queue poller (CLI mode)
├── slack_server.py     # Slack bot server + notification delivery (Slack mode)
├── sessions/           # Created at runtime — stores conversation JSON files
└── queue.json          # Created at runtime — notification queue
```

### File Breakdown

| File | Purpose |
|------|---------|
| **`prompts.py`** | Contains `SYSTEM_PROMPT` (agent behavior rules) and `PRE_EXIT_PROMPT` (instructs the LLM to generate 5 timezone-aware notifications across different time horizons) |
| **`tools/`** | Modular package with 29 tools across 10 modules. Includes web search, scheduling, Google Calendar, Gmail, Notion, file system operations (with search and pagination for large files), structured data (JSON/CSV with row-based pagination), and chart generation. Re-exports `TOOLS_SCHEMA`, `dispatch_tool_call`, `set_current_session_id`, and `QUEUE_FILE` from `__init__.py` |
| **`agent.py`** | The `Agent` class — manages conversation history (text + images), serializes OpenAI message objects to JSON for persistence, saves/loads sessions from disk, and runs the pre-exit flow on timeout |
| **`main.py`** | CLI entrypoint — uses `select.select()` for input with timeout (macOS/Linux), manages session selection, clears stale queue entries on resume |
| **`scheduler.py`** | FastAPI app that runs independently — polls `queue.json` every 30s and prints due notifications to the terminal. Used in CLI mode only |
| **`slack_server.py`** | Slack bot server — receives DMs via Slack Events API, runs the agent, posts responses back as DMs. Also polls `queue.json` and delivers due notifications as Slack DMs |

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

# Slack mode (optional — only needed if using Slack interface)
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_USER_ID=U...
```

- `OPENAI_API_KEY` — your OpenAI API key (uses GPT-4o)
- `TAVILY_API_KEY` — your [Tavily](https://tavily.com/) API key for web search
- `AGENT_TIMEOUT` — seconds of idle time before the agent auto-exits (default: 300 = 5 minutes)
- `SLACK_BOT_TOKEN` — Bot User OAuth Token from your Slack app (starts with `xoxb-`)
- `SLACK_SIGNING_SECRET` — Signing Secret from your Slack app's Basic Information page
- `SLACK_USER_ID` — your Slack user ID (find it in your Slack profile → three dots → Copy member ID)

## Running

There are two modes: **CLI mode** (local terminal) and **Slack mode** (DM a Slack bot). Pick one.

### CLI Mode

You need **two terminals**:

**Terminal 1 — Start the scheduler (keep running):**

```bash
python scheduler.py
```

This starts a FastAPI server on port 8000 and begins polling `queue.json` every 30 seconds.

**Terminal 2 — Start the agent CLI:**

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

### Sending Images (CLI)

Use `/image` or `/img` followed by a file path:

```
You: /image /path/to/screenshot.png
Caption (or press Enter): What's wrong with this error message?

Agent: The error is a NullPointerException on line 42...
```

If you press Enter without a caption, it defaults to "What's in this image?". Supports PNG, JPEG, GIF, and WebP.

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

### Slack Mode

You need **two terminals** and a Slack app:

**Terminal 1 — Start the Slack server:**

```bash
python slack_server.py
```

**Terminal 2 — Expose via ngrok:**

```bash
ngrok http 8000
```

Copy the HTTPS URL from ngrok (e.g. `https://abc123.ngrok.io`) and set it as your Slack app's Event Subscriptions Request URL: `https://abc123.ngrok.io/slack/events`

Then just DM the bot in Slack. All conversations share a single persistent session (`slack_<your_user_id>`), and scheduled notifications are delivered as Slack DMs. You can also send images — just attach a photo to your message and the agent will see it.

#### Slack App Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. **OAuth & Permissions** — add Bot Token Scopes: `chat:write`, `im:history`, `im:read`, `im:write`, `files:read`
3. **Event Subscriptions** — enable events, set Request URL to your ngrok HTTPS URL + `/slack/events`
4. **Subscribe to bot events** — add `message.im`
5. **Install to workspace** — copy the Bot User OAuth Token and Signing Secret into your `.env`

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

This endpoint is also available on `slack_server.py` for compatibility.

## Quick Test

### CLI mode (low timeout)

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

### Slack mode

1. Start `python slack_server.py` + `ngrok http 8000`
2. Configure the ngrok URL in Slack Event Subscriptions
3. DM the bot "hello" — you should get a response
4. Send an image with a question — the agent should describe/analyze it
5. Ask the agent to schedule a reminder for 1 minute from now — it should arrive as a Slack DM
