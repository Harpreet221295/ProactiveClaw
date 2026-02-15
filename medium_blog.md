# I Built an AI Agent That Doesn't Wait for You to Come Back — It Comes to You

## The Problem: AI Agents Are Reactive

Every AI chatbot I've used has the same flaw: you close the tab and it forgets you exist. You were mid-research on a topic, got distracted, and three days later you can't even remember what you were looking into. The agent had the context. It could have nudged you. But it just sat there, waiting.

I wanted an agent that doesn't just answer questions — one that remembers conversations, connects to my actual productivity tools, and proactively follows up when I drop the ball.

So I built **ProactiveClaw**.

## What Is ProactiveClaw?

It's a persistent conversational agent that lives in Slack (or runs in a terminal). You chat with it like any other AI assistant, but with three key differences:

1. **It remembers everything.** Full conversation history is saved to disk. Close Slack, come back tomorrow, pick up exactly where you left off.

2. **It's plugged into your life.** Google Calendar, Gmail, Notion, web search, a sandboxed file system for its own notes, chart generation — 29 tools across 10 modules. When you say "schedule a meeting about this," it actually creates the calendar event.

3. **It follows up on its own.** When you stop chatting, it doesn't just go idle. It reviews the conversation, checks your calendar for context, looks at your email, and decides whether to ping you later. Then it schedules notifications at times you're most likely to respond.

That last part — the proactive follow-up — is what makes it different from everything else I've used. And the "times you're most likely to respond" part is where it gets interesting.

## The Architecture

### How a Conversation Works

```
You DM the bot in Slack
  → Slack Events API delivers the message
  → Agent runs a ReAct loop (think → use tools → think → respond)
  → Response posted back as a DM
  → Conversation saved to disk
```

The agent uses OpenAI's function calling under the hood. When you ask "what's on my calendar tomorrow?", it calls `list_calendar_events`. When you say "look up the latest on SpaceX", it calls `tavily_search`. It chains tool calls until it has enough information to answer.

Images work too — attach a screenshot to your Slack message and the agent sees it via the vision API. It auto-detects MIME types from magic bytes and base64-encodes on the fly.

### The Tool Stack

The agent isn't just a chatbot with web search bolted on. It has deep integrations:

- **Google Calendar** — list, create, update, delete events. The agent uses this proactively during scheduling to avoid booking notifications during your meetings.
- **Gmail** — list, read, send, reply to emails. It never sends email without your explicit request, but it reads proactively when relevant ("you have 3 unread emails about that project").
- **Notion** — search pages/databases, read content, create pages, query databases, add entries. Your task lists and notes are accessible.
- **File system** — a sandboxed directory where the agent stores its own notes, drafts, and data between sessions. It can write JSON, CSV, search through files, generate matplotlib charts.
- **Web search** — Tavily-powered search with up to 5 results per query.

All 29 tools are defined as OpenAI function-calling schemas and dispatched through a unified handler. Adding a new tool module is straightforward — define the schema, write the function, register it.

### Session Persistence

Every conversation turn is serialized to a JSON file in `sessions/`. OpenAI message objects get converted to plain dicts, preserving tool call metadata. When you reconnect, the full history loads back in.

In Slack mode, all conversations share a single persistent session tied to your user ID. In CLI mode, you pick from existing sessions or start a new one.

This matters because the agent's pre-exit flow needs conversation context to decide what's worth following up on. Without persistence, every session starts cold.

## The Pre-Exit Flow: Where Proactiveness Happens

This is the core of ProactiveClaw. Here's what happens when you stop chatting:

1. An idle monitor tracks your last message timestamp
2. After `AGENT_TIMEOUT` seconds (default: 5 minutes), it triggers the pre-exit flow
3. The agent runs a full ReAct loop with a special prompt that says:

> "The user has gone idle. Review the conversation. Check their Google Calendar for upcoming events. Check Gmail for relevant emails. Check Notion for tasks and deadlines. Decide whether follow-up notifications would be useful. If so, schedule 1-5 of them."

The key word is **decide**. The agent doesn't blindly schedule 5 notifications every time. If the conversation was casual and there's nothing to follow up on, it says goodbye and schedules nothing. If you were deep into a research topic with a deadline tomorrow, it might schedule a reminder tonight and another one tomorrow morning.

After scheduling, the agent enters a "sleeping" state. When you message it again, it wakes up, clears any pending notifications that are no longer relevant, and resumes the conversation.

### But *When* Should It Notify?

This was the hard part. My first version used fixed time horizons:
- 15-30 minutes: "Welcome back" nudge
- 1-2 hours: Follow-up on the most recent topic
- 3-6 hours: Check for updates
- 8-12 hours: End-of-day recap
- 18-24 hours: Next-day reminder

It worked, but it was dumb. It had no idea when I actually check Slack. The 2 PM notification got instant replies. The 8 AM Saturday notification sat unread for 12 hours.

The agent had the data — it knew when I responded — but it wasn't learning from it.

## Enter the Multi-Armed Bandit

### The Idea

Imagine 168 slot machines — one for each hour of the week. Every time you send a notification at a particular time, you're "pulling" that machine's lever. If the user responds, it pays out. Over time, you learn which machines are worth pulling.

This is a textbook [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem. And for this use case, it's the perfect fit:
- Sparse feedback (a few interactions per day)
- Non-stationary environment (schedules change)
- Single user (no collaborative filtering needed)
- No complex state (the optimal time doesn't depend on previous notifications)

### Two Reward Signals

**1. Notification response (reward = 1.0)**

You respond to a notification within 10 minutes. This is the gold standard — you were available, the timing was right, and you chose to engage.

**2. Organic chat (reward = 0.5)**

You start a conversation on your own, unprompted. The agent was awake and you messaged it. A weaker signal — you were available, but the notification didn't cause it. Still valuable for mapping your active hours.

Both signals are captured automatically. No surveys. No configuration. Just behavior.

### Time-Decayed Updates

People's schedules change. I used to be a morning person on weekdays; now I'm not. A standard running average would take weeks to catch up because all historical data weighs equally.

Instead, I use exponential time decay:

```python
time_since = (now - last_updated[arm]) in days
decay = exp(-lambda * time_since)
Q[arm] = decay * Q_old + (1 - decay) * reward
```

With lambda = 0.1:
- Yesterday's data retains 90% weight
- 10-day-old data retains 37%
- 23-day-old data retains 10%

Stale arms fade to zero on their own. If I stop responding at 3 PM on Wednesdays, that arm decays without me doing anything.

### But Here's the Thing — The Bandit Doesn't Schedule

This is the most important design decision: **the bandit provides recommendations, not decisions.**

It generates a ranked list:

```
1. Wednesday 2:00 PM (score: 0.85)
2. Thursday 10:00 AM (score: 0.68)
3. Friday 9:00 AM (score: 0.50)
```

This gets injected into the LLM's prompt alongside explicit instructions:

> "These are recommendations — adjust based on task urgency and calendar availability. For urgent deadlines, schedule more aggressively regardless of scores. For casual tasks, use only the top 1-2 slots."

The LLM then does what it does best — it **reasons**:

- "The user has a meeting from 2-3 PM on Wednesday, so I'll schedule for 3:15 PM instead"
- "There's a report due Thursday at noon, so I'll add a reminder Thursday morning even though it's not in the top slots"
- "This was a casual conversation, no need for more than one follow-up"
- "It's 11 PM — I'm not going to schedule a notification for midnight regardless of the score"

The bandit provides **data**. The LLM provides **judgment**. Calendar, email, and Notion provide **context**. All three together produce better scheduling than any one alone.

### Cold Start

When there's no bandit data (fresh install), the recommendations list is empty. The prompt placeholder resolves to nothing, and the LLM falls back to general time horizon guidelines:

1. Short-term (15min-2hrs): resume nudge or topic follow-up
2. Medium-term (3-6hrs): check for updates
3. Long-term (8-24hrs): end-of-day recap or next-morning reminder

This means day-one behavior is identical to the pre-bandit version. As the user interacts over days, recommendations gradually appear. No configuration needed.

## Implementation Details

### The Bandit (~120 lines)

The entire MAB module is four functions:

```python
load_state()           # Read from bandit_state.json, defaults if missing
save_state(state)      # Write to bandit_state.json
update_arm(day, hour, reward)  # Apply time-decayed update
get_recommendations(top_k=10) # Return top arms by effective Q-value
```

State is a JSON file:

```json
{
  "q_values": {"1_14": 0.85, "3_10": 0.68},
  "last_updated": {"1_14": "2026-02-14T14:30:00-08:00"},
  "lambda": 0.1
}
```

No ML frameworks. No database. No external dependencies.

### Reward Tracking (in slack_server.py)

```python
# User responds to notification within 10 minutes
if _is_sleeping and time.time() - _last_notification_time <= 600:
    bandit.update_arm(*_last_notification_arm, reward=1.0)

# User starts an organic conversation
elif not _is_sleeping:
    bandit.update_arm(now.weekday(), now.hour, reward=0.5)
```

When a notification fires, the delivery time and arm are recorded. If the user responds within 10 minutes, that arm gets reward 1.0. Every organic message records the current time slot with reward 0.5.

### Pre-Exit Integration

```python
# In the idle monitor, right before pre-exit:
recommendations = bandit.get_recommendations()
rec_text = bandit.format_recommendations(recommendations)
agent.run_pre_exit(bandit_recommendations=rec_text)
```

The LLM then sees the recommendations in its prompt and uses them — alongside calendar lookups, email checks, and Notion queries — to make its scheduling decision.

## The Full Session Lifecycle

1. **You message the bot** → organic reward recorded for current time slot
2. **Agent responds** → tool calls as needed, conversation saved
3. **You keep chatting** → each message resets idle timer, records rewards
4. **You go idle** → after 5 minutes:
   - Bandit generates recommendations
   - LLM checks your calendar, email, Notion
   - LLM schedules 0-5 notifications using all signals
   - Agent sleeps
5. **Notification fires** → delivered as Slack DM, delivery time tracked
6. **You respond within 10 min** → notification response reward recorded
7. **Agent wakes up** → pending notifications cleared, conversation resumes

Over time, the bandit learns your patterns. The LLM gets better scheduling data. Notifications become more relevant.

## Why Not Just [Alternative]?

**"Just use cron / fixed schedules"**
Fixed schedules don't adapt. My 8 AM is different from your 8 AM. My Tuesday is different from my Saturday.

**"Let the LLM figure it out"**
LLMs don't have memory across sessions. They can't track that I've responded to Wednesday 2 PM notifications 85% of the time over the last 3 weeks. They're great at reasoning but terrible at statistics.

**"Use a neural network"**
For 168 arms with sparse, binary-ish rewards and a single user, a neural network is massive overkill. The bandit converges in days with a few interactions per day.

**"Use a sliding window instead of decay"**
Sliding windows create hard cutoffs — all data older than N days vanishes at once. Decay is smoother. If I only chat 3 times a week, a 14-day window has 6 data points. Decay preserves all of them with appropriate weighting.

**"Just ask the user when they want notifications"**
People are bad at predicting their own behavior. "I'm a morning person" says the person who responds to every 2 PM notification within a minute. Implicit feedback beats explicit preferences.

## What I Learned

**1. Proactive > Reactive, but only if the timing is right.**
The difference between a useful follow-up and an annoying notification is about 4 hours. Same message, different time, completely different user experience.

**2. Hybrid architectures beat pure ones.**
The bandit can't reason about deadlines. The LLM can't remember cross-session response patterns. Calendar data can't predict availability without historical context. All three together produce scheduling that none could achieve alone.

**3. Simple models ship.**
A 168-arm bandit in 120 lines of Python, persisted as JSON, with no dependencies. It's been running for weeks without issues. The complex ML pipeline I considered first would probably still be in a notebook somewhere.

**4. Implicit feedback is the only feedback.**
I tried adding a "was this notification useful?" button. Nobody clicks it. But "user responded within 10 minutes" is a signal you get for free, every time.

**5. Tool integration is the moat.**
The agent without calendar/email/Notion access is a chatbot with reminders. The agent *with* those tools is a genuine assistant that knows your schedule, reads your email, and follows up on your tasks. The tools are what make proactive behavior actually useful rather than just persistent.

## Try It

ProactiveClaw is open source. Set up a Slack app, add your API keys, and let it run. The first day it schedules notifications on fixed horizons. By the end of the week, it's learned your patterns and the follow-ups start landing at exactly the right time.

The code is straightforward — `agent.py` for the ReAct loop, `slack_server.py` for the Slack integration, `bandit.py` for the MAB, `prompts.py` for the prompt engineering, and `tools/` for the 29-tool modular architecture. No magic, no complex infrastructure, just an agent that actually remembers you exist after you close the tab.
