import asyncio
import hashlib
import hmac
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests as http_requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from slack_sdk import WebClient

import bandit

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_USER_ID = os.environ["SLACK_USER_ID"]
SLACK_SESSION_ID = f"slack_{SLACK_USER_ID}"

slack_client = WebClient(token=SLACK_BOT_TOKEN)

QUEUE_FILE = os.path.join(os.path.dirname(__file__), "queue.json")
REMINDERS_FILE = os.path.join(os.path.dirname(__file__), "reminders.json")
CRON_JOBS_FILE = os.path.join(os.path.dirname(__file__), "cron_jobs.json")
REENGAGEMENT_FILE = os.path.join(os.path.dirname(__file__), "reengagement.json")
REENGAGEMENT_INTERVALS = [72 * 3600, 7 * 86400, 14 * 86400, 30 * 86400]  # 72h, 1w, 2w, 1mo
POLL_INTERVAL = 30
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))

# Deduplication: track seen event IDs
_seen_events: set[str] = set()
_MAX_SEEN = 1000

# Serialize agent calls so session file doesn't race
_agent_lock = asyncio.Lock()

# Idle timeout tracking
_last_message_time: float = 0.0
_is_sleeping: bool = False

# Bandit reward tracking
_last_notification_time: float = 0.0
_last_notification_arm: tuple[int, int] | None = None

# Previous invocation queue — stashed on wake so next pre-exit has context
_previous_invocations: list[dict] = []


# ---------------------------------------------------------------------------
# Slack signature verification
# ---------------------------------------------------------------------------

def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(my_sig, signature)


# ---------------------------------------------------------------------------
# Background message handler
# ---------------------------------------------------------------------------

def _download_slack_file(url: str) -> bytes:
    """Download a file from Slack using the bot token for auth."""
    resp = http_requests.get(url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"})
    resp.raise_for_status()
    return resp.content


_IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/jpg"}


_AGENT_FS_BASE = os.path.join(os.path.dirname(__file__), "agent_file_system")

# Matches tool output patterns: "File written: <path>", "JSON written: <path>",
# "CSV written: <path>", "Chart saved: <path>"
_FILE_OUTPUT_PATTERN = re.compile(
    r"(?:File written|JSON written|CSV written|Chart saved):\s*([\w\-./]+)"
)

# Matches any relative path with a file extension (e.g. "reports/q1.csv", "charts/plot.png")
_PATH_PATTERN = re.compile(r"(?:^|\s)([\w\-./]+\.[\w]+)", re.MULTILINE)


def _find_output_files(response_text: str) -> list[str]:
    """Extract file paths from agent response and return existing ones in agent_file_system."""
    # First: explicit tool output patterns (highest confidence)
    matches = _FILE_OUTPUT_PATTERN.findall(response_text)
    # Second: any path-like string with a file extension mentioned in the response
    matches += _PATH_PATTERN.findall(response_text)

    paths = []
    seen = set()
    for match in matches:
        full = os.path.join(_AGENT_FS_BASE, match)
        if full not in seen and os.path.isfile(full):
            paths.append(full)
            seen.add(full)
    return paths


def _clear_invocation_queue() -> None:
    """Clear the invocation queue on wake, stashing previous entries for next pre-exit."""
    global _previous_invocations
    _previous_invocations = _load_json_file(QUEUE_FILE)
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "w") as f:
            json.dump([], f)


def _load_reengagement_state() -> dict:
    if not os.path.exists(REENGAGEMENT_FILE):
        return {"invocations_exhausted_at": None, "attempt_number": 0, "messages": []}
    with open(REENGAGEMENT_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"invocations_exhausted_at": None, "attempt_number": 0, "messages": []}


def _save_reengagement_state(state: dict) -> None:
    with open(REENGAGEMENT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _mark_invocations_exhausted(now: datetime) -> None:
    """Record when the invocation queue drains to empty while sleeping."""
    state = _load_reengagement_state()
    if not state.get("invocations_exhausted_at"):
        state["invocations_exhausted_at"] = now.isoformat()
        _save_reengagement_state(state)
        print(f"[reengagement] Invocations exhausted at {now.isoformat()}")


def _reset_reengagement() -> None:
    """Reset re-engagement timer on user wake. Preserves message history."""
    state = _load_reengagement_state()
    state["invocations_exhausted_at"] = None
    state["attempt_number"] = 0
    _save_reengagement_state(state)
    print("[reengagement] Reset — user responded")


async def _handle_message(channel: str, text: str, images: list[bytes] | None = None) -> None:
    global _last_message_time, _is_sleeping, _last_notification_time, _last_notification_arm
    from agent import Agent
    from tools import set_current_session_id

    async with _agent_lock:
        try:
            _last_message_time = time.time()
            now_dt = datetime.now().astimezone()

            # Record bandit rewards
            if _is_sleeping and _last_notification_time > 0:
                # User responded — check if within 10 min of last notification
                if time.time() - _last_notification_time <= 600 and _last_notification_arm:
                    bandit.update_arm(*_last_notification_arm, reward=1.0)
                    print(f"[bandit] Notification response reward: arm {_last_notification_arm}")
                _last_notification_time = 0.0
                _last_notification_arm = None
            elif not _is_sleeping:
                # Organic chat — reward current time slot
                bandit.update_arm(now_dt.weekday(), now_dt.hour, reward=0.5)

            # Wake up if sleeping — start fresh conversation with summary context
            if _is_sleeping:
                _is_sleeping = False
                _clear_invocation_queue()
                _reset_reengagement()
                print(f"[wake] Agent woke up — cleared invocation queue (reminders preserved)")

            loop = asyncio.get_running_loop()
            set_current_session_id(SLACK_SESSION_ID)

            # Check if we need to start a fresh conversation with summary
            session_path = os.path.join(
                os.path.dirname(__file__), "sessions", f"{SLACK_SESSION_ID}.json"
            )
            if not os.path.exists(session_path):
                # No active session — start fresh with summary context
                summary = Agent.load_summary(SLACK_SESSION_ID)
                agent = Agent(session_id=SLACK_SESSION_ID, chat_summary=summary or None)
            else:
                # Active session exists — continue mid-conversation
                agent = Agent(session_id=SLACK_SESSION_ID)

            response = await loop.run_in_executor(
                None, lambda: agent.run(text, images=images or None)
            )

            # Split long messages (Slack limit ~4000 chars)
            chunks = [response[i : i + 3900] for i in range(0, len(response), 3900)]
            for chunk in chunks:
                slack_client.chat_postMessage(channel=channel, text=chunk)

            # Upload any files the agent created
            for file_path in _find_output_files(response):
                slack_client.files_upload_v2(
                    channel=channel,
                    file=file_path,
                    title=os.path.basename(file_path),
                )
        except Exception as e:
            slack_client.chat_postMessage(
                channel=channel,
                text=f"Something went wrong: {e}",
            )


# ---------------------------------------------------------------------------
# Notification queue polling (ported from scheduler.py)
# ---------------------------------------------------------------------------

STALE_THRESHOLD_SECONDS = 600  # 10 minutes


def _load_json_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_json_file(path: str, data: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_queue() -> list[dict]:
    return _load_json_file(QUEUE_FILE)


def save_queue(queue: list[dict]) -> None:
    _save_json_file(QUEUE_FILE, queue)


async def poll_notifications(dm_channel: str) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        now = datetime.now(timezone.utc)
        now_local = now.astimezone()
        global _last_notification_time, _last_notification_arm

        # --- Poll reminders (always deliver, even if late) ---
        reminders = _load_json_file(REMINDERS_FILE)
        if reminders:
            remaining_reminders = []
            for entry in reminders:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.astimezone()
                except (ValueError, KeyError):
                    print(f"[warning] Dropping malformed reminder: {entry}")
                    continue

                if ts <= now:
                    msg = f":bell: *Reminder:* {entry['message']}"
                    slack_client.chat_postMessage(channel=dm_channel, text=msg)
                    print(f"[reminder] delivered: {entry['message']}")
                    _last_notification_time = time.time()
                    _last_notification_arm = (now_local.weekday(), now_local.hour)
                else:
                    remaining_reminders.append(entry)

            if len(remaining_reminders) != len(reminders):
                _save_json_file(REMINDERS_FILE, remaining_reminders)

        # --- Poll invocation queue (drop if >10 min stale) ---
        queue = _load_json_file(QUEUE_FILE)
        if queue:
            remaining_queue = []
            for entry in queue:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.astimezone()
                except (ValueError, KeyError):
                    print(f"[warning] Dropping malformed queue entry: {entry}")
                    continue

                if ts <= now:
                    age_seconds = (now - ts).total_seconds()
                    if age_seconds <= STALE_THRESHOLD_SECONDS:
                        msg = f":loudspeaker: {entry['message']}"
                        slack_client.chat_postMessage(channel=dm_channel, text=msg)
                        print(f"[invocation] sent: {entry['message']}")
                        _last_notification_time = time.time()
                        _last_notification_arm = (now_local.weekday(), now_local.hour)
                    else:
                        print(f"[invocation] dropped stale ({age_seconds:.0f}s old): {entry['message']}")
                else:
                    remaining_queue.append(entry)

            if len(remaining_queue) != len(queue):
                _save_json_file(QUEUE_FILE, remaining_queue)

            # Detect queue exhaustion: queue was non-empty, now drained, agent sleeping
            if not remaining_queue and _is_sleeping:
                _mark_invocations_exhausted(now_local)


# ---------------------------------------------------------------------------
# Idle timeout monitor
# ---------------------------------------------------------------------------

async def _idle_monitor(dm_channel: str) -> None:
    """Check for idle timeout and trigger pre-exit flow when the agent sleeps."""
    global _is_sleeping, _last_message_time
    from agent import Agent
    from tools import set_current_session_id

    while True:
        await asyncio.sleep(10)

        # Don't trigger if no messages yet, already sleeping, or not idle long enough
        if _last_message_time == 0.0 or _is_sleeping:
            continue
        if time.time() - _last_message_time <= AGENT_TIMEOUT:
            continue

        async with _agent_lock:
            # Re-check after acquiring lock (a message may have arrived)
            if _is_sleeping or time.time() - _last_message_time <= AGENT_TIMEOUT:
                continue

            try:
                print(f"[timeout] Session idle for {AGENT_TIMEOUT}s — running pre-exit flow...")
                loop = asyncio.get_running_loop()
                set_current_session_id(SLACK_SESSION_ID)
                agent = Agent(session_id=SLACK_SESSION_ID)

                recommendations = bandit.get_recommendations()
                rec_text = bandit.format_recommendations(recommendations)

                # Load existing reminders so the agent avoids duplicating them
                from tools.scheduling import load_reminders
                existing = load_reminders()
                if existing:
                    lines = [f"• {r['timestamp']} — {r['message']}" for r in existing]
                    reminders_text = "The following reminders are already scheduled:\n" + "\n".join(lines)
                else:
                    reminders_text = "No reminders are currently scheduled."

                # Build previous invocations context
                if _previous_invocations:
                    inv_lines = [f"• {inv['timestamp']} — {inv['message']}" for inv in _previous_invocations]
                    prev_invocations_text = "The following invocations were scheduled in the previous session:\n" + "\n".join(inv_lines)
                else:
                    prev_invocations_text = "No invocations were scheduled in the previous session."

                response = await loop.run_in_executor(
                    None, lambda: agent.run_pre_exit(
                        bandit_recommendations=rec_text,
                        existing_reminders=reminders_text,
                        previous_invocations=prev_invocations_text,
                    )
                )

                # Generate conversation summary and archive session
                try:
                    new_summary = await loop.run_in_executor(
                        None, agent.generate_summary
                    )
                    Agent.save_summary(SLACK_SESSION_ID, new_summary)

                    # Store conversation in SimpleMem long-term memory
                    try:
                        from memory import store_dialogues
                        dialogues = agent.format_conversation_for_memory()
                        if dialogues:
                            store_dialogues(dialogues)
                            print(f"[memory] Stored {len(dialogues)} dialogue turns in Mem0")
                    except Exception as mem_err:
                        print(f"[warning] SimpleMem storage failed: {mem_err}")

                    agent.archive_session()
                    print(f"[summary] Conversation summarized and session archived")
                except Exception as e:
                    print(f"[warning] Summary generation failed: {e}")

                _is_sleeping = True

                if response:
                    chunks = [response[i : i + 3900] for i in range(0, len(response), 3900)]
                    for chunk in chunks:
                        slack_client.chat_postMessage(channel=dm_channel, text=chunk)

                    for file_path in _find_output_files(response):
                        slack_client.files_upload_v2(
                            channel=dm_channel,
                            file=file_path,
                            title=os.path.basename(file_path),
                        )

                print(f"[sleep] Agent is now sleeping")
            except Exception as e:
                print(f"[error] Pre-exit flow failed: {e}")


# ---------------------------------------------------------------------------
# Cron job polling
# ---------------------------------------------------------------------------

def _save_cron_jobs(jobs: list[dict]) -> None:
    with open(CRON_JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


async def _poll_cron_jobs(dm_channel: str) -> None:
    """Poll cron_jobs.json every 60s and execute due jobs with a fresh agent."""
    from agent import Agent, DEFAULT_MODEL
    from prompts import CRON_JOB_PROMPT

    while True:
        await asyncio.sleep(60)
        try:
            from tools.scheduling import load_cron_jobs, save_cron_jobs

            now = datetime.now().astimezone()
            current_day = now.strftime("%A").lower()
            current_time = now.strftime("%H:%M")

            jobs = load_cron_jobs()
            for job in jobs:
                # Skip if not scheduled for today
                if job.get("days") and current_day not in job["days"]:
                    continue
                # Skip if not the right time
                if job["time"] != current_time:
                    continue
                # Skip if already ran this cycle (within 2 minutes)
                if job.get("last_run"):
                    last = datetime.fromisoformat(job["last_run"])
                    if (now - last).total_seconds() < 120:
                        continue

                # Update last_run immediately to prevent double-fire
                job["last_run"] = now.isoformat()
                save_cron_jobs(jobs)

                print(f"[cron] Executing job {job['id']}: {job['task'][:80]}")

                # Execute with a fresh agent (no session_id = no history)
                async with _agent_lock:
                    loop = asyncio.get_running_loop()
                    agent = Agent(model=DEFAULT_MODEL)
                    cron_prompt = CRON_JOB_PROMPT.format(
                        current_time=now.isoformat(),
                        task=job["task"],
                    )
                    response = await loop.run_in_executor(
                        None, lambda: agent.run(cron_prompt)
                    )

                # Deliver result via DM
                header = f":gear: *Scheduled task:* {job['task'][:80]}"
                slack_client.chat_postMessage(channel=dm_channel, text=header)
                for chunk in [response[i:i+3900] for i in range(0, len(response), 3900)]:
                    slack_client.chat_postMessage(channel=dm_channel, text=chunk)

                # Upload any files the agent created
                for file_path in _find_output_files(response):
                    slack_client.files_upload_v2(
                        channel=dm_channel,
                        file=file_path,
                        title=os.path.basename(file_path),
                    )

                print(f"[cron] Job {job['id']} completed")

        except Exception as e:
            print(f"[cron] Error in cron polling: {e}")


# ---------------------------------------------------------------------------
# Re-engagement monitor
# ---------------------------------------------------------------------------

async def _reengagement_monitor(dm_channel: str) -> None:
    """Periodically check if re-engagement DM should be sent after invocations exhaust."""
    from agent import Agent, DEFAULT_MODEL
    from prompts import REENGAGEMENT_PROMPT

    while True:
        await asyncio.sleep(300)  # check every 5 minutes

        if not _is_sleeping:
            continue

        try:
            state = _load_reengagement_state()
            if not state.get("invocations_exhausted_at"):
                continue

            exhausted_at = datetime.fromisoformat(state["invocations_exhausted_at"])
            attempt = state.get("attempt_number", 0)
            interval = REENGAGEMENT_INTERVALS[min(attempt, len(REENGAGEMENT_INTERVALS) - 1)]
            now = datetime.now().astimezone()

            if (now - exhausted_at).total_seconds() < interval:
                continue

            # Backoff elapsed — check bandit for optimal send time
            recommendations = bandit.get_recommendations(top_k=5)
            if recommendations:
                current_day = now.weekday()
                current_hour = now.hour
                # Check if current (day, hour) is among top recommended slots
                top_slots = {(r["day"], r["hour"]) for r in recommendations}
                if (current_day, current_hour) not in top_slots:
                    # Not an optimal time — skip and check again next cycle
                    continue
                print(f"[reengagement] Current slot ({current_day}, {current_hour}) matches bandit recommendation — proceeding")

            # Time to re-engage — spawn fresh agent
            rec_text = bandit.format_recommendations(recommendations)
            print(f"[reengagement] Sending re-engagement DM (attempt {attempt + 1})")

            # Build context from previous re-engagement messages
            prev_messages = state.get("messages", [])
            if prev_messages:
                prev_lines = [f"• {m['sent_at']}: {m['message']}" for m in prev_messages]
                prev_text = "You have already sent these re-engagement DMs (do NOT repeat topics):\n" + "\n".join(prev_lines)
            else:
                prev_text = "This is your first re-engagement DM — no previous ones sent."

            async with _agent_lock:
                loop = asyncio.get_running_loop()
                agent = Agent(model=DEFAULT_MODEL)  # fresh, no session
                prompt = REENGAGEMENT_PROMPT.format(
                    current_time=now.isoformat(),
                    previous_reengagement_dms=prev_text,
                    bandit_recommendations=rec_text,
                )
                response = await loop.run_in_executor(None, lambda: agent.run(prompt))

            # Send DM
            slack_client.chat_postMessage(channel=dm_channel, text=response[:3900])
            print(f"[reengagement] DM sent (attempt {attempt + 1})")

            # Update state: increment attempt, advance clock for next interval
            state["attempt_number"] = attempt + 1
            state["invocations_exhausted_at"] = now.isoformat()
            state.setdefault("messages", []).append({
                "sent_at": now.isoformat(),
                "message": response[:500],
            })
            _save_reengagement_state(state)

        except Exception as e:
            print(f"[reengagement] Error: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class Notification(BaseModel):
    timestamp: str
    message: str
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open DM channel for notifications
    resp = slack_client.conversations_open(users=[SLACK_USER_ID])
    dm_channel = resp["channel"]["id"]
    print(f"Slack DM channel: {dm_channel}")

    queue_task = asyncio.create_task(poll_notifications(dm_channel))
    idle_task = asyncio.create_task(_idle_monitor(dm_channel))
    cron_task = asyncio.create_task(_poll_cron_jobs(dm_channel))
    reengagement_task = asyncio.create_task(_reengagement_monitor(dm_channel))
    print(f"Scheduler running — polling reminders + invocations every {POLL_INTERVAL}s")
    print(f"Idle monitor running — timeout {AGENT_TIMEOUT}s")
    print(f"Cron job poller running — checking every 60s")
    print(f"Re-engagement monitor running — checking every 5m")
    yield
    queue_task.cancel()
    idle_task.cancel()
    cron_task.cancel()
    reengagement_task.cancel()
    for t in (queue_task, idle_task, cron_task, reengagement_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


@app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(body, timestamp, signature):
        return {"error": "invalid signature"}, 403

    data = json.loads(body)

    # URL verification challenge
    if data.get("type") == "url_verification":
        return {"challenge": data["challenge"]}

    # Event callback
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        event_id = data.get("event_id", "")

        # Deduplicate retries
        if event_id in _seen_events:
            return {"ok": True}
        _seen_events.add(event_id)
        if len(_seen_events) > _MAX_SEEN:
            _seen_events.clear()

        # Only handle DMs from our user (not bot messages, not subtypes)
        if (
            event.get("type") == "message"
            and event.get("channel_type") == "im"
            and event.get("user") == SLACK_USER_ID
            and "subtype" not in event
            and "bot_id" not in event
        ):
            channel = event["channel"]
            text = event.get("text", "")

            # Download any image attachments
            images = []
            for f in event.get("files", []):
                if f.get("mimetype", "") in _IMAGE_MIMES:
                    url = f.get("url_private_download") or f.get("url_private")
                    if url:
                        try:
                            images.append(_download_slack_file(url))
                        except Exception as e:
                            print(f"[warning] Failed to download image: {e}")

            asyncio.create_task(_handle_message(channel, text, images or None))

    return {"ok": True}


@app.post("/notify")
async def notify(notification: Notification):
    queue = load_queue()
    queue.append(notification.model_dump())
    save_queue(queue)
    print(f"[queued] {notification.message} @ {notification.timestamp}")
    return {"status": "queued"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
