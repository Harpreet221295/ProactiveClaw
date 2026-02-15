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


def _clear_queue_for_session(session_id: str) -> None:
    """Remove all queued notifications for the given session."""
    if not os.path.exists(QUEUE_FILE):
        return
    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            queue = []
    queue = [entry for entry in queue if entry.get("session_id") != session_id]
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


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
                _clear_queue_for_session(SLACK_SESSION_ID)
                print(f"[wake] Agent woke up — cleared notification queue")

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

def load_queue() -> list[dict]:
    if not os.path.exists(QUEUE_FILE):
        return []
    with open(QUEUE_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_queue(queue: list[dict]) -> None:
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


async def poll_queue(dm_channel: str) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        queue = load_queue()
        if not queue:
            continue

        now = datetime.now(timezone.utc)
        remaining = []
        for entry in queue:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.astimezone()
            except (ValueError, KeyError):
                print(f"[warning] Dropping malformed queue entry: {entry}")
                continue

            if ts <= now:
                global _last_notification_time, _last_notification_arm
                msg = f":bell: *Reminder:* {entry['message']}"
                slack_client.chat_postMessage(channel=dm_channel, text=msg)
                print(f"[notification] sent: {entry['message']}")
                # Track for bandit reward
                _last_notification_time = time.time()
                _last_notification_arm = (now.astimezone().weekday(), now.astimezone().hour)
            else:
                remaining.append(entry)

        if len(remaining) != len(queue):
            save_queue(remaining)


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
                response = await loop.run_in_executor(
                    None, lambda: agent.run_pre_exit(bandit_recommendations=rec_text)
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
                            print(f"[memory] Stored {len(dialogues)} dialogue turns in SimpleMem")
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

    queue_task = asyncio.create_task(poll_queue(dm_channel))
    idle_task = asyncio.create_task(_idle_monitor(dm_channel))
    print(f"Scheduler running — polling {QUEUE_FILE} every {POLL_INTERVAL}s")
    print(f"Idle monitor running — timeout {AGENT_TIMEOUT}s")
    yield
    queue_task.cancel()
    idle_task.cancel()
    for t in (queue_task, idle_task):
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
