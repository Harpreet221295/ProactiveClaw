import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

QUEUE_FILE = os.path.join(os.path.dirname(__file__), "queue.json")
POLL_INTERVAL = 30  # seconds


class Notification(BaseModel):
    timestamp: str
    message: str
    session_id: str | None = None


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


async def poll_queue() -> None:
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
                    # Assume local timezone if LLM omits offset
                    ts = ts.astimezone()
            except (ValueError, KeyError):
                print(f"[warning] Dropping malformed queue entry: {entry}")
                continue

            if ts <= now:
                sid = entry.get("session_id", "unknown")
                print(f"\n[notification] (session {sid}) {entry['message']}")
            else:
                remaining.append(entry)

        if len(remaining) != len(queue):
            save_queue(remaining)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(poll_queue())
    print(f"Scheduler running — polling {QUEUE_FILE} every {POLL_INTERVAL}s")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.post("/notify")
async def notify(notification: Notification):
    queue = load_queue()
    queue.append(notification.model_dump())
    save_queue(queue)
    print(f"[queued] {notification.message} @ {notification.timestamp}")
    return {"status": "queued"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
