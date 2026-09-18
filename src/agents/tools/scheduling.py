from __future__ import annotations
import json
import os
import secrets
from datetime import datetime, timedelta

from ._state import QUEUE_FILE, REMINDERS_FILE, CRON_JOBS_FILE, get_current_session_id
from core import config as _config


def _load_queue() -> list[dict]:
    if not os.path.exists(QUEUE_FILE):
        return []
    with open(QUEUE_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_queue(queue: list[dict]) -> None:
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def _parse_ts(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt
    except (ValueError, TypeError):
        return None


MIN_GAP_SECONDS = 30 * 60


def schedule_notifications(notifications: list[dict]) -> str:
    """Schedule proactive nudges (invocations). Enforces the proactiveness level:
    daily cap, quiet hours, and 'off'. Nudges tied to a care item bump its nudge count."""
    if len(notifications) > 10:
        return "Error: maximum 10 notifications per call."

    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    if not eff["invocations_enabled"]:
        return "Proactiveness level is 'off' — no nudges scheduled. (The user can change this in Settings.)"

    from care.nudges import budget_for_day
    from care.registry import CareRegistry

    now = datetime.now().astimezone()
    queue = _load_queue()
    scheduled, skipped, shifted = [], [], []
    budgets: dict[str, int] = {}
    registry = CareRegistry()
    # Existing pending timestamps, so new nudges are spaced out from them
    taken: list[datetime] = [t for t in (_parse_ts(e.get("timestamp", "")) for e in queue
                                        if e.get("source") != "subagent_complete") if t]

    for entry in sorted(notifications, key=lambda e: e.get("timestamp", "")):
        ts = _parse_ts(entry.get("timestamp", ""))
        msg = (entry.get("message") or "").strip()
        if ts is None or not msg:
            skipped.append(f"malformed entry: {entry}")
            continue
        if ts <= now:
            ts = now.replace(second=0, microsecond=0)
        if entry.get("priority") != "critical":
            moved = _config.next_allowed_time(ts, eff["quiet_hours"])
            if moved != ts:
                shifted.append(f"'{msg[:40]}…' moved from {ts.strftime('%H:%M')} to {moved.strftime('%a %H:%M')} (quiet hours)")
                ts = moved
        # Space nudges at least MIN_GAP apart (avoids three pings in the same minute)
        original = ts
        for _ in range(12):
            if any(abs((ts - t).total_seconds()) < MIN_GAP_SECONDS for t in taken):
                ts = ts + timedelta(seconds=MIN_GAP_SECONDS)
                if entry.get("priority") != "critical":
                    ts = _config.next_allowed_time(ts, eff["quiet_hours"])
            else:
                break
        if ts != original:
            shifted.append(f"'{msg[:40]}…' moved from {original.strftime('%H:%M')} to {ts.strftime('%a %H:%M')} (spacing)")
        day = ts.strftime("%Y-%m-%d")
        if day not in budgets:
            cap = eff["max_nudges_per_day"]
            # The morning review leaves headroom for the rest of the day (pre-exit nudges share the cap)
            if (entry.get("source") == "morning_review") and cap > 1:
                cap = cap - 1
            budgets[day] = budget_for_day(day, cap, now)
        if budgets[day] <= 0:
            skipped.append(f"'{msg[:40]}…' on {day} — daily cap of {eff['max_nudges_per_day']} reached")
            continue
        budgets[day] -= 1
        taken.append(ts)

        item = {
            "id": f"inv_{secrets.token_hex(3)}",
            "timestamp": ts.isoformat(),
            "message": msg,
            "session_id": get_current_session_id(),
            "source": entry.get("source") or "pre_exit",
            "created_at": now.isoformat(),
        }
        if entry.get("item_id"):
            item["item_id"] = entry["item_id"]
            registry.record_nudge(entry["item_id"], ts)
        if entry.get("priority"):
            item["priority"] = entry["priority"]
        queue.append(item)
        scheduled.append(f"{ts.strftime('%a %b %d %H:%M')} — {msg[:60]}")

    _save_queue(queue)
    lines = [f"Scheduled {len(scheduled)} nudge(s)."] + [f"  • {s}" for s in scheduled]
    if shifted:
        lines.append("Adjusted for quiet hours:")
        lines += [f"  • {s}" for s in shifted]
    if skipped:
        lines.append("Not scheduled:")
        lines += [f"  • {s}" for s in skipped]
    return "\n".join(lines)


def list_scheduled_nudges() -> str:
    """List pending proactive nudges in the queue."""
    now = datetime.now().astimezone()
    pending = []
    for e in _load_queue():
        if e.get("source") == "subagent_complete":
            continue
        ts = _parse_ts(e.get("timestamp", ""))
        if ts and ts > now:
            pending.append((ts, e))
    if not pending:
        return "No nudges are scheduled."
    pending.sort(key=lambda x: x[0])
    return "\n".join(
        f"• [{e.get('id', '?')}] {ts.strftime('%a %b %d %H:%M')} — {e['message']}" + (f" (item {e['item_id']})" if e.get("item_id") else "")
        for ts, e in pending
    )


def cancel_scheduled_nudge(nudge_id: str) -> str:
    queue = _load_queue()
    new_queue = [e for e in queue if e.get("id") != nudge_id]
    if len(new_queue) == len(queue):
        return f"No scheduled nudge with id {nudge_id}."
    _save_queue(new_queue)
    return f"Nudge {nudge_id} cancelled."


def get_current_datetime() -> str:
    now = datetime.now().astimezone()
    return now.isoformat()


# ---------------------------------------------------------------------------
# Reminders — persistent, user-facing
# ---------------------------------------------------------------------------

def _load_reminders_file() -> list[dict]:
    if not os.path.exists(REMINDERS_FILE):
        return []
    with open(REMINDERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_reminders_file(reminders: list[dict]) -> None:
    with open(REMINDERS_FILE, "w") as f:
        json.dump(reminders, f, indent=2)


def set_reminder(timestamp: str, message: str) -> str:
    """Create a persistent reminder that survives across sessions."""
    reminder_id = f"rem_{secrets.token_hex(2)}"
    reminders = _load_reminders_file()
    reminders.append({
        "id": reminder_id,
        "timestamp": timestamp,
        "message": message,
        "session_id": get_current_session_id(),
    })
    _save_reminders_file(reminders)
    return f"Reminder set (ID: {reminder_id}): {message} at {timestamp}"


def list_reminders() -> str:
    """List all future reminders."""
    reminders = _load_reminders_file()
    now = datetime.now().astimezone()
    future = []
    for r in reminders:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.astimezone()
            if ts > now:
                future.append(r)
        except (ValueError, KeyError):
            continue
    if not future:
        return "No active reminders."
    lines = []
    for r in future:
        lines.append(f"• [{r['id']}] {r['timestamp']} — {r['message']}")
    return "\n".join(lines)


def cancel_reminder(reminder_id: str) -> str:
    """Cancel a reminder by its ID."""
    reminders = _load_reminders_file()
    new_reminders = [r for r in reminders if r.get("id") != reminder_id]
    if len(new_reminders) == len(reminders):
        return f"Reminder not found: {reminder_id}"
    _save_reminders_file(new_reminders)
    return f"Reminder {reminder_id} cancelled."


def load_reminders() -> list[dict]:
    """Return all reminders (used by slack_server to pass into pre-exit context)."""
    return _load_reminders_file()


# ---------------------------------------------------------------------------
# Cron Jobs — persistent, recurring autonomous tasks
# ---------------------------------------------------------------------------

ALL_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _load_cron_jobs_file() -> list[dict]:
    if not os.path.exists(CRON_JOBS_FILE):
        return []
    with open(CRON_JOBS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_cron_jobs_file(jobs: list[dict]) -> None:
    with open(CRON_JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def create_cron_job(time: str, task: str, days: list[str] | None = None) -> str:
    """Create a recurring cron job that executes a task autonomously on schedule."""
    job_id = f"cron_{secrets.token_hex(2)}"
    if days:
        days = [d.lower() for d in days]
        invalid = [d for d in days if d not in ALL_DAYS]
        if invalid:
            return f"Error: invalid day(s): {', '.join(invalid)}. Use full lowercase day names."
    else:
        days = list(ALL_DAYS)

    jobs = _load_cron_jobs_file()
    jobs.append({
        "id": job_id,
        "time": time,
        "days": days,
        "task": task,
        "created_at": datetime.now().astimezone().isoformat(),
        "last_run": None,
    })
    _save_cron_jobs_file(jobs)

    days_str = ", ".join(days) if len(days) < 7 else "every day"
    return f"Cron job created (ID: {job_id}): \"{task}\" at {time} on {days_str}."


def list_cron_jobs() -> str:
    """List all active cron jobs."""
    jobs = _load_cron_jobs_file()
    if not jobs:
        return "No cron jobs scheduled."
    lines = []
    for j in jobs:
        days_str = ", ".join(j["days"]) if len(j["days"]) < 7 else "every day"
        lines.append(f"• [{j['id']}] {j['time']} {days_str} — {j['task']}")
    return "\n".join(lines)


def delete_cron_job(job_id: str) -> str:
    """Delete a cron job by its ID."""
    # Protect system cron jobs from accidental deletion
    if job_id.startswith("cron_deep_think"):
        return "Error: This is a system cron job and cannot be deleted via tools."
    jobs = _load_cron_jobs_file()
    # Also check the protected flag on any job
    for j in jobs:
        if j.get("id") == job_id and j.get("protected"):
            return "Error: This is a system cron job and cannot be deleted via tools."
    new_jobs = [j for j in jobs if j.get("id") != job_id]
    if len(new_jobs) == len(jobs):
        return f"Cron job not found: {job_id}"
    _save_cron_jobs_file(new_jobs)
    return f"Cron job {job_id} deleted."


def load_cron_jobs() -> list[dict]:
    """Return all cron jobs (used by slack_server for polling)."""
    return _load_cron_jobs_file()


def save_cron_jobs(jobs: list[dict]) -> None:
    """Save cron jobs list (used by slack_server to update last_run)."""
    _save_cron_jobs_file(jobs)


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "schedule_notifications",
            "description": "Schedule proactive nudges (invocations) to re-engage the user later. Each has a timestamp (ISO 8601 with offset) and a message. The system enforces the user's proactiveness level: a daily cap and quiet hours (nudges inside quiet hours are moved to the end of the window unless priority='critical'). Link a nudge to a care item with item_id so its nudge count is tracked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notifications": {
                        "type": "array",
                        "description": "List of nudges to schedule (max 10; the daily cap may allow fewer).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {
                                    "type": "string",
                                    "description": "ISO 8601 datetime when the notification should fire (e.g. '2025-01-15T14:30:00').",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "The notification message to show the user.",
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Optional source identifier ('pre_exit' default, 'morning_review') for staleness handling.",
                                },
                                "item_id": {
                                    "type": "string",
                                    "description": "Optional care registry item id this nudge is about.",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["normal", "critical"],
                                    "description": "critical bypasses quiet hours (hard deadlines only).",
                                },
                            },
                            "required": ["timestamp", "message"],
                        },
                    }
                },
                "required": ["notifications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time with timezone. Use this whenever you need to know the exact current time, e.g. for scheduling events or checking relative times.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a persistent reminder for the user. Use this when the user asks to be reminded about something (e.g. 'remind me to...', 'text me at...'). Reminders persist across sessions and are delivered even if the agent is sleeping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 8601 datetime with timezone when the reminder should fire (e.g. '2025-01-15T14:30:00-08:00').",
                    },
                    "message": {
                        "type": "string",
                        "description": "The reminder message to deliver to the user.",
                    },
                },
                "required": ["timestamp", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List all active (future) reminders. Use when the user asks what reminders they have scheduled.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel a reminder by its ID. Use when the user wants to remove a scheduled reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "The reminder ID to cancel (e.g. 'rem_a3f8').",
                    },
                },
                "required": ["reminder_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cron_job",
            "description": "Create a recurring cron job that runs autonomously on a schedule. The agent will wake up at the scheduled time, execute the task using available tools, and DM the results. Use this for recurring tasks like 'every weekday at 9am, search for AI news and send me a summary'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time to run in HH:MM 24-hour format (e.g. '09:00', '14:30').",
                    },
                    "task": {
                        "type": "string",
                        "description": "Description of the task to execute. Be specific — the agent will run this with no conversation context.",
                    },
                    "days": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Days of the week to run (lowercase, e.g. ['monday', 'friday']). Omit for every day.",
                    },
                },
                "required": ["time", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cron_jobs",
            "description": "List all active recurring cron jobs. Use when the user asks what scheduled tasks they have.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_cron_job",
            "description": "Delete a recurring cron job by its ID. Use when the user wants to stop a scheduled recurring task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The cron job ID to delete (e.g. 'cron_a3f8').",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_nudges",
            "description": "List proactive nudges already queued for delivery. Check before scheduling to avoid duplicates.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_scheduled_nudge",
            "description": "Cancel a queued nudge by id (from list_scheduled_nudges).",
            "parameters": {"type": "object", "properties": {"nudge_id": {"type": "string"}}, "required": ["nudge_id"]},
        },
    },
]
