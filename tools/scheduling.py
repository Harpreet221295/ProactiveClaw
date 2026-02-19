import json
import os
import secrets
from datetime import datetime

from ._state import QUEUE_FILE, REMINDERS_FILE, CRON_JOBS_FILE, get_current_session_id


def schedule_notifications(notifications: list[dict]) -> str:
    if len(notifications) > 5:
        return "Error: maximum 5 notifications allowed."

    # Load existing queue
    queue = []
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []

    for entry in notifications:
        queue.append({
            "timestamp": entry["timestamp"],
            "message": entry["message"],
            "session_id": get_current_session_id(),
        })

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

    return f"Scheduled {len(notifications)} notification(s)."


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
    jobs = _load_cron_jobs_file()
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
            "description": "Schedule follow-up notifications to re-engage the user later. Use this when the conversation is ending and there are pending topics, reminders, or follow-ups the user would benefit from. Each notification has a timestamp (ISO 8601) and a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notifications": {
                        "type": "array",
                        "description": "List of notifications to schedule (max 5).",
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
]
