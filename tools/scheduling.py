import json
import os

from ._state import QUEUE_FILE, get_current_session_id


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
    from datetime import datetime
    now = datetime.now().astimezone()
    return now.isoformat()


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
]
