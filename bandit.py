"""
Multi-Armed Bandit for Smart Notification Scheduling.

168-arm bandit (7 days x 24 hours). Each arm = (day_of_week, hour).
Values updated with time-decayed running average so stale arms fade
toward 0 and fresh rewards dominate.
"""

import json
import math
import os
from datetime import datetime

STATE_FILE = os.path.join(os.path.dirname(__file__), "bandit_state.json")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEFAULT_LAMBDA = 0.1


def arm_key(day: int, hour: int) -> str:
    """Build key string for an arm. day=0..6 (Mon-Sun), hour=0..23."""
    return f"{day}_{hour}"


def load_state() -> dict:
    """Load bandit state from disk, returning defaults if missing."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {"q_values": {}, "last_updated": {}, "lambda": DEFAULT_LAMBDA}


def save_state(state: dict) -> None:
    """Persist bandit state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def update_arm(day: int, hour: int, reward: float) -> None:
    """Apply time-decayed update to an arm.

    Q[arm] = decay * Q_old + (1 - decay) * reward
    where decay = exp(-lambda * days_since_last_update)
    """
    state = load_state()
    key = arm_key(day, hour)
    now = datetime.now().astimezone()
    now_iso = now.isoformat()

    q_old = state["q_values"].get(key, 0.0)
    last = state["last_updated"].get(key)

    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            time_since = (now - last_dt).total_seconds() / 86400  # days
        except ValueError:
            time_since = 0.0
    else:
        time_since = 0.0

    decay = math.exp(-state["lambda"] * time_since)
    q_new = decay * q_old + (1 - decay) * reward

    state["q_values"][key] = round(q_new, 4)
    state["last_updated"][key] = now_iso
    save_state(state)


def get_recommendations(top_k: int = 10) -> list[dict]:
    """Return top arms sorted by Q-value.

    Each entry: {"day": int, "hour": int, "day_name": str, "time": str, "score": float}
    """
    state = load_state()
    now = datetime.now().astimezone()
    lam = state["lambda"]

    scored = []
    for key, q_val in state["q_values"].items():
        parts = key.split("_")
        if len(parts) != 2:
            continue
        day, hour = int(parts[0]), int(parts[1])

        last = state["last_updated"].get(key)
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                time_since = (now - last_dt).total_seconds() / 86400
            except ValueError:
                time_since = 0.0
        else:
            time_since = 0.0

        # Apply decay to get current effective Q-value
        effective_q = q_val * math.exp(-lam * time_since)
        if effective_q < 0.01:
            continue

        hour_12 = hour % 12 or 12
        ampm = "AM" if hour < 12 else "PM"
        time_str = f"{hour_12}:00 {ampm}"

        scored.append({
            "day": day,
            "hour": hour,
            "day_name": DAY_NAMES[day],
            "time": time_str,
            "score": round(effective_q, 2),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_recommendations(recommendations: list[dict]) -> str:
    """Format recommendations as a string for the LLM prompt."""
    if not recommendations:
        return ""

    lines = [
        "## User Availability Predictions (based on past engagement)",
        "The following time slots have the highest probability of user response.",
        "Schedule notifications at or near these times when possible.",
        "Only schedule forward chronologically from the current time.",
        "",
    ]
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec['day_name']} {rec['time']} (score: {rec['score']})")

    lines.append("")
    lines.append("These are recommendations — adjust based on task urgency and calendar availability.")
    lines.append("For urgent deadlines, schedule more aggressively regardless of scores.")
    lines.append("For casual/habit tasks, use only the top 1-2 slots.")

    return "\n".join(lines)
