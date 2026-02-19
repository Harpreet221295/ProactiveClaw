"""
Multi-Armed Bandit for Smart Notification Scheduling.

168-arm bandit (7 days x 24 hours). Each arm = (day_of_week, hour).
Values updated with time-decayed running average so stale arms fade
toward 0 and fresh rewards dominate.

Reward propagation: when one arm receives a reward, nearby arms on the
(day, hour) grid receive a kernel-weighted portion of that reward.  The
kernel uses periodic (wrap-around) distances in both dimensions and
inflates the distance for weekday-to-weekend transitions.
"""

import json
import math
import os
from datetime import datetime

STATE_FILE = os.path.join(os.path.dirname(__file__), "bandit_state.json")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEFAULT_LAMBDA = 0.1

# Kernel hyperparameters for reward propagation
SIGMA_HOUR = 1.5              # hour lengthscale  (correlation ~3-4 hours)
SIGMA_DAY_WITHIN = 1.5       # day lengthscale within same cluster (weekday↔weekday, weekend↔weekend)
SIGMA_DAY_CROSS = 0.8        # day lengthscale across clusters (weekday↔weekend) — tighter = weaker similarity
KERNEL_THRESHOLD = 0.05      # skip arms with weight below this


def arm_key(day: int, hour: int) -> str:
    """Build key string for an arm. day=0..6 (Mon-Sun), hour=0..23."""
    return f"{day}_{hour}"


def _periodic_kernel(diff: float, period: float, lengthscale: float) -> float:
    """ExpSineSquared (periodic) kernel for a single dimension.

        k(x, x') = exp( -2 sin²(π·|x-x'| / p) / ℓ² )

    Maps inputs onto the unit circle via sin, so values separated by
    exactly one period have similarity 1.0 (perfect wrap-around).
    """
    return math.exp(
        -2.0 * math.sin(math.pi * diff / period) ** 2 / (lengthscale ** 2)
    )


def _kernel_weight(d1: int, h1: int, d2: int, h2: int) -> float:
    """Product of periodic kernels over (day, hour) with two-cluster day model.

        w = k_day(d1, d2) · k_hour(h1, h2)

    Hour kernel: ExpSineSquared with period 24 (wraps 23↔0).

    Day kernel: weekdays (Mon-Fri) and weekends (Sat-Sun) are treated as
    two separate clusters.  Within a cluster the lengthscale is tight
    (SIGMA_DAY_WITHIN) so nearby days strongly correlate.  Across clusters
    the lengthscale is wide (SIGMA_DAY_CROSS) so the correlation is weak —
    a Wednesday reward barely touches Saturday, but Saturday and Sunday
    share strongly.
    """
    # Hour kernel — period 24
    k_hour = _periodic_kernel(abs(h1 - h2), 24.0, SIGMA_HOUR)

    # Day kernel — two-cluster: weekdays vs weekends
    day_diff = abs(d1 - d2)
    is_same_cluster = (d1 <= 4) == (d2 <= 4)
    sigma_day = SIGMA_DAY_WITHIN if is_same_cluster else SIGMA_DAY_CROSS
    k_day = _periodic_kernel(day_diff, 7.0, sigma_day)

    return k_day * k_hour


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
    """Apply kernel-propagated, time-decayed update.

    The observed arm gets the full reward.  Neighboring arms on the
    (day, hour) grid receive a kernel-weighted fraction:

        For each arm (d', h'):
            w  = kernel((day, hour), (d', h'))       # ∈ (0, 1]
            α  = (1 - decay) * w                     # kernel-scaled learning rate
            Q' = decay * Q_old  +  α * reward

    Only the observed arm's last_updated timestamp is refreshed so that
    the time-decay of neighbor arms is driven by their own direct
    observations (or lack thereof).
    """
    state = load_state()
    now = datetime.now().astimezone()
    now_iso = now.isoformat()
    lam = state["lambda"]

    for d in range(7):
        for h in range(24):
            w = _kernel_weight(day, hour, d, h)
            if w < KERNEL_THRESHOLD:
                continue

            key = arm_key(d, h)
            q_old = state["q_values"].get(key, 0.0)

            # Time decay since this arm's own last update
            last = state["last_updated"].get(key)
            if last:
                try:
                    last_dt = datetime.fromisoformat(last)
                    time_since = (now - last_dt).total_seconds() / 86400
                except ValueError:
                    time_since = 0.0
            else:
                time_since = 0.0

            decay = math.exp(-lam * time_since)
            alpha = (1 - decay) * w
            q_new = decay * q_old + alpha * reward

            state["q_values"][key] = round(q_new, 4)

    # Only the directly-observed arm gets its timestamp refreshed
    state["last_updated"][arm_key(day, hour)] = now_iso
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
