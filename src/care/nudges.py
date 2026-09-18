"""Nudge budget: how many proactive messages have gone out today.

Backed by engagement_data/nudge_log.json. Both scheduled (queued) and
delivered nudges count toward the daily cap so the agent cannot over-schedule.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import NUDGE_LOG_FILE, QUEUE_FILE

_lock = threading.RLock()


def _load() -> list[dict[str, Any]]:
    p = Path(NUDGE_LOG_FILE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    p = Path(NUDGE_LOG_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries[-500:], indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def record_delivery(message: str, source: str = "", item_id: str = "", when: datetime | None = None) -> None:
    when = when or datetime.now().astimezone()
    with _lock:
        entries = _load()
        entries.append({"at": when.isoformat(), "date": when.strftime("%Y-%m-%d"),
                        "message": message[:200], "source": source, "item_id": item_id})
        _save(entries)


def delivered_today(now: datetime | None = None) -> int:
    now = now or datetime.now().astimezone()
    today = now.strftime("%Y-%m-%d")
    return sum(1 for e in _load() if e.get("date") == today)


def queued_for_day(day: str, exclude_sources: tuple[str, ...] = ("subagent_complete",)) -> int:
    p = Path(QUEUE_FILE)
    if not p.exists():
        return 0
    try:
        queue = json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return 0
    n = 0
    for e in queue:
        if e.get("source") in exclude_sources:
            continue
        ts = e.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.astimezone()
            if dt.astimezone().strftime("%Y-%m-%d") == day:
                n += 1
        except ValueError:
            continue
    return n


def budget_for_day(day: str, max_per_day: int, now: datetime | None = None) -> int:
    """Remaining nudges allowed for `day` (YYYY-MM-DD)."""
    now = now or datetime.now().astimezone()
    used = queued_for_day(day)
    if day == now.strftime("%Y-%m-%d"):
        used += delivered_today(now)
    return max(0, max_per_day - used)
