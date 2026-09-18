from __future__ import annotations

import json
from datetime import datetime, timedelta

from agents.tools import scheduling
from agents.tools.scheduling import schedule_notifications, list_scheduled_nudges, cancel_scheduled_nudge
from care.registry import CareRegistry
from care.nudges import record_delivery, budget_for_day
from care import brief


def _ts(dt):
    return dt.isoformat()


def test_off_level_schedules_nothing(write_config):
    write_config(proactiveness={"level": "off"})
    out = schedule_notifications([{"timestamp": _ts(datetime.now().astimezone() + timedelta(hours=1)), "message": "hi"}])
    assert "off" in out and list_scheduled_nudges() == "No nudges are scheduled."


def test_daily_cap_enforced(write_config):
    write_config(proactiveness={"level": "minimal"})  # cap 1
    now = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
    base = now + timedelta(days=1)
    out = schedule_notifications([
        {"timestamp": _ts(base + timedelta(hours=1)), "message": "first"},
        {"timestamp": _ts(base + timedelta(hours=2)), "message": "second"},
    ])
    assert "Scheduled 1 nudge" in out and "daily cap" in out
    q = json.load(open(scheduling.QUEUE_FILE))
    assert len(q) == 1 and q[0]["message"] == "first" and q[0]["id"].startswith("inv_")


def test_delivered_nudges_count_toward_cap(write_config):
    write_config(proactiveness={"level": "balanced"})  # cap 3
    now = datetime.now().astimezone()
    for i in range(3):
        record_delivery(f"sent {i}", when=now)
    assert budget_for_day(now.strftime("%Y-%m-%d"), 3, now) == 0
    later = now.replace(hour=12) + timedelta(hours=1)
    if later.date() != now.date():
        later = now + timedelta(minutes=5)
    out = schedule_notifications([{"timestamp": _ts(later), "message": "x"}])
    assert "Scheduled 0" in out


def test_quiet_hours_shift_and_critical_bypass(write_config):
    write_config(proactiveness={"level": "active"}, overrides={"quiet_hours": {"start": "22:00", "end": "08:00"}})
    tomorrow = (datetime.now().astimezone() + timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    out = schedule_notifications([
        {"timestamp": _ts(tomorrow), "message": "late one"},
        {"timestamp": _ts(tomorrow + timedelta(minutes=5)), "message": "critical one", "priority": "critical"},
    ])
    q = {e["message"]: e for e in json.load(open(scheduling.QUEUE_FILE))}
    assert datetime.fromisoformat(q["late one"]["timestamp"]).hour == 8
    assert datetime.fromisoformat(q["critical one"]["timestamp"]).hour == 23
    assert "quiet hours" in out


def test_item_link_bumps_nudge_count_and_cancel(write_config):
    write_config()
    reg = CareRegistry()
    item, _ = reg.add(type="task", title="thing")
    out = schedule_notifications([{"timestamp": _ts(datetime.now().astimezone().replace(hour=12) + timedelta(days=1)), "message": "do thing", "item_id": item["id"]}])
    assert "Scheduled 1" in out
    assert CareRegistry().get(item["id"])["reminder_count"] == 1
    nid = json.load(open(scheduling.QUEUE_FILE))[0]["id"]
    assert "cancelled" in cancel_scheduled_nudge(nid)
    assert "No scheduled nudge" in cancel_scheduled_nudge(nid)


def test_generate_brief_respects_threshold_and_writes_file(write_config):
    write_config(care_mode="focus")  # urgency threshold high
    reg = CareRegistry()
    reg.add(type="task", title="Big deal", urgency="high")
    reg.add(type="task", title="Small thing", urgency="low")
    b = brief.generate_brief(reg)
    assert [i["summary"] for i in b["priority_items"]] == ["Big deal"]
    assert brief.has_todays_brief()
    assert reg.get_meta("last_review")
    data = json.loads(brief.DAILY_BRIEF_FILE.read_text())
    assert data["care_mode"] == "focus"


def test_session_context_and_digest(write_config):
    write_config()
    assert brief.session_context() == ""          # empty registry → nothing injected
    reg = CareRegistry()
    reg.add(type="commitment", title="Call Simran", deadline=(datetime.now().astimezone() + timedelta(hours=2)).isoformat())
    ctx = brief.session_context()
    assert ctx.startswith("<care_context>") and "Call Simran" in ctx and "Due within 36h" in ctx
    write_config(care_mode="heads_down")
    assert brief.session_context() == ""          # surfacing disabled in heads_down


def test_spacing_and_morning_review_headroom(write_config):
    write_config(proactiveness={"level": "balanced"})  # cap 3 → morning review may use 2
    day = (datetime.now().astimezone() + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    out = schedule_notifications([
        {"timestamp": _ts(day), "message": "a", "source": "morning_review"},
        {"timestamp": _ts(day), "message": "b", "source": "morning_review"},
        {"timestamp": _ts(day), "message": "c", "source": "morning_review"},
    ])
    q = sorted(json.load(open(scheduling.QUEUE_FILE)), key=lambda e: e["timestamp"])
    assert [e["message"] for e in q] == ["a", "b"]           # headroom of 1 left for pre-exit
    t0, t1 = (datetime.fromisoformat(e["timestamp"]) for e in q)
    assert (t1 - t0).total_seconds() >= 30 * 60               # spaced apart
    assert "spacing" in out and "daily cap" in out
    # pre-exit can still use the last slot
    out = schedule_notifications([{"timestamp": _ts(day + timedelta(hours=3)), "message": "d"}])
    assert "Scheduled 1" in out
