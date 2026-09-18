from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from care.registry import CareRegistry, OPEN_STATUSES


def _now():
    return datetime.now().astimezone()


def test_add_and_persist(isolated_paths):
    reg = CareRegistry()
    item, created = reg.add(type="commitment", title="Call Riya tonight", urgency="medium")
    assert created and item["status"] == "new" and item["id"].startswith("care_")
    reg2 = CareRegistry()
    assert reg2.get(item["id"])["title"] == "Call Riya tonight"


def test_dedupe_by_title_and_ref():
    reg = CareRegistry()
    a, _ = reg.add(type="email", title="Reply to Maya re term sheet", urgency="medium", source_ref="msg1")
    b, created = reg.add(type="email", title="Reply to Maya re: term sheet", urgency="high", source_ref="msg1")
    assert not created and b["id"] == a["id"] and b["urgency"] == "high"   # urgency only goes up
    c, created = reg.add(type="email", title="reply to maya re term sheet", urgency="low")
    assert not created and c["id"] == a["id"] and c["urgency"] == "high"
    d, created = reg.add(type="task", title="Reply to Maya re term sheet")   # different type → new
    assert created


def test_validation():
    reg = CareRegistry()
    with pytest.raises(ValueError):
        reg.add(type="bogus", title="x")
    with pytest.raises(ValueError):
        reg.add(type="task", title="")
    item, _ = reg.add(type="task", title="x")
    with pytest.raises(ValueError):
        reg.update(item["id"], status="weird")
    with pytest.raises(KeyError):
        reg.update("care_nope", status="done")


def test_lifecycle_and_intent():
    reg = CareRegistry()
    item, _ = reg.add(type="commitment", title="Sort out visa")
    reg.update(item["id"], user_intent="will do tonight", status="deferred")
    reg.update(item["id"], status="deferred")
    it = reg.get(item["id"])
    assert it["deferral_count"] == 2 and it["user_intent"] == "will do tonight" and it["intent_set_at"]
    reg.resolve(item["id"], note="done via chat")
    it = reg.get(item["id"])
    assert it["status"] == "done" and it["resolved_at"] and "done via chat" in it["notes"]
    assert reg.list("open") == [] and len(reg.list("closed")) == 1


def test_find_fuzzy():
    reg = CareRegistry()
    reg.add(type="email", title="Reply to Maya about the term sheet", sender="maya@sequoia.com")
    reg.add(type="task", title="Finish Q3 board deck")
    assert reg.find("maya")[0][1]["title"].startswith("Reply to Maya")
    assert reg.find("board deck")[0][1]["title"] == "Finish Q3 board deck"
    assert reg.find("zzz") == []


def test_tend_snooze_expiry_and_escalation(write_config):
    write_config()  # escalate_after_deferrals=3 by default
    reg = CareRegistry()
    now = _now()
    snoozed, _ = reg.add(type="task", title="Snoozed thing")
    reg.snooze(snoozed["id"], now - timedelta(minutes=1), intent="later")
    deferred, _ = reg.add(type="task", title="Deferred thing", urgency="low")
    for _ in range(3):
        reg.update(deferred["id"], status="deferred")
    report = reg.tend(now=now)
    assert any(snoozed["id"] in s for s in report["resurfaced"])
    assert reg.get(snoozed["id"])["status"] == "acknowledged"
    assert any(deferred["id"] in s for s in report["escalated"])
    assert reg.get(deferred["id"])["urgency"] == "medium"
    # escalation happens once
    assert reg.tend(now=now)["escalated"] == []


def test_tend_overdue_promise_and_deadline():
    reg = CareRegistry()
    now = _now()
    promise, _ = reg.add(type="commitment", title="Call mom")
    reg.update(promise["id"], user_intent="tonight")
    dl, _ = reg.add(type="task", title="Submit form", deadline=(now - timedelta(hours=1)).isoformat(), urgency="low")
    report = reg.tend(now=now + timedelta(days=1))
    ids = " ".join(report["overdue"])
    assert promise["id"] in ids and dl["id"] in ids
    assert reg.get(dl["id"])["urgency"] == "high"
    assert reg.get(promise["id"])["_overdue"] is True


def test_tend_archives_stale_low_items_keeps_high(write_config):
    write_config(overrides={"item_max_age_days": 5})
    reg = CareRegistry()
    now = _now()
    low, _ = reg.add(type="email", title="Old newsletter reply", urgency="low")
    high, _ = reg.add(type="task", title="Old but important", urgency="high")
    report = reg.tend(now=now + timedelta(days=6))
    assert reg.get(low["id"])["status"] == "archived"
    assert reg.get(high["id"])["status"] in OPEN_STATUSES
    assert any(high["id"] in s for s in report["stale"])


def test_due_soon_and_sorting():
    reg = CareRegistry()
    now = _now()
    reg.add(type="task", title="Later", urgency="low", deadline=(now + timedelta(days=5)).isoformat())
    reg.add(type="task", title="Soon", urgency="low", deadline=(now + timedelta(hours=3)).isoformat())
    reg.add(type="task", title="Urgent no deadline", urgency="high")
    assert [i["title"] for i in reg.due_soon(36, now)] == ["Soon"]
    assert reg.list("open")[0]["title"] == "Urgent no deadline"


def test_record_nudge_and_meta():
    reg = CareRegistry()
    item, _ = reg.add(type="task", title="x")
    reg.record_nudge(item["id"])
    assert reg.get(item["id"])["reminder_count"] == 1
    reg.mark_source_checked("gmail")
    assert reg.get_meta("last_gmail_check")
    with pytest.raises(ValueError):
        reg.mark_source_checked("fax")
