"""care_registry.json — the source of truth for everything the assistant is tracking.

Item lifecycle:

    new → acknowledged / in_progress / deferred / snoozed / dismissed / done / archived

Each item records what the user said about it (`user_intent`), when it was last
nudged, how many times it was deferred, and where it came from. The daily brief
is a *view* generated from this registry, never the other way round.

`tend()` is the deterministic housekeeping pass that the morning review runs
before any LLM call: it resurfaces expired snoozes, escalates repeatedly
deferred items, flags overdue "will do tonight" promises and archives stale
low-value items.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from core.paths import CARE_REGISTRY_FILE
from core.config import URGENCY_RANK, effective_settings, load_config

ITEM_TYPES = ("email", "task", "commitment", "followup", "calendar")
OPEN_STATUSES = ("new", "acknowledged", "in_progress", "deferred", "snoozed")
CLOSED_STATUSES = ("done", "dismissed", "archived")
ALL_STATUSES = OPEN_STATUSES + CLOSED_STATUSES
SOURCES = ("gmail", "notion", "calendar", "conversation", "morning_review", "manual")

_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt
    except (ValueError, TypeError):
        return None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _similar(a: str, b: str) -> float:
    """Cheap token-overlap similarity in [0, 1] for de-duplication."""
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class CareRegistry:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else Path(CARE_REGISTRY_FILE)
        self._data: dict[str, Any] | None = None

    # ── persistence ─────────────────────────────────────────────────────────

    def _empty(self) -> dict[str, Any]:
        return {
            "items": [],
            "meta": {
                "last_gmail_check": None,
                "last_notion_check": None,
                "last_calendar_check": None,
                "last_tended": None,
                "last_review": None,
            },
        }

    def load(self) -> dict[str, Any]:
        with _lock:
            if self._data is None:
                if self.path.exists():
                    try:
                        self._data = json.loads(self.path.read_text(encoding="utf-8")) or self._empty()
                    except json.JSONDecodeError:
                        self._data = self._empty()
                else:
                    self._data = self._empty()
                self._data.setdefault("items", [])
                self._data.setdefault("meta", self._empty()["meta"])
            return self._data

    def save(self) -> None:
        with _lock:
            data = self.load()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)

    def reload(self) -> None:
        self._data = None
        self.load()

    # ── meta ────────────────────────────────────────────────────────────────

    def get_meta(self, key: str) -> Any:
        return self.load()["meta"].get(key)

    def set_meta(self, key: str, value: Any) -> None:
        self.load()["meta"][key] = value
        self.save()

    def mark_source_checked(self, source: str, when: datetime | None = None) -> None:
        key = {"gmail": "last_gmail_check", "notion": "last_notion_check",
               "calendar": "last_calendar_check"}.get(source)
        if not key:
            raise ValueError(f"Unknown source '{source}'")
        self.set_meta(key, _iso(when or _now()))

    # ── queries ─────────────────────────────────────────────────────────────

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.load()["items"]

    def get(self, item_id: str) -> dict[str, Any] | None:
        return next((i for i in self.items if i.get("id") == item_id), None)

    def open_items(self) -> list[dict[str, Any]]:
        return [i for i in self.items if i.get("status") in OPEN_STATUSES]

    def list(self, status: str | Iterable[str] | None = "open", item_type: str | None = None,
             limit: int = 0) -> list[dict[str, Any]]:
        if status == "open":
            wanted: set[str] | None = set(OPEN_STATUSES)
        elif status == "closed":
            wanted = set(CLOSED_STATUSES)
        elif status in (None, "all"):
            wanted = None
        elif isinstance(status, str):
            wanted = {status}
        else:
            wanted = set(status)
        out = [i for i in self.items
               if (wanted is None or i.get("status") in wanted)
               and (item_type is None or i.get("type") == item_type)]
        out.sort(key=self._sort_key)
        return out[:limit] if limit else out

    @staticmethod
    def _sort_key(item: dict[str, Any]):
        urg = URGENCY_RANK.get(item.get("urgency", "low"), 1)
        deadline = _parse(item.get("deadline"))
        # sort: higher urgency first, sooner deadline first, older first
        return (-urg, deadline or datetime.max.replace(tzinfo=_now().tzinfo), item.get("created_at") or "")

    def find(self, query: str, threshold: float = 0.34, include_closed: bool = False,
             limit: int = 5) -> list[tuple[float, dict[str, Any]]]:
        """Fuzzy search by title/sender/details. Returns [(score, item)] best-first."""
        q = _norm(query)
        if not q:
            return []
        pool = self.items if include_closed else self.open_items()
        scored: list[tuple[float, dict[str, Any]]] = []
        for it in pool:
            hay = " ".join(str(it.get(k, "")) for k in ("title", "sender", "details", "notes"))
            s = _similar(q, it.get("title", ""))
            if q in _norm(hay):
                s = max(s, 0.9)
            for tok in q.split():
                if len(tok) > 3 and tok in _norm(hay):
                    s = max(s, 0.45)
            if s >= threshold:
                scored.append((round(s, 2), it))
        scored.sort(key=lambda x: (-x[0], self._sort_key(x[1])))
        return scored[:limit]

    def find_by_ref(self, source_ref: str) -> dict[str, Any] | None:
        if not source_ref:
            return None
        return next((i for i in self.items if i.get("source_ref") == source_ref), None)

    # ── mutations ───────────────────────────────────────────────────────────

    def add(self, *, type: str, title: str, urgency: str = "medium", source: str = "conversation",
            details: str = "", deadline: str | None = None, sender: str = "", topics: list[str] | None = None,
            source_ref: str = "", confidence: float = 1.0, notes: str = "", status: str = "new",
            dedupe: bool = True) -> tuple[dict[str, Any], bool]:
        """Add an item. Returns (item, created). With dedupe, a near-duplicate open
        item is updated instead and created=False."""
        if type not in ITEM_TYPES:
            raise ValueError(f"type must be one of {ITEM_TYPES}")
        if urgency not in URGENCY_RANK:
            raise ValueError("urgency must be high, medium or low")
        if status not in ALL_STATUSES:
            raise ValueError(f"status must be one of {ALL_STATUSES}")
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")

        with _lock:
            if dedupe:
                existing = self.find_by_ref(source_ref) if source_ref else None
                if existing is None:
                    for score, cand in self.find(title, threshold=0.6, include_closed=False, limit=3):
                        if cand.get("type") == type:
                            existing = cand
                            break
                if existing is not None:
                    changed = False
                    if URGENCY_RANK[urgency] > URGENCY_RANK.get(existing.get("urgency"), 0):
                        existing["urgency"] = urgency
                        changed = True
                    if deadline and not existing.get("deadline"):
                        existing["deadline"] = deadline
                        changed = True
                    if details and details not in (existing.get("details") or ""):
                        existing["details"] = ((existing.get("details") or "") + "\n" + details).strip()
                        changed = True
                    if changed:
                        existing["last_updated"] = _iso(_now())
                        self.save()
                    return existing, False

            now = _iso(_now())
            item = {
                "id": f"care_{secrets.token_hex(3)}",
                "type": type,
                "source": source,
                "source_ref": source_ref or "",
                "title": title,
                "details": details or "",
                "sender": sender or "",
                "topics": [t.lower() for t in (topics or [])],
                "urgency": urgency,
                "confidence": float(confidence),
                "status": status,
                "user_intent": "",
                "intent_set_at": None,
                "snooze_until": None,
                "deadline": deadline,
                "reminder_count": 0,
                "deferral_count": 0,
                "last_nudged_at": None,
                "notes": notes or "",
                "created_at": now,
                "last_updated": now,
                "resolved_at": None,
                "history": [{"at": now, "event": "created", "status": status}],
            }
            self.items.append(item)
            self.save()
            return item, True

    def update(self, item_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"status", "user_intent", "snooze_until", "urgency", "notes", "deadline",
                   "details", "title", "topics", "sender", "confidence"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {sorted(bad)}")
        with _lock:
            item = self.get(item_id)
            if item is None:
                raise KeyError(f"No care item with id '{item_id}'")
            now = _iso(_now())
            events: list[str] = []

            new_status = fields.get("status")
            if new_status is not None:
                if new_status not in ALL_STATUSES:
                    raise ValueError(f"status must be one of {ALL_STATUSES}")
                if new_status == "deferred":
                    item["deferral_count"] = int(item.get("deferral_count", 0)) + 1
                    events.append(f"deferred #{item['deferral_count']}")
                if new_status != item.get("status"):
                    events.append(f"status:{item.get('status')}->{new_status}")
                    if new_status in CLOSED_STATUSES:
                        item["resolved_at"] = now
                    else:
                        item["resolved_at"] = None
                    item["status"] = new_status

            if "user_intent" in fields and fields["user_intent"] is not None:
                item["user_intent"] = str(fields["user_intent"]).strip()
                item["intent_set_at"] = now
                events.append("intent")

            if "snooze_until" in fields:
                su = fields["snooze_until"]
                item["snooze_until"] = su
                if su:
                    item["status"] = "snoozed"
                    events.append("snoozed")

            if "urgency" in fields and fields["urgency"] is not None:
                if fields["urgency"] not in URGENCY_RANK:
                    raise ValueError("urgency must be high, medium or low")
                if fields["urgency"] != item.get("urgency"):
                    events.append(f"urgency:{item.get('urgency')}->{fields['urgency']}")
                item["urgency"] = fields["urgency"]

            for k in ("notes", "deadline", "details", "title", "sender", "confidence", "topics"):
                if k in fields and fields[k] is not None:
                    if k == "notes" and fields[k]:
                        item["notes"] = (item.get("notes") + "\n" + fields[k]).strip() if item.get("notes") else fields[k]
                    else:
                        item[k] = fields[k]

            item["last_updated"] = now
            if events:
                item.setdefault("history", []).append({"at": now, "event": ", ".join(events)})
            self.save()
            return item

    def resolve(self, item_id: str, note: str = "") -> dict[str, Any]:
        return self.update(item_id, status="done", notes=note or None)

    def dismiss(self, item_id: str, note: str = "") -> dict[str, Any]:
        return self.update(item_id, status="dismissed", notes=note or None)

    def snooze(self, item_id: str, until: datetime | str, intent: str | None = None) -> dict[str, Any]:
        until_iso = until if isinstance(until, str) else _iso(until)
        return self.update(item_id, snooze_until=until_iso, user_intent=intent)

    def record_nudge(self, item_id: str, when: datetime | None = None) -> None:
        with _lock:
            item = self.get(item_id)
            if item is None:
                return
            item["reminder_count"] = int(item.get("reminder_count", 0)) + 1
            item["last_nudged_at"] = _iso(when or _now())
            item["last_updated"] = _iso(_now())
            self.save()

    def remove(self, item_id: str) -> bool:
        with _lock:
            before = len(self.items)
            self.load()["items"] = [i for i in self.items if i.get("id") != item_id]
            self.save()
            return len(self.items) < before

    # ── tending ─────────────────────────────────────────────────────────────

    def tend(self, cfg: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, list[str]]:
        """Deterministic housekeeping. Returns a report of what changed.

        - snoozed items whose snooze expired → back to `acknowledged`
        - deferred >= escalate_after_deferrals times → urgency bumped one step
        - "tonight / today" intents still open next day → flagged overdue
        - deadline passed and still open → urgency high, flagged overdue
        - low/medium items untouched for item_max_age_days → archived
        - high items untouched that long → flagged stale (kept)
        """
        eff = effective_settings(cfg or load_config())
        now = now or _now()
        report: dict[str, list[str]] = {
            "resurfaced": [], "escalated": [], "overdue": [], "archived": [], "stale": [],
        }
        with _lock:
            for item in self.items:
                if item.get("status") not in OPEN_STATUSES:
                    continue
                iid, title = item["id"], item.get("title", "")

                # snooze expiry
                if item.get("status") == "snoozed":
                    su = _parse(item.get("snooze_until"))
                    if su is None or su <= now:
                        item["status"] = "acknowledged"
                        item["snooze_until"] = None
                        item.setdefault("history", []).append({"at": _iso(now), "event": "snooze expired"})
                        report["resurfaced"].append(f"{iid}: {title}")

                # deferral escalation
                if (item.get("deferral_count", 0) >= eff["escalate_after_deferrals"]
                        and not item.get("_escalated")):
                    cur = item.get("urgency", "low")
                    nxt = {"low": "medium", "medium": "high", "high": "high"}[cur]
                    if nxt != cur:
                        item["urgency"] = nxt
                        item["_escalated"] = True
                        item.setdefault("history", []).append({"at": _iso(now), "event": f"escalated {cur}->{nxt} after {item['deferral_count']} deferrals"})
                        report["escalated"].append(f"{iid}: {title} ({cur}→{nxt})")

                # overdue deadline
                dl = _parse(item.get("deadline"))
                if dl and dl < now and not item.get("_overdue"):
                    item["_overdue"] = True
                    item["urgency"] = "high"
                    item.setdefault("history", []).append({"at": _iso(now), "event": "deadline passed"})
                    report["overdue"].append(f"{iid}: {title} (deadline {dl.strftime('%b %d %H:%M')})")

                # "tonight/today" promises still open the next day
                intent = (item.get("user_intent") or "").lower()
                set_at = _parse(item.get("intent_set_at"))
                if set_at and not item.get("_overdue") and any(w in intent for w in ("tonight", "today", "this evening", "in an hour", "right now", "asap")):
                    if now.date() > set_at.date():
                        item["_overdue"] = True
                        if URGENCY_RANK[item.get("urgency", "low")] < URGENCY_RANK["high"]:
                            item["urgency"] = "high"
                        item.setdefault("history", []).append({"at": _iso(now), "event": f"promised '{intent}' on {set_at.date()} — still open"})
                        report["overdue"].append(f"{iid}: {title} (you said '{intent}' on {set_at.strftime('%b %d')})")

                # staleness / auto-archive
                last = _parse(item.get("last_updated")) or _parse(item.get("created_at")) or now
                age_days = (now - last).total_seconds() / 86400
                if age_days >= eff["item_max_age_days"]:
                    if item.get("urgency") == "high" or dl:
                        if not item.get("_stale"):
                            item["_stale"] = True
                            report["stale"].append(f"{iid}: {title} (untouched {int(age_days)}d)")
                    else:
                        item["status"] = "archived"
                        item["resolved_at"] = _iso(now)
                        item.setdefault("history", []).append({"at": _iso(now), "event": f"auto-archived after {int(age_days)}d"})
                        report["archived"].append(f"{iid}: {title}")

            self.load()["meta"]["last_tended"] = _iso(now)
            self.save()
        return report

    # ── views ───────────────────────────────────────────────────────────────

    def due_soon(self, hours: int = 36, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or _now()
        horizon = now + timedelta(hours=hours)
        out = []
        for it in self.open_items():
            dl = _parse(it.get("deadline"))
            if dl and dl <= horizon:
                out.append(it)
        out.sort(key=lambda i: _parse(i.get("deadline")) or now)
        return out

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for it in self.items:
            counts[it.get("status", "?")] = counts.get(it.get("status", "?"), 0) + 1
        counts["open"] = len(self.open_items())
        counts["total"] = len(self.items)
        return counts
