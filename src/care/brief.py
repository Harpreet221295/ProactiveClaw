"""Views over the registry: the daily brief file and prompt digests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from core.paths import AGENT_FS_DIR
from core.config import URGENCY_RANK, effective_settings, load_config
from .registry import CareRegistry, _parse
from .patterns import CarePatterns

DAILY_BRIEF_FILE = AGENT_FS_DIR / "daily_brief.json"


def _fmt_deadline(item: dict[str, Any], now: datetime) -> str:
    dl = _parse(item.get("deadline"))
    if not dl:
        return ""
    delta = dl - now
    if delta.total_seconds() < 0:
        return f"OVERDUE since {dl.strftime('%b %d')}"
    if delta < timedelta(hours=36):
        return f"due {dl.strftime('%a %H:%M')}"
    return f"due {dl.strftime('%b %d')}"


def item_line(item: dict[str, Any], now: datetime | None = None, with_id: bool = True) -> str:
    now = now or datetime.now().astimezone()
    bits = [f"[{item.get('urgency','?')}]", item.get("title", "")]
    meta = []
    if item.get("sender"):
        meta.append(f"from {item['sender']}")
    dl = _fmt_deadline(item, now)
    if dl:
        meta.append(dl)
    st = item.get("status")
    if st and st not in ("new",):
        meta.append(st)
    if item.get("user_intent"):
        meta.append(f"you said: \"{item['user_intent']}\"")
    if item.get("_overdue"):
        meta.append("⚠ overdue")
    if item.get("reminder_count"):
        meta.append(f"nudged {item['reminder_count']}x")
    line = " ".join(bits)
    if meta:
        line += "  (" + "; ".join(meta) + ")"
    if with_id:
        line += f"  id={item['id']}"
    return line


def generate_brief(registry: CareRegistry | None = None, cfg: dict[str, Any] | None = None,
                   now: datetime | None = None, write: bool = True) -> dict[str, Any]:
    """Build daily_brief.json as a curated view of the registry's open items."""
    cfg = cfg or load_config()
    eff = effective_settings(cfg)
    reg = registry or CareRegistry()
    now = now or datetime.now().astimezone()
    threshold = URGENCY_RANK[eff["urgency_threshold"]]

    candidates = [i for i in reg.open_items()
                  if URGENCY_RANK.get(i.get("urgency", "low"), 1) >= threshold
                  and i.get("status") != "snoozed"]
    candidates.sort(key=CareRegistry._sort_key)
    top = candidates[: eff["max_brief_items"]]

    priority_items = []
    for it in top:
        priority_items.append({
            "id": it["id"],
            "type": it.get("type"),
            "summary": it.get("title"),
            "urgency": it.get("urgency"),
            "source": it.get("source"),
            "deadline": it.get("deadline"),
            "status": it.get("status"),
            "user_intent": it.get("user_intent") or "",
            "details": (it.get("details") or "")[:300],
        })

    high = [i for i in top if i.get("urgency") == "high"]
    due = reg.due_soon(36, now)
    overview_bits = []
    if high:
        overview_bits.append(f"{len(high)} high-urgency item{'s' if len(high) != 1 else ''}")
    if due:
        overview_bits.append(f"{len(due)} due within 36h")
    remaining = len(candidates) - len(top)
    if remaining > 0:
        overview_bits.append(f"{remaining} more lower-priority items tracked")
    overview = ("Today: " + ", ".join(overview_bits) + ".") if overview_bits else "Nothing pressing is being tracked right now."

    brief = {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "level": eff["level"],
        "care_mode": eff["care_mode"],
        "priority_items": priority_items,
        "day_overview": overview,
        "counts": reg.stats(),
    }
    if write:
        DAILY_BRIEF_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAILY_BRIEF_FILE.write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reg.set_meta("last_review", now.isoformat())
    return brief


def has_todays_brief(now: datetime | None = None) -> bool:
    now = now or datetime.now().astimezone()
    if not DAILY_BRIEF_FILE.exists():
        return False
    try:
        data = json.loads(DAILY_BRIEF_FILE.read_text(encoding="utf-8"))
        return data.get("date") == now.strftime("%Y-%m-%d")
    except (json.JSONDecodeError, OSError):
        return False


def registry_digest(registry: CareRegistry | None = None, cfg: dict[str, Any] | None = None,
                    now: datetime | None = None, max_items: int = 15) -> str:
    """Compact text digest of the registry for injection into prompts."""
    cfg = cfg or load_config()
    eff = effective_settings(cfg)
    reg = registry or CareRegistry()
    now = now or datetime.now().astimezone()

    open_items = reg.list("open")
    if not open_items:
        return "Care registry: nothing is being tracked right now."

    lines = [f"Care registry: {len(open_items)} open item(s) (level={eff['level']}, mode={eff['care_mode']})."]
    due = reg.due_soon(36, now)
    if due:
        lines.append("Due within 36h:")
        lines += [f"  • {item_line(i, now)}" for i in due[:6]]
    overdue = [i for i in open_items if i.get("_overdue") and i not in due]
    if overdue:
        lines.append("Overdue / broken promises:")
        lines += [f"  • {item_line(i, now)}" for i in overdue[:6]]
    snoozed = [i for i in open_items if i.get("status") == "snoozed"]
    soon = []
    for i in snoozed:
        su = _parse(i.get("snooze_until"))
        if su and su <= now + timedelta(hours=24):
            soon.append(i)
    if soon:
        lines.append("Snoozed, resurfacing within 24h:")
        lines += [f"  • {item_line(i, now)}" for i in soon[:4]]
    shown = set(i["id"] for i in due + overdue + soon)
    rest = [i for i in open_items if i["id"] not in shown and i.get("status") != "snoozed"]
    if rest:
        lines.append("Other open items (by urgency):")
        lines += [f"  • {item_line(i, now)}" for i in rest[:max_items]]
        if len(rest) > max_items:
            lines.append(f"  … and {len(rest) - max_items} more (use care_list to see all)")
    return "\n".join(lines)


def session_context(cfg: dict[str, Any] | None = None) -> str:
    """The <care_context> block prepended to the first message of a session."""
    cfg = cfg or load_config()
    eff = effective_settings(cfg)
    if not eff.get("surface_care_items_in_chat", True):
        return ""
    reg = CareRegistry()
    if not reg.open_items():
        return ""
    digest = registry_digest(reg, cfg)
    return (
        "<care_context>\n"
        "Background from your care registry (things you are tracking for the user). "
        "Use it to be thoughtfully proactive: after answering, weave in at most 1-2 relevant items. "
        "Never dump the list. Update items with the care_* tools when the user tells you something about them.\n"
        f"{digest}\n"
        "</care_context>"
    )


def tend_report_text(report: dict[str, list[str]]) -> str:
    parts = []
    labels = {"resurfaced": "Resurfaced (snooze expired)", "escalated": "Escalated (repeated deferrals)",
              "overdue": "Overdue / broken promises", "archived": "Auto-archived (stale)",
              "stale": "Stale but kept (high urgency or has deadline)"}
    for k, label in labels.items():
        if report.get(k):
            parts.append(f"{label}:\n" + "\n".join(f"  • {x}" for x in report[k]))
    return "\n".join(parts) if parts else "No housekeeping changes."


def patterns_digest() -> str:
    return CarePatterns().digest()
