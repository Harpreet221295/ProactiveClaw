"""Agent-facing tools for the care registry, patterns and proactiveness config."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from care.registry import CareRegistry, ITEM_TYPES, ALL_STATUSES
from care.patterns import CarePatterns, KINDS, SIGNALS
from care.brief import generate_brief, registry_digest, item_line
from core import config as _config


def _reg() -> CareRegistry:
    return CareRegistry()


def _pat() -> CarePatterns:
    return CarePatterns()


def _fmt(item: dict) -> str:
    return item_line(item)


# ── Registry ────────────────────────────────────────────────────────────────

def care_list(status: str = "open", item_type: str = "", limit: int = 25) -> str:
    reg = _reg()
    items = reg.list(status=status or "open", item_type=item_type or None, limit=limit)
    if not items:
        return f"No care items with status '{status}'."
    lines = [f"{len(items)} item(s):"] + [f"• {_fmt(i)}" for i in items]
    return "\n".join(lines)


def care_digest() -> str:
    return registry_digest(_reg())


def care_find(query: str) -> str:
    matches = _reg().find(query, include_closed=False)
    if not matches:
        closed = _reg().find(query, include_closed=True)
        if closed:
            return "No open match. Closed matches:\n" + "\n".join(f"• ({s}) {_fmt(i)}" for s, i in closed)
        return f"No care item matches '{query}'."
    return "\n".join(f"• (match {s}) {_fmt(i)}" for s, i in matches)


def care_add_item(title: str, type: str = "commitment", urgency: str = "medium", deadline: str = "",
                  details: str = "", source: str = "conversation", sender: str = "", topics: list[str] | None = None,
                  source_ref: str = "", confidence: float = 1.0, category: str = "") -> str:
    if type not in ITEM_TYPES:
        return f"Error: type must be one of {', '.join(ITEM_TYPES)}"
    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    if eff["commitment_capture"] == "off" and source == "conversation" and type == "commitment":
        return "Commitment capture is off at the current proactiveness level — not tracked. (The user can raise the level in Settings.)"
    probe = {"title": title, "details": details, "sender": sender, "topics": topics or [], "source": source, "category": category}
    adj_urgency, filtered, reasons = _pat().apply(urgency if urgency in ("high", "medium", "low") else "medium", probe, cfg)
    if filtered and source != "conversation":
        return f"Filtered — not tracked ({'; '.join(reasons)})."
    try:
        item, created = _reg().add(type=type, title=title, urgency=adj_urgency, source=source, details=details,
                                   deadline=deadline or None, sender=sender, topics=topics or [],
                                   source_ref=source_ref, confidence=confidence)
    except (ValueError, KeyError) as e:
        return f"Error: {e}"
    note = f" (urgency adjusted: {'; '.join(reasons)})" if reasons and adj_urgency != urgency else ""
    if created:
        return f"Tracked: {_fmt(item)}{note}"
    return f"Already tracked (merged): {_fmt(item)}"


def care_update_item(item_id: str, status: str = "", user_intent: str = "", snooze_until: str = "",
                     urgency: str = "", notes: str = "", deadline: str = "", feedback_signal: str = "") -> str:
    reg = _reg()
    item = reg.get(item_id)
    if item is None:
        return f"Error: no care item '{item_id}'. Use care_find to locate it."
    fields: dict = {}
    if status:
        if status not in ALL_STATUSES:
            return f"Error: status must be one of {', '.join(ALL_STATUSES)}"
        fields["status"] = status
    if user_intent:
        fields["user_intent"] = user_intent
    if snooze_until:
        fields["snooze_until"] = snooze_until
    if urgency:
        fields["urgency"] = urgency
    if notes:
        fields["notes"] = notes
    if deadline:
        fields["deadline"] = deadline
    try:
        item = reg.update(item_id, **fields)
    except (ValueError, KeyError) as e:
        return f"Error: {e}"
    # learn from the reaction
    signal = feedback_signal or {"done": "acted", "in_progress": "acted", "dismissed": "dismissed",
                                 "deferred": "deferred", "snoozed": "deferred"}.get(status, "")
    if signal in SIGNALS:
        _pat().observe_item(item, signal, note=user_intent or notes)
    return f"Updated: {_fmt(item)}"


def care_resolve_item(item_id: str, note: str = "") -> str:
    return care_update_item(item_id, status="done", notes=note)


def care_record_feedback(kind: str, key: str, action: str, note: str = "") -> str:
    """Record an explicit user preference: mute/boost a sender, topic, category or source,
    or a one-off reaction (acted/dismissed/deferred)."""
    if kind not in KINDS:
        return f"Error: kind must be one of {', '.join(KINDS)}"
    pat = _pat()
    if action in ("mute", "boost"):
        rule = pat.add_rule(kind, key, action, note)
        return f"Rule saved: {rule['action']} {rule['kind']} '{rule['key']}'."
    if action == "clear":
        return "Rule removed." if pat.remove_rule(kind, key) else "No such rule."
    if action in SIGNALS:
        pat.observe(kind, key, action, note)
        return f"Noted: {action} for {kind} '{key}'."
    return "Error: action must be mute, boost, clear, acted, dismissed or deferred"


def care_patterns() -> str:
    return _pat().digest()


def care_generate_brief() -> str:
    b = generate_brief(_reg())
    lines = [f"Daily brief written for {b['date']} — {len(b['priority_items'])} priority item(s).", b["day_overview"]]
    for it in b["priority_items"]:
        lines.append(f"• [{it['urgency']}] {it['summary']}  id={it['id']}")
    return "\n".join(lines)


def care_mark_source_checked(source: str) -> str:
    try:
        _reg().mark_source_checked(source)
    except ValueError as e:
        return f"Error: {e}"
    return f"Marked {source} as checked at {datetime.now().astimezone().isoformat()}"


# ── Config ──────────────────────────────────────────────────────────────────

def care_get_config() -> str:
    return _config.summary()


def care_set_proactiveness(level: str) -> str:
    try:
        lvl = _config.set_level(level)
    except ValueError as e:
        return f"Error: {e}\n\nLevels:\n{_config.describe_levels()}"
    return f"Proactiveness level set to '{lvl}'.\n{_config.summary()}"


def care_set_mode(mode: str) -> str:
    try:
        m = _config.set_care_mode(mode)
    except ValueError as e:
        return f"Error: {e}\n\nModes:\n{_config.describe_modes()}"
    return f"Care mode set to '{m}'.\n{_config.summary()}"


def care_set_override(key: str, value_json: str) -> str:
    try:
        value = json.loads(value_json)
    except json.JSONDecodeError:
        value = value_json
    try:
        if value is None or value == "":
            _config.clear_override(key)
            return f"Override for '{key}' cleared."
        _config.set_override(key, value)
    except ValueError as e:
        return f"Error: {e}"
    return f"Override set: {key} = {json.dumps(value)}\n{_config.summary()}"


SCHEMA = [
    {"type": "function", "function": {
        "name": "care_list",
        "description": "List items in the care registry (things being tracked for the user: emails needing action, tasks, commitments, follow-ups). Default shows open items sorted by urgency.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "description": "open (default) | closed | all | or a specific status: new, acknowledged, in_progress, deferred, snoozed, dismissed, done, archived"},
            "item_type": {"type": "string", "description": "Optional filter: email | task | commitment | followup | calendar"},
            "limit": {"type": "integer", "description": "Max items (default 25)"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "care_digest",
        "description": "Compact digest of the care registry: what's due, overdue, snoozed-and-resurfacing, and other open items. Use at the start of a session or before scheduling nudges.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "care_find",
        "description": "Fuzzy-find a care item by words from its title, sender or details. Use this when the user mentions something you might be tracking ('I replied to Maya', 'the visa thing') so you can update the right item.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "care_add_item",
        "description": "Start tracking something. Use for commitments the user makes in conversation ('I have to call Riya tonight'), emails/tasks that need action, and follow-ups you owe the user. Near-duplicates are merged automatically.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Short, specific title, e.g. 'Call Riya' or 'Reply to Maya re: term sheet'"},
            "type": {"type": "string", "enum": list(ITEM_TYPES), "description": "commitment (default) for things the user said they'd do; email; task; followup; calendar"},
            "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
            "deadline": {"type": "string", "description": "ISO 8601 with timezone offset, if the user implied one ('tonight' → today 21:00, 'before Thursday' → Thursday 09:00)"},
            "details": {"type": "string"},
            "source": {"type": "string", "enum": ["conversation", "gmail", "notion", "calendar", "morning_review", "manual"]},
            "sender": {"type": "string", "description": "For emails: sender address or name"},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "Lowercase topic tags, e.g. ['investor', 'series a']"},
            "source_ref": {"type": "string", "description": "Stable external id (Gmail message id, Notion page id) for de-duplication"},
            "confidence": {"type": "number", "description": "0-1 how sure you are this matters (default 1)"},
            "category": {"type": "string", "description": "For emails: newsletter | promotion | automated | social | receipt | personal | work"},
        }, "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "care_update_item",
        "description": "Record what the user said or did about a tracked item. Set status (done, dismissed, deferred, in_progress, acknowledged, snoozed), capture their words as user_intent ('will reply tonight', 'not important'), snooze until a time, adjust urgency or deadline, add notes. Reactions feed pattern learning.",
        "parameters": {"type": "object", "properties": {
            "item_id": {"type": "string"},
            "status": {"type": "string", "enum": list(ALL_STATUSES)},
            "user_intent": {"type": "string", "description": "The user's own words about this item"},
            "snooze_until": {"type": "string", "description": "ISO 8601 with offset; sets status=snoozed"},
            "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {"type": "string"},
            "deadline": {"type": "string", "description": "ISO 8601 with offset"},
            "feedback_signal": {"type": "string", "enum": ["acted", "dismissed", "deferred"], "description": "Optional explicit learning signal; inferred from status when omitted"},
        }, "required": ["item_id"]}}},
    {"type": "function", "function": {
        "name": "care_resolve_item",
        "description": "Mark a tracked item done (e.g. user says they replied / finished it).",
        "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}, "note": {"type": "string"}}, "required": ["item_id"]}}},
    {"type": "function", "function": {
        "name": "care_record_feedback",
        "description": "Save a standing preference about what matters: mute or boost a sender, topic, category or source ('ignore newsletters from X', 'always flag anything about the term sheet'), or log a one-off reaction.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "key": {"type": "string", "description": "e.g. 'news@company.com', 'investor', 'newsletter', 'notion'"},
            "action": {"type": "string", "enum": ["mute", "boost", "clear", "acted", "dismissed", "deferred"]},
            "note": {"type": "string"},
        }, "required": ["kind", "key", "action"]}}},
    {"type": "function", "function": {
        "name": "care_patterns",
        "description": "Show learned patterns and explicit rules about what the user cares about.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "care_generate_brief",
        "description": "Regenerate today's daily brief as a view of the care registry (morning review step). Returns the priority items.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "care_mark_source_checked",
        "description": "Record that a source (gmail | notion | calendar) has just been scanned, so the next review only pulls newer items.",
        "parameters": {"type": "object", "properties": {"source": {"type": "string", "enum": ["gmail", "notion", "calendar"]}}, "required": ["source"]}}},
    {"type": "function", "function": {
        "name": "care_get_config",
        "description": "Show the current proactiveness level, care mode, nudge limits, quiet hours and active connectors. Use when the user asks how proactive you are or what's configured.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "care_set_proactiveness",
        "description": "Change how proactive you are. Levels: off (reactive only), minimal (deadline-only, 1 nudge/day), balanced (default, 3/day), active (6/day, captures commitments aggressively), max (10/day). Use when the user says things like 'be less pushy', 'nudge me more', 'stop reaching out'. Confirm the change in your reply.",
        "parameters": {"type": "object", "properties": {"level": {"type": "string", "enum": ["off", "minimal", "balanced", "active", "max"]}}, "required": ["level"]}}},
    {"type": "function", "function": {
        "name": "care_set_mode",
        "description": "Switch the situational care mode: normal | focus (heads-down week, only urgent) | fundraising (boost investor/board/legal) | travel (long quiet hours) | heads_down (near silent). Use when the user says 'I'm heads down this week', 'we're fundraising', 'I'm travelling till Friday'.",
        "parameters": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["normal", "focus", "fundraising", "travel", "heads_down"]}}, "required": ["mode"]}}},
    {"type": "function", "function": {
        "name": "care_set_override",
        "description": "Fine-tune one setting on top of the level/mode presets: max_nudges_per_day (int), quiet_hours ({\"start\":\"22:00\",\"end\":\"08:00\"}), urgency_threshold (high|medium|low), commitment_capture (off|conservative|aggressive), ask_on_borderline (bool), boosted_topics / muted_topics (list of strings), morning_review_dm (always|if_items|if_high_urgency|never). Pass an empty value to clear.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value_json": {"type": "string", "description": "JSON-encoded value"}}, "required": ["key", "value_json"]}}},
]
