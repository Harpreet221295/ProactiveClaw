"""ProactiveClaw configuration.

Two dials control how the assistant behaves:

1. **Proactiveness level** — *how much* the assistant reaches out on its own.
   `off` → `minimal` → `balanced` → `active` → `max`.
2. **Care mode** — *what kind of things matter right now* (topic boosts,
   filtering strictness, quiet hours). `normal`, `focus`, `fundraising`,
   `travel`, `heads_down`.

Effective settings are resolved as:

    DEFAULTS  ←  level preset  ←  care-mode overrides  ←  explicit user overrides

Connectors (Gmail, Calendar, Notion, web search, browser) are optional. A
connector is *active* only when it is enabled in config AND its credentials
are present. Everything else in the codebase asks this module instead of
assuming an integration exists.

The config file lives at `config/config.json` (see `config/config.example.json`).
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from .paths import (
    CONFIG_FILE, CONFIG_EXAMPLE_FILE, GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Proactiveness levels
# ─────────────────────────────────────────────────────────────────────────────

PROACTIVENESS_LEVELS: dict[str, dict[str, Any]] = {
    "off": {
        "description": "Reactive only. Answers when asked, never reaches out. Reminders and cron jobs you set explicitly still fire.",
        "invocations_enabled": False,
        "morning_review_enabled": False,
        "morning_review_dm": "never",
        "reengagement_enabled": False,
        "reengagement_intervals_hours": [],
        "max_nudges_per_day": 0,
        "quiet_hours": {"start": "22:00", "end": "08:00"},
        "commitment_capture": "off",
        "ask_on_borderline": False,
        "research_followups": False,
        "surface_care_items_in_chat": False,
    },
    "minimal": {
        "description": "Deadline-driven only. One nudge a day at most, morning review runs silently, no re-engagement DMs.",
        "invocations_enabled": True,
        "morning_review_enabled": True,
        "morning_review_dm": "if_high_urgency",
        "reengagement_enabled": False,
        "reengagement_intervals_hours": [],
        "max_nudges_per_day": 1,
        "quiet_hours": {"start": "21:00", "end": "09:00"},
        "commitment_capture": "conservative",
        "ask_on_borderline": False,
        "research_followups": False,
        "surface_care_items_in_chat": True,
    },
    "balanced": {
        "description": "The default. Daily morning review, up to 3 nudges a day, gentle re-engagement after a week of silence.",
        "invocations_enabled": True,
        "morning_review_enabled": True,
        "morning_review_dm": "if_items",
        "reengagement_enabled": True,
        "reengagement_intervals_hours": [168, 336, 720],
        "max_nudges_per_day": 3,
        "quiet_hours": {"start": "22:00", "end": "08:00"},
        "commitment_capture": "conservative",
        "ask_on_borderline": True,
        "research_followups": False,
        "surface_care_items_in_chat": True,
    },
    "active": {
        "description": "A real chief of staff. Up to 6 nudges a day, captures commitments aggressively, follows up on research, re-engages after 3 days.",
        "invocations_enabled": True,
        "morning_review_enabled": True,
        "morning_review_dm": "always",
        "reengagement_enabled": True,
        "reengagement_intervals_hours": [72, 168, 336, 720],
        "max_nudges_per_day": 6,
        "quiet_hours": {"start": "23:00", "end": "07:00"},
        "commitment_capture": "aggressive",
        "ask_on_borderline": False,
        "research_followups": True,
        "surface_care_items_in_chat": True,
    },
    "max": {
        "description": "Relentless. Up to 10 nudges a day, short quiet hours, tracks everything that sounds like a commitment, re-engages daily.",
        "invocations_enabled": True,
        "morning_review_enabled": True,
        "morning_review_dm": "always",
        "reengagement_enabled": True,
        "reengagement_intervals_hours": [24, 72, 168, 336],
        "max_nudges_per_day": 10,
        "quiet_hours": {"start": "00:00", "end": "06:00"},
        "commitment_capture": "aggressive",
        "ask_on_borderline": False,
        "research_followups": True,
        "surface_care_items_in_chat": True,
    },
}

LEVEL_ORDER = ["off", "minimal", "balanced", "active", "max"]
LEVEL_ALIASES = {"0": "off", "1": "minimal", "2": "balanced", "3": "active", "4": "max",
                 "none": "off", "quiet": "minimal", "low": "minimal", "normal": "balanced",
                 "default": "balanced", "medium": "balanced", "high": "active", "aggressive": "max",
                 "relentless": "max"}

# ─────────────────────────────────────────────────────────────────────────────
# Care modes — situational presets layered on top of the level
# ─────────────────────────────────────────────────────────────────────────────

CARE_MODES: dict[str, dict[str, Any]] = {
    "normal": {
        "description": "Balanced defaults. No topic boosts or extra filtering.",
    },
    "focus": {
        "description": "Heads-down work week. Only high-urgency items surface, nudges capped at 2/day, conservative commitment capture.",
        "urgency_threshold": "high",
        "max_nudges_per_day_cap": 2,
        "commitment_capture_cap": "conservative",
        "ask_on_borderline": False,
    },
    "fundraising": {
        "description": "Investor, term sheet, board and legal items are boosted and escalate faster. Nudge cap raised.",
        "urgency_threshold": "medium",
        "boosted_topics": ["investor", "series a", "series b", "term sheet", "board", "legal", "due diligence", "cap table", "fundrais"],
        "max_nudges_per_day_floor": 5,
        "escalate_after_deferrals": 2,
    },
    "travel": {
        "description": "Longer quiet hours, only truly urgent items, commitment capture kept conservative.",
        "urgency_threshold": "high",
        "quiet_hours": {"start": "21:00", "end": "09:00"},
        "commitment_capture_cap": "conservative",
        "max_nudges_per_day_cap": 2,
    },
    "heads_down": {
        "description": "Near-silent. Only hard deadlines and calendar-driven nudges. Commitment capture off.",
        "urgency_threshold": "high",
        "max_nudges_per_day_cap": 1,
        "commitment_capture": "off",
        "ask_on_borderline": False,
        "surface_care_items_in_chat": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Defaults for every effective key (level/mode/overrides refine these)
# ─────────────────────────────────────────────────────────────────────────────

EFFECTIVE_DEFAULTS: dict[str, Any] = {
    "invocations_enabled": True,
    "morning_review_enabled": True,
    "morning_review_dm": "if_items",           # always | if_items | if_high_urgency | never
    "reengagement_enabled": True,
    "reengagement_intervals_hours": [168, 336, 720],
    "max_nudges_per_day": 3,
    "quiet_hours": {"start": "22:00", "end": "08:00"},
    "urgency_threshold": "medium",             # high | medium | low — minimum urgency to track/surface
    "commitment_capture": "conservative",      # off | conservative | aggressive
    "ask_on_borderline": True,
    "research_followups": False,
    "surface_care_items_in_chat": True,
    "boosted_topics": [],
    "muted_topics": [],
    "auto_filter_categories": ["newsletter", "promotion", "automated", "social", "receipt"],
    "min_confidence": 0.6,
    "resurface_after_hours": 24,
    "escalate_after_deferrals": 3,
    "item_max_age_days": 14,
    "max_brief_items": 8,
    "pattern_learning": {"enabled": True, "min_observations": 3, "decay_days": 30},
}

DEFAULT_CONFIG: dict[str, Any] = {
    "user": {"name": "", "timezone": ""},
    "llm": {"model": ""},
    "proactiveness": {"level": "balanced"},
    "care_mode": "normal",
    "connectors": {
        "gmail": False,
        "calendar": False,
        "notion": False,
        "web_search": True,
        "browser": False,
    },
    "notion": {"tasks_database_id": "", "tasks_schema_hint": ""},
    "morning_review": {"time": "07:30", "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
    "overrides": {},
}

URGENCY_RANK = {"high": 3, "medium": 2, "low": 1}
CAPTURE_RANK = {"off": 0, "conservative": 1, "aggressive": 2}

# ─────────────────────────────────────────────────────────────────────────────
# Loading / saving
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict[str, Any] | None = None
_cache_mtime: float = -1.0


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_timezone(tz: str) -> None:
    """Make the process clock match the configured timezone (POSIX only)."""
    if not tz or os.environ.get("TZ") == tz:
        return
    os.environ["TZ"] = tz
    if hasattr(time, "tzset"):
        try:
            time.tzset()
        except Exception:
            pass


def load_config(force: bool = False) -> dict[str, Any]:
    """Load config/config.json merged over defaults. Cached by file mtime."""
    global _cache, _cache_mtime
    path = CONFIG_FILE
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if not force and _cache is not None and mtime == _cache_mtime:
        return _cache

    raw: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError as e:
            print(f"[config] {path} is not valid JSON ({e}) — using defaults")
            raw = {}

    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    cfg["proactiveness"]["level"] = normalize_level(cfg["proactiveness"].get("level"))
    if cfg.get("care_mode") not in CARE_MODES:
        cfg["care_mode"] = "normal"
    _apply_timezone(cfg["user"].get("timezone", ""))

    _cache, _cache_mtime = cfg, mtime
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    global _cache, _cache_mtime
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _cache = None
    _cache_mtime = -1.0


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def normalize_level(level: Any) -> str:
    s = str(level or "balanced").strip().lower()
    if s in PROACTIVENESS_LEVELS:
        return s
    return LEVEL_ALIASES.get(s, "balanced")


# ─────────────────────────────────────────────────────────────────────────────
# Effective settings resolution
# ─────────────────────────────────────────────────────────────────────────────

def effective_settings(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve defaults ← level ← care mode ← overrides into one flat dict."""
    cfg = cfg or load_config()
    level = normalize_level(cfg["proactiveness"]["level"])
    mode = cfg.get("care_mode", "normal")

    eff = copy.deepcopy(EFFECTIVE_DEFAULTS)
    preset = {k: v for k, v in PROACTIVENESS_LEVELS[level].items() if k != "description"}
    eff.update(copy.deepcopy(preset))

    mode_over = {k: v for k, v in CARE_MODES.get(mode, {}).items() if k != "description"}
    # Special caps/floors from mode
    cap = mode_over.pop("max_nudges_per_day_cap", None)
    floor = mode_over.pop("max_nudges_per_day_floor", None)
    capture_cap = mode_over.pop("commitment_capture_cap", None)
    eff.update(copy.deepcopy(mode_over))
    if cap is not None:
        eff["max_nudges_per_day"] = min(eff["max_nudges_per_day"], cap)
    if floor is not None and eff["max_nudges_per_day"] > 0:
        eff["max_nudges_per_day"] = max(eff["max_nudges_per_day"], floor)
    if capture_cap is not None and CAPTURE_RANK[eff["commitment_capture"]] > CAPTURE_RANK[capture_cap]:
        eff["commitment_capture"] = capture_cap

    # Explicit overrides always win
    for k, v in (cfg.get("overrides") or {}).items():
        if k in eff:
            if isinstance(v, dict) and isinstance(eff[k], dict):
                eff[k] = _deep_merge(eff[k], v)
            else:
                eff[k] = copy.deepcopy(v)

    # Level "off" is absolute — no override can turn proactive features back on
    if level == "off":
        eff["invocations_enabled"] = False
        eff["morning_review_enabled"] = False
        eff["reengagement_enabled"] = False
        eff["max_nudges_per_day"] = 0
        eff["commitment_capture"] = "off"

    eff["level"] = level
    eff["care_mode"] = mode
    return eff


# ─────────────────────────────────────────────────────────────────────────────
# Mutators used by the setup wizard and the agent's config tools
# ─────────────────────────────────────────────────────────────────────────────

def set_level(level: str) -> str:
    cfg = load_config()
    norm = normalize_level(level)
    if str(level).strip().lower() not in PROACTIVENESS_LEVELS and str(level).strip().lower() not in LEVEL_ALIASES:
        raise ValueError(f"Unknown proactiveness level '{level}'. Choose one of: {', '.join(LEVEL_ORDER)}")
    cfg["proactiveness"]["level"] = norm
    save_config(cfg)
    return norm


def set_care_mode(mode: str) -> str:
    cfg = load_config()
    m = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if m not in CARE_MODES:
        raise ValueError(f"Unknown care mode '{mode}'. Choose one of: {', '.join(CARE_MODES)}")
    cfg["care_mode"] = m
    save_config(cfg)
    return m


OVERRIDABLE_KEYS = {
    "max_nudges_per_day": int,
    "quiet_hours": dict,
    "urgency_threshold": str,
    "commitment_capture": str,
    "ask_on_borderline": bool,
    "research_followups": bool,
    "boosted_topics": list,
    "muted_topics": list,
    "auto_filter_categories": list,
    "resurface_after_hours": int,
    "escalate_after_deferrals": int,
    "item_max_age_days": int,
    "morning_review_dm": str,
    "surface_care_items_in_chat": bool,
}


def set_override(key: str, value: Any) -> None:
    if key not in OVERRIDABLE_KEYS:
        raise ValueError(f"'{key}' cannot be overridden. Overridable: {', '.join(sorted(OVERRIDABLE_KEYS))}")
    expected = OVERRIDABLE_KEYS[key]
    if expected is int and isinstance(value, bool):
        raise ValueError(f"'{key}' expects an integer")
    if not isinstance(value, expected):
        raise ValueError(f"'{key}' expects {expected.__name__}, got {type(value).__name__}")
    if key == "urgency_threshold" and value not in URGENCY_RANK:
        raise ValueError("urgency_threshold must be high, medium or low")
    if key == "commitment_capture" and value not in CAPTURE_RANK:
        raise ValueError("commitment_capture must be off, conservative or aggressive")
    if key == "morning_review_dm" and value not in ("always", "if_items", "if_high_urgency", "never"):
        raise ValueError("morning_review_dm must be always, if_items, if_high_urgency or never")
    if key == "quiet_hours":
        for k in ("start", "end"):
            if not re.fullmatch(r"\d{2}:\d{2}", str(value.get(k, ""))):
                raise ValueError("quiet_hours needs 'start' and 'end' as HH:MM")
    cfg = load_config()
    cfg.setdefault("overrides", {})[key] = value
    save_config(cfg)


def clear_override(key: str) -> bool:
    cfg = load_config()
    existed = key in (cfg.get("overrides") or {})
    cfg.get("overrides", {}).pop(key, None)
    save_config(cfg)
    return existed


def set_connector(name: str, enabled: bool) -> None:
    cfg = load_config()
    if name not in cfg["connectors"]:
        raise ValueError(f"Unknown connector '{name}'. Known: {', '.join(cfg['connectors'])}")
    cfg["connectors"][name] = bool(enabled)
    save_config(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Connectors
# ─────────────────────────────────────────────────────────────────────────────

CONNECTOR_TOOLS: dict[str, list[str]] = {
    "gmail": ["list_emails", "read_email", "send_email", "reply_to_email"],
    "calendar": ["list_calendar_events", "create_calendar_event", "update_calendar_event", "delete_calendar_event"],
    "notion": ["search_notion", "read_notion_page", "create_notion_page", "update_notion_page",
               "query_notion_database", "create_database_entry"],
    "web_search": ["tavily_search"],
    "browser": ["browser_navigate", "browser_snapshot", "browser_click", "browser_type",
                "browser_screenshot", "browser_get_page_info", "browser_scroll"],
}


def _google_configured() -> tuple[bool, str]:
    if GOOGLE_TOKEN_FILE.exists():
        return True, ""
    if GOOGLE_CREDENTIALS_FILE.exists():
        return True, "token.json missing — first use will open a browser for Google sign-in"
    return False, f"credentials.json not found at {GOOGLE_CREDENTIALS_FILE}"


def connector_status(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """For each connector: enabled (config), configured (credentials), active (both)."""
    cfg = cfg or load_config()
    enabled = cfg.get("connectors", {})
    status: dict[str, dict[str, Any]] = {}

    g_ok, g_reason = _google_configured()
    checks = {
        "gmail": (g_ok, g_reason),
        "calendar": (g_ok, g_reason),
        "notion": (bool(os.getenv("NOTION_API_KEY")), "NOTION_API_KEY not set in .env"),
        "web_search": (bool(os.getenv("TAVILY_API_KEY")), "TAVILY_API_KEY not set in .env"),
        "browser": (bool(os.getenv("BROWSER_TOKEN")), "BROWSER_TOKEN not set in .env"),
    }
    for name, (configured, reason) in checks.items():
        is_enabled = bool(enabled.get(name, False))
        active = is_enabled and configured
        why = ""
        if not is_enabled:
            why = "disabled in config"
        elif not configured:
            why = reason
        elif reason:
            why = reason  # informational (e.g. token will be created on first use)
        status[name] = {"enabled": is_enabled, "configured": configured, "active": active, "reason": why}
    return status


def active_connectors(cfg: dict[str, Any] | None = None) -> set[str]:
    return {n for n, s in connector_status(cfg).items() if s["active"]}


def active_tool_names(cfg: dict[str, Any] | None = None) -> set[str] | None:
    """Return the set of connector-gated tool names that are currently allowed.

    Tools not listed in CONNECTOR_TOOLS are always allowed.
    """
    allowed: set[str] = set()
    for name in active_connectors(cfg):
        allowed.update(CONNECTOR_TOOLS[name])
    return allowed


def gated_tool_names() -> set[str]:
    out: set[str] = set()
    for names in CONNECTOR_TOOLS.values():
        out.update(names)
    return out


def is_tool_allowed(tool_name: str, cfg: dict[str, Any] | None = None) -> bool:
    if tool_name not in gated_tool_names():
        return True
    return tool_name in active_tool_names(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessors
# ─────────────────────────────────────────────────────────────────────────────

def user_name(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return (cfg.get("user", {}).get("name") or "").strip() or "the user"


def user_slug(cfg: dict[str, Any] | None = None) -> str:
    """Lowercase identifier used as the memory user_id and graph self-node."""
    name = user_name(cfg)
    first = name.split()[0] if name and name != "the user" else "user"
    slug = re.sub(r"[^a-z0-9]+", "_", first.lower()).strip("_")
    return slug or "user"


def default_model(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    model = (cfg.get("llm", {}).get("model") or os.getenv("AGENT_MODEL") or "").strip()
    if model:
        return model
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-5"
    return "gpt-4o-mini"


def in_quiet_hours(dt: datetime, quiet: dict[str, str]) -> bool:
    start, end = quiet.get("start", "22:00"), quiet.get("end", "08:00")
    if start == end:
        return False
    cur = dt.strftime("%H:%M")
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end


def next_allowed_time(dt: datetime, quiet: dict[str, str]) -> datetime:
    """If `dt` falls inside quiet hours, push it to the end of the quiet window."""
    if not in_quiet_hours(dt, quiet):
        return dt
    end_h, end_m = map(int, quiet.get("end", "08:00").split(":"))
    candidate = dt.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if candidate <= dt:
        from datetime import timedelta
        candidate += timedelta(days=1)
    return candidate


def describe_levels() -> str:
    lines = []
    for lvl in LEVEL_ORDER:
        p = PROACTIVENESS_LEVELS[lvl]
        lines.append(f"{lvl:9s} — {p['description']}")
    return "\n".join(lines)


def describe_modes() -> str:
    return "\n".join(f"{m:11s} — {d['description']}" for m, d in CARE_MODES.items())


def summary(cfg: dict[str, Any] | None = None) -> str:
    """Human-readable summary of the effective configuration."""
    cfg = cfg or load_config()
    eff = effective_settings(cfg)
    conns = connector_status(cfg)
    active = [n for n, s in conns.items() if s["active"]]
    inactive = [f"{n} ({s['reason']})" for n, s in conns.items() if not s["active"]]
    lines = [
        f"Proactiveness level: {eff['level']} — {PROACTIVENESS_LEVELS[eff['level']]['description']}",
        f"Care mode: {eff['care_mode']} — {CARE_MODES[eff['care_mode']]['description']}",
        f"Nudges per day: {eff['max_nudges_per_day']} | Quiet hours: {eff['quiet_hours']['start']}–{eff['quiet_hours']['end']}",
        f"Urgency threshold: {eff['urgency_threshold']} | Commitment capture: {eff['commitment_capture']}"
        + (" (asks on borderline)" if eff['ask_on_borderline'] else ""),
        f"Morning review: {'on at ' + cfg['morning_review']['time'] if eff['morning_review_enabled'] else 'off'}"
        f" | Re-engagement: {'on' if eff['reengagement_enabled'] else 'off'}",
        f"Active connectors: {', '.join(active) or 'none'}",
    ]
    if inactive:
        lines.append(f"Inactive connectors: {'; '.join(inactive)}")
    if eff.get("boosted_topics"):
        lines.append(f"Boosted topics: {', '.join(eff['boosted_topics'])}")
    if eff.get("muted_topics"):
        lines.append(f"Muted topics: {', '.join(eff['muted_topics'])}")
    if cfg.get("overrides"):
        lines.append(f"Explicit overrides: {json.dumps(cfg['overrides'])}")
    return "\n".join(lines)
