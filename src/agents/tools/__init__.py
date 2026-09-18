"""Tool registry.

`TOOLS_SCHEMA` / `TOOL_FUNCTIONS` describe every tool the codebase knows about.
`active_tools_schema()` returns only the tools whose connector is enabled AND
configured (see core.config.connector_status) — that is what the agent sees.
"""
from __future__ import annotations

import json

from core import config as _config

from ._state import QUEUE_FILE, set_current_session_id
from .web import tavily_search, SCHEMA as _web_schema
from .scheduling import schedule_notifications, get_current_datetime, set_reminder, list_reminders, cancel_reminder, create_cron_job, list_cron_jobs, delete_cron_job, list_scheduled_nudges, cancel_scheduled_nudge, SCHEMA as _scheduling_schema
from .calendar import list_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event, SCHEMA as _calendar_schema
from .gmail import list_emails, read_email, send_email, reply_to_email, SCHEMA as _gmail_schema
from .notion import search_notion, read_notion_page, create_notion_page, update_notion_page, query_notion_database, create_database_entry, SCHEMA as _notion_schema
from .filesystem import fs_list_files, fs_read_file, fs_write_file, fs_create_directory, fs_delete, fs_move, fs_search_file, SCHEMA as _filesystem_schema
from .data import fs_read_json, fs_write_json, fs_read_csv, fs_write_csv, SCHEMA as _data_schema
from .charts import generate_chart, SCHEMA as _charts_schema
from .memory import query_long_term_memory, retrieve_long_term_memory, SCHEMA as _memory_schema
from .browser import browser_navigate, browser_snapshot, browser_click, browser_type, browser_screenshot, browser_get_page_info, browser_scroll, SCHEMA as _browser_schema
from .subagent import spawn_subagent, list_subagents, cancel_subagent, read_subagent_output, SCHEMA as _subagent_schema
from .care import (
    care_list, care_digest, care_find, care_add_item, care_update_item, care_resolve_item,
    care_record_feedback, care_patterns, care_generate_brief, care_mark_source_checked,
    care_get_config, care_set_proactiveness, care_set_mode, care_set_override,
    SCHEMA as _care_schema,
)

TOOLS_SCHEMA = (
    _web_schema
    + _scheduling_schema
    + _calendar_schema
    + _gmail_schema
    + _notion_schema
    + _filesystem_schema
    + _data_schema
    + _charts_schema
    + _memory_schema
    + _browser_schema
    + _subagent_schema
    + _care_schema
)

TOOL_FUNCTIONS = {
    "tavily_search": tavily_search,
    "schedule_notifications": schedule_notifications,
    "get_current_datetime": get_current_datetime,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "cancel_reminder": cancel_reminder,
    "create_cron_job": create_cron_job,
    "list_cron_jobs": list_cron_jobs,
    "delete_cron_job": delete_cron_job,
    "list_scheduled_nudges": list_scheduled_nudges,
    "cancel_scheduled_nudge": cancel_scheduled_nudge,
    "list_calendar_events": list_calendar_events,
    "create_calendar_event": create_calendar_event,
    "update_calendar_event": update_calendar_event,
    "delete_calendar_event": delete_calendar_event,
    "list_emails": list_emails,
    "read_email": read_email,
    "send_email": send_email,
    "reply_to_email": reply_to_email,
    "search_notion": search_notion,
    "read_notion_page": read_notion_page,
    "create_notion_page": create_notion_page,
    "update_notion_page": update_notion_page,
    "query_notion_database": query_notion_database,
    "create_database_entry": create_database_entry,
    "fs_list_files": fs_list_files,
    "fs_read_file": fs_read_file,
    "fs_write_file": fs_write_file,
    "fs_create_directory": fs_create_directory,
    "fs_delete": fs_delete,
    "fs_move": fs_move,
    "fs_search_file": fs_search_file,
    "fs_read_json": fs_read_json,
    "fs_write_json": fs_write_json,
    "fs_read_csv": fs_read_csv,
    "fs_write_csv": fs_write_csv,
    "generate_chart": generate_chart,
    "query_long_term_memory": query_long_term_memory,
    "retrieve_long_term_memory": retrieve_long_term_memory,
    "browser_navigate": browser_navigate,
    "browser_snapshot": browser_snapshot,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_screenshot": browser_screenshot,
    "browser_get_page_info": browser_get_page_info,
    "browser_scroll": browser_scroll,
    "spawn_subagent": spawn_subagent,
    "list_subagents": list_subagents,
    "cancel_subagent": cancel_subagent,
    "read_subagent_output": read_subagent_output,
    "care_list": care_list,
    "care_digest": care_digest,
    "care_find": care_find,
    "care_add_item": care_add_item,
    "care_update_item": care_update_item,
    "care_resolve_item": care_resolve_item,
    "care_record_feedback": care_record_feedback,
    "care_patterns": care_patterns,
    "care_generate_brief": care_generate_brief,
    "care_mark_source_checked": care_mark_source_checked,
    "care_get_config": care_get_config,
    "care_set_proactiveness": care_set_proactiveness,
    "care_set_mode": care_set_mode,
    "care_set_override": care_set_override,
}


def _schema_name(entry: dict) -> str:
    return entry.get("function", {}).get("name", entry.get("name", ""))


def active_tools_schema(cfg: dict | None = None, exclude: set[str] | None = None) -> list[dict]:
    """Tools the agent may use right now: connector-gated tools only if their connector is active."""
    allowed = _config.active_tool_names(cfg)
    gated = _config.gated_tool_names()
    out = []
    for entry in TOOLS_SCHEMA:
        name = _schema_name(entry)
        if exclude and name in exclude:
            continue
        if name in gated and name not in allowed:
            continue
        out.append(entry)
    return out


def dispatch_tool_call(name: str, args: str) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    if not _config.is_tool_allowed(name):
        status = _config.connector_status()
        for conn, names in _config.CONNECTOR_TOOLS.items():
            if name in names:
                return f"Error: '{name}' is unavailable — the {conn} connector is not active ({status[conn]['reason']}). Tell the user they can enable it in Settings."
        return f"Error: '{name}' is unavailable."
    try:
        parsed_args = json.loads(args) if args else {}
    except json.JSONDecodeError as e:
        return f"Error: could not parse arguments for {name}: {e}"
    try:
        return func(**parsed_args)
    except TypeError as e:
        return f"Error calling {name}: {e}"
    except Exception as e:  # tools must never crash the agent loop
        return f"Error in {name}: {type(e).__name__}: {e}"
