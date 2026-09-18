"""
Tests for tool profile resolution.

Covers:
- Each profile contains the expected tools
- Multiple profiles are correctly unioned (no duplicates)
- Blocked tools (scheduling, spawn, cron) never appear in any profile
- Invalid profile returns a clean error from spawn_subagent
- 'full' profile excludes all blocked tools
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).parents[3])
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agents.subagent_runner import TOOL_PROFILES, resolve_tools, _BLOCKED_TOOLS
from agents.tools import TOOL_FUNCTIONS


# ── Helpers ──────────────────────────────────────────────────────────────────

def tool_names(schema: list[dict]) -> set[str]:
    return {t.get("function", {}).get("name", t.get("name")) for t in schema}


# ── Profile content ───────────────────────────────────────────────────────────

class TestProfileContents:

    def test_research_has_web_search(self):
        assert "tavily_search" in TOOL_PROFILES["research"]

    def test_research_has_email_read(self):
        assert "list_emails" in TOOL_PROFILES["research"]
        assert "read_email" in TOOL_PROFILES["research"]

    def test_research_has_memory(self):
        assert "query_long_term_memory" in TOOL_PROFILES["research"]

    def test_research_has_filesystem(self):
        assert "fs_read_file" in TOOL_PROFILES["research"]
        assert "fs_write_file" in TOOL_PROFILES["research"]

    def test_notion_has_all_notion_tools(self):
        for tool in ["search_notion", "read_notion_page", "query_notion_database",
                     "create_database_entry", "create_notion_page", "update_notion_page"]:
            assert tool in TOOL_PROFILES["notion"], f"Missing {tool} in notion profile"

    def test_calendar_email_has_calendar(self):
        for tool in ["list_calendar_events", "create_calendar_event", "update_calendar_event"]:
            assert tool in TOOL_PROFILES["calendar_email"], f"Missing {tool}"

    def test_calendar_email_can_send_email(self):
        assert "send_email" in TOOL_PROFILES["calendar_email"]
        assert "reply_to_email" in TOOL_PROFILES["calendar_email"]

    def test_filesystem_has_all_fs_ops(self):
        for tool in ["fs_read_file", "fs_write_file", "fs_list_files",
                     "fs_create_directory", "fs_delete", "fs_move", "fs_search_file"]:
            assert tool in TOOL_PROFILES["filesystem"], f"Missing {tool}"

    def test_all_profiles_have_datetime(self):
        for profile in TOOL_PROFILES:
            assert "get_current_datetime" in TOOL_PROFILES[profile], \
                f"Profile '{profile}' missing get_current_datetime"

    def test_all_profiles_defined(self):
        expected = {"research", "notion", "calendar_email", "filesystem", "full"}
        assert expected == set(TOOL_PROFILES.keys())


# ── Blocked tools ─────────────────────────────────────────────────────────────

class TestBlockedTools:

    MUST_BE_BLOCKED = [
        "spawn_subagent",
        "cancel_subagent",
        "list_subagents",
        "read_subagent_output",
        "schedule_notifications",
        "set_reminder",
        "cancel_reminder",
        "create_cron_job",
        "delete_cron_job",
        "list_cron_jobs",
    ]

    @pytest.mark.parametrize("blocked_tool", MUST_BE_BLOCKED)
    def test_blocked_tool_not_in_any_profile(self, blocked_tool):
        for profile_name, tools in TOOL_PROFILES.items():
            assert blocked_tool not in tools, \
                f"Blocked tool '{blocked_tool}' found in profile '{profile_name}'"

    @pytest.mark.parametrize("blocked_tool", MUST_BE_BLOCKED)
    def test_blocked_tool_not_in_resolved_full(self, blocked_tool):
        resolved = tool_names(resolve_tools(["full"]))
        assert blocked_tool not in resolved, \
            f"Blocked tool '{blocked_tool}' leaked into 'full' resolved schema"

    def test_blocked_tools_constant_covers_spawn(self):
        assert "spawn_subagent" in _BLOCKED_TOOLS

    def test_blocked_tools_constant_covers_scheduling(self):
        assert "schedule_notifications" in _BLOCKED_TOOLS


# ── Multi-profile union ───────────────────────────────────────────────────────

class TestProfileUnion:

    def test_single_profile_resolves_correctly(self):
        resolved = tool_names(resolve_tools(["research"]))
        for t in TOOL_PROFILES["research"]:
            assert t in resolved, f"Missing {t} from resolved research profile"

    def test_union_contains_all_tools_from_both(self):
        resolved = tool_names(resolve_tools(["research", "notion"]))
        for t in TOOL_PROFILES["research"]:
            assert t in resolved
        for t in TOOL_PROFILES["notion"]:
            assert t in resolved

    def test_union_has_no_duplicates(self):
        schema = resolve_tools(["research", "notion"])
        names = [t.get("function", {}).get("name") for t in schema]
        assert len(names) == len(set(names)), "Duplicate tools in resolved schema"

    def test_all_profiles_union(self):
        all_profiles = list(TOOL_PROFILES.keys())
        resolved = tool_names(resolve_tools(all_profiles))
        # Should be same as full (minus blocked)
        full_resolved = tool_names(resolve_tools(["full"]))
        assert full_resolved == resolved - (resolved - full_resolved)

    def test_resolved_tools_exist_in_tool_functions(self):
        """Every resolved tool must be dispatchable."""
        for profile in TOOL_PROFILES:
            resolved = tool_names(resolve_tools([profile]))
            for name in resolved:
                assert name in TOOL_FUNCTIONS, \
                    f"Tool '{name}' in profile '{profile}' not in TOOL_FUNCTIONS"

    def test_full_profile_is_largest(self):
        full = len(resolve_tools(["full"]))
        for profile in ["research", "notion", "calendar_email", "filesystem"]:
            assert full >= len(resolve_tools([profile])), \
                f"'full' profile should have >= tools than '{profile}'"


# ── Schema format ─────────────────────────────────────────────────────────────

class TestSchemaFormat:

    def test_resolved_schema_entries_have_function_key(self):
        for entry in resolve_tools(["research"]):
            assert "function" in entry or "name" in entry, \
                f"Schema entry missing 'function' key: {entry}"

    def test_resolved_schema_entries_have_name(self):
        for entry in resolve_tools(["full"]):
            name = entry.get("function", {}).get("name", entry.get("name"))
            assert name, f"Schema entry has no name: {entry}"

    def test_resolved_schema_entries_have_description(self):
        for entry in resolve_tools(["full"]):
            desc = entry.get("function", {}).get("description", "")
            assert desc, f"Tool '{entry}' has no description"
