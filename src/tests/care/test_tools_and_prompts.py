from __future__ import annotations

import json
from datetime import datetime, timedelta

from agents.tools import active_tools_schema, dispatch_tool_call, TOOLS_SCHEMA
from agents.tools import care as care_tools
from agents.prompts.prompts import build_system_prompt, build_pre_exit_prompt, build_morning_review_prompt
from care.registry import CareRegistry


def _names(schema):
    return {t["function"]["name"] for t in schema}


def test_tools_gated_by_connectors(write_config, monkeypatch):
    write_config(connectors={"notion": False, "gmail": False, "web_search": False})
    names = _names(active_tools_schema())
    assert "care_list" in names and "schedule_notifications" in names
    assert "search_notion" not in names and "tavily_search" not in names and "list_emails" not in names
    assert "Error" in dispatch_tool_call("tavily_search", json.dumps({"query": "x"}))
    monkeypatch.setenv("NOTION_API_KEY", "k")
    write_config(connectors={"notion": True})
    assert "search_notion" in _names(active_tools_schema())
    assert len(TOOLS_SCHEMA) > len(active_tools_schema())


def test_care_tools_roundtrip(write_config):
    write_config(proactiveness={"level": "balanced"})
    out = care_tools.care_add_item(title="Call Simran tonight", deadline=(datetime.now().astimezone() + timedelta(hours=3)).isoformat())
    assert out.startswith("Tracked:")
    item = CareRegistry().list("open")[0]
    assert "Call Simran" in care_tools.care_find("simran")
    assert "Updated" in care_tools.care_update_item(item["id"], user_intent="will do tonight", status="deferred")
    assert "Updated" in care_tools.care_resolve_item(item["id"], note="called her")
    assert CareRegistry().get(item["id"])["status"] == "done"
    assert "No care items" in care_tools.care_list("open")
    assert "Daily brief written" in care_tools.care_generate_brief()


def test_commitment_capture_off_blocks_conversation_items(write_config):
    write_config(proactiveness={"level": "off"})
    out = care_tools.care_add_item(title="Call Simran", source="conversation")
    assert "off" in out and CareRegistry().list("open") == []
    # explicit sources are still allowed
    assert care_tools.care_add_item(title="Reply to Maya", type="email", source="gmail", sender="maya@x.com").startswith("Tracked")


def test_feedback_rules_filter_future_items(write_config):
    write_config()
    assert "Rule saved" in care_tools.care_record_feedback("sender", "news@x.com", "mute")
    out = care_tools.care_add_item(title="Weekly digest", type="email", source="gmail", sender="news@x.com")
    assert out.startswith("Filtered")
    assert "mute sender" in care_tools.care_patterns()


def test_config_tools(write_config):
    write_config()
    assert "active" in care_tools.care_set_proactiveness("active")
    assert "Error" in care_tools.care_set_proactiveness("ultra")
    assert "fundraising" in care_tools.care_set_mode("fundraising")
    assert "Override set" in care_tools.care_set_override("max_nudges_per_day", "4")
    assert "Error" in care_tools.care_set_override("max_nudges_per_day", "\"four\"")
    assert "cleared" in care_tools.care_set_override("max_nudges_per_day", "")
    summary = care_tools.care_get_config()
    assert "Proactiveness level: active" in summary and "fundraising" in summary


def test_dispatch_never_raises(write_config):
    write_config()
    assert "Error" in dispatch_tool_call("care_update_item", json.dumps({"item_id": "nope", "status": "done"}))
    assert "Error" in dispatch_tool_call("care_list", "{not json")
    assert "unknown tool" in dispatch_tool_call("nope", "{}")


def test_prompts_reflect_config(write_config, monkeypatch):
    write_config(user={"name": "Ada Lovelace"}, proactiveness={"level": "max"}, care_mode="fundraising",
                 connectors={"notion": True, "gmail": False}, notion={"tasks_database_id": "db-123"})
    monkeypatch.setenv("NOTION_API_KEY", "k")
    sp = build_system_prompt()
    assert "Ada Lovelace" in sp and "db-123" in sp and "level: **max**" in sp
    assert "Capture anything that sounds like a commitment" in sp
    assert "Gmail**" not in sp.split("## Available integrations")[1].split("## Care registry")[0].split("*Not connected*")[0]
    assert "Not connected" in sp and "gmail" in sp

    pe = build_pre_exit_prompt(current_time="now", care_digest="digest!")
    assert "digest!" in pe and "Step 2" in pe and "next-day follow-up" in pe

    write_config(proactiveness={"level": "off"})
    pe = build_pre_exit_prompt(current_time="now")
    assert "must NOT schedule" in pe

    write_config(proactiveness={"level": "minimal"}, connectors={"gmail": False, "notion": False})
    mr = build_morning_review_prompt(current_time="now", care_digest="d", tend_report="t", patterns="p")
    assert "NO_BRIEF" in mr and "No external sources are connected" in mr
