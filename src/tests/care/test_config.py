from __future__ import annotations

from datetime import datetime

import pytest

import core.config as config


def test_defaults_when_no_config_file():
    cfg = config.load_config(force=True)
    assert cfg["proactiveness"]["level"] == "balanced"
    eff = config.effective_settings(cfg)
    assert eff["max_nudges_per_day"] == 3
    assert eff["commitment_capture"] == "conservative"


@pytest.mark.parametrize("level,cap,capture,reeng", [
    ("off", 0, "off", False),
    ("minimal", 1, "conservative", False),
    ("balanced", 3, "conservative", True),
    ("active", 6, "aggressive", True),
    ("max", 10, "aggressive", True),
])
def test_level_presets(write_config, level, cap, capture, reeng):
    cfg = write_config(proactiveness={"level": level})
    eff = config.effective_settings(cfg)
    assert eff["level"] == level
    assert eff["max_nudges_per_day"] == cap
    assert eff["commitment_capture"] == capture
    assert eff["reengagement_enabled"] is reeng


def test_level_aliases():
    assert config.normalize_level("2") == "balanced"
    assert config.normalize_level("HIGH") == "active"
    assert config.normalize_level("garbage") == "balanced"


def test_mode_caps_and_floors(write_config):
    cfg = write_config(proactiveness={"level": "active"}, care_mode="focus")
    eff = config.effective_settings(cfg)
    assert eff["max_nudges_per_day"] == 2            # capped by focus
    assert eff["commitment_capture"] == "conservative"  # capped by focus
    assert eff["urgency_threshold"] == "high"

    cfg = write_config(proactiveness={"level": "minimal"}, care_mode="fundraising")
    eff = config.effective_settings(cfg)
    assert eff["max_nudges_per_day"] == 5             # floor raised by fundraising
    assert "investor" in eff["boosted_topics"]


def test_overrides_win_but_off_is_absolute(write_config):
    cfg = write_config(proactiveness={"level": "balanced"}, overrides={"max_nudges_per_day": 7, "quiet_hours": {"start": "23:00", "end": "06:00"}})
    eff = config.effective_settings(cfg)
    assert eff["max_nudges_per_day"] == 7
    assert eff["quiet_hours"]["start"] == "23:00"

    cfg = write_config(proactiveness={"level": "off"}, overrides={"max_nudges_per_day": 7})
    eff = config.effective_settings(cfg)
    assert eff["max_nudges_per_day"] == 0
    assert eff["invocations_enabled"] is False


def test_setters_validate(write_config):
    write_config()
    assert config.set_level("active") == "active"
    with pytest.raises(ValueError):
        config.set_level("bogus")
    assert config.set_care_mode("Heads Down") == "heads_down"
    with pytest.raises(ValueError):
        config.set_override("max_nudges_per_day", "three")
    with pytest.raises(ValueError):
        config.set_override("quiet_hours", {"start": "22"})
    config.set_override("boosted_topics", ["investor"])
    assert config.effective_settings()["boosted_topics"] == ["investor"]
    assert config.clear_override("boosted_topics") is True


def test_connector_status_requires_config_and_credentials(write_config, monkeypatch):
    cfg = write_config(connectors={"notion": True, "web_search": True, "gmail": True})
    st = config.connector_status(cfg)
    assert st["notion"]["enabled"] and not st["notion"]["configured"] and not st["notion"]["active"]
    assert "NOTION_API_KEY" in st["notion"]["reason"]
    monkeypatch.setenv("NOTION_API_KEY", "secret_x")
    st = config.connector_status(cfg)
    assert st["notion"]["active"]
    assert not st["gmail"]["active"]
    assert config.is_tool_allowed("search_notion", cfg)
    assert not config.is_tool_allowed("list_emails", cfg)
    assert config.is_tool_allowed("care_list", cfg)   # not gated


def test_quiet_hours_helpers():
    q = {"start": "22:00", "end": "08:00"}
    assert config.in_quiet_hours(datetime(2026, 1, 1, 23, 30), q)
    assert config.in_quiet_hours(datetime(2026, 1, 1, 3, 0), q)
    assert not config.in_quiet_hours(datetime(2026, 1, 1, 12, 0), q)
    moved = config.next_allowed_time(datetime(2026, 1, 1, 23, 30), q)
    assert (moved.day, moved.hour, moved.minute) == (2, 8, 0)
    assert config.next_allowed_time(datetime(2026, 1, 1, 12, 0), q).hour == 12


def test_user_slug(write_config):
    cfg = write_config(user={"name": "Ada Lovelace"})
    assert config.user_slug(cfg) == "ada"
    cfg = write_config(user={"name": ""})
    assert config.user_slug(cfg) == "user"
