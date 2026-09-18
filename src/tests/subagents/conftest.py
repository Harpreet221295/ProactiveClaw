"""
Shared fixtures for sub-agent tests.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure src/ is on path
SRC = str(Path(__file__).parents[3])
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True)
def all_connectors_active(monkeypatch):
    """Sub-agent tests exercise tool *profiles*; treat every connector as active
    so results don't depend on which API keys exist on the test machine."""
    import core.config as cfg
    monkeypatch.setattr(cfg, "active_tool_names", lambda cfg_=None: cfg.gated_tool_names())
    monkeypatch.setattr(cfg, "is_tool_allowed", lambda name, cfg_=None: True)


@pytest.fixture()
def tmp_agent_fs(monkeypatch, tmp_path):
    """
    Redirect all sub-agent file I/O to a temp directory.
    Patches _PROJECT_ROOT in _state and subagent modules so no real files are touched.
    """
    # Create expected subdirs
    (tmp_path / "agent_file_system" / "subagents").mkdir(parents=True)
    (tmp_path / "engagement_data").mkdir()

    # Patch _PROJECT_ROOT in _state
    import agents.tools._state as _state
    monkeypatch.setattr(_state, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_state, "_AGENT_FS_BASE", tmp_path / "agent_file_system")

    # Patch paths in subagent tool module
    import agents.tools.subagent as subagent_tool
    monkeypatch.setattr(subagent_tool, "_AGENT_FS", tmp_path / "agent_file_system")
    monkeypatch.setattr(subagent_tool, "_SUBAGENTS_DIR", tmp_path / "agent_file_system" / "subagents")
    monkeypatch.setattr(subagent_tool, "_REGISTRY_FILE", str(tmp_path / "agent_file_system" / "currently_running_subagents.json"))

    # Patch paths in subagent runner module
    import agents.subagent_runner as runner
    monkeypatch.setattr(runner, "_AGENT_FS", tmp_path / "agent_file_system")
    monkeypatch.setattr(runner, "_SUBAGENTS_DIR", tmp_path / "agent_file_system" / "subagents")
    monkeypatch.setattr(runner, "_HISTORY_FILE", str(tmp_path / "agent_file_system" / "subagents" / "history.jsonl"))
    monkeypatch.setattr(runner, "_REGISTRY_FILE", str(tmp_path / "agent_file_system" / "currently_running_subagents.json"))
    monkeypatch.setattr(runner, "_QUEUE_FILE", str(tmp_path / "engagement_data" / "queue.json"))

    return tmp_path


@pytest.fixture()
def registry_file(tmp_agent_fs):
    return str(tmp_agent_fs / "agent_file_system" / "currently_running_subagents.json")


@pytest.fixture()
def queue_file(tmp_agent_fs):
    return str(tmp_agent_fs / "engagement_data" / "queue.json")


@pytest.fixture()
def history_file(tmp_agent_fs):
    return str(tmp_agent_fs / "agent_file_system" / "subagents" / "history.jsonl")


def make_registry_entry(
    subagent_id="abc123",
    name="test_agent",
    status="running",
    pid=None,
    task="Do something",
    profiles=None,
    max_runtime=1800,
    notify=True,
):
    import os
    from datetime import datetime
    return {
        "id": subagent_id,
        "name": name,
        "status": status,
        "task": task,
        "tool_profiles": profiles or ["research"],
        "model": "gpt-4o-mini",
        "max_runtime_seconds": max_runtime,
        "notify_on_completion": notify,
        "pid": pid or os.getpid(),  # use current PID so it's "alive"
        "started_at": datetime.now().astimezone().isoformat(),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
