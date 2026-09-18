"""
Tests for the four sub-agent tool functions (spawn/list/cancel/read).

Focuses on argument validation, concurrency limit, and message contract —
does NOT test actual subprocess execution (covered in test_e2e.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

SRC = str(Path(__file__).parents[3])
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tests.subagents.conftest import make_registry_entry


# ── spawn_subagent validation ─────────────────────────────────────────────────

class TestSpawnValidation:

    def test_invalid_profile_returns_error(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.spawn_subagent(
            task="Do something",
            tool_profiles=["bogus_profile"],
            name="test",
        )
        assert "Error" in result
        assert "bogus_profile" in result

    def test_mixed_valid_invalid_profiles_returns_error(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.spawn_subagent(
            task="Do something",
            tool_profiles=["research", "INVALID"],
            name="test",
        )
        assert "Error" in result

    def test_concurrency_limit_enforced(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        # Fill registry with 3 running entries
        entries = [
            make_registry_entry(f"id{i}", pid=os.getpid(), status="running")
            for i in range(3)
        ]
        sa._save_registry(entries)

        result = sa.spawn_subagent(
            task="Fourth task",
            tool_profiles=["research"],
            name="overflow",
        )
        assert "Error" in result
        assert "concurrency limit" in result.lower()

    def test_concurrency_limit_only_counts_running(self, tmp_agent_fs):
        """Completed entries don't count toward the concurrency limit."""
        import agents.tools.subagent as sa
        entries = [
            make_registry_entry(f"id{i}", status="completed") for i in range(3)
        ]
        sa._save_registry(entries)

        # Should NOT be blocked (no running agents)
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 9999
            mock_proc.return_value = mock_instance

            result = sa.spawn_subagent(
                task="Task with no running agents",
                tool_profiles=["research"],
                name="allowed",
            )
        assert "Error" not in result

    def test_spawn_success_returns_id_and_pid(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 12345
            mock_proc.return_value = mock_instance

            result = sa.spawn_subagent(
                task="Do something",
                tool_profiles=["research"],
                name="my_agent",
            )
        assert "my_agent" in result
        assert "pid=12345" in result

    def test_spawn_writes_task_json(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 42
            mock_proc.return_value = mock_instance

            sa.spawn_subagent(
                task="Research quantum computing",
                tool_profiles=["research"],
                name="quantum_researcher",
                max_rounds=10,
            )

        # Find the created subagent dir
        subagents_dir = tmp_agent_fs / "agent_file_system" / "subagents"
        subdirs = [d for d in subagents_dir.iterdir() if d.is_dir() and d.name != "subagents"]
        assert len(subdirs) == 1
        task_file = subdirs[0] / "task.json"
        assert task_file.exists()
        spec = json.loads(task_file.read_text())
        assert spec["task"] == "Research quantum computing"
        assert spec["name"] == "quantum_researcher"
        assert spec["max_rounds"] == 10
        assert "research" in spec["tool_profiles"]

    def test_spawn_registers_entry(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 777
            mock_proc.return_value = mock_instance

            sa.spawn_subagent(
                task="Some task",
                tool_profiles=["filesystem"],
                name="fs_agent",
            )

        reg = sa._load_registry()
        assert len(reg) == 1
        assert reg[0]["name"] == "fs_agent"
        assert reg[0]["status"] == "running"
        assert reg[0]["pid"] == 777

    def test_spawn_with_all_profiles(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 1
            mock_proc.return_value = mock_instance

            result = sa.spawn_subagent(
                task="Full access task",
                tool_profiles=["full"],
                name="full_agent",
            )
        assert "Error" not in result

    def test_spawn_task_truncated_in_registry(self, tmp_agent_fs):
        """Long tasks are stored as a truncated preview in the registry."""
        import agents.tools.subagent as sa
        long_task = "x" * 500
        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_instance = MagicMock()
            mock_instance.pid = 1
            mock_proc.return_value = mock_instance
            sa.spawn_subagent(task=long_task, tool_profiles=["research"], name="t")

        reg = sa._load_registry()
        assert len(reg[0]["task"]) <= 210  # 200 + "..."


# ── list_subagents ────────────────────────────────────────────────────────────

class TestListSubagentsTool:

    def test_no_entries_returns_expected_string(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.list_subagents()
        assert "No sub-agents" in result

    def test_returns_string_always(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.list_subagents()
        assert isinstance(result, str)

    def test_profiles_included_in_output(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", profiles=["notion", "research"])
        sa._save_registry([entry])
        result = sa.list_subagents()
        assert "notion" in result
        assert "research" in result


# ── SCHEMA structure ──────────────────────────────────────────────────────────

class TestSubagentSchema:

    def test_schema_has_four_tools(self):
        from agents.tools.subagent import SCHEMA
        assert len(SCHEMA) == 4

    def test_schema_tool_names(self):
        from agents.tools.subagent import SCHEMA
        names = {entry["function"]["name"] for entry in SCHEMA}
        expected = {"spawn_subagent", "list_subagents", "cancel_subagent", "read_subagent_output"}
        assert names == expected

    def test_spawn_requires_task_profiles_name(self):
        from agents.tools.subagent import SCHEMA
        spawn = next(e for e in SCHEMA if e["function"]["name"] == "spawn_subagent")
        required = spawn["function"]["parameters"]["required"]
        assert "task" in required
        assert "tool_profiles" in required
        assert "name" in required

    def test_spawn_profiles_has_enum(self):
        from agents.tools.subagent import SCHEMA
        spawn = next(e for e in SCHEMA if e["function"]["name"] == "spawn_subagent")
        profiles_prop = spawn["function"]["parameters"]["properties"]["tool_profiles"]
        assert "enum" in profiles_prop["items"]

    def test_cancel_requires_subagent_id(self):
        from agents.tools.subagent import SCHEMA
        cancel = next(e for e in SCHEMA if e["function"]["name"] == "cancel_subagent")
        assert "subagent_id" in cancel["function"]["parameters"]["required"]

    def test_read_requires_subagent_id(self):
        from agents.tools.subagent import SCHEMA
        read = next(e for e in SCHEMA if e["function"]["name"] == "read_subagent_output")
        assert "subagent_id" in read["function"]["parameters"]["required"]

    def test_list_has_no_required_params(self):
        from agents.tools.subagent import SCHEMA
        lst = next(e for e in SCHEMA if e["function"]["name"] == "list_subagents")
        assert lst["function"]["parameters"].get("required", []) == []


# ── TOOL_FUNCTIONS registration ───────────────────────────────────────────────

class TestToolFunctionRegistration:

    def test_all_four_tools_registered(self):
        from agents.tools import TOOL_FUNCTIONS
        for name in ["spawn_subagent", "list_subagents", "cancel_subagent", "read_subagent_output"]:
            assert name in TOOL_FUNCTIONS, f"'{name}' missing from TOOL_FUNCTIONS"

    def test_all_four_tools_in_schema(self):
        from agents.tools import TOOLS_SCHEMA
        names = {
            t.get("function", {}).get("name", t.get("name"))
            for t in TOOLS_SCHEMA
        }
        for name in ["spawn_subagent", "list_subagents", "cancel_subagent", "read_subagent_output"]:
            assert name in names, f"'{name}' missing from TOOLS_SCHEMA"

    def test_spawn_callable(self):
        from agents.tools import TOOL_FUNCTIONS
        fn = TOOL_FUNCTIONS["spawn_subagent"]
        assert callable(fn)
