"""
End-to-end sub-agent lifecycle tests.

These tests exercise the full lifecycle using real file I/O but with:
- Mocked LLM (no real API calls)
- Mocked multiprocessing.Process (no real subprocesses for spawn tests)
- Real file system redirected to tmp_path via conftest fixtures

Scenarios:
1. spawn → task.json written, registry populated
2. spawn → run_subagent() → output.json + history.jsonl + queue.json
3. spawn → run → read_subagent_output → registry entry removed
4. spawn → cancel → registry shows cancelled + history appended
5. Multiple profiles union correctly produces larger tool set than individual
6. Blocked tools never surfaced across all lifecycle stages
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = str(Path(__file__).parents[3])
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tests.subagents.conftest import make_registry_entry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_llm_response(text="Task complete.", usage_in=20, usage_out=10):
    from llms.base import ChatResponse, Usage
    return ChatResponse(
        text=text,
        tool_calls=[],
        usage=Usage(input_tokens=usage_in, output_tokens=usage_out),
        stop_reason="end_turn",
    )


def _run_subagent_mocked(tmp_agent_fs, subagent_id, extra_spec=None, llm_text="Done."):
    """Create task.json and run run_subagent with a mocked LLM."""
    import agents.subagent_runner as runner

    subagent_dir = tmp_agent_fs / "agent_file_system" / "subagents" / subagent_id
    subagent_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "id": subagent_id,
        "name": f"agent_{subagent_id}",
        "task": "Do the research task",
        "tool_profiles": ["research"],
        "model": "gpt-4o-mini",
        "max_rounds": 2,
        "max_runtime_seconds": 1800,
        "notify_on_completion": True,
    }
    if extra_spec:
        spec.update(extra_spec)
    (subagent_dir / "task.json").write_text(json.dumps(spec))

    with patch("agents.subagent_runner.get_llm_client") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _fake_llm_response(llm_text)
        mock_factory.return_value = mock_llm
        runner.run_subagent(subagent_id)

    return spec


# ── Lifecycle: spawn ──────────────────────────────────────────────────────────

class TestSpawnLifecycle:

    def test_spawn_creates_task_json(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_proc.return_value.pid = 1001
            sa.spawn_subagent("My task", ["research"], "researcher")

        subagents_dir = tmp_agent_fs / "agent_file_system" / "subagents"
        task_files = list(subagents_dir.glob("*/task.json"))
        assert len(task_files) == 1
        spec = json.loads(task_files[0].read_text())
        assert spec["task"] == "My task"
        assert spec["name"] == "researcher"

    def test_spawn_populates_registry(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_proc.return_value.pid = 2002
            sa.spawn_subagent("Build something", ["filesystem"], "builder")

        reg = sa._load_registry()
        assert len(reg) == 1
        assert reg[0]["name"] == "builder"
        assert reg[0]["status"] == "running"
        assert reg[0]["pid"] == 2002
        assert "filesystem" in reg[0]["tool_profiles"]

    def test_spawn_result_message_parseable(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        with patch.object(sa.multiprocessing, "Process") as mock_proc:
            mock_proc.return_value.pid = 3003
            result = sa.spawn_subagent("Do stuff", ["research"], "stuff_doer")

        assert isinstance(result, str)
        assert "stuff_doer" in result
        assert "pid=3003" in result


# ── Lifecycle: run_subagent ───────────────────────────────────────────────────

class TestRunLifecycle:

    def test_run_creates_output_json(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        _run_subagent_mocked(tmp_agent_fs, "run01")
        output = tmp_agent_fs / "agent_file_system" / "subagents" / "run01" / "output.json"
        assert output.exists()

    def test_run_creates_run_log(self, tmp_agent_fs):
        _run_subagent_mocked(tmp_agent_fs, "run02")
        log = tmp_agent_fs / "agent_file_system" / "subagents" / "run02" / "run.log"
        assert log.exists()
        content = log.read_text()
        assert "SUB-AGENT run02 STARTED" in content

    def test_run_appends_history(self, tmp_agent_fs, history_file):
        _run_subagent_mocked(tmp_agent_fs, "run03")
        lines = open(history_file).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "run03"
        assert record["status"] in ("success", "crashed")

    def test_run_writes_queue_entry(self, tmp_agent_fs, queue_file):
        _run_subagent_mocked(tmp_agent_fs, "run04")
        queue = json.loads(open(queue_file).read())
        assert len(queue) == 1
        entry = queue[0]
        assert entry["source"] == "subagent_complete"
        assert entry["subagent_id"] == "run04"

    def test_run_updates_registry_status(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        import agents.tools.subagent as sa

        sid = "run05"
        # Pre-populate registry
        sa._save_registry([make_registry_entry(sid, status="running")])
        _run_subagent_mocked(tmp_agent_fs, sid)

        reg = runner._load_registry()
        entry = next((e for e in reg if e["id"] == sid), None)
        assert entry is not None
        assert entry["status"] in ("completed", "crashed")

    def test_run_log_contains_token_info(self, tmp_agent_fs):
        _run_subagent_mocked(tmp_agent_fs, "run06")
        log = tmp_agent_fs / "agent_file_system" / "subagents" / "run06" / "run.log"
        content = log.read_text()
        assert "in=" in content and "out=" in content


# ── Lifecycle: read_subagent_output ──────────────────────────────────────────

class TestReadLifecycle:

    def test_read_returns_output_content(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        sid = "read01"
        # Run first
        _run_subagent_mocked(tmp_agent_fs, sid, llm_text="Great findings!")
        # Pre-populate registry with completed status
        sa._save_registry([make_registry_entry(sid, status="completed")])

        # Point to the output file
        reg = sa._load_registry()
        reg[0]["output_file"] = f"subagents/{sid}/output.json"
        sa._save_registry(reg)

        result = sa.read_subagent_output(sid)
        assert isinstance(result, str)
        assert sid in result

    def test_read_removes_registry_entry(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        sid = "read02"
        _run_subagent_mocked(tmp_agent_fs, sid)
        sa._save_registry([make_registry_entry(sid, status="completed")])

        sa.read_subagent_output(sid)
        assert sa._load_registry() == []

    def test_read_twice_returns_error(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        sid = "read03"
        sa._save_registry([make_registry_entry(sid, status="completed")])
        sa.read_subagent_output(sid)

        # Second read — entry is gone
        result = sa.read_subagent_output(sid)
        assert "Error" in result


# ── Lifecycle: cancel_subagent ────────────────────────────────────────────────

class TestCancelLifecycle:

    def test_cancel_updates_registry(self, tmp_agent_fs):
        import agents.tools.subagent as sa

        sid = "cancel01"
        sa._save_registry([make_registry_entry(sid, pid=99999999, status="running")])
        sa.cancel_subagent(sid)

        reg = sa._load_registry()
        assert reg[0]["status"] == "cancelled"

    def test_cancel_appends_history(self, tmp_agent_fs, history_file):
        import agents.tools.subagent as sa

        sid = "cancel02"
        sa._save_registry([make_registry_entry(sid, pid=99999999, status="running")])
        sa.cancel_subagent(sid)

        lines = open(history_file).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["status"] == "cancelled"

    def test_cancel_completed_does_not_modify_history(self, tmp_agent_fs, history_file):
        import agents.tools.subagent as sa

        sid = "cancel03"
        sa._save_registry([make_registry_entry(sid, status="completed")])
        sa.cancel_subagent(sid)

        # History should NOT be appended — agent was not running
        if os.path.exists(history_file):
            lines = open(history_file).readlines()
            assert len(lines) == 0


# ── Profile union correctness ─────────────────────────────────────────────────

class TestProfileUnionE2E:

    def test_combined_profiles_larger_than_individual(self):
        from agents.subagent_runner import resolve_tools
        research = len(resolve_tools(["research"]))
        notion = len(resolve_tools(["notion"]))
        combined = len(resolve_tools(["research", "notion"]))
        assert combined >= max(research, notion)

    def test_combined_has_tools_from_both(self):
        from agents.subagent_runner import resolve_tools
        combined = {
            t.get("function", {}).get("name", t.get("name"))
            for t in resolve_tools(["research", "notion"])
        }
        assert "tavily_search" in combined  # research
        assert "search_notion" in combined  # notion

    def test_full_is_superset_of_all_others(self):
        from agents.subagent_runner import TOOL_PROFILES, resolve_tools

        full_names = {
            t.get("function", {}).get("name", t.get("name"))
            for t in resolve_tools(["full"])
        }
        for profile in ["research", "notion", "calendar_email", "filesystem"]:
            profile_names = {
                t.get("function", {}).get("name", t.get("name"))
                for t in resolve_tools([profile])
            }
            missing = profile_names - full_names
            assert not missing, f"Full profile missing tools from '{profile}': {missing}"


# ── Blocked tool invariants across lifecycle ──────────────────────────────────

class TestBlockedToolsE2E:

    BLOCKED = [
        "spawn_subagent", "cancel_subagent", "list_subagents", "read_subagent_output",
        "schedule_notifications", "set_reminder", "cancel_reminder",
        "create_cron_job", "delete_cron_job", "list_cron_jobs",
    ]

    @pytest.mark.parametrize("blocked", BLOCKED)
    def test_blocked_tool_not_in_full_resolved(self, blocked):
        from agents.subagent_runner import resolve_tools
        names = {t.get("function", {}).get("name", t.get("name")) for t in resolve_tools(["full"])}
        assert blocked not in names

    @pytest.mark.parametrize("blocked", BLOCKED)
    def test_blocked_tool_not_granted_during_run(self, tmp_agent_fs, blocked):
        """The runner should not hand blocked tools to the LLM."""
        import agents.subagent_runner as runner

        sid = "block_check"
        subagent_dir = tmp_agent_fs / "agent_file_system" / "subagents" / sid
        subagent_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "id": sid, "name": "blocker", "task": "check",
            "tool_profiles": ["full"], "model": "gpt-4o-mini",
            "max_rounds": 1, "max_runtime_seconds": 1800, "notify_on_completion": False,
        }
        (subagent_dir / "task.json").write_text(json.dumps(spec))

        tools_seen = []

        def capture_complete(messages, tools=None, max_tokens=None):
            if tools:
                tools_seen.extend(tools)
            return _fake_llm_response()

        with patch("agents.subagent_runner.get_llm_client") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.complete.side_effect = capture_complete
            mock_factory.return_value = mock_llm
            runner.run_subagent(sid)

        seen_names = {t.get("function", {}).get("name", t.get("name")) for t in tools_seen}
        assert blocked not in seen_names, f"Blocked tool '{blocked}' was passed to LLM"
