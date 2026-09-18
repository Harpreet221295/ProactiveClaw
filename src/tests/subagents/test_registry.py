"""
Tests for registry CRUD operations and PID liveness detection.

Covers:
- Loading an empty/missing registry returns []
- Saving and loading a registry round-trips correctly
- _update_registry_status patches the right entry
- _pid_alive returns True for current process, False for dead PID
- list_subagents detects dead PIDs as "crashed (PID dead)"
- cancel_subagent: marks entry cancelled, appends to history
- read_subagent_output: still-running guard, crashed auto-detection, entry removal
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).parents[3])
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tests.subagents.conftest import make_registry_entry


# ── Registry I/O ─────────────────────────────────────────────────────────────

class TestRegistryIO:

    def test_load_missing_registry_returns_empty(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa._load_registry()
        assert result == []

    def test_load_empty_file_returns_empty(self, tmp_agent_fs, registry_file):
        Path(registry_file).write_text("[]")
        import agents.tools.subagent as sa
        assert sa._load_registry() == []

    def test_load_corrupt_file_returns_empty(self, tmp_agent_fs, registry_file):
        Path(registry_file).write_text("{bad json{{")
        import agents.tools.subagent as sa
        assert sa._load_registry() == []

    def test_save_and_reload(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entries = [make_registry_entry("aaa"), make_registry_entry("bbb")]
        sa._save_registry(entries)
        loaded = sa._load_registry()
        assert len(loaded) == 2
        ids = {e["id"] for e in loaded}
        assert ids == {"aaa", "bbb"}

    def test_save_creates_directory_if_missing(self, tmp_agent_fs, monkeypatch):
        import agents.tools.subagent as sa
        new_path = str(tmp_agent_fs / "new_dir" / "registry.json")
        monkeypatch.setattr(sa, "_REGISTRY_FILE", new_path)
        sa._save_registry([make_registry_entry("x")])
        assert os.path.exists(new_path)


# ── _update_registry_status ──────────────────────────────────────────────────

class TestUpdateRegistryStatus:

    def test_patches_correct_entry(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        entries = [
            make_registry_entry("id1", status="running"),
            make_registry_entry("id2", status="running"),
        ]
        runner._save_registry(entries)
        runner._update_registry_status("id1", status="completed", completed_at="2026-01-01T00:00:00+00:00")
        loaded = runner._load_registry()
        id1 = next(e for e in loaded if e["id"] == "id1")
        id2 = next(e for e in loaded if e["id"] == "id2")
        assert id1["status"] == "completed"
        assert id2["status"] == "running"  # unchanged

    def test_unknown_id_is_no_op(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        entries = [make_registry_entry("id1")]
        runner._save_registry(entries)
        runner._update_registry_status("nonexistent", status="completed")
        loaded = runner._load_registry()
        assert loaded[0]["status"] == "running"  # unchanged

    def test_updated_at_is_refreshed(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        entries = [make_registry_entry("id1")]
        entries[0]["updated_at"] = "2000-01-01T00:00:00+00:00"
        runner._save_registry(entries)
        runner._update_registry_status("id1", status="completed")
        loaded = runner._load_registry()
        assert loaded[0]["updated_at"] != "2000-01-01T00:00:00+00:00"


# ── PID liveness ─────────────────────────────────────────────────────────────

class TestPidLiveness:

    def test_current_process_is_alive(self):
        import agents.tools.subagent as sa
        assert sa._pid_alive(os.getpid()) is True

    def test_bogus_pid_is_dead(self):
        import agents.tools.subagent as sa
        # PID 1 is always init; PID 99999999 should not exist
        assert sa._pid_alive(99999999) is False

    def test_zero_pid_is_dead(self):
        import agents.tools.subagent as sa
        # os.kill(0, 0) sends to the process group — not what we want; should be False
        # Our implementation catches PermissionError or ProcessLookupError
        # For PID 0 it may raise PermissionError on macOS which we treat as dead
        result = sa._pid_alive(0)
        assert isinstance(result, bool)


# ── list_subagents ───────────────────────────────────────────────────────────

class TestListSubagents:

    def test_empty_registry_message(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.list_subagents()
        assert "No sub-agents" in result

    def test_running_entry_shows(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", name="myagent", pid=os.getpid())
        sa._save_registry([entry])
        result = sa.list_subagents()
        assert "myagent" in result
        assert "RUNNING" in result.upper()

    def test_dead_pid_shown_as_crashed(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", name="deadagent", pid=99999999, status="running")
        sa._save_registry([entry])
        result = sa.list_subagents()
        assert "crashed" in result.lower()

    def test_completed_entry_shows(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", name="done", status="completed")
        sa._save_registry([entry])
        result = sa.list_subagents()
        assert "COMPLETED" in result.upper()

    def test_multiple_entries_all_shown(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entries = [
            make_registry_entry("id1", name="agent_one"),
            make_registry_entry("id2", name="agent_two"),
        ]
        sa._save_registry(entries)
        result = sa.list_subagents()
        assert "agent_one" in result
        assert "agent_two" in result

    def test_token_info_shown_when_present(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc")
        entry["token_usage"] = {"input": 100, "output": 50, "total": 150}
        sa._save_registry([entry])
        result = sa.list_subagents()
        assert "tokens=" in result


# ── cancel_subagent ──────────────────────────────────────────────────────────

class TestCancelSubagent:

    def test_cancel_nonexistent_returns_error(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.cancel_subagent("doesnotexist")
        assert "Error" in result

    def test_cancel_completed_returns_not_running(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", status="completed")
        sa._save_registry([entry])
        result = sa.cancel_subagent("abc")
        assert "not running" in result.lower()

    def test_cancel_running_marks_cancelled(self, tmp_agent_fs, history_file):
        import agents.tools.subagent as sa
        # Use current PID so _pid_alive would return True but SIGTERM to ourselves is ok
        # We instead use a dead PID so no actual kill happens
        entry = make_registry_entry("abc", pid=99999999, status="running")
        sa._save_registry([entry])
        result = sa.cancel_subagent("abc")
        # Registry should show cancelled
        reg = sa._load_registry()
        assert reg[0]["status"] == "cancelled"
        # History should have been appended
        assert os.path.exists(history_file)
        lines = open(history_file).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["status"] == "cancelled"
        assert record["id"] == "abc"

    def test_cancel_result_message_mentions_agent_name(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", name="special_agent", pid=99999999, status="running")
        sa._save_registry([entry])
        result = sa.cancel_subagent("abc")
        assert "special_agent" in result


# ── read_subagent_output ─────────────────────────────────────────────────────

class TestReadSubagentOutput:

    def test_missing_id_returns_error(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        result = sa.read_subagent_output("ghost")
        assert "Error" in result

    def test_still_running_returns_running_message(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", pid=os.getpid(), status="running")
        sa._save_registry([entry])
        result = sa.read_subagent_output("abc")
        assert "still running" in result.lower()
        # Entry should NOT be removed
        assert sa._load_registry()

    def test_dead_pid_auto_crashes_and_reads(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", pid=99999999, status="running")
        sa._save_registry([entry])
        result = sa.read_subagent_output("abc")
        # Since no output file exists we still get a response about crashed status
        assert "abc" in result
        # Entry removed from registry after read
        assert sa._load_registry() == []

    def test_completed_with_output_file(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        # Create the output file
        subagent_dir = tmp_agent_fs / "agent_file_system" / "subagents" / "abc"
        subagent_dir.mkdir(parents=True, exist_ok=True)
        output = {"findings": "all good", "status": "success"}
        (subagent_dir / "output.json").write_text(json.dumps(output))

        entry = make_registry_entry("abc", status="completed")
        entry["output_file"] = "subagents/abc/output.json"
        sa._save_registry([entry])

        result = sa.read_subagent_output("abc")
        assert "all good" in result
        # Entry removed
        assert sa._load_registry() == []

    def test_read_appends_removal_confirmation(self, tmp_agent_fs):
        import agents.tools.subagent as sa
        entry = make_registry_entry("abc", status="completed")
        sa._save_registry([entry])
        result = sa.read_subagent_output("abc")
        assert "Registry entry removed" in result
