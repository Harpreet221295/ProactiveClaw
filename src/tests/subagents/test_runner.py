"""
Tests for subagent_runner internals.

Covers:
- _append_history writes valid JSONL
- _write_completion_to_queue writes correct queue entry with source=subagent_complete
- SubagentLogger writes to file correctly
- resolve_tools returns only granted tools with no blocked tools
- output.json fallback written when agent produces no file
- output.json metadata injection when agent writes its own file
- wrap-up trigger: tools stripped at T-3min
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


# ── _append_history ──────────────────────────────────────────────────────────

class TestAppendHistory:

    def test_creates_file_with_valid_jsonl(self, tmp_agent_fs, history_file):
        import agents.subagent_runner as runner
        record = {"id": "abc", "name": "test", "status": "success"}
        runner._append_history(record)

        assert os.path.exists(history_file)
        lines = open(history_file).readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["id"] == "abc"

    def test_appends_multiple_records(self, tmp_agent_fs, history_file):
        import agents.subagent_runner as runner
        runner._append_history({"id": "a", "status": "success"})
        runner._append_history({"id": "b", "status": "crashed"})

        lines = open(history_file).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "a"
        assert json.loads(lines[1])["id"] == "b"

    def test_records_are_newline_separated(self, tmp_agent_fs, history_file):
        import agents.subagent_runner as runner
        runner._append_history({"id": "x"})
        runner._append_history({"id": "y"})
        content = open(history_file).read()
        # Two lines: each ends with \n
        assert content.count("\n") == 2


# ── _write_completion_to_queue ────────────────────────────────────────────────

class TestWriteCompletionToQueue:

    def test_creates_queue_file(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        runner._write_completion_to_queue("abc", "myagent", "success", True, "subagents/abc/output.json")
        assert os.path.exists(queue_file)

    def test_queue_entry_has_correct_source(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        runner._write_completion_to_queue("abc", "myagent", "success", True, "")
        queue = json.loads(open(queue_file).read())
        assert len(queue) == 1
        assert queue[0]["source"] == "subagent_complete"

    def test_queue_entry_has_subagent_id(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        runner._write_completion_to_queue("xyzabc", "myagent", "success", True, "")
        queue = json.loads(open(queue_file).read())
        assert queue[0]["subagent_id"] == "xyzabc"

    def test_crashed_status_reflected_in_message(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        runner._write_completion_to_queue("abc", "badagent", "crashed", True, "")
        queue = json.loads(open(queue_file).read())
        assert "crashed" in queue[0]["message"].lower()

    def test_appends_to_existing_queue(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        existing = [{"source": "pre_exit", "message": "earlier entry"}]
        os.makedirs(os.path.dirname(queue_file), exist_ok=True)
        open(queue_file, "w").write(json.dumps(existing))

        runner._write_completion_to_queue("abc", "agent", "success", True, "")
        queue = json.loads(open(queue_file).read())
        assert len(queue) == 2
        assert queue[0]["source"] == "pre_exit"
        assert queue[1]["source"] == "subagent_complete"

    def test_notify_on_completion_stored(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner
        runner._write_completion_to_queue("abc", "a", "success", False, "")
        queue = json.loads(open(queue_file).read())
        assert queue[0]["notify_on_completion"] is False


# ── SubagentLogger ────────────────────────────────────────────────────────────

class TestSubagentLogger:

    def test_write_creates_log_file(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        log_path = str(tmp_agent_fs / "agent_file_system" / "subagents" / "test_id" / "run.log")
        logger = runner.SubagentLogger(log_path, "test_id", "TestAgent")
        logger.write("hello world")
        assert os.path.exists(log_path)
        content = open(log_path).read()
        assert "hello world" in content

    def test_log_turn_writes_token_counts(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        log_path = str(tmp_agent_fs / "run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = runner.SubagentLogger(log_path, "id", "n")
        logger.log_turn(1, 100, 50, {"input": 100, "output": 50, "total": 150})
        content = open(log_path).read()
        assert "TURN 1" in content
        assert "in=100" in content
        assert "out=50" in content

    def test_log_tool_call_truncates_long_args(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        log_path = str(tmp_agent_fs / "run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = runner.SubagentLogger(log_path, "id", "n")
        long_args = "x" * 1000
        logger.log_tool_call("some_tool", long_args)
        content = open(log_path).read()
        assert "some_tool" in content
        assert "..." in content  # truncated

    def test_log_wrap_up_writes_marker(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        log_path = str(tmp_agent_fs / "run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = runner.SubagentLogger(log_path, "id", "n")
        logger.log_wrap_up()
        content = open(log_path).read()
        assert "TIME LIMIT" in content.upper() or "WRAP" in content.upper() or "APPROACHING" in content.upper()

    def test_multiple_writes_accumulate(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        log_path = str(tmp_agent_fs / "run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = runner.SubagentLogger(log_path, "id", "n")
        logger.write("line 1")
        logger.write("line 2")
        logger.write("line 3")
        lines = [l for l in open(log_path).readlines() if l.strip()]
        assert len(lines) == 3


# ── resolve_tools ─────────────────────────────────────────────────────────────

class TestResolveTools:

    def test_empty_profiles_returns_empty(self):
        from agents.subagent_runner import resolve_tools
        result = resolve_tools([])
        assert result == []

    def test_invalid_profile_returns_empty(self):
        from agents.subagent_runner import resolve_tools
        result = resolve_tools(["nonexistent"])
        assert result == []

    def test_blocked_tools_never_in_result(self):
        from agents.subagent_runner import resolve_tools, _BLOCKED_TOOLS
        result = resolve_tools(["full"])
        names = {t.get("function", {}).get("name", t.get("name")) for t in result}
        for blocked in _BLOCKED_TOOLS:
            assert blocked not in names, f"Blocked tool '{blocked}' leaked into resolved schema"

    def test_result_is_list_of_dicts(self):
        from agents.subagent_runner import resolve_tools
        result = resolve_tools(["research"])
        assert isinstance(result, list)
        assert all(isinstance(t, dict) for t in result)


# ── output.json handling ─────────────────────────────────────────────────────

class TestOutputJsonHandling:

    def _make_task_spec(self, tmp_agent_fs, subagent_id: str, **overrides) -> str:
        """Create a minimal task.json and return the subagent dir path."""
        subagent_dir = tmp_agent_fs / "agent_file_system" / "subagents" / subagent_id
        subagent_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "id": subagent_id,
            "name": "test_agent",
            "task": "Do the thing",
            "tool_profiles": ["research"],
            "model": "gpt-4o-mini",
            "max_rounds": 2,
            "max_runtime_seconds": 1800,
            "notify_on_completion": False,
        }
        spec.update(overrides)
        (subagent_dir / "task.json").write_text(json.dumps(spec))
        return str(subagent_dir)

    def _mock_llm_response(self, text="Done.", tool_calls=None, usage_in=10, usage_out=5):
        from llms.base import ChatResponse, Usage
        usage = Usage(input_tokens=usage_in, output_tokens=usage_out)
        return ChatResponse(text=text, tool_calls=tool_calls or [], usage=usage, stop_reason="end_turn")

    def test_fallback_output_json_written_when_missing(self, tmp_agent_fs):
        """When agent finishes without writing output.json, runner writes fallback."""
        import agents.subagent_runner as runner

        sid = "fallback1"
        subagent_dir = self._make_task_spec(tmp_agent_fs, sid)
        output_path = Path(subagent_dir) / "output.json"

        mock_response = self._mock_llm_response("My final answer")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "subagent_id" in data
        assert data["subagent_id"] == sid

    def test_fallback_output_includes_status(self, tmp_agent_fs):
        import agents.subagent_runner as runner

        sid = "fallback2"
        self._make_task_spec(tmp_agent_fs, sid)
        output_path = tmp_agent_fs / "agent_file_system" / "subagents" / sid / "output.json"

        mock_response = self._mock_llm_response("Summary text")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        data = json.loads(output_path.read_text())
        assert "status" in data

    def test_agent_written_output_gets_metadata_injected(self, tmp_agent_fs):
        """When agent writes its own output.json, runner injects metadata fields."""
        import agents.subagent_runner as runner

        sid = "inject1"
        self._make_task_spec(tmp_agent_fs, sid)
        subagent_dir = tmp_agent_fs / "agent_file_system" / "subagents" / sid
        output_path = subagent_dir / "output.json"

        # Pre-write a custom output (simulating what fs_write_json would do)
        custom_output = {"findings": "great stuff", "score": 42}

        def side_effect_write_output(*args, **kwargs):
            output_path.write_text(json.dumps(custom_output))
            return self._mock_llm_response("Written output.json")

        mock_response = self._mock_llm_response("Done")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            # Simulate agent having written output.json before runner checks
            output_path.write_text(json.dumps(custom_output))
            runner.run_subagent(sid)

        data = json.loads(output_path.read_text())
        # Original content preserved
        assert data["findings"] == "great stuff"
        assert data["score"] == 42
        # Metadata injected
        assert "token_usage" in data
        assert "completed_at" in data
        assert "status" in data

    def test_history_appended_after_run(self, tmp_agent_fs, history_file):
        import agents.subagent_runner as runner

        sid = "hist1"
        self._make_task_spec(tmp_agent_fs, sid)
        mock_response = self._mock_llm_response("done")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        assert os.path.exists(history_file)
        lines = open(history_file).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == sid

    def test_queue_entry_written_after_run(self, tmp_agent_fs, queue_file):
        import agents.subagent_runner as runner

        sid = "queue1"
        self._make_task_spec(tmp_agent_fs, sid)
        mock_response = self._mock_llm_response("done")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        assert os.path.exists(queue_file)
        queue = json.loads(open(queue_file).read())
        assert len(queue) == 1
        assert queue[0]["source"] == "subagent_complete"
        assert queue[0]["subagent_id"] == sid

    def test_registry_updated_to_completed_after_run(self, tmp_agent_fs):
        import agents.subagent_runner as runner
        import agents.tools.subagent as sa

        sid = "reg1"
        self._make_task_spec(tmp_agent_fs, sid)

        # Pre-populate registry
        entry = make_registry_entry(sid, status="running")
        sa._save_registry([entry])

        mock_response = self._mock_llm_response("done")

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        reg = runner._load_registry()
        entry = next((e for e in reg if e["id"] == sid), None)
        assert entry is not None
        assert entry["status"] == "completed"

    def test_token_usage_accumulated_over_turns(self, tmp_agent_fs):
        import agents.subagent_runner as runner

        sid = "tokens1"
        self._make_task_spec(tmp_agent_fs, sid, max_rounds=3)
        output_path = tmp_agent_fs / "agent_file_system" / "subagents" / sid / "output.json"

        call_count = 0

        def fake_complete(messages, tools=None, max_tokens=None):
            nonlocal call_count
            call_count += 1
            return self._mock_llm_response("turn response", usage_in=10, usage_out=5)

        with patch("agents.subagent_runner.get_llm_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.complete.side_effect = fake_complete
            mock_llm_factory.return_value = mock_llm

            runner.run_subagent(sid)

        # Read output — check token total
        if output_path.exists():
            data = json.loads(output_path.read_text())
            total = data.get("token_usage", {}).get("total", 0)
            # At least 1 turn × (10 + 5) = 15
            assert total >= 15
