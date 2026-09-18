from __future__ import annotations

"""
Sub-agent management tools for the main agent.

Tools:
  spawn_subagent        — launch a sub-agent process
  list_subagents        — inspect registry (running / completed / crashed)
  cancel_subagent       — kill a running sub-agent
  read_subagent_output  — read results and remove registry entry
"""

import json
import multiprocessing
import os
import signal
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from ._state import _PROJECT_ROOT

# ---- Paths ----------------------------------------------------------------
_AGENT_FS = _PROJECT_ROOT / "agent_file_system"
_SUBAGENTS_DIR = _AGENT_FS / "subagents"
_REGISTRY_FILE = str(_AGENT_FS / "currently_running_subagents.json")
_MAX_CONCURRENT = 3

# ---- Registry I/O ----------------------------------------------------------

def _load_registry() -> list[dict]:
    if not os.path.exists(_REGISTRY_FILE):
        return []
    with open(_REGISTRY_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_registry(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_FILE), exist_ok=True)
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---- Tool: spawn_subagent --------------------------------------------------

def spawn_subagent(
    task: str,
    tool_profiles: list[str],
    name: str,
    notify_on_completion: bool = True,
    max_runtime_seconds: int = 1800,
    max_rounds: int = 20,
    model: str | None = None,
) -> str:
    from core.config import default_model
    DEFAULT_MODEL = default_model()

    # Check concurrency limit
    registry = _load_registry()
    running = [e for e in registry if e.get("status") == "running"]
    if len(running) >= _MAX_CONCURRENT:
        names = [e["name"] for e in running]
        return f"Error: concurrency limit reached ({_MAX_CONCURRENT} sub-agents already running): {names}. Cancel one before spawning a new one."

    # Validate profiles
    from agents.subagent_runner import TOOL_PROFILES
    invalid = [p for p in tool_profiles if p not in TOOL_PROFILES]
    if invalid:
        valid = list(TOOL_PROFILES.keys())
        return f"Error: unknown tool profiles: {invalid}. Valid profiles: {valid}"

    subagent_id = secrets.token_hex(6)
    subagent_dir = _SUBAGENTS_DIR / subagent_id
    subagent_dir.mkdir(parents=True, exist_ok=True)
    (_SUBAGENTS_DIR / "subagents").mkdir(parents=True, exist_ok=True)

    # Write task spec
    task_spec = {
        "id": subagent_id,
        "name": name,
        "task": task,
        "tool_profiles": tool_profiles,
        "model": model or DEFAULT_MODEL,
        "max_rounds": max_rounds,
        "max_runtime_seconds": max_runtime_seconds,
        "notify_on_completion": notify_on_completion,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    with open(subagent_dir / "task.json", "w") as f:
        json.dump(task_spec, f, indent=2)

    # Spawn subprocess
    runner_path = str(Path(__file__).parents[1] / "subagent_runner.py")
    src_path = str(Path(__file__).parents[2])  # src/

    def _run():
        # Child process: insert src/ on path and run
        sys.path.insert(0, src_path)
        from agents.subagent_runner import run_subagent
        run_subagent(subagent_id)

    proc = multiprocessing.Process(target=_run, name=f"subagent-{name}", daemon=False)
    proc.start()
    pid = proc.pid

    # Register
    now = datetime.now().astimezone().isoformat()
    registry.append({
        "id": subagent_id,
        "name": name,
        "status": "running",
        "task": task[:200] + ("..." if len(task) > 200 else ""),
        "tool_profiles": tool_profiles,
        "model": model or DEFAULT_MODEL,
        "max_runtime_seconds": max_runtime_seconds,
        "notify_on_completion": notify_on_completion,
        "pid": pid,
        "started_at": now,
        "updated_at": now,
    })
    _save_registry(registry)

    profiles_str = ", ".join(tool_profiles)
    return (
        f"Sub-agent '{name}' spawned (id={subagent_id}, pid={pid}).\n"
        f"Profiles: {profiles_str} | Max runtime: {max_runtime_seconds}s | "
        f"Notify on completion: {notify_on_completion}.\n"
        f"Use list_subagents() to check status."
    )


# ---- Tool: list_subagents --------------------------------------------------

def list_subagents() -> str:
    registry = _load_registry()
    if not registry:
        return "No sub-agents in registry."

    lines = []
    for e in registry:
        pid = e.get("pid", 0)
        status = e.get("status", "unknown")

        # Auto-detect crashed: status=running but PID is dead
        if status == "running" and pid and not _pid_alive(pid):
            status = "crashed (PID dead)"

        age = ""
        started = e.get("started_at")
        if started:
            try:
                dt = datetime.fromisoformat(started)
                secs = (datetime.now().astimezone() - dt).total_seconds()
                age = f", running {int(secs)}s" if status == "running" else f", ran {int(secs)}s"
            except Exception:
                pass

        token_info = ""
        if e.get("token_usage"):
            t = e["token_usage"]
            token_info = f", tokens={t.get('total', 0)}"

        lines.append(
            f"• [{status.upper()}] {e['name']} (id={e['id']}, pid={pid}{age}{token_info})\n"
            f"  Task: {e.get('task', '')[:120]}\n"
            f"  Profiles: {', '.join(e.get('tool_profiles', []))}"
        )

    return "\n\n".join(lines)


# ---- Tool: cancel_subagent -------------------------------------------------

def cancel_subagent(subagent_id: str) -> str:
    registry = _load_registry()
    entry = next((e for e in registry if e["id"] == subagent_id), None)

    if not entry:
        return f"Error: no sub-agent with id '{subagent_id}' found in registry."

    if entry.get("status") != "running":
        return f"Sub-agent '{entry['name']}' is not running (status={entry['status']}). Nothing to cancel."

    pid = entry.get("pid")
    killed = False
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            pass  # Already dead
        except Exception as e:
            return f"Error killing PID {pid}: {e}"

    # Update registry
    now = datetime.now().astimezone().isoformat()
    entry["status"] = "cancelled"
    entry["updated_at"] = now
    entry["completed_at"] = now
    entry["error"] = "Cancelled by main agent"
    _save_registry(registry)

    # Append to history
    from agents.subagent_runner import _append_history
    _append_history({
        "id": subagent_id,
        "name": entry["name"],
        "status": "cancelled",
        "task": entry.get("task", ""),
        "tool_profiles": entry.get("tool_profiles", []),
        "model": entry.get("model", ""),
        "started_at": entry.get("started_at"),
        "completed_at": now,
        "duration_seconds": None,
        "token_usage": entry.get("token_usage"),
        "output_file": None,
        "error": "Cancelled by main agent",
    })

    return f"Sub-agent '{entry['name']}' (pid={pid}) {'killed' if killed else 'was already dead'} and marked as cancelled."


# ---- Tool: read_subagent_output --------------------------------------------

def read_subagent_output(subagent_id: str) -> str:
    registry = _load_registry()
    entry = next((e for e in registry if e["id"] == subagent_id), None)

    if not entry:
        return f"Error: no sub-agent with id '{subagent_id}' in registry. It may have already been read."

    status = entry.get("status", "running")

    if status == "running":
        pid = entry.get("pid")
        if pid and _pid_alive(pid):
            started = entry.get("started_at", "")
            try:
                dt = datetime.fromisoformat(started)
                elapsed = int((datetime.now().astimezone() - dt).total_seconds())
            except Exception:
                elapsed = -1
            return (
                f"Sub-agent '{entry['name']}' is still running "
                f"(elapsed={elapsed}s). Check back later or call list_subagents()."
            )
        else:
            # PID dead but status still running — update to crashed
            entry["status"] = "crashed"
            entry["error"] = "Process died unexpectedly (PID not found)"
            _save_registry(registry)
            status = "crashed"

    # Read output file
    output_file = entry.get("output_file")
    output_contents = None
    if output_file:
        full_path = str(_AGENT_FS / output_file.replace("subagents/", "", 1))
        # Handle path like "subagents/abc123/output.json"
        full_path = str(_AGENT_FS / Path(output_file))
        if os.path.exists(full_path):
            with open(full_path) as f:
                try:
                    output_contents = json.load(f)
                except json.JSONDecodeError:
                    with open(full_path) as f2:
                        output_contents = f2.read()

    # Remove from registry (main agent has read it)
    _save_registry([e for e in registry if e["id"] != subagent_id])

    # Build response
    name = entry["name"]
    token_usage = entry.get("token_usage", {})
    completed_at = entry.get("completed_at", "unknown")

    result_lines = [
        f"Sub-agent '{name}' (id={subagent_id})",
        f"Status: {status} | Completed: {completed_at}",
        f"Token usage: in={token_usage.get('input', 0)} out={token_usage.get('output', 0)} total={token_usage.get('total', 0)}",
    ]

    if entry.get("error"):
        result_lines.append(f"Error: {entry['error']}")

    if output_contents:
        result_lines.append("\n--- OUTPUT ---")
        if isinstance(output_contents, dict):
            result_lines.append(json.dumps(output_contents, indent=2))
        else:
            result_lines.append(str(output_contents))
    else:
        result_lines.append("\nNo output file found.")

    result_lines.append("\n[Registry entry removed — sub-agent acknowledged.]")
    return "\n".join(result_lines)


# ---- Schema ----------------------------------------------------------------

SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": (
                "Spawn a sub-agent process to handle a long-running, complex, or parallel task. "
                "The sub-agent runs independently and notifies you when done. "
                "Use this for tasks that would consume too much context in the main conversation, "
                "need deep research, or can run in parallel while you continue talking to the user. "
                "Max 3 concurrent sub-agents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Fully self-contained task for the sub-agent. "
                            "It has ZERO access to your conversation history, so include ALL context it needs. "
                            "Structure your task as follows:\n"
                            "1. SYSTEM PROMPT: Start with a clear role/persona for the sub-agent "
                            "(e.g. 'You are a research assistant specialising in email triage...'). "
                            "Define its goal, what a good output looks like, and any constraints.\n"
                            "2. AVAILABLE TOOLS: List the tools it has access to (derived from the profiles you selected) "
                            "and how it should use each one. E.g. 'Use tavily_search for web lookups, "
                            "list_emails + read_email to check Gmail, fs_write_json to save your findings.'\n"
                            "3. TASK: The specific work to perform, with enough detail to be actionable without you.\n"
                            "4. OUTPUT FORMAT: Exactly what to write to output.json — fields, structure, level of detail."
                        ),
                    },
                    "tool_profiles": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["research", "notion", "calendar_email", "filesystem", "full"],
                        },
                        "description": "Tool profiles to grant. Multiple profiles are unioned. 'research'=web+email+memory, 'notion'=notion+fs, 'calendar_email'=calendar+email, 'filesystem'=fs only, 'full'=everything except scheduling/spawn.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short descriptive name, e.g. 'deep_care_email_check'. Used in notifications and logs.",
                    },
                    "notify_on_completion": {
                        "type": "boolean",
                        "description": "Whether to send a Slack DM to the user when done. Main agent always gets notified regardless. Default true.",
                    },
                    "max_runtime_seconds": {
                        "type": "integer",
                        "description": "Hard runtime limit in seconds. Default 1800 (30 min). At T-3min tools are stripped and agent writes output.",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Max agent loop iterations. Default 20.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use. Defaults to same as parent agent.",
                    },
                },
                "required": ["task", "tool_profiles", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_subagents",
            "description": "List all sub-agents in the registry with their status (running/completed/crashed/cancelled), elapsed time, and token usage. Use to check progress or avoid spawning duplicates.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_subagent",
            "description": "Cancel and kill a running sub-agent. Use when the user asks to stop it, or if it's no longer needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subagent_id": {
                        "type": "string",
                        "description": "The sub-agent ID returned by spawn_subagent.",
                    },
                },
                "required": ["subagent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_subagent_output",
            "description": (
                "Check status and read the output of a completed sub-agent. "
                "Returns 'still running' if not done yet. "
                "If completed or crashed, returns the full output and removes the registry entry. "
                "Call this when you receive a sub-agent completion notification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subagent_id": {
                        "type": "string",
                        "description": "The sub-agent ID returned by spawn_subagent.",
                    },
                },
                "required": ["subagent_id"],
            },
        },
    },
]
