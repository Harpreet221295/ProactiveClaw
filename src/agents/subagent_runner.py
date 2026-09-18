from __future__ import annotations

"""
Sub-agent subprocess entry point.

Usage:
    python3 subagent_runner.py <subagent_id>

Reads task spec from:
    agent_file_system/subagents/<id>/task.json

Writes output to:
    agent_file_system/subagents/<id>/output.json

Logs execution to:
    agent_file_system/subagents/<id>/run.log

Appends completion record to:
    agent_file_system/subagents/history.jsonl

Notifies main agent via:
    engagement_data/queue.json
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

# Resolve src/ on path
_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_SRC))

from agents.tools._state import _PROJECT_ROOT
from llms import get_llm_client
from agents.tools import TOOLS_SCHEMA, TOOL_FUNCTIONS, dispatch_tool_call

# ---- Paths ----------------------------------------------------------------
_AGENT_FS = _PROJECT_ROOT / "agent_file_system"
_SUBAGENTS_DIR = _AGENT_FS / "subagents"
_HISTORY_FILE = str(_SUBAGENTS_DIR / "history.jsonl")
_REGISTRY_FILE = str(_AGENT_FS / "currently_running_subagents.json")
_QUEUE_FILE = str(_PROJECT_ROOT / "engagement_data" / "queue.json")

# ---- Tool profiles --------------------------------------------------------
# Hard rules: scheduling/cron/queue tools and spawn_subagent are NEVER granted
_BLOCKED_TOOLS = {
    "schedule_notifications", "set_reminder", "cancel_reminder",
    "create_cron_job", "delete_cron_job", "list_cron_jobs",
    "spawn_subagent", "cancel_subagent", "list_subagents", "read_subagent_output",
    "care_set_proactiveness", "care_set_mode", "care_set_override",
}

TOOL_PROFILES: dict[str, list[str]] = {
    "research": [
        "tavily_search",
        "list_emails", "read_email",
        "query_long_term_memory", "retrieve_long_term_memory",
        "fs_read_file", "fs_write_file", "fs_list_files",
        "fs_read_json", "fs_write_json",
        "get_current_datetime",
    ],
    "notion": [
        "search_notion", "read_notion_page", "create_notion_page",
        "update_notion_page", "query_notion_database", "create_database_entry",
        "fs_read_file", "fs_write_file", "fs_list_files",
        "fs_read_json", "fs_write_json",
        "get_current_datetime",
    ],
    "calendar_email": [
        "list_calendar_events", "create_calendar_event", "update_calendar_event",
        "list_emails", "read_email", "send_email", "reply_to_email",
        "fs_read_file", "fs_write_file",
        "get_current_datetime",
    ],
    "filesystem": [
        "fs_read_file", "fs_write_file", "fs_list_files",
        "fs_create_directory", "fs_delete", "fs_move", "fs_search_file",
        "fs_read_json", "fs_write_json", "fs_read_csv", "fs_write_csv",
        "get_current_datetime",
    ],
    "full": [
        t for t in TOOL_FUNCTIONS.keys() if t not in _BLOCKED_TOOLS
    ],
}


def resolve_tools(profiles: list[str]) -> list[dict]:
    """Union tool schemas from multiple profiles, preserving order, no duplicates.
    Tools whose connector is not active are dropped as well."""
    from core.config import is_tool_allowed
    granted: set[str] = set()
    for profile in profiles:
        granted.update(TOOL_PROFILES.get(profile, []))
    granted -= _BLOCKED_TOOLS
    granted = {t for t in granted if is_tool_allowed(t)}

    def tool_name(schema_entry: dict) -> str:
        fn = schema_entry.get("function", {})
        return fn.get("name", schema_entry.get("name", ""))

    return [t for t in TOOLS_SCHEMA if tool_name(t) in granted]


# ---- Registry helpers -----------------------------------------------------

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


def _update_registry_status(subagent_id: str, **kwargs) -> None:
    entries = _load_registry()
    for e in entries:
        if e["id"] == subagent_id:
            e.update(kwargs)
            e["updated_at"] = datetime.now().astimezone().isoformat()
            break
    _save_registry(entries)


# ---- Queue helper ----------------------------------------------------------

def _write_completion_to_queue(subagent_id: str, name: str, status: str,
                                notify: bool, output_file: str) -> None:
    if status == "success":
        msg = (f"Sub-agent '{name}' completed successfully. "
               f"Call read_subagent_output('{subagent_id}') to read its findings.")
    else:
        msg = (f"Sub-agent '{name}' {status}. "
               f"Call read_subagent_output('{subagent_id}') for details.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": msg,
        "source": "subagent_complete",
        "subagent_id": subagent_id,
        "subagent_name": name,
        "notify_on_completion": notify,
        "output_file": output_file,
    }
    queue: list[dict] = []
    if os.path.exists(_QUEUE_FILE):
        with open(_QUEUE_FILE) as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []
    queue.append(entry)
    os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
    with open(_QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


# ---- History helper --------------------------------------------------------

def _append_history(record: dict) -> None:
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    with open(_HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---- Logger ----------------------------------------------------------------

class SubagentLogger:
    def __init__(self, log_path: str, subagent_id: str, name: str):
        self.log_path = log_path
        self.subagent_id = subagent_id
        self.name = name
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _ts(self) -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    def write(self, msg: str) -> None:
        line = f"[{self._ts()}] {msg}\n"
        print(line, end="")
        with open(self.log_path, "a") as f:
            f.write(line)

    def log_turn(self, turn: int, usage_in: int, usage_out: int, cumulative: dict) -> None:
        self.write(
            f"TURN {turn} | in={usage_in} out={usage_out} "
            f"turn_total={usage_in + usage_out} | "
            f"cumulative: in={cumulative['input']} out={cumulative['output']} "
            f"total={cumulative['total']}"
        )

    def log_tool_call(self, name: str, args: str) -> None:
        truncated = args[:300] + ("..." if len(args) > 300 else "")
        self.write(f"  TOOL CALL: {name}({truncated})")

    def log_tool_result(self, name: str, result: str) -> None:
        truncated = result[:400] + ("..." if len(result) > 400 else "")
        self.write(f"  TOOL RESULT [{name}]: {truncated}")

    def log_agent_text(self, text: str) -> None:
        if text:
            truncated = text[:600] + ("..." if len(text) > 600 else "")
            self.write(f"  AGENT: {truncated}")

    def log_wrap_up(self) -> None:
        self.write(">>> TIME LIMIT APPROACHING — stripping tools, forcing final output <<<")

    def log_hard_timeout(self) -> None:
        self.write(">>> HARD TIMEOUT — writing partial output and exiting <<<")


# ---- Main runner -----------------------------------------------------------

def run_subagent(subagent_id: str) -> None:
    # Load task spec
    task_file = str(_SUBAGENTS_DIR / subagent_id / "task.json")
    with open(task_file) as f:
        spec = json.load(f)

    name = spec["name"]
    task = spec["task"]
    profiles = spec["tool_profiles"]
    model = spec.get("model") or None
    if not model:
        from core.config import default_model
        model = default_model()
    max_rounds = spec.get("max_rounds", 20)
    max_runtime = spec.get("max_runtime_seconds", 1800)  # 30 min default
    notify = spec.get("notify_on_completion", True)

    subagent_dir = str(_SUBAGENTS_DIR / subagent_id)
    log_path = os.path.join(subagent_dir, "run.log")
    output_path = os.path.join(subagent_dir, "output.json")

    logger = SubagentLogger(log_path, subagent_id, name)
    logger.write(f"{'='*60}")
    logger.write(f"SUB-AGENT {subagent_id} STARTED")
    logger.write(f"Name: {name}")
    logger.write(f"Model: {model}")
    logger.write(f"Profiles: {', '.join(profiles)}")
    logger.write(f"Max rounds: {max_rounds} | Max runtime: {max_runtime}s")
    logger.write(f"Task: {task}")
    logger.write(f"{'='*60}")

    start_time = time.time()
    started_at = datetime.now().astimezone().isoformat()
    token_usage = {"input": 0, "output": 0, "total": 0}
    status = "crashed"
    error_msg = None
    final_output = None

    # Resolve tool subset
    tools_schema = resolve_tools(profiles)
    granted_names = [t.get("function", {}).get("name", t.get("name")) for t in tools_schema]
    logger.write(f"Granted tools: {', '.join(granted_names)}")
    logger.write("")

    # Build system prompt — always inject granted tool list so sub-agent knows exactly what it has
    now_local = datetime.now().astimezone()
    tool_list_lines = []
    for t in tools_schema:
        fn = t.get("function", {})
        tname = fn.get("name", "")
        tdesc = fn.get("description", "")
        tool_list_lines.append(f"  - {tname}: {tdesc[:120]}")
    tool_list_str = "\n".join(tool_list_lines)

    system_prompt = (
        f"You are a focused sub-agent. Current time: {now_local.isoformat()}\n\n"
        f"## Your granted tools ({len(tools_schema)} total)\n"
        f"{tool_list_str}\n\n"
        f"## Output instructions\n"
        f"When your analysis is complete, write your findings to "
        f"'subagents/{subagent_id}/output.json' using fs_write_json. "
        f"Include: findings, key_data, conclusions, unfinished_work (if any), "
        f"and recommended_next_steps. Then provide a brief summary as your final text response.\n\n"
        f"## Task instructions follow\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    llm = get_llm_client(model)
    wrap_up_triggered = False
    turn = 0

    try:
        while turn < max_rounds:
            elapsed = time.time() - start_time
            current_tools = tools_schema

            # Phase 1: T - 3min → strip tools, inject wrap-up nudge
            if elapsed > (max_runtime - 180) and not wrap_up_triggered:
                logger.log_wrap_up()
                current_tools = []  # No tools — forces text output
                messages.append({
                    "role": "user",
                    "content": (
                        "[System: You are approaching your time limit. "
                        "You no longer have access to tools. "
                        "Write your complete findings to "
                        f"subagents/{subagent_id}/output.json using fs_write_json — "
                        "but since you have no tools, instead provide your full report "
                        "as your text response right now. Include everything you found, "
                        "key data points, conclusions, and any unfinished work.]"
                    ),
                })
                wrap_up_triggered = True

            turn += 1
            response = llm.complete(messages, tools=current_tools or None)

            # Track tokens
            token_usage["input"] += response.usage.input_tokens
            token_usage["output"] += response.usage.output_tokens
            token_usage["total"] += response.usage.input_tokens + response.usage.output_tokens
            logger.log_turn(turn, response.usage.input_tokens, response.usage.output_tokens, token_usage)

            # Log agent text
            if response.text:
                logger.log_agent_text(response.text)

            # Build and store assistant message
            assistant_msg: dict = {"role": "assistant", "content": response.text}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

            # No tool calls → final response
            if not response.tool_calls:
                final_output = response.text
                status = "success"
                logger.write(f"Sub-agent finished cleanly after {turn} turns")
                break

            # Execute tool calls
            for tc in response.tool_calls:
                logger.log_tool_call(tc.name, tc.arguments)
                result = dispatch_tool_call(tc.name, tc.arguments)
                logger.log_tool_result(tc.name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        else:
            # Exhausted max_rounds — force final output with no tools
            logger.write(f"Hit max_rounds={max_rounds} — forcing final response")
            messages.append({
                "role": "user",
                "content": (
                    "[System: You have reached your maximum turn limit. "
                    "No more tool calls are allowed. "
                    "Provide your complete findings and conclusions as a text response now.]"
                ),
            })
            response = llm.complete(messages)
            if response.text:
                token_usage["input"] += response.usage.input_tokens
                token_usage["output"] += response.usage.output_tokens
                token_usage["total"] += response.usage.input_tokens + response.usage.output_tokens
                final_output = response.text
                status = "success"
                logger.log_agent_text(response.text)

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.write(f"EXCEPTION: {e}\n{error_msg}")
        status = "crashed"

    # Write output.json if not already written by agent via fs_write_json
    completed_at = datetime.now().astimezone().isoformat()
    if not os.path.exists(output_path):
        output_data = {
            "subagent_id": subagent_id,
            "name": name,
            "status": status,
            "completed_at": completed_at,
            "summary": final_output or "No output generated.",
            "error": error_msg,
            "token_usage": token_usage,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.write(f"Wrote output.json (fallback — agent did not write it directly)")
    else:
        # Agent wrote its own output — inject metadata
        try:
            with open(output_path) as f:
                existing = json.load(f)
            existing.setdefault("subagent_id", subagent_id)
            existing.setdefault("name", name)
            existing["status"] = status
            existing["completed_at"] = completed_at
            existing["token_usage"] = token_usage
            if error_msg:
                existing["error"] = error_msg
            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass

    elapsed_total = time.time() - start_time
    logger.write("")
    logger.write(f"{'='*60}")
    logger.write(f"SUB-AGENT {subagent_id} FINISHED — status={status}")
    logger.write(f"Total turns: {turn} | Runtime: {elapsed_total:.1f}s")
    logger.write(f"Token usage: in={token_usage['input']} out={token_usage['output']} total={token_usage['total']}")
    logger.write(f"{'='*60}")

    # Update registry entry to completed
    _update_registry_status(
        subagent_id,
        status="completed" if status == "success" else "crashed",
        completed_at=completed_at,
        output_file=f"subagents/{subagent_id}/output.json",
        token_usage=token_usage,
        error=error_msg,
    )

    # Append to history.jsonl
    _append_history({
        "id": subagent_id,
        "name": name,
        "status": "success" if status == "success" else "crashed",
        "task": task,
        "tool_profiles": profiles,
        "model": model,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": round(elapsed_total, 1),
        "token_usage": token_usage,
        "output_file": f"subagents/{subagent_id}/output.json",
        "error": error_msg,
    })

    # Notify main agent via queue
    _write_completion_to_queue(
        subagent_id=subagent_id,
        name=name,
        status="success" if status == "success" else "crashed",
        notify=notify,
        output_file=f"subagents/{subagent_id}/output.json",
    )

    logger.write(f"Completion notification written to queue.json (notify_on_completion={notify})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 subagent_runner.py <subagent_id>")
        sys.exit(1)
    run_subagent(sys.argv[1])
