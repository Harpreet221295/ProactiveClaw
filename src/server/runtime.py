"""The proactive runtime: one persistent conversation plus the background loops
that make the assistant reach out on its own.

  handle_message        user → agent → reply (wakes the agent if asleep)
  idle_monitor          silence for AGENT_TIMEOUT → pre-exit flow → sleep
  poll_notifications    delivers due reminders and nudges, feeds the bandit
  poll_cron_jobs        user cron jobs + the protected morning-review job
  run_morning_review    tend registry → gather new items → brief → nudges
  reengagement_monitor  after nudges are exhausted, reach out with backoff
  subagent_watchdog     detect crashed / timed-out sub-agents

Everything is channel-agnostic: output goes through server.channels.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core import config as _config
from core.paths import (
    QUEUE_FILE, REMINDERS_FILE, CRON_JOBS_FILE, REENGAGEMENT_FILE, AGENT_FS_DIR,
)
from personalized_bandits import bandit
from . import channels

SESSION_ID = "main"
POLL_INTERVAL = 30
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300") or 300)
STALE_THRESHOLD_SECONDS = 600
MORNING_REVIEW_JOB_ID = "cron_deep_think"   # historical id, kept so old data keeps working
MORNING_REVIEW_MAX_ROUNDS = 40

_agent_lock = asyncio.Lock()

# Session state
_last_message_time: float = 0.0
_is_sleeping: bool = True
_first_message: bool = True
_current_activity: str = "idle"

# Bandit reward tracking
_last_notification_time: float = 0.0
_last_notification_arm: tuple[int, int] | None = None

# Context stashed for the next pre-exit / wake
_previous_invocations: list[dict] = []
_last_delivered_notification: dict | None = None

_morning_review_running: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_list(path: os.PathLike | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return []


def _save_json(path: os.PathLike | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_FILE_OUTPUT_PATTERN = re.compile(r"(?:File written|JSON written|CSV written|Chart saved):\s*([\w\-./]+)")
_PATH_PATTERN = re.compile(r"(?:^|[\s(`])([\w\-./]+\.[A-Za-z0-9]{1,5})(?=[\s)`.,;:]|$)", re.MULTILINE)


def find_output_files(response_text: str) -> list[str]:
    """Relative agent_file_system paths mentioned in a response that actually exist."""
    matches = _FILE_OUTPUT_PATTERN.findall(response_text) + _PATH_PATTERN.findall(response_text)
    out, seen = [], set()
    for m in matches:
        rel = m.lstrip("/")
        if rel.startswith("agent_file_system/"):
            rel = rel[len("agent_file_system/"):]
        full = AGENT_FS_DIR / rel
        try:
            full.resolve().relative_to(AGENT_FS_DIR.resolve())
        except ValueError:
            continue
        if rel not in seen and full.is_file():
            out.append(rel)
            seen.add(rel)
    return out


def is_sleeping() -> bool:
    return _is_sleeping


def activity() -> str:
    return _current_activity


def last_message_time() -> float:
    return _last_message_time


_phase: str = "chat"   # chat | pre_exit | morning_review — which status to show between tool calls


def _agent_event_callback(event: str, payload: dict) -> None:
    """Forward tool activity from the agent thread to the UI."""
    global _current_activity
    if event == "tool_start":
        _current_activity = f"tool:{payload.get('name')}"
        channels.set_status_threadsafe("tool", payload.get("name", ""))
    elif event == "tool_end":
        channels.set_status_threadsafe({"chat": "thinking", "pre_exit": "pre_exit", "morning_review": "morning_review"}.get(_phase, "thinking"), "")


def _set_phase(phase: str) -> None:
    global _phase
    _phase = phase


def _bandit_text() -> str:
    return bandit.format_recommendations(bandit.get_recommendations())


# ─────────────────────────────────────────────────────────────────────────────
# Re-engagement state
# ─────────────────────────────────────────────────────────────────────────────

def _load_reengagement_state() -> dict:
    p = Path(REENGAGEMENT_FILE)
    default = {"invocations_exhausted_at": None, "attempt_number": 0, "messages": []}
    if not p.exists():
        return default
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data else default
    except json.JSONDecodeError:
        return default


def _save_reengagement_state(state: dict) -> None:
    _save_json(REENGAGEMENT_FILE, state)


def _mark_invocations_exhausted(now: datetime) -> None:
    state = _load_reengagement_state()
    if not state.get("invocations_exhausted_at"):
        state["invocations_exhausted_at"] = now.isoformat()
        _save_reengagement_state(state)
        print(f"[reengagement] Invocations exhausted at {now.isoformat()}")


def _reset_reengagement() -> None:
    state = _load_reengagement_state()
    state["invocations_exhausted_at"] = None
    state["attempt_number"] = 0
    _save_reengagement_state(state)


def _clear_invocation_queue() -> None:
    """On wake: drop pending nudges (they're about a conversation that just resumed)."""
    global _previous_invocations
    queue = _load_json_list(QUEUE_FILE)
    keep = [e for e in queue if e.get("source") == "subagent_complete"]
    _previous_invocations = [e for e in queue if e.get("source") != "subagent_complete"]
    _save_json(QUEUE_FILE, keep)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────────────────────────────────────

async def handle_message(text: str, images: list[bytes] | None = None, *, origin: str = "web") -> str:
    """Process one user message. Returns the assistant reply (already broadcast)."""
    global _last_message_time, _is_sleeping, _last_notification_time, _last_notification_arm
    global _first_message, _last_delivered_notification, _current_activity
    from agents.agent import Agent
    from agents.tools import set_current_session_id

    await channels.post_message("user", text, source=origin, slack=False,
                                meta={"images": len(images or [])})

    async with _agent_lock:
        try:
            _last_message_time = time.time()
            now_dt = datetime.now().astimezone()
            cfg = _config.load_config()
            eff = _config.effective_settings(cfg)
            _current_activity = "thinking"
            _set_phase("chat")
            await channels.set_status("thinking")

            woke = False
            notification_context = None
            if _is_sleeping:
                woke = True
                bandit.update_arm(now_dt.weekday(), now_dt.hour, reward=1.0)
                _last_notification_time = 0.0
                _last_notification_arm = None
                _is_sleeping = False
                _clear_invocation_queue()
                _reset_reengagement()
                try:
                    from memory_tier1 import clear_session_cache
                    clear_session_cache(SESSION_ID)
                except Exception:
                    pass
                print("[wake] Agent woke up — cleared pending nudges (reminders preserved)")
                if _last_delivered_notification:
                    notification_context = {**_last_delivered_notification, "_was_sleeping": True}
                    _last_delivered_notification = None

            if not woke and _last_delivered_notification:
                notification_context = _last_delivered_notification
                _last_delivered_notification = None

            need_review = False
            if _first_message:
                _first_message = False
                need_review = True
            if woke:
                need_review = True
            if need_review and eff["morning_review_enabled"]:
                from care.brief import has_todays_brief
                if not has_todays_brief() and _morning_review_due_today(cfg):
                    print("[morning-review] No brief for today — scheduling catch-up review")
                    asyncio.create_task(run_morning_review(reason="catch-up"))

            loop = asyncio.get_running_loop()
            set_current_session_id(SESSION_ID)

            def _run_tier1():
                try:
                    from memory_tier1 import build_tier1_context
                    return build_tier1_context(text, SESSION_ID)
                except Exception as _e:
                    print(f"[tier1 memory] skipped: {_e}")
                    return ""
            tier1_future = loop.run_in_executor(None, _run_tier1)

            effective_text = text
            if notification_context:
                ntype = notification_context.get("type", "notification")
                nmsg = notification_context.get("message", "")
                ntime = notification_context.get("delivered_at", "")
                if notification_context.get("_was_sleeping"):
                    effective_text = (f"[System context: You sent the user a {ntype} at {ntime}: \"{nmsg}\". "
                                      f"The user is now responding at {now_dt.isoformat()}. They may be replying to it or messaging organically — use your judgment.]\n\n{text}")
                else:
                    effective_text = (f"[System context: You just sent the user a {ntype}: \"{nmsg}\". "
                                      f"They may be replying to it or this may be unrelated — use your judgment.]\n\n{text}")

            new_session = not Agent.session_exists(SESSION_ID)
            if new_session:
                summary = Agent.load_summary(SESSION_ID)
                agent = Agent(session_id=SESSION_ID, chat_summary=summary or None, event_callback=_agent_event_callback)
                try:
                    from care.brief import session_context
                    care_ctx = session_context(cfg)
                except Exception as e:
                    print(f"[care] context skipped: {e}")
                    care_ctx = ""
                if care_ctx:
                    effective_text = f"{care_ctx}\n\n{effective_text}"
            else:
                agent = Agent(session_id=SESSION_ID, event_callback=_agent_event_callback)

            tier1_ctx = await tier1_future
            if tier1_ctx:
                effective_text = f"{tier1_ctx}\n\n{effective_text}"

            response = await loop.run_in_executor(None, lambda: agent.run(effective_text, images=images or None))
            files = find_output_files(response)
            await channels.post_message("assistant", response, files=files)
            await channels.broadcast({"type": "care_updated"})
            await channels.broadcast({"type": "queue_updated"})
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = f"Something went wrong: {type(e).__name__}: {e}"
            await channels.post_message("system", msg)
            return msg
        finally:
            _current_activity = "idle"
            await channels.set_status("awake" if not _is_sleeping else "sleeping")


async def handle_system_event(text: str, *, deliver: bool) -> None:
    """Feed a system event (e.g. sub-agent completion) to the agent. If `deliver`,
    its response is shown to the user; otherwise it's processed silently."""
    from agents.agent import Agent
    from agents.tools import set_current_session_id
    async with _agent_lock:
        try:
            loop = asyncio.get_running_loop()
            set_current_session_id(SESSION_ID)
            agent = Agent(session_id=SESSION_ID, event_callback=_agent_event_callback)
            prompt = (f"[System notification: {text}]\n"
                      + ("Handle it, then write a short message for the user summarising what came of it."
                         if deliver else
                         "Handle it silently: read any output, update the care registry if relevant, and reply with just `OK`."))
            response = await loop.run_in_executor(None, lambda: agent.run(prompt))
            if deliver and response and response.strip() != "OK":
                await channels.post_message("assistant", response, files=find_output_files(response), source="subagent")
                await channels.broadcast({"type": "care_updated"})
        except Exception as e:
            print(f"[system-event] failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Idle → pre-exit → sleep
# ─────────────────────────────────────────────────────────────────────────────

async def run_pre_exit(reason: str = "idle") -> str | None:
    """Run the pre-exit flow (registry update, nudges, summary, memory) and sleep."""
    global _is_sleeping, _current_activity
    from agents.agent import Agent
    from agents.tools import set_current_session_id
    from agents.tools.scheduling import load_reminders, list_scheduled_nudges

    async with _agent_lock:
        if _is_sleeping:
            return None
        try:
            _current_activity = "pre-exit"
            _set_phase("pre_exit")
            await channels.set_status("pre_exit", reason)
            print(f"[timeout] Session idle — running pre-exit flow ({reason})...")
            loop = asyncio.get_running_loop()
            set_current_session_id(SESSION_ID)
            agent = Agent(session_id=SESSION_ID, event_callback=_agent_event_callback)

            existing = load_reminders()
            reminders_text = ("Reminders already scheduled (user-requested, don't duplicate):\n" +
                              "\n".join(f"• {r['timestamp']} — {r['message']}" for r in existing)) if existing else "No reminders are currently scheduled."
            if _previous_invocations:
                prev = "Nudges scheduled at the end of the previous session (already cleared; re-schedule only those still relevant):\n" + \
                       "\n".join(f"• {i.get('timestamp')} — {i.get('message')}" for i in _previous_invocations)
            else:
                prev = "No nudges were scheduled in the previous session."

            response = await loop.run_in_executor(None, lambda: agent.run_pre_exit(
                bandit_recommendations=_bandit_text(),
                existing_reminders=reminders_text,
                previous_invocations=prev,
                scheduled_nudges="Nudges already queued:\n" + list_scheduled_nudges(),
            ))

            try:
                new_summary = await loop.run_in_executor(None, agent.generate_summary)
                Agent.save_summary(SESSION_ID, new_summary)
                try:
                    (AGENT_FS_DIR / "last_session_summary.txt").write_text(new_summary, encoding="utf-8")
                except Exception as sf_err:
                    print(f"[warning] Failed to write last_session_summary.txt: {sf_err}")
                try:
                    from memory import store_dialogues
                    dialogues = agent.format_conversation_for_memory()
                    if dialogues:
                        store_dialogues(dialogues)
                        print(f"[memory] Stored {len(dialogues)} dialogue turns")
                except Exception as mem_err:
                    print(f"[warning] Long-term memory storage failed: {mem_err}")
                agent.archive_session()
            except Exception as e:
                print(f"[warning] Summary generation failed: {e}")

            _is_sleeping = True
            if response:
                await channels.post_message("assistant", response, files=find_output_files(response), source="pre_exit")
            await channels.broadcast({"type": "queue_updated"})
            await channels.broadcast({"type": "care_updated"})
            print("[sleep] Agent is now sleeping")
            return response
        except Exception as e:
            print(f"[error] Pre-exit flow failed: {e}")
            _is_sleeping = True
            return None
        finally:
            _current_activity = "idle"
            await channels.set_status("sleeping")


async def idle_monitor() -> None:
    while True:
        await asyncio.sleep(10)
        if _last_message_time == 0.0 or _is_sleeping:
            continue
        if time.time() - _last_message_time <= AGENT_TIMEOUT:
            continue
        if _agent_lock.locked():
            continue
        await run_pre_exit("idle")


# ─────────────────────────────────────────────────────────────────────────────
# Notification delivery
# ─────────────────────────────────────────────────────────────────────────────

async def poll_notifications() -> None:
    global _last_notification_time, _last_notification_arm, _last_delivered_notification
    from care.nudges import record_delivery
    PENALTY_WINDOW = 1800
    ACTIVE_WINDOW = 1800

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            now = datetime.now(timezone.utc)
            now_local = now.astimezone()

            if (_last_notification_time > 0 and _last_notification_arm is not None and not _is_sleeping
                    and time.time() - _last_notification_time > PENALTY_WINDOW
                    and (_last_message_time == 0.0 or time.time() - _last_message_time > ACTIVE_WINDOW)):
                bandit.update_arm(*_last_notification_arm, reward=-0.5)
                _last_notification_time = 0.0
                _last_notification_arm = None

            # Reminders — always deliver, even if late
            reminders = _load_json_list(REMINDERS_FILE)
            if reminders:
                remaining = []
                for entry in reminders:
                    try:
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.astimezone()
                    except (ValueError, KeyError):
                        print(f"[warning] Dropping malformed reminder: {entry}")
                        continue
                    if ts <= now:
                        await channels.post_message("reminder", entry["message"], meta={"id": entry.get("id")})
                        _last_notification_time = time.time()
                        _last_notification_arm = (now_local.weekday(), now_local.hour)
                        _last_delivered_notification = {"type": "reminder", "message": entry["message"], "delivered_at": now_local.isoformat()}
                        if not _is_sleeping:
                            bandit.update_arm(now_local.weekday(), now_local.hour, reward=0.5)
                    else:
                        remaining.append(entry)
                if len(remaining) != len(reminders):
                    _save_json(REMINDERS_FILE, remaining)

            # Nudges / sub-agent events
            queue = _load_json_list(QUEUE_FILE)
            if queue:
                remaining_queue = []
                delivered_any = False
                for entry in queue:
                    try:
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.astimezone()
                    except (ValueError, KeyError):
                        print(f"[warning] Dropping malformed queue entry: {entry}")
                        continue
                    if ts > now:
                        remaining_queue.append(entry)
                        continue
                    source = entry.get("source", "")
                    age = (now - ts).total_seconds()
                    if source == "subagent_complete":
                        notify = bool(entry.get("notify_on_completion", True))
                        asyncio.create_task(handle_system_event(entry["message"], deliver=notify))
                        delivered_any = True
                        continue
                    stale_limit = 21600 if source == "morning_review" else STALE_THRESHOLD_SECONDS
                    if age > stale_limit:
                        print(f"[nudge] dropped stale ({age:.0f}s): {entry['message'][:60]}")
                        continue
                    await channels.post_message("nudge", entry["message"], source=source or "pre_exit",
                                                meta={"id": entry.get("id"), "item_id": entry.get("item_id")})
                    record_delivery(entry["message"], source=source, item_id=entry.get("item_id", ""), when=now_local)
                    delivered_any = True
                    _last_notification_time = time.time()
                    _last_notification_arm = (now_local.weekday(), now_local.hour)
                    _last_delivered_notification = {"type": "nudge", "message": entry["message"], "source": source, "delivered_at": now_local.isoformat()}
                    if not _is_sleeping:
                        bandit.update_arm(now_local.weekday(), now_local.hour, reward=0.5)

                if len(remaining_queue) != len(queue):
                    _save_json(QUEUE_FILE, remaining_queue)
                if delivered_any:
                    await channels.broadcast({"type": "queue_updated"})
                if not [e for e in remaining_queue if e.get("source") != "subagent_complete"] and _is_sleeping:
                    _mark_invocations_exhausted(now_local)
        except Exception as e:
            print(f"[poll] error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Morning review (protected cron job)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_morning_review_job(cfg: dict | None = None) -> None:
    """Create/update/remove the protected morning-review cron job to match config."""
    cfg = cfg or _config.load_config()
    eff = _config.effective_settings(cfg)
    jobs = _load_json_list(CRON_JOBS_FILE)
    others = [j for j in jobs if j.get("id") != MORNING_REVIEW_JOB_ID]
    existing = next((j for j in jobs if j.get("id") == MORNING_REVIEW_JOB_ID), None)
    if not eff["morning_review_enabled"]:
        if existing:
            _save_json(CRON_JOBS_FILE, others)
            print("[morning-review] Disabled at this level — removed cron job")
        return
    mr = cfg.get("morning_review", {})
    job = {
        "id": MORNING_REVIEW_JOB_ID,
        "kind": "morning_review",
        "protected": True,
        "time": mr.get("time", "07:30"),
        "days": mr.get("days") or ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "task": "Morning review (system) — tends the care registry, pulls new items, writes the daily brief and schedules nudges.",
        "created_at": (existing or {}).get("created_at") or datetime.now().astimezone().isoformat(),
        "last_run": (existing or {}).get("last_run"),
    }
    _save_json(CRON_JOBS_FILE, others + [job])


def _morning_review_due_today(cfg: dict) -> bool:
    """True if the scheduled review time has already passed today (so a catch-up makes sense)."""
    now = datetime.now().astimezone()
    mr = cfg.get("morning_review", {})
    days = mr.get("days") or []
    if days and now.strftime("%A").lower() not in days:
        return False
    return now.strftime("%H:%M") >= mr.get("time", "07:30")


async def run_morning_review(reason: str = "scheduled") -> str | None:
    """Tend the registry, gather what's new, generate the brief, schedule nudges."""
    global _morning_review_running, _current_activity
    if _morning_review_running:
        return None
    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    if not eff["morning_review_enabled"] and reason != "manual":
        return None
    from agents.agent import Agent
    from agents.tools import set_current_session_id
    from agents.tools.scheduling import list_scheduled_nudges
    from care.registry import CareRegistry
    from care.patterns import CarePatterns
    from care.brief import registry_digest, tend_report_text, generate_brief, has_todays_brief

    _morning_review_running = True
    try:
        now = datetime.now().astimezone()
        registry = CareRegistry()
        report = registry.tend(cfg, now)
        summary_path = AGENT_FS_DIR / "last_session_summary.txt"
        last_summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        prompt = _build_review_prompt(cfg, now, registry, report, last_summary, list_scheduled_nudges())

        log_path = AGENT_FS_DIR / "morning_review_log.txt"
        AGENT_FS_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"=== Morning review ({reason}) — {now.isoformat()} ===\n\n", encoding="utf-8")

        def _logger(name: str, args: str, result: str) -> None:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {name}({args[:200]})\n  → {(result or '')[:300]}\n")
            except Exception:
                pass

        await channels.broadcast({"type": "morning_review", "state": "running", "reason": reason})
        async with _agent_lock:
            _current_activity = "morning-review"
            _set_phase("morning_review")
            await channels.set_status("morning_review", reason)
            loop = asyncio.get_running_loop()
            set_current_session_id(SESSION_ID)
            agent = Agent(max_rounds=MORNING_REVIEW_MAX_ROUNDS, tool_logger=_logger, event_callback=_agent_event_callback)
            response = await loop.run_in_executor(None, lambda: agent.run(prompt))
            _current_activity = "idle"
            await channels.set_status("sleeping" if _is_sleeping else "awake")

        # Safety nets: brief + source timestamps even if the model skipped them
        registry.reload()
        if not has_todays_brief(now):
            generate_brief(registry, cfg, now)
        for src in ("gmail", "notion", "calendar"):
            if src in _config.active_connectors(cfg):
                registry.mark_source_checked(src, now)

        _mark_job_ran(MORNING_REVIEW_JOB_ID, now)
        text = (response or "").strip()
        deliver = text and text != "NO_BRIEF" and eff["morning_review_dm"] != "never"
        if deliver:
            await channels.post_message("morning_review", text, source=reason)
        await channels.broadcast({"type": "morning_review", "state": "done", "reason": reason, "delivered": bool(deliver)})
        await channels.broadcast({"type": "care_updated"})
        await channels.broadcast({"type": "queue_updated"})
        print(f"[morning-review] complete ({reason})")
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        await channels.broadcast({"type": "morning_review", "state": "error", "error": str(e)})
        return None
    finally:
        _morning_review_running = False


def _build_review_prompt(cfg, now, registry, report, last_summary, scheduled) -> str:
    from agents.prompts.prompts import build_morning_review_prompt
    from care.brief import registry_digest, tend_report_text
    from care.patterns import CarePatterns
    return build_morning_review_prompt(
        current_time=now.isoformat(),
        care_digest=registry_digest(registry, cfg, now, max_items=25),
        tend_report=tend_report_text(report),
        patterns=CarePatterns().digest(),
        last_session_summary=last_summary[:4000],
        bandit_recommendations=_bandit_text(),
        scheduled_nudges=scheduled,
        cfg=cfg,
    )


def _mark_job_ran(job_id: str, when: datetime) -> None:
    jobs = _load_json_list(CRON_JOBS_FILE)
    for j in jobs:
        if j.get("id") == job_id:
            j["last_run"] = when.isoformat()
    _save_json(CRON_JOBS_FILE, jobs)


# ─────────────────────────────────────────────────────────────────────────────
# Cron jobs
# ─────────────────────────────────────────────────────────────────────────────

async def poll_cron_jobs() -> None:
    from agents.agent import Agent
    from agents.tools import set_current_session_id
    from agents.prompts.prompts import build_cron_job_prompt

    while True:
        await asyncio.sleep(60)
        try:
            now = datetime.now().astimezone()
            current_day = now.strftime("%A").lower()
            current_time = now.strftime("%H:%M")
            jobs = _load_json_list(CRON_JOBS_FILE)
            for job in jobs:
                if job.get("days") and current_day not in job["days"]:
                    continue
                if current_time < job.get("time", "99:99"):
                    continue
                try:
                    h, m = map(int, job["time"].split(":"))
                except (ValueError, KeyError):
                    continue
                scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if (now - scheduled_dt).total_seconds() > 21600:
                    continue
                if job.get("last_run"):
                    try:
                        if datetime.fromisoformat(job["last_run"]).date() == now.date():
                            continue
                    except ValueError:
                        pass

                _mark_job_ran(job["id"], now)
                if job.get("kind") == "morning_review" or job["id"] == MORNING_REVIEW_JOB_ID:
                    await run_morning_review(reason="scheduled")
                    continue

                print(f"[cron] Executing job {job['id']}: {job.get('task', '')[:80]}")
                async with _agent_lock:
                    loop = asyncio.get_running_loop()
                    set_current_session_id(SESSION_ID)
                    agent = Agent(event_callback=_agent_event_callback)
                    prompt = build_cron_job_prompt(current_time=now.isoformat(), task=job.get("task", ""))
                    response = await loop.run_in_executor(None, lambda: agent.run(prompt))
                header = f"**Scheduled task:** {job.get('task', '')[:80]}\n\n"
                await channels.post_message("assistant", header + (response or ""), files=find_output_files(response or ""), source="cron")
        except Exception as e:
            print(f"[cron] error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Re-engagement
# ─────────────────────────────────────────────────────────────────────────────

async def reengagement_monitor() -> None:
    from agents.agent import Agent
    from agents.prompts.prompts import build_reengagement_prompt
    from care.brief import registry_digest

    while True:
        await asyncio.sleep(300)
        if not _is_sleeping:
            continue
        try:
            eff = _config.effective_settings()
            intervals = eff.get("reengagement_intervals_hours") or []
            if not eff["reengagement_enabled"] or not intervals:
                continue
            state = _load_reengagement_state()
            if not state.get("invocations_exhausted_at"):
                continue
            exhausted_at = datetime.fromisoformat(state["invocations_exhausted_at"])
            attempt = state.get("attempt_number", 0)
            interval = intervals[min(attempt, len(intervals) - 1)] * 3600
            now = datetime.now().astimezone()
            if (now - exhausted_at).total_seconds() < interval:
                continue
            if _config.in_quiet_hours(now, eff["quiet_hours"]):
                continue
            recommendations = bandit.get_recommendations(top_k=5)
            if recommendations:
                top = {(r["day"], r["hour"]) for r in recommendations}
                if (now.weekday(), now.hour) not in top:
                    continue
            prev = state.get("messages", [])
            prev_text = ("You have already sent these re-engagement messages (do NOT repeat topics):\n" +
                         "\n".join(f"• {m['sent_at']}: {m['message']}" for m in prev)) if prev else "This is your first re-engagement message."
            async with _agent_lock:
                loop = asyncio.get_running_loop()
                agent = Agent(event_callback=_agent_event_callback)
                prompt = build_reengagement_prompt(current_time=now.isoformat(), bandit_recommendations=_bandit_text(),
                                                   previous_reengagement_dms=prev_text, care_digest=registry_digest())
                response = await loop.run_in_executor(None, lambda: agent.run(prompt))
            await channels.post_message("nudge", response[:3900], source="reengagement")
            state["attempt_number"] = attempt + 1
            state["invocations_exhausted_at"] = now.isoformat()
            state.setdefault("messages", []).append({"sent_at": now.isoformat(), "message": response[:500]})
            _save_reengagement_state(state)
            print(f"[reengagement] message sent (attempt {attempt + 1})")
        except Exception as e:
            print(f"[reengagement] error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agent watchdog
# ─────────────────────────────────────────────────────────────────────────────

async def subagent_watchdog() -> None:
    import signal as _signal
    registry_path = AGENT_FS_DIR / "currently_running_subagents.json"
    while True:
        await asyncio.sleep(60)
        try:
            if not registry_path.exists():
                continue
            registry = _load_json_list(registry_path)
            changed = False
            now_local = datetime.now().astimezone()
            for entry in registry:
                if entry.get("status") != "running":
                    continue
                pid = entry.get("pid", 0)
                alive = False
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except (ProcessLookupError, PermissionError):
                        pass
                if not alive:
                    entry.update(status="crashed", updated_at=now_local.isoformat(), completed_at=now_local.isoformat(), error="Process died unexpectedly")
                    changed = True
                    _watchdog_notify(entry["id"], entry["name"], "crashed", entry.get("notify_on_completion", True))
                    continue
                started = entry.get("started_at")
                max_runtime = entry.get("max_runtime_seconds", 1800)
                if started:
                    try:
                        started_dt = datetime.fromisoformat(started)
                        if started_dt.tzinfo is None:
                            started_dt = started_dt.astimezone()
                        if (now_local - started_dt).total_seconds() > max_runtime:
                            try:
                                os.kill(pid, _signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            entry.update(status="crashed", updated_at=now_local.isoformat(), completed_at=now_local.isoformat(), error=f"Hard timeout exceeded ({max_runtime}s)")
                            changed = True
                            _watchdog_notify(entry["id"], entry["name"], "timeout", entry.get("notify_on_completion", True))
                    except Exception as e:
                        print(f"[watchdog] {e}")
            if changed:
                _save_json(registry_path, registry)
        except Exception as e:
            print(f"[watchdog] error: {e}")


def _watchdog_notify(subagent_id: str, name: str, reason: str, notify: bool) -> None:
    msg = (f"Sub-agent '{name}' was hard-killed (timeout). Call read_subagent_output('{subagent_id}') for partial results."
           if reason == "timeout" else
           f"Sub-agent '{name}' crashed unexpectedly. Call read_subagent_output('{subagent_id}') for details.")
    queue = _load_json_list(QUEUE_FILE)
    queue.append({"timestamp": datetime.now(timezone.utc).isoformat(), "message": msg, "source": "subagent_complete",
                  "subagent_id": subagent_id, "subagent_name": name, "notify_on_completion": notify, "output_file": None})
    _save_json(QUEUE_FILE, queue)


# ─────────────────────────────────────────────────────────────────────────────
# State snapshot for the UI
# ─────────────────────────────────────────────────────────────────────────────

def snapshot() -> dict[str, Any]:
    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    now = datetime.now().astimezone()
    pending = []
    for e in _load_json_list(QUEUE_FILE):
        if e.get("source") == "subagent_complete":
            continue
        pending.append({"id": e.get("id"), "timestamp": e.get("timestamp"), "message": e.get("message"),
                        "source": e.get("source", "pre_exit"), "item_id": e.get("item_id")})
    reminders = [{"id": r.get("id"), "timestamp": r.get("timestamp"), "message": r.get("message")} for r in _load_json_list(REMINDERS_FILE)]
    jobs = _load_json_list(CRON_JOBS_FILE)
    mr = next((j for j in jobs if j.get("id") == MORNING_REVIEW_JOB_ID), None)
    subagents = [e for e in _load_json_list(AGENT_FS_DIR / "currently_running_subagents.json") if e.get("status") == "running"]
    from care.nudges import delivered_today
    from care.registry import CareRegistry
    return {
        "now": now.isoformat(),
        "sleeping": _is_sleeping,
        "activity": _current_activity,
        "morning_review_running": _morning_review_running,
        "last_message_time": _last_message_time,
        "idle_timeout": AGENT_TIMEOUT,
        "level": eff["level"],
        "care_mode": eff["care_mode"],
        "effective": eff,
        "connectors": _config.connector_status(cfg),
        "pending_nudges": sorted(pending, key=lambda x: x.get("timestamp") or ""),
        "reminders": sorted(reminders, key=lambda x: x.get("timestamp") or ""),
        "nudges_delivered_today": delivered_today(now),
        "morning_review": {"enabled": eff["morning_review_enabled"], "time": cfg.get("morning_review", {}).get("time"),
                            "last_run": (mr or {}).get("last_run")},
        "cron_jobs": [j for j in jobs if j.get("id") != MORNING_REVIEW_JOB_ID],
        "subagents_running": len(subagents),
        "care_counts": CareRegistry().stats(),
        "slack": channels.slack_enabled(),
        "model": _config.default_model(cfg),
        "user": cfg.get("user", {}),
    }
