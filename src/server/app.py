"""FastAPI application: web UI, JSON API, WebSockets, optional Slack inbound."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
load_dotenv(_SRC.parent / ".env")

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import config as _config
from core.paths import AGENT_FS_DIR, ensure_runtime_dirs
from . import channels, runtime, browser_bridge

WEB_DIR = _SRC / "web"
BROWSER_TOKEN = os.getenv("BROWSER_TOKEN", "")
_bg_tasks: list[asyncio.Task] = []


def _check_llm_keys() -> None:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n[error] No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env "
              "(run ./setup.sh or python src/setup_wizard.py).\n")
        sys.exit(1)
    if not _config.config_exists():
        print("[config] config/config.json not found — using defaults (level=balanced, no connectors). "
              "Run python src/setup_wizard.py to configure.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()
    _check_llm_keys()
    import agents.tools._state as _state
    loop = asyncio.get_running_loop()
    _state._event_loop = loop
    channels.set_loop(loop)
    channels.init_slack()
    runtime.ensure_morning_review_job()

    _bg_tasks[:] = [
        asyncio.create_task(runtime.poll_notifications()),
        asyncio.create_task(runtime.idle_monitor()),
        asyncio.create_task(runtime.poll_cron_jobs()),
        asyncio.create_task(runtime.reengagement_monitor()),
        asyncio.create_task(runtime.subagent_watchdog()),
    ]
    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    print(f"ProactiveClaw ready — level={eff['level']} mode={eff['care_mode']} model={_config.default_model(cfg)}")
    print(f"  connectors: {', '.join(_config.active_connectors(cfg)) or 'none'}")
    print(f"  idle timeout {runtime.AGENT_TIMEOUT}s · morning review {'on at ' + cfg['morning_review']['time'] if eff['morning_review_enabled'] else 'off'}")
    yield
    for t in _bg_tasks:
        t.cancel()
    for t in _bg_tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(title="ProactiveClaw", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/files/{path:path}")
async def serve_agent_file(path: str):
    full = (AGENT_FS_DIR / path).resolve()
    try:
        full.relative_to(AGENT_FS_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "forbidden")
    if not full.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(full))


# ─────────────────────────────────────────────────────────────────────────────
# Chat WebSocket
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    channels.register(ws)
    try:
        await ws.send_json({"type": "hello", "state": runtime.snapshot(), "history": channels.load_transcript(200)})
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "message":
                text = (msg.get("text") or "").strip()
                images = []
                for b64 in msg.get("images") or []:
                    try:
                        if "," in b64:
                            b64 = b64.split(",", 1)[1]
                        images.append(base64.b64decode(b64))
                    except Exception:
                        continue
                if not text and not images:
                    continue
                asyncio.create_task(runtime.handle_message(text or "What's in this image?", images or None))
            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] {type(e).__name__}: {e}")
    finally:
        channels.unregister(ws)


# ─────────────────────────────────────────────────────────────────────────────
# JSON API
# ─────────────────────────────────────────────────────────────────────────────

class ChatIn(BaseModel):
    text: str


@app.post("/api/chat")
async def api_chat(body: ChatIn):
    reply = await runtime.handle_message(body.text)
    return {"reply": reply}


@app.get("/api/state")
async def api_state():
    return runtime.snapshot()


@app.get("/api/history")
async def api_history(limit: int = 200):
    return channels.load_transcript(limit)


@app.delete("/api/history")
async def api_clear_history():
    channels.clear_transcript()
    return {"ok": True}


@app.post("/api/sleep")
async def api_sleep():
    if runtime.is_sleeping():
        return {"ok": True, "already_sleeping": True}
    asyncio.create_task(runtime.run_pre_exit("manual"))
    return {"ok": True}


@app.post("/api/morning-review/run")
async def api_run_review():
    asyncio.create_task(runtime.run_morning_review(reason="manual"))
    return {"ok": True}


# ── Config ──────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def api_get_config():
    cfg = _config.load_config()
    return {
        "config": cfg,
        "effective": _config.effective_settings(cfg),
        "connectors": _config.connector_status(cfg),
        "levels": {k: v["description"] for k, v in _config.PROACTIVENESS_LEVELS.items()},
        "level_order": _config.LEVEL_ORDER,
        "modes": {k: v["description"] for k, v in _config.CARE_MODES.items()},
        "overridable": sorted(_config.OVERRIDABLE_KEYS),
        "summary": _config.summary(cfg),
        "env": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "slack": channels.slack_enabled(),
        },
    }


@app.put("/api/config")
async def api_put_config(request: Request):
    body = await request.json()
    cfg = _config.load_config()
    try:
        if "level" in body:
            cfg["proactiveness"]["level"] = _config.normalize_level(body["level"])
        if "care_mode" in body:
            if body["care_mode"] not in _config.CARE_MODES:
                raise ValueError(f"unknown care mode {body['care_mode']}")
            cfg["care_mode"] = body["care_mode"]
        if "connectors" in body:
            for k, v in (body["connectors"] or {}).items():
                if k in cfg["connectors"]:
                    cfg["connectors"][k] = bool(v)
        if "overrides" in body:
            new_over = {}
            for k, v in (body["overrides"] or {}).items():
                if v in (None, "", []):
                    continue
                if k not in _config.OVERRIDABLE_KEYS:
                    raise ValueError(f"'{k}' is not overridable")
                new_over[k] = v
            cfg["overrides"] = new_over
        for section in ("user", "llm", "notion", "morning_review"):
            if section in body and isinstance(body[section], dict):
                cfg[section].update({k: v for k, v in body[section].items() if k in cfg[section]})
        _config.save_config(cfg)
        # validate overrides through the setter path (raises on bad values)
        for k, v in (cfg.get("overrides") or {}).items():
            _config.set_override(k, v)
    except ValueError as e:
        raise HTTPException(400, str(e))
    runtime.ensure_morning_review_job()
    await channels.broadcast({"type": "config_updated"})
    return await api_get_config()


# ── Care ────────────────────────────────────────────────────────────────────

@app.get("/api/care")
async def api_care(status: str = "open"):
    from care.registry import CareRegistry
    reg = CareRegistry()
    items = reg.list(status=status)
    return {"items": items, "counts": reg.stats(), "meta": reg.load()["meta"]}


class CareAction(BaseModel):
    action: str
    until: str | None = None
    note: str | None = None
    urgency: str | None = None
    intent: str | None = None


@app.post("/api/care/{item_id}")
async def api_care_action(item_id: str, body: CareAction):
    from care.registry import CareRegistry
    from care.patterns import CarePatterns
    reg = CareRegistry()
    if reg.get(item_id) is None:
        raise HTTPException(404, "no such item")
    try:
        a = body.action
        if a == "done":
            item = reg.update(item_id, status="done", notes=body.note, user_intent=body.intent)
            CarePatterns().observe_item(item, "acted")
        elif a == "dismiss":
            item = reg.update(item_id, status="dismissed", notes=body.note, user_intent=body.intent)
            CarePatterns().observe_item(item, "dismissed")
        elif a == "defer":
            item = reg.update(item_id, status="deferred", user_intent=body.intent or "later")
            CarePatterns().observe_item(item, "deferred")
        elif a == "snooze":
            until = body.until
            if not until:
                until = (datetime.now().astimezone() + timedelta(hours=3)).isoformat()
            item = reg.snooze(item_id, until, intent=body.intent)
            CarePatterns().observe_item(item, "deferred")
        elif a == "acknowledge":
            item = reg.update(item_id, status="acknowledged", user_intent=body.intent)
        elif a == "in_progress":
            item = reg.update(item_id, status="in_progress", user_intent=body.intent)
        elif a == "reopen":
            item = reg.update(item_id, status="acknowledged")
        elif a == "urgency":
            item = reg.update(item_id, urgency=body.urgency)
        elif a == "delete":
            reg.remove(item_id)
            item = None
        else:
            raise ValueError(f"unknown action {a}")
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    await channels.broadcast({"type": "care_updated"})
    return {"ok": True, "item": item}


class CareNew(BaseModel):
    title: str
    type: str = "commitment"
    urgency: str = "medium"
    deadline: str | None = None
    details: str = ""


@app.post("/api/care")
async def api_care_add(body: CareNew):
    from care.registry import CareRegistry
    try:
        item, created = CareRegistry().add(type=body.type, title=body.title, urgency=body.urgency,
                                           source="manual", deadline=body.deadline, details=body.details)
    except ValueError as e:
        raise HTTPException(400, str(e))
    await channels.broadcast({"type": "care_updated"})
    return {"ok": True, "created": created, "item": item}


@app.get("/api/care/patterns")
async def api_patterns():
    from care.patterns import CarePatterns
    return CarePatterns().load()


@app.delete("/api/care/patterns/rules/{rule_id}")
async def api_delete_rule(rule_id: str):
    from care.patterns import CarePatterns
    kind, _, key = rule_id.partition(":")
    return {"ok": CarePatterns().remove_rule(kind, key)}


class RuleIn(BaseModel):
    kind: str
    key: str
    action: str
    note: str = ""


@app.post("/api/care/patterns/rules")
async def api_add_rule(body: RuleIn):
    from care.patterns import CarePatterns
    try:
        return {"ok": True, "rule": CarePatterns().add_rule(body.kind, body.key, body.action, body.note)}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Nudges & reminders ──────────────────────────────────────────────────────

@app.delete("/api/nudges/{nudge_id}")
async def api_cancel_nudge(nudge_id: str):
    from agents.tools.scheduling import cancel_scheduled_nudge, cancel_reminder
    result = cancel_scheduled_nudge(nudge_id)
    if "No scheduled nudge" in result:
        result = cancel_reminder(nudge_id)
    await channels.broadcast({"type": "queue_updated"})
    return {"ok": "cancelled" in result, "message": result}


# ── Connectors ──────────────────────────────────────────────────────────────

@app.post("/api/connectors/google/connect")
async def api_google_connect():
    from agents.tools._google_auth import run_oauth_flow, GoogleNotConfigured
    loop = asyncio.get_running_loop()
    try:
        token_path = await loop.run_in_executor(None, run_oauth_flow)
    except GoogleNotConfigured as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Google sign-in failed: {e}")
    await channels.broadcast({"type": "config_updated"})
    return {"ok": True, "token": token_path}


@app.get("/api/health")
async def api_health(fast: bool = True):
    """Lightweight readiness info. For the full report run ./health_check.sh."""
    cfg = _config.load_config()
    return {
        "ok": True,
        "model": _config.default_model(cfg),
        "connectors": _config.connector_status(cfg),
        "web_clients": channels.client_count(),
        "browser_extension": browser_bridge.is_connected(),
        "sleeping": runtime.is_sleeping(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Slack inbound (only if configured)
# ─────────────────────────────────────────────────────────────────────────────

_seen_events: set[str] = set()


@app.post("/slack/events")
async def slack_events(request: Request):
    if not channels.slack_enabled():
        raise HTTPException(404, "Slack is not configured")
    body = await request.body()
    ts = request.headers.get("X-Slack-Request-Timestamp", "0")
    sig = request.headers.get("X-Slack-Signature", "")
    secret = os.getenv("SLACK_SIGNING_SECRET", "")
    try:
        if abs(time.time() - int(ts)) > 300:
            raise ValueError
    except ValueError:
        return JSONResponse({"error": "invalid timestamp"}, status_code=403)
    mine = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:{body.decode('utf-8')}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mine, sig):
        return JSONResponse({"error": "invalid signature"}, status_code=403)
    data = json.loads(body)
    if data.get("type") == "url_verification":
        return {"challenge": data["challenge"]}
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        eid = data.get("event_id", "")
        if eid in _seen_events:
            return {"ok": True}
        _seen_events.add(eid)
        if len(_seen_events) > 1000:
            _seen_events.clear()
        if (event.get("type") == "message" and event.get("channel_type") == "im"
                and event.get("user") == os.getenv("SLACK_USER_ID") and "subtype" not in event and "bot_id" not in event):
            asyncio.create_task(runtime.handle_message(event.get("text", ""), origin="slack"))
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Browser extension relay
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/extension")
async def ws_extension(ws: WebSocket):
    await ws.accept()
    if not BROWSER_TOKEN:
        await ws.close(code=4003, reason="BROWSER_TOKEN not configured")
        return
    await browser_bridge.serve_extension(ws, BROWSER_TOKEN)


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Open http://{host}:{port} in your browser")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
