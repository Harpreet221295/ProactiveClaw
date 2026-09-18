#!/usr/bin/env python3
"""
ProactiveClaw health check — run this before starting the server.

Checks every integration for both configuration (env vars / files present)
and liveness (can we actually reach the service?). Exits 0 if all critical
checks pass, 1 if any critical check fails.

Usage:
    python src/health_check/check.py           # full check
    python src/health_check/check.py --fast    # skip live API calls (env/files only)
    python src/health_check/check.py --json    # machine-readable JSON output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Project root & path ───────────────────────────────────────────────────────
# src/health_check/check.py → health_check → src → project_root
_PROJECT_ROOT = Path(__file__).parents[2]
_SRC = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC))

# Load .env (best-effort)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from core import config as _config
from core.paths import GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE, CONFIG_FILE

_CFG = _config.load_config()
_CONN = _config.connector_status(_CFG)


def _connector_off(name: str) -> tuple[str, str, str] | None:
    """Return a skip/warn result if the connector is disabled or unconfigured."""
    st = _CONN.get(name, {})
    if not st.get("enabled"):
        return "skip", f"{name} connector is off (enable it in Settings if you want it)", ""
    if not st.get("configured"):
        return "fail", f"{name} is enabled but not configured: {st.get('reason')}", ""
    return None

# ── Terminal colours ──────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
CYAN   = lambda t: _c("36", t)

OK_MARK   = GREEN("✓")
WARN_MARK = YELLOW("⚠")
FAIL_MARK = RED("✗")
SKIP_MARK = DIM("–")


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str          # "ok" | "warn" | "fail" | "skip"
    message: str
    detail: str = ""
    elapsed_ms: int = 0
    critical: bool = True


@dataclass
class Section:
    title: str
    results: list[CheckResult] = field(default_factory=list)


# ── Runner ────────────────────────────────────────────────────────────────────

def _run(
    name: str,
    fn: Callable[[], tuple[str, str, str]],
    critical: bool = True,
    skip: bool = False,
) -> CheckResult:
    if skip:
        return CheckResult(name=name, status="skip", message="Skipped (--fast mode)", critical=critical)
    t0 = time.monotonic()
    try:
        status, message, detail = fn()
    except Exception as exc:
        status, message, detail = "fail", f"Unexpected error: {type(exc).__name__}", str(exc)
    elapsed = int((time.monotonic() - t0) * 1000)
    return CheckResult(name=name, status=status, message=message, detail=detail,
                       elapsed_ms=elapsed, critical=critical)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_env_var(var: str, min_len: int = 8) -> tuple[str, str, str]:
    val = os.getenv(var, "")
    if not val:
        return "fail", f"{var} is not set", "Add it to your .env file"
    if len(val) < min_len:
        return "warn", f"{var} is set but suspiciously short ({len(val)} chars)", ""
    return "ok", f"{var} is set ({len(val)} chars)", ""


# ── Python dependencies ───────────────────────────────────────────────────────

def check_dependencies() -> tuple[str, str, str]:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "slack_sdk": "slack-sdk",
        "openai": "openai",
        "anthropic": "anthropic",
        "requests": "requests",
        "tavily": "tavily-python",
        "google.oauth2": "google-auth",
        "googleapiclient": "google-api-python-client",
        "mem0": "mem0ai",
        "dotenv": "python-dotenv",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        return "fail", f"Missing packages: {', '.join(missing)}", f"Run: pip install {' '.join(missing)}"
    return "ok", f"All {len(required)} required packages importable", ""


# ── LLM: OpenAI ──────────────────────────────────────────────────────────────

def check_openai_models() -> tuple[str, str, str]:
    """Verify the API key is accepted and list models."""
    import urllib.request, urllib.error
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "fail", "OPENAI_API_KEY not set", ""
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            count = len(data.get("data", []))
            return "ok", f"OpenAI API key accepted — {count} models available", ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "fail", "OpenAI rejected key (401 Unauthorized)", "Check OPENAI_API_KEY"
        return "warn", f"OpenAI responded with HTTP {e.code}", str(e)
    except Exception as e:
        return "fail", f"OpenAI unreachable: {type(e).__name__}", str(e)


def check_openai_completion() -> tuple[str, str, str]:
    """Do a real (cheap) completion call to verify the configured model works end-to-end."""
    from agents.agent import DEFAULT_MODEL
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "fail", "OPENAI_API_KEY not set — cannot test completion", ""

    # Only run this check for OpenAI models
    if DEFAULT_MODEL.startswith("claude-"):
        return "skip", f"Default model is {DEFAULT_MODEL} (Anthropic) — skipping OpenAI completion", ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly the word: ok"}],
            max_tokens=5,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = resp.choices[0].message.content.strip()
        tokens_in  = resp.usage.prompt_tokens
        tokens_out = resp.usage.completion_tokens
        return (
            "ok",
            f"Completion successful using '{DEFAULT_MODEL}' ({latency_ms}ms)",
            f"Response: '{text}' | tokens: {tokens_in} in / {tokens_out} out",
        )
    except Exception as e:
        msg = str(e)
        if "model_not_found" in msg or "does not exist" in msg:
            return "fail", f"Model '{DEFAULT_MODEL}' not found in your OpenAI account", msg
        if "Incorrect API key" in msg or "invalid_api_key" in msg:
            return "fail", "OpenAI rejected key during completion", "Check OPENAI_API_KEY"
        return "fail", f"OpenAI completion failed: {type(e).__name__}", msg


# ── LLM: Anthropic ───────────────────────────────────────────────────────────

def check_anthropic_models() -> tuple[str, str, str]:
    import urllib.request, urllib.error
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return "warn", "ANTHROPIC_API_KEY not set (optional if using OpenAI)", ""
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            count = len(data.get("data", []))
            return "ok", f"Anthropic API key accepted — {count} models available", ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "fail", "Anthropic rejected key (401 Unauthorized)", "Check ANTHROPIC_API_KEY"
        return "warn", f"Anthropic responded with HTTP {e.code}", str(e)
    except Exception as e:
        return "fail", f"Anthropic unreachable: {type(e).__name__}", str(e)


def check_anthropic_completion() -> tuple[str, str, str]:
    """Do a real (cheap) completion call to verify the configured Claude model works."""
    from agents.agent import DEFAULT_MODEL
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return "warn", "ANTHROPIC_API_KEY not set — skipping Anthropic completion test", ""

    # Only run this as the primary completion test if model is Claude
    model = DEFAULT_MODEL if DEFAULT_MODEL.startswith("claude-") else "claude-haiku-4-5"
    label = DEFAULT_MODEL if DEFAULT_MODEL.startswith("claude-") else f"claude-haiku-4-5 (probe, default is {DEFAULT_MODEL})"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        t0 = time.monotonic()
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with exactly the word: ok"}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = resp.content[0].text.strip() if resp.content else "(empty)"
        tokens_in  = resp.usage.input_tokens
        tokens_out = resp.usage.output_tokens
        return (
            "ok",
            f"Completion successful using '{label}' ({latency_ms}ms)",
            f"Response: '{text}' | tokens: {tokens_in} in / {tokens_out} out",
        )
    except Exception as e:
        msg = str(e)
        if "model_not_found" in msg or "not_found_error" in msg:
            return "fail", f"Model '{model}' not found in your Anthropic account", msg
        if "authentication_error" in msg or "401" in msg:
            return "fail", "Anthropic rejected key during completion", "Check ANTHROPIC_API_KEY"
        return "fail", f"Anthropic completion failed: {type(e).__name__}", msg


def check_agent_model() -> tuple[str, str, str]:
    """Confirm the configured DEFAULT_MODEL is reachable with the right provider."""
    try:
        from agents.agent import DEFAULT_MODEL
        from llms import get_llm_client
        key_openai    = os.getenv("OPENAI_API_KEY", "")
        key_anthropic = os.getenv("ANTHROPIC_API_KEY", "")

        if DEFAULT_MODEL.startswith("claude-") and not key_anthropic:
            return "fail", (
                f"DEFAULT_MODEL is '{DEFAULT_MODEL}' (Anthropic) but ANTHROPIC_API_KEY is not set"
            ), "Set ANTHROPIC_API_KEY in .env or change DEFAULT_MODEL to an OpenAI model"

        if not DEFAULT_MODEL.startswith("claude-") and not key_openai:
            return "fail", (
                f"DEFAULT_MODEL is '{DEFAULT_MODEL}' (OpenAI) but OPENAI_API_KEY is not set"
            ), "Set OPENAI_API_KEY in .env"

        provider = "Anthropic" if DEFAULT_MODEL.startswith("claude-") else "OpenAI"
        return "ok", f"DEFAULT_MODEL = '{DEFAULT_MODEL}' ({provider}) — provider key present", ""
    except ImportError as e:
        return "warn", f"Could not import agent module: {e}", ""


# ── Tavily ────────────────────────────────────────────────────────────────────

def check_tavily_live() -> tuple[str, str, str]:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return "fail", "TAVILY_API_KEY not set", "Add it to .env"
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        t0 = time.monotonic()
        results = client.search("site:anthropic.com Claude", max_results=1)
        latency_ms = int((time.monotonic() - t0) * 1000)
        hits = results.get("results", [])
        if not hits:
            return "warn", f"Tavily reachable but returned 0 results ({latency_ms}ms)", "Key may be valid but quota exhausted"
        title = hits[0].get("title", "")[:60]
        return "ok", f"Tavily search working ({latency_ms}ms) — top result: '{title}'", ""
    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower() or "invalid" in msg.lower():
            return "fail", "Tavily rejected key — check TAVILY_API_KEY", msg
        if "429" in msg or "quota" in msg.lower():
            return "warn", "Tavily rate-limited or quota exceeded", msg
        return "fail", f"Tavily check failed: {type(e).__name__}", msg


def check_tavily_content_quality() -> tuple[str, str, str]:
    """Verify Tavily returns structured results with expected fields."""
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return "fail", "TAVILY_API_KEY not set", ""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        results = client.search("Python programming language", max_results=2)
        hits = results.get("results", [])
        if not hits:
            return "warn", "Search returned no results — cannot verify content quality", ""
        missing_fields = []
        for h in hits:
            for field in ("title", "url", "content"):
                if not h.get(field):
                    missing_fields.append(field)
        if missing_fields:
            return "warn", f"Search results missing fields: {set(missing_fields)}", "Results may be incomplete"
        urls_valid = all(h["url"].startswith("http") for h in hits)
        if not urls_valid:
            return "warn", "Some result URLs look malformed", ""
        return (
            "ok",
            f"Search results well-formed — {len(hits)} results with title, url, content",
            f"Sample URL: {hits[0]['url'][:80]}",
        )
    except Exception as e:
        return "warn", f"Content quality check failed: {type(e).__name__}", str(e)


# ── Notion ────────────────────────────────────────────────────────────────────

def check_notion_live() -> tuple[str, str, str]:
    import urllib.request, urllib.error
    key = os.getenv("NOTION_API_KEY", "")
    if not key:
        return "fail", "NOTION_API_KEY not set", "Add it to .env"
    req = urllib.request.Request(
        "https://api.notion.com/v1/users/me",
        headers={
            "Authorization": f"Bearer {key}",
            "Notion-Version": "2022-06-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            name = data.get("name") or data.get("id", "unknown")
            return "ok", f"Notion authenticated as: {name}", ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "fail", "Notion rejected key (401)", "Verify NOTION_API_KEY is an internal integration token"
        if e.code == 403:
            return "warn", "Notion key valid but no user access (403)", "Grant the integration access to pages in Notion UI"
        return "fail", f"Notion API error: HTTP {e.code}", str(e)
    except Exception as e:
        return "fail", f"Notion unreachable: {type(e).__name__}", str(e)


# ── Google ────────────────────────────────────────────────────────────────────

def check_google_files() -> tuple[str, str, str]:
    creds_path = _PROJECT_ROOT / "credentials.json"
    token_path  = _PROJECT_ROOT / "token.json"
    missing = []
    if not creds_path.exists():
        missing.append("credentials.json")
    if not token_path.exists():
        missing.append("token.json")
    if missing:
        hint = (
            "Download credentials.json from Google Cloud Console (OAuth 2.0 Client)"
            if "credentials.json" in missing else
            "Connect Google from Settings in the web UI, or run: python src/setup_wizard.py --google"
        )
        return "fail", f"Missing Google files: {', '.join(missing)}", hint
    try:
        data = json.loads(token_path.read_text())
        expiry = data.get("expiry", "unknown")
        has_refresh = bool(data.get("refresh_token"))
        extra = f"refresh_token={'present' if has_refresh else 'MISSING'}, expiry={expiry}"
        if not has_refresh:
            return "warn", "token.json has no refresh_token — may need re-auth", extra
        return "ok", "credentials.json + token.json present", extra
    except json.JSONDecodeError:
        return "fail", "token.json is corrupt (invalid JSON)", "Delete it and re-run OAuth flow"


def _get_google_creds():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    token_path = GOOGLE_TOKEN_FILE
    scopes = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/calendar",
    ]
    creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Token invalid and cannot be refreshed — re-run OAuth flow")
    return creds


def check_gmail_live() -> tuple[str, str, str]:
    try:
        from googleapiclient.discovery import build
        creds = _get_google_creds()
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        address = profile.get("emailAddress", "unknown")
        total = profile.get("messagesTotal", "?")
        return "ok", f"Gmail authenticated as: {address} ({total} messages)", ""
    except Exception as e:
        msg = str(e)
        if "invalid_grant" in msg or "Token has been expired" in msg:
            return "fail", "Google token expired — re-run OAuth flow", (
                "Delete token.json and run: python src/agents/tools/_google_auth.py"
            )
        return "fail", f"Gmail check failed: {type(e).__name__}", msg


def check_calendar_live() -> tuple[str, str, str]:
    try:
        from googleapiclient.discovery import build
        creds = _get_google_creds()
        service = build("calendar", "v3", credentials=creds)
        result = service.calendarList().list(maxResults=1).execute()
        calendars = result.get("items", [])
        primary = next((c for c in calendars if c.get("primary")), None)
        name = primary.get("summary", "primary") if primary else "no calendars found"
        return "ok", f"Google Calendar authenticated — primary: {name}", ""
    except Exception as e:
        msg = str(e)
        if "invalid_grant" in msg or "Token has been expired" in msg:
            return "fail", "Google token expired — re-run OAuth flow", (
                "Delete token.json and run: python src/agents/tools/_google_auth.py"
            )
        return "fail", f"Calendar check failed: {type(e).__name__}", msg


# ── Slack ─────────────────────────────────────────────────────────────────────

def check_slack_env() -> tuple[str, str, str]:
    missing = [v for v in ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "SLACK_USER_ID"] if not os.getenv(v)]
    if missing:
        return "fail", f"Missing Slack env vars: {', '.join(missing)}", "Add them to .env"
    user_id = os.getenv("SLACK_USER_ID", "")
    if not user_id.startswith("U"):
        return "warn", f"SLACK_USER_ID looks wrong (got '{user_id}') — should start with 'U'", ""
    return "ok", "SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_USER_ID all set", ""


def check_slack_live() -> tuple[str, str, str]:
    import urllib.request, urllib.error
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        return "fail", "SLACK_BOT_TOKEN not set", ""
    req = urllib.request.Request(
        "https://slack.com/api/auth.test",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        if not data.get("ok"):
            error = data.get("error", "unknown")
            return "fail", f"Slack auth.test failed: {error}", "Check SLACK_BOT_TOKEN"
        bot_name = data.get("user", "unknown")
        team = data.get("team", "unknown")
        return "ok", f"Slack authenticated — bot '{bot_name}' in workspace '{team}'", ""
    except Exception as e:
        return "fail", f"Slack unreachable: {type(e).__name__}", str(e)


def check_slack_can_dm_user() -> tuple[str, str, str]:
    """Verify the bot can actually open a DM channel with the configured user."""
    import urllib.request, urllib.error, urllib.parse
    token   = os.getenv("SLACK_BOT_TOKEN", "")
    user_id = os.getenv("SLACK_USER_ID", "")
    if not token or not user_id:
        return "fail", "SLACK_BOT_TOKEN or SLACK_USER_ID not set", ""
    payload = json.dumps({"users": user_id}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/conversations.open",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        if not data.get("ok"):
            error = data.get("error", "unknown")
            if error == "cannot_dm_bot":
                return "warn", f"SLACK_USER_ID '{user_id}' appears to be a bot — should be a real user ID", ""
            return "fail", f"Cannot open DM with user '{user_id}': {error}", "Check SLACK_USER_ID in .env"
        channel_id = data.get("channel", {}).get("id", "unknown")
        return "ok", f"Bot can DM user '{user_id}' via channel {channel_id}", ""
    except Exception as e:
        return "fail", f"Slack DM check failed: {type(e).__name__}", str(e)


# ── Memory (mem0) ─────────────────────────────────────────────────────────────

def check_memory_import() -> tuple[str, str, str]:
    mem0_dir = _PROJECT_ROOT / "mem0_data"
    try:
        from memory import query_memory  # noqa: F401
        db_path = mem0_dir / "collection" / "agent_memory" / "storage.sqlite"
        if db_path.exists():
            size_kb = db_path.stat().st_size // 1024
            return "ok", f"Memory module importable — SQLite DB exists ({size_kb} KB)", ""
        return "ok", "Memory module importable (DB not yet created — will init on first use)", ""
    except ImportError as e:
        return "fail", f"Memory module import failed: {e}", "Check src/memory/ and mem0 package"
    except Exception as e:
        return "warn", f"Memory check partial: {type(e).__name__}", str(e)


def check_memory_query() -> tuple[str, str, str]:
    """Actually try to query memory to confirm the vector store is operational."""
    try:
        from memory import query_memory
        t0 = time.monotonic()
        results = query_memory("health check probe")
        latency_ms = int((time.monotonic() - t0) * 1000)
        count = len(results) if isinstance(results, list) else 0
        return "ok", f"Memory query successful ({latency_ms}ms) — {count} result(s) returned", ""
    except Exception as e:
        msg = str(e)
        if "no such table" in msg or "database" in msg.lower():
            return "warn", "Memory DB not yet initialised — will be created on first use", str(e)
        return "fail", f"Memory query failed: {type(e).__name__}", msg


# ── Filesystem ────────────────────────────────────────────────────────────────

def check_filesystem() -> tuple[str, str, str]:
    fs_base = _PROJECT_ROOT / "agent_file_system"
    if not fs_base.exists():
        return "fail", "agent_file_system/ directory is missing", "It should be created automatically on first run"

    test_file = fs_base / ".health_check_probe"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        return "fail", f"agent_file_system/ is not writable: {e}", ""

    # Ensure expected subdirs exist
    for sub in ["subagents"]:
        (fs_base / sub).mkdir(exist_ok=True)

    try:
        entries = list(fs_base.rglob("*"))
        return "ok", f"agent_file_system/ writable — {len(entries)} files/dirs inside", ""
    except Exception:
        return "ok", "agent_file_system/ writable", ""


def check_engagement_data() -> tuple[str, str, str]:
    ed = _PROJECT_ROOT / "engagement_data"
    if not ed.exists():
        return "warn", "engagement_data/ missing — will be created on first run", ""
    data_files = ["queue.json", "reminders.json", "cron_jobs.json"]
    issues = []
    for fname in data_files:
        fpath = ed / fname
        if fpath.exists():
            try:
                json.loads(fpath.read_text())
            except json.JSONDecodeError:
                issues.append(f"{fname} is corrupt JSON")
    if issues:
        return "warn", f"Engagement data issues: {'; '.join(issues)}", "Delete the corrupt files — they'll be recreated"
    found = sum(1 for f in data_files if (ed / f).exists())
    return "ok", f"engagement_data/ OK — {found}/{len(data_files)} data files present", ""


# ── Browser ───────────────────────────────────────────────────────────────────

def check_browser_token() -> tuple[str, str, str]:
    token = os.getenv("BROWSER_TOKEN", "")
    if not token:
        return "warn", "BROWSER_TOKEN not set — browser tools will be unavailable", "Set BROWSER_TOKEN in .env to enable browser use"
    return "ok", f"BROWSER_TOKEN set ({len(token)} chars) — extension auth ready", ""


# ── LLM provider gate ─────────────────────────────────────────────────────────

def check_llm_provider() -> tuple[str, str, str]:
    openai_key    = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not openai_key and not anthropic_key:
        return "fail", "No LLM API key found — set OPENAI_API_KEY or ANTHROPIC_API_KEY", ""
    providers = []
    if openai_key:
        providers.append("OpenAI")
    if anthropic_key:
        providers.append("Anthropic")
    return "ok", f"LLM provider(s) configured: {', '.join(providers)}", ""


# ── Build sections ────────────────────────────────────────────────────────────

def build_sections(fast: bool) -> list[Section]:
    sections: list[Section] = []
    eff = _config.effective_settings(_CFG)

    s = Section("Python Environment")
    s.results.append(_run("Dependencies", check_dependencies))
    sections.append(s)

    s = Section("Configuration")
    def _cfg_check():
        if not CONFIG_FILE.exists():
            return "warn", "config/config.json not found — using defaults (run python src/setup_wizard.py)", ""
        active = ", ".join(_config.active_connectors(_CFG)) or "none"
        return "ok", f"level={eff['level']} mode={eff['care_mode']} connectors={active}", f"model={_config.default_model(_CFG)}"
    s.results.append(_run("config/config.json", _cfg_check, critical=False))
    sections.append(s)

    s = Section("LLM Providers")
    s.results.append(_run("LLM provider (any)", check_llm_provider, critical=True))
    s.results.append(_run("Configured agent model", check_agent_model, critical=True))
    def _optional_key(var: str, other: str):
        def _f():
            if not os.getenv(var) and os.getenv(other):
                return "skip", f"{var} not set (using {other.split('_')[0].title()} instead)", ""
            return _check_env_var(var)
        return _f
    s.results.append(_run("OpenAI API key", _optional_key("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), critical=False))
    s.results.append(_run("OpenAI models reachable", check_openai_models, critical=False, skip=fast or not os.getenv("OPENAI_API_KEY")))
    s.results.append(_run("OpenAI completion test", check_openai_completion, critical=False, skip=fast or not os.getenv("OPENAI_API_KEY")))
    s.results.append(_run("Anthropic API key", _optional_key("ANTHROPIC_API_KEY", "OPENAI_API_KEY"), critical=False))
    s.results.append(_run("Anthropic models reachable", check_anthropic_models, critical=False, skip=fast or not os.getenv("ANTHROPIC_API_KEY")))
    s.results.append(_run("Anthropic completion test", check_anthropic_completion, critical=False, skip=fast or not os.getenv("ANTHROPIC_API_KEY")))
    sections.append(s)

    s = Section("Slack (optional delivery channel)")
    slack_on = bool(os.getenv("SLACK_BOT_TOKEN"))
    if slack_on:
        s.results.append(_run("Slack env vars", check_slack_env, critical=False))
        s.results.append(_run("Slack API live", check_slack_live, critical=False, skip=fast))
        s.results.append(_run("Slack can DM user", check_slack_can_dm_user, critical=False, skip=fast))
    else:
        s.results.append(CheckResult("Slack", "skip", "Not configured — web UI only (set SLACK_* in .env to add Slack)", critical=False))
    sections.append(s)

    s = Section("Google (Gmail & Calendar)")
    g_off = _connector_off("gmail") if _CONN["gmail"]["enabled"] else _connector_off("calendar")
    if not _CONN["gmail"]["enabled"] and not _CONN["calendar"]["enabled"]:
        s.results.append(CheckResult("Google", "skip", "gmail and calendar connectors are off", critical=False))
    else:
        s.results.append(_run("Google credential files", check_google_files, critical=True))
        if _CONN["gmail"]["enabled"]:
            s.results.append(_run("Gmail API live", check_gmail_live, critical=True, skip=fast))
        if _CONN["calendar"]["enabled"]:
            s.results.append(_run("Google Calendar API live", check_calendar_live, critical=True, skip=fast))
    sections.append(s)

    s = Section("Notion")
    off = _connector_off("notion")
    if off:
        s.results.append(CheckResult("Notion", off[0], off[1], critical=off[0] == "fail"))
    else:
        s.results.append(_run("NOTION_API_KEY", lambda: _check_env_var("NOTION_API_KEY"), critical=True))
        s.results.append(_run("Notion API live", check_notion_live, critical=True, skip=fast))
    sections.append(s)

    s = Section("Web Search (Tavily)")
    off = _connector_off("web_search")
    if off:
        s.results.append(CheckResult("Web search", off[0], off[1], critical=off[0] == "fail"))
    else:
        s.results.append(_run("TAVILY_API_KEY", lambda: _check_env_var("TAVILY_API_KEY"), critical=True))
        s.results.append(_run("Tavily search live", check_tavily_live, critical=True, skip=fast))
        s.results.append(_run("Tavily result quality", check_tavily_content_quality, critical=False, skip=fast))
    sections.append(s)

    s = Section("Storage & Filesystem")
    s.results.append(_run("agent_file_system/", check_filesystem, critical=True))
    s.results.append(_run("engagement_data/", check_engagement_data, critical=True))
    s.results.append(_run("Memory module import", check_memory_import, critical=False))
    s.results.append(_run("Memory query", check_memory_query, critical=False, skip=fast or not os.getenv("OPENAI_API_KEY")))
    sections.append(s)

    s = Section("Browser Extension (optional)")
    if _CONN["browser"]["enabled"]:
        s.results.append(_run("BROWSER_TOKEN", check_browser_token, critical=False))
    else:
        s.results.append(CheckResult("Browser", "skip", "browser connector is off", critical=False))
    sections.append(s)

    return sections


# ── Output ────────────────────────────────────────────────────────────────────

def _mark(status: str) -> str:
    return {"ok": OK_MARK, "warn": WARN_MARK, "fail": FAIL_MARK, "skip": SKIP_MARK}.get(status, "?")


def print_report(sections: list[Section]) -> None:
    print()
    print(BOLD("━━━  ProactiveClaw Health Check  ━━━"))
    print()
    for section in sections:
        print(CYAN(f"▸ {section.title}"))
        for r in section.results:
            timing    = DIM(f"  [{r.elapsed_ms}ms]") if r.elapsed_ms and r.status != "skip" else ""
            crit_tag  = "" if r.critical else DIM(" (optional)")
            print(f"  {_mark(r.status)}  {r.name}{crit_tag}{timing}")
            print(f"     {r.message}")
            if r.detail:
                print(f"     {DIM(r.detail)}")
        print()


def print_summary(sections: list[Section]) -> int:
    all_results = [r for s in sections for r in s.results]
    ok    = sum(1 for r in all_results if r.status == "ok")
    warn  = sum(1 for r in all_results if r.status == "warn")
    fail  = sum(1 for r in all_results if r.status == "fail")
    skip  = sum(1 for r in all_results if r.status == "skip")
    total = len(all_results)

    critical_failures = [r for r in all_results if r.status == "fail" and r.critical]
    optional_failures = [r for r in all_results if r.status == "fail" and not r.critical]

    print(BOLD("━━━  Summary  ━━━"))
    print(f"  {OK_MARK}  {ok} passed   "
          f"{WARN_MARK}  {warn} warnings   "
          f"{FAIL_MARK}  {fail} failed   "
          f"{SKIP_MARK}  {skip} skipped   "
          f"(total: {total})")
    print()

    if critical_failures:
        print(RED(BOLD("  Critical failures (must fix before starting server):")))
        for r in critical_failures:
            print(f"    {FAIL_MARK}  {r.name}: {r.message}")
            if r.detail:
                print(f"       {DIM(r.detail)}")
        print()
        print(RED("  Server should NOT be started until these are resolved."))
        print()
        return 1

    if optional_failures:
        print(YELLOW("  Optional features not configured:"))
        for r in optional_failures:
            print(f"    {FAIL_MARK}  {r.name}: {r.message}")
        print()

    if warn > 0 or optional_failures:
        print(YELLOW("  All critical checks passed. Some optional features may be degraded."))
        print(GREEN("  Server can be started."))
    else:
        print(GREEN(BOLD("  All checks passed — ready to start!")))
    print()
    return 0


def json_report(sections: list[Section]) -> int:
    all_results = [r for s in sections for r in s.results]
    critical_failures = [r for r in all_results if r.status == "fail" and r.critical]
    output = {
        "pass": not critical_failures,
        "sections": [
            {
                "title": s.title,
                "checks": [
                    {
                        "name": r.name,
                        "status": r.status,
                        "message": r.message,
                        "detail": r.detail,
                        "elapsed_ms": r.elapsed_ms,
                        "critical": r.critical,
                    }
                    for r in s.results
                ],
            }
            for s in sections
        ],
        "summary": {
            "ok":   sum(1 for r in all_results if r.status == "ok"),
            "warn": sum(1 for r in all_results if r.status == "warn"),
            "fail": sum(1 for r in all_results if r.status == "fail"),
            "skip": sum(1 for r in all_results if r.status == "skip"),
        },
        "critical_failures": [
            {"name": r.name, "message": r.message}
            for r in critical_failures
        ],
    }
    print(json.dumps(output, indent=2))
    return 0 if not critical_failures else 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="ProactiveClaw health check")
    parser.add_argument("--fast", action="store_true",
                        help="Skip live API calls — only check env vars and files")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of terminal report")
    args = parser.parse_args()

    global _NO_COLOR
    if args.json:
        _NO_COLOR = True

    if not args.json:
        mode = "fast (env/file checks only)" if args.fast else "full (including live API calls)"
        print(f"\n{DIM(f'Mode: {mode}')}")

    sections = build_sections(fast=args.fast)

    if args.json:
        return json_report(sections)

    print_report(sections)
    return print_summary(sections)


if __name__ == "__main__":
    code = main()
    # Qdrant's __del__ fires after atexit, during final GC, when sys.meta_path is
    # already gone — producing harmless but ugly tracebacks on Python 3.13+.
    # Redirect fd 2 at the OS level so __del__ noise is silenced on exit.
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    except OSError:
        pass
    sys.exit(code)
