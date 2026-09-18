#!/usr/bin/env python3
"""Interactive first-run setup.

Writes .env and config/config.json, optionally signs in to Google, and
finishes with a fast health check. Safe to re-run: existing values are
offered as defaults.

    python src/setup_wizard.py            # full wizard
    python src/setup_wizard.py --google   # just (re)run the Google sign-in
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from core import config as _config
from core.paths import ENV_FILE, CONFIG_FILE, CONFIG_EXAMPLE_FILE, GOOGLE_CREDENTIALS_FILE, PROJECT_ROOT

try:
    from rich.console import Console
    from rich.panel import Panel
    _c = Console()
    def say(s=""): _c.print(s)
    def head(s): _c.print(Panel.fit(f"[bold]{s}[/bold]", border_style="cyan"))
except ImportError:  # pragma: no cover
    def say(s=""): print(s)
    def head(s): print(f"\n=== {s} ===")


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    shown = f" [{'•' * 6 if (secret and default) else default}]" if default else ""
    try:
        val = input(f"{prompt}{shown}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return val or default


def yes(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    v = ask(f"{prompt} ({d})").lower()
    if not v:
        return default
    return v in ("y", "yes")


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


def write_env(env: dict[str, str]) -> None:
    template = (PROJECT_ROOT / ".env.example").read_text() if (PROJECT_ROOT / ".env.example").exists() else ""
    lines = []
    seen = set()
    for line in template.splitlines():
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
        if m:
            k = m.group(1)
            seen.add(k)
            lines.append(f"{k}={env.get(k, '')}")
        else:
            lines.append(line)
    for k, v in env.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines).rstrip() + "\n")


def guess_timezone() -> str:
    try:
        tzfile = Path("/etc/localtime")
        if tzfile.is_symlink():
            target = str(tzfile.resolve())
            if "zoneinfo/" in target:
                return target.split("zoneinfo/", 1)[1]
    except Exception:
        pass
    return os.environ.get("TZ", "")


def google_signin() -> bool:
    from agents.tools._google_auth import run_oauth_flow, GoogleNotConfigured
    try:
        path = run_oauth_flow()
        say(f"  ✓ Google connected — token saved to {path}")
        return True
    except GoogleNotConfigured as e:
        say(f"  ✗ {e}")
    except Exception as e:
        say(f"  ✗ Google sign-in failed: {e}")
    return False


def main() -> int:
    if "--google" in sys.argv:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        return 0 if google_signin() else 1

    head("ProactiveClaw setup")
    say("Answer a few questions. Press Enter to keep a default. Everything can be changed later in the web UI (⚙️ Settings) or by editing .env / config/config.json.\n")

    env = read_env()
    cfg = _config.load_config() if CONFIG_FILE.exists() else _config._deep_merge(_config.DEFAULT_CONFIG, {})

    # ── You ────────────────────────────────────────────────────────────────
    head("1 · You")
    cfg["user"]["name"] = ask("Your name", cfg["user"].get("name") or "")
    cfg["user"]["timezone"] = ask("Timezone (IANA, e.g. America/Los_Angeles)", cfg["user"].get("timezone") or guess_timezone())

    # ── LLM ────────────────────────────────────────────────────────────────
    head("2 · Language model (one key is enough)")
    env["OPENAI_API_KEY"] = ask("OpenAI API key (leave empty to skip)", env.get("OPENAI_API_KEY", ""), secret=True)
    env["ANTHROPIC_API_KEY"] = ask("Anthropic API key (leave empty to skip)", env.get("ANTHROPIC_API_KEY", ""), secret=True)
    if not env["OPENAI_API_KEY"] and not env["ANTHROPIC_API_KEY"]:
        say("[red]At least one LLM key is required.[/red]")
        return 1
    default_model = cfg["llm"].get("model") or ("gpt-4o-mini" if env["OPENAI_API_KEY"] else "claude-sonnet-5")
    cfg["llm"]["model"] = ask("Model", default_model)
    if not env["OPENAI_API_KEY"]:
        say("  Note: long-term memory (mem0) and tier-1 memory use OpenAI embeddings; without an OpenAI key they stay disabled. Chat and care tracking work fine.")

    # ── Proactiveness ─────────────────────────────────────────────────────
    head("3 · How proactive should it be?")
    for i, lvl in enumerate(_config.LEVEL_ORDER):
        say(f"  {i}. [bold]{lvl:9s}[/bold] {_config.PROACTIVENESS_LEVELS[lvl]['description']}")
    cur = cfg["proactiveness"].get("level", "balanced")
    choice = ask("Level (number or name)", cur)
    cfg["proactiveness"]["level"] = _config.normalize_level(choice)
    cfg["morning_review"]["time"] = ask("Morning review time (HH:MM)", cfg["morning_review"].get("time", "07:30"))

    # ── Connectors ────────────────────────────────────────────────────────
    head("4 · Optional connectors (you can skip all of these)")
    say("Web search (Tavily, free tier at tavily.com) lets the assistant look things up.")
    env["TAVILY_API_KEY"] = ask("Tavily API key (empty = off)", env.get("TAVILY_API_KEY", ""), secret=True)
    cfg["connectors"]["web_search"] = bool(env["TAVILY_API_KEY"])

    say("\nNotion: create an internal integration at notion.so/my-integrations and share the pages/databases with it.")
    env["NOTION_API_KEY"] = ask("Notion integration token (empty = off)", env.get("NOTION_API_KEY", ""), secret=True)
    cfg["connectors"]["notion"] = bool(env["NOTION_API_KEY"])
    if cfg["connectors"]["notion"]:
        cfg["notion"]["tasks_database_id"] = ask("Notion tasks database id (optional)", cfg["notion"].get("tasks_database_id", ""))

    say(f"\nGmail & Google Calendar: download an OAuth client (Desktop app) from Google Cloud Console → save as {GOOGLE_CREDENTIALS_FILE.name} in the project root.")
    has_creds = GOOGLE_CREDENTIALS_FILE.exists()
    say(f"  credentials.json {'found ✓' if has_creds else 'not found'}")
    want_gmail = yes("Enable Gmail?", cfg["connectors"].get("gmail", False) or has_creds)
    want_cal = yes("Enable Google Calendar?", cfg["connectors"].get("calendar", False) or has_creds)
    cfg["connectors"]["gmail"], cfg["connectors"]["calendar"] = want_gmail, want_cal

    say("\nSlack (optional): also deliver messages to a Slack DM. Leave empty to use only the web UI.")
    env["SLACK_BOT_TOKEN"] = ask("Slack bot token (empty = off)", env.get("SLACK_BOT_TOKEN", ""), secret=True)
    if env["SLACK_BOT_TOKEN"]:
        env["SLACK_SIGNING_SECRET"] = ask("Slack signing secret", env.get("SLACK_SIGNING_SECRET", ""), secret=True)
        env["SLACK_USER_ID"] = ask("Your Slack member id", env.get("SLACK_USER_ID", ""))

    env.setdefault("AGENT_TIMEOUT", "300")
    env["AGENT_TIMEOUT"] = ask("\nIdle seconds before the assistant 'sleeps' and plans follow-ups", env.get("AGENT_TIMEOUT") or "300")

    # ── Write ─────────────────────────────────────────────────────────────
    write_env(env)
    _config.save_config(cfg)
    say(f"\n✓ Wrote {ENV_FILE.relative_to(PROJECT_ROOT)} and {CONFIG_FILE.relative_to(PROJECT_ROOT)}")

    for k, v in env.items():
        if v:
            os.environ[k] = v
    if (want_gmail or want_cal) and has_creds and yes("\nSign in to Google now? (opens a browser)", True):
        google_signin()

    head("Summary")
    say(_config.summary(_config.load_config(force=True)))

    say("\nRunning a quick health check…")
    try:
        from health_check.check import build_sections, print_report, print_summary
        sections = build_sections(fast=True)
        print_report(sections)
        print_summary(sections)
    except Exception as e:
        say(f"(health check skipped: {e})")

    say("\n[bold green]Done.[/bold green] Start it with  ./run.sh  and open http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
