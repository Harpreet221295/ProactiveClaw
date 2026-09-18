"""CLI mode — chat with ProactiveClaw in the terminal (no web UI, no scheduler).

Useful for a quick first test. Proactive nudges are only *delivered* by the
server (python src/serve.py), but everything else — tools, care registry,
memory, pre-exit flow — works here too.
"""
from __future__ import annotations
import os
import select
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agents.agent import Agent, SESSIONS_DIR
from agents.tools import set_current_session_id
from core import config as _config


def timed_input(prompt: str, timeout: float) -> str | None:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().rstrip("\n")
    return None


def list_sessions() -> list[str]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    return [f.removesuffix(".json") for f in sorted(os.listdir(SESSIONS_DIR))
            if f.endswith(".json") and not f.endswith("_summary.json")]


def pick_session() -> str:
    sessions = list_sessions()
    if not sessions:
        new_id = uuid.uuid4().hex[:8]
        print(f"No existing sessions. Starting new session: {new_id}")
        return new_id
    print("\nExisting sessions:")
    for i, sid in enumerate(sessions, 1):
        print(f"  {i}. {sid}")
    print(f"  {len(sessions) + 1}. Start new session")
    choice = input("\nPick a session number: ").strip()
    try:
        idx = int(choice)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
    except ValueError:
        pass
    new_id = uuid.uuid4().hex[:8]
    print(f"Starting new session: {new_id}")
    return new_id


def main():
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("No LLM API key found. Run ./setup.sh (or python src/setup_wizard.py) first.")
        sys.exit(1)
    timeout = int(os.getenv("AGENT_TIMEOUT", "300"))
    cfg = _config.load_config()
    eff = _config.effective_settings(cfg)
    print("ProactiveClaw CLI (type 'quit' to exit)")
    print(f"level={eff['level']} mode={eff['care_mode']} model={_config.default_model(cfg)} connectors={', '.join(_config.active_connectors(cfg)) or 'none'}")
    print("-" * 40)

    session_id = pick_session()
    set_current_session_id(session_id)
    new_session = not Agent.session_exists(session_id)
    agent = Agent(session_id=session_id)
    care_ctx = ""
    if new_session:
        try:
            from care.brief import session_context
            care_ctx = session_context(cfg)
        except Exception:
            care_ctx = ""
    print(f"\nSession: {session_id} | Timeout: {timeout}s")
    print("-" * 40)

    while True:
        try:
            query = timed_input("\nYou: ", timeout)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if query is None:
            agent.run_pre_exit()
            print("\n[session saved — exiting due to inactivity]")
            break
        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        images = None
        if query.lower().startswith(("/image ", "/img ")):
            parts = query.split(maxsplit=1)
            if len(parts) == 2 and os.path.isfile(parts[1].strip()):
                with open(parts[1].strip(), "rb") as f:
                    images = [f.read()]
                query = (timed_input("Caption (or press Enter): ", timeout) or "").strip() or "What's in this image?"
            else:
                print("File not found.")
                continue
        if care_ctx:
            query = f"{care_ctx}\n\n{query}"
            care_ctx = ""
        print()
        answer = agent.run(query, images=images)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
