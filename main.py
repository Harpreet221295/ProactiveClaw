import json
import os
import select
import sys
import uuid

from dotenv import load_dotenv

from agent import Agent, SESSIONS_DIR
from tools import set_current_session_id, QUEUE_FILE


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
    return [
        f.removesuffix(".json")
        for f in sorted(os.listdir(SESSIONS_DIR))
        if f.endswith(".json")
    ]


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


def clear_queue_for_session(session_id: str) -> None:
    if not os.path.exists(QUEUE_FILE):
        return
    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            queue = []
    queue = [entry for entry in queue if entry.get("session_id") != session_id]
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def main():
    load_dotenv()
    timeout = int(os.getenv("AGENT_TIMEOUT", "300"))

    print("ReACT Agent (type 'quit' to exit)")
    print("-" * 40)

    session_id = pick_session()
    set_current_session_id(session_id)
    clear_queue_for_session(session_id)

    agent = Agent(session_id=session_id)
    print(f"\nSession: {session_id} | Timeout: {timeout}s")
    print("-" * 40)

    while True:
        try:
            query = timed_input("\nYou: ", timeout)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query is None:
            # Timeout — run pre-exit flow and exit
            agent.run_pre_exit()
            print("\n[session saved — exiting due to inactivity]")
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Check for image attachment: /image <path> or /img <path>
        images = None
        if query.lower().startswith(("/image ", "/img ")):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                img_path = parts[1].strip()
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as f:
                        images = [f.read()]
                    query = timed_input("Caption (or press Enter): ", timeout)
                    if query is None:
                        agent.run_pre_exit()
                        print("\n[session saved — exiting due to inactivity]")
                        break
                    query = query.strip() or "What's in this image?"
                else:
                    print(f"File not found: {img_path}")
                    continue

        print()
        answer = agent.run(query, images=images)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
