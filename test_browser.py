"""Quick test script for browser tools — no Slack needed."""
import asyncio
import sys
import os

# Must run within the server process's event loop context.
# Usage: python test_browser.py

async def main():
    # Import after event loop is running
    from dotenv import load_dotenv
    load_dotenv()

    import slack_server

    print(f"[test] _extension_ws = {slack_server._extension_ws}")
    print(f"[test] _extension_ws is None = {slack_server._extension_ws is None}")

    if slack_server._extension_ws is None:
        print("[test] Waiting up to 15s for extension to connect...")
        for i in range(30):
            await asyncio.sleep(0.5)
            if slack_server._extension_ws is not None:
                print(f"[test] Extension connected after {(i+1)*0.5}s")
                break
        else:
            print("[test] Extension never connected. Check:")
            print("  1. Is the extension loaded in Chrome?")
            print("  2. Did you click the icon on a tab?")
            print("  3. Check offscreen.html console for errors")
            return

    print(f"[test] Extension connected: {slack_server._extension_ws is not None}")

    # Test: get page info
    try:
        print("\n[test] Calling browser_get_page_info via send_cdp_command...")
        js = "JSON.stringify({url: document.location.href, title: document.title})"
        result = await slack_server.send_cdp_command("Runtime.evaluate", {"expression": js}, timeout=10)
        print(f"[test] Result: {result}")
    except Exception as e:
        print(f"[test] Error: {type(e).__name__}: {e}")

    # Test: snapshot
    try:
        print("\n[test] Calling snapshot...")
        result = await slack_server.send_cdp_command("snapshot", {}, timeout=10)
        snapshot = result.get("snapshot", "")
        print(f"[test] Snapshot ({len(snapshot)} chars):")
        print(snapshot[:500])
    except Exception as e:
        print(f"[test] Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # We need uvicorn running to accept the WS, so start the FastAPI app
    # and run our test after a delay
    import uvicorn
    from slack_server import app
    import tools._state

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tools._state._event_loop = loop

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)

    async def run_test():
        # Start server in background
        task = asyncio.create_task(server.serve())
        # Wait for server to be ready
        await asyncio.sleep(2)
        print("\n" + "="*50)
        print("SERVER READY — waiting for extension connection")
        print("="*50 + "\n")

        await main()

        print("\n[test] Done. Press Ctrl+C to exit.")
        await task

    try:
        loop.run_until_complete(run_test())
    except KeyboardInterrupt:
        print("\nShutting down.")
