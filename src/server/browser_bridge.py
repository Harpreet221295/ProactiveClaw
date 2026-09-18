"""WebSocket relay between the agent's browser tools and the Chrome extension."""
from __future__ import annotations

import asyncio
from typing import Any

_extension_ws: Any = None
_pending_cdp: dict[str, asyncio.Future] = {}
_cdp_counter: int = 0


def is_connected() -> bool:
    return _extension_ws is not None


async def send_cdp_command(method: str, params: dict | None = None, timeout: float = 30) -> dict:
    """Send a CDP command to the browser extension and await the result."""
    global _cdp_counter
    if _extension_ws is None:
        # MV3 service workers restart; give the extension a moment to reconnect
        for _ in range(10):
            await asyncio.sleep(0.5)
            if _extension_ws is not None:
                break
        if _extension_ws is None:
            raise ConnectionError("Browser extension is not connected")

    _cdp_counter += 1
    cmd_id = str(_cdp_counter)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending_cdp[cmd_id] = future
    await _extension_ws.send_json({"type": "cdp_command", "id": cmd_id, "method": method, "params": params or {}})
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        _pending_cdp.pop(cmd_id, None)
        raise TimeoutError(f"CDP command '{method}' timed out after {timeout}s")


async def serve_extension(ws: Any, expected_token: str) -> None:
    """Handle one extension connection for its whole lifetime."""
    global _extension_ws
    try:
        auth_msg = await asyncio.wait_for(ws.receive_json(), timeout=10)
        if auth_msg.get("token") != expected_token:
            await ws.close(code=4001, reason="Invalid token")
            return
    except Exception:
        try:
            await ws.close(code=4002, reason="Auth timeout")
        except Exception:
            pass
        return

    _extension_ws = ws
    print("[browser] Extension connected")
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "cdp_result":
                future = _pending_cdp.pop(msg.get("id"), None)
                if future and not future.done():
                    if "error" in msg:
                        future.set_exception(RuntimeError(msg["error"]))
                    else:
                        future.set_result(msg.get("result", {}))
    except Exception as e:
        if type(e).__name__ != "WebSocketDisconnect":
            print(f"[browser] Extension WS error: {type(e).__name__}: {e}")
    finally:
        if _extension_ws is ws:
            _extension_ws = None
        for fut in _pending_cdp.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Extension disconnected"))
        _pending_cdp.clear()
        print("[browser] Extension disconnected")
