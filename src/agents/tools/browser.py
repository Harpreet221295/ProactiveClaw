from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path

from . import _state


def _run_cdp(method: str, params: dict | None = None, timeout: float = 30) -> dict:
    """Synchronous bridge: send a CDP command via the WebSocket relay and block for the result."""
    loop = _state._event_loop
    if loop is None:
        raise RuntimeError("Event loop not set — is the server running?")

    from server import browser_bridge
    coro = browser_bridge.send_cdp_command(method, params, timeout=timeout)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout + 5)


def browser_navigate(url: str) -> str:
    """Navigate the attached browser tab to a URL."""
    try:
        _run_cdp("Page.navigate", {"url": url})
    except RuntimeError as e:
        return f"Browser error: {e}. Use tavily_search as an alternative to look up web content."
    time.sleep(2)  # wait for page load
    return f"Navigated to {url}"


def browser_snapshot(**_kwargs) -> str:
    """Get a text snapshot of the current page with interactive element refs."""
    result = _run_cdp("snapshot")
    return result.get("snapshot", "Error: no snapshot returned")


def browser_click(ref: int) -> str:
    """Click an interactive element by its ref number from the snapshot."""
    js = f"window.__pclaw_refs && window.__pclaw_refs[{ref}] ? (window.__pclaw_refs[{ref}].click(), 'clicked') : 'ref not found'"
    result = _run_cdp("Runtime.evaluate", {"expression": js})
    value = result.get("result", {}).get("value", "unknown")
    if value == "ref not found":
        return f"Error: ref {ref} not found — try taking a new snapshot"
    return f"Clicked element [ref:{ref}]"


def browser_type(ref: int, text: str) -> str:
    """Type text into an input element by its ref number."""
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    js = f"""
    (function() {{
        var el = window.__pclaw_refs && window.__pclaw_refs[{ref}];
        if (!el) return 'ref not found';
        el.focus();
        el.value = '{escaped}';
        el.dispatchEvent(new Event('input', {{bubbles: true}}));
        el.dispatchEvent(new Event('change', {{bubbles: true}}));
        return 'typed';
    }})()
    """
    result = _run_cdp("Runtime.evaluate", {"expression": js})
    value = result.get("result", {}).get("value", "unknown")
    if value == "ref not found":
        return f"Error: ref {ref} not found — try taking a new snapshot"
    return f"Typed '{text}' into element [ref:{ref}]"


def browser_screenshot(**_kwargs) -> str:
    """Take a screenshot of the current page and save it."""
    result = _run_cdp("Page.captureScreenshot", {"format": "png"})
    data = result.get("data", "")
    if not data:
        return "Error: no screenshot data returned"

    screenshots_dir = Path(_state._AGENT_FS_BASE) / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    filename = f"screenshot_{int(time.time())}.png"
    filepath = screenshots_dir / filename
    filepath.write_bytes(base64.b64decode(data))
    return f"Screenshot saved: screenshots/{filename}"


def browser_get_page_info(**_kwargs) -> str:
    """Get the current page URL and title."""
    js = "JSON.stringify({url: document.location.href, title: document.title})"
    result = _run_cdp("Runtime.evaluate", {"expression": js})
    value = result.get("result", {}).get("value", "{}")
    return f"Page info: {value}"


def browser_scroll(direction: str = "down") -> str:
    """Scroll the page up or down by 500px."""
    delta = 500 if direction == "down" else -500
    js = f"window.scrollBy(0, {delta}); 'scrolled {direction}'"
    result = _run_cdp("Runtime.evaluate", {"expression": js})
    return f"Scrolled {direction}"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate the attached browser tab to a URL. Waits 2 seconds for the page to load.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": "Get a text snapshot of the current browser page. Returns a text tree of visible elements with [ref:N] markers on interactive elements (links, buttons, inputs). Use the ref numbers with browser_click and browser_type.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an interactive element on the page by its ref number from browser_snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "integer",
                        "description": "The ref number of the element to click (from browser_snapshot output).",
                    }
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input field by its ref number from browser_snapshot. Sets the value and dispatches input/change events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "integer",
                        "description": "The ref number of the input element (from browser_snapshot output).",
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type into the input field.",
                    },
                },
                "required": ["ref", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a PNG screenshot of the current browser page. Saves to agent_file_system/screenshots/ and returns the file path.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_page_info",
            "description": "Get the current page URL and title from the attached browser tab.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the browser page up or down by 500 pixels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction: 'up' or 'down'. Defaults to 'down'.",
                    }
                },
                "required": [],
            },
        },
    },
]
