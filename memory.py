"""SimpleMem MCP client — JSON-RPC 2.0 over Streamable HTTP."""

import os
import uuid

import requests

_SIMPLEMEM_URL = os.getenv("SIMPLEMEM_URL", "")
_SIMPLEMEM_TOKEN = os.getenv("SIMPLEMEM_TOKEN", "")

_client: "SimpleMemClient | None" = None


class SimpleMemClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session_id: str | None = None

    def initialize(self) -> bool:
        """Send MCP initialize handshake. Returns True on success."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ProactiveClaw", "version": "1.0.0"},
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.session_id = resp.headers.get("Mcp-Session-Id")

            # Send initialized notification
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            requests.post(
                f"{self.base_url}/mcp",
                json=notif,
                headers=self._headers(),
                timeout=10,
            )
            return "result" in data
        except Exception as e:
            print(f"[memory] MCP initialize failed: {e}")
            return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool via JSON-RPC 2.0. Lazy-initializes if needed."""
        if self.session_id is None:
            if not self.initialize():
                return "Error: SimpleMem is not available"

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return f"Error: {data['error'].get('message', 'unknown error')}"
            result = data.get("result", {})
            content = result.get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else str(result)
        except Exception as e:
            return f"Error: SimpleMem call failed: {e}"

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers


def _get_client() -> SimpleMemClient | None:
    """Return the singleton client, or None if not configured."""
    global _client
    if not _SIMPLEMEM_URL:
        return None
    if _client is None:
        _client = SimpleMemClient(_SIMPLEMEM_URL, _SIMPLEMEM_TOKEN)
    return _client


def store_dialogues(dialogues: list[dict]) -> str:
    """Store conversation dialogues in SimpleMem via memory_add_batch."""
    client = _get_client()
    if client is None:
        return "SimpleMem not configured"
    return client.call_tool("memory_add_batch", {"dialogues": dialogues})


def query_memory(query: str) -> str:
    """Semantic search over long-term memory."""
    client = _get_client()
    if client is None:
        return "Long-term memory is not configured."
    return client.call_tool("memory_query", {"query": query})


def retrieve_memory(query: str) -> str:
    """Direct retrieval from long-term memory by topic/keyword."""
    client = _get_client()
    if client is None:
        return "Long-term memory is not configured."
    return client.call_tool("memory_retrieve", {"query": query})
