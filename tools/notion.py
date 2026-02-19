import json
import logging
import os
import re
import time
from pathlib import Path

import requests

from ._state import _AGENT_FS_BASE

logger = logging.getLogger(__name__)

_NOTION_PAGES_DIR = _AGENT_FS_BASE / "notion_pages"
_NOTION_PAGES_DIR.mkdir(exist_ok=True)
_READ_PREVIEW_LINES = 20
_CONTENT_ARG_MAX_CHARS = 200

_NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
_NOTION_BASE_URL = "https://api.notion.com/v1"
_NOTION_HEADERS = {
    "Authorization": f"Bearer {_NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]
_NOTION_BLOCK_CHUNK_SIZE = 100
_NOTION_RICH_TEXT_LIMIT = 2000


def _notion_request(method: str, endpoint: str, body: dict = None) -> dict:
    url = f"{_NOTION_BASE_URL}{endpoint}"
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.request(method, url, headers=_NOTION_HEADERS, json=body)
        if resp.ok:
            return resp.json()
        if resp.status_code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_RETRIES:
            error_msg = resp.json().get("message", resp.text)
            raise RuntimeError(f"Notion API error ({resp.status_code}): {error_msg}")
        # Determine backoff delay
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else _BACKOFF_SECONDS[attempt]
        else:
            delay = _BACKOFF_SECONDS[attempt]
        logger.warning(
            "Notion API %s %s returned %s, retrying in %ss (attempt %d/%d)",
            method, endpoint, resp.status_code, delay, attempt + 1, _MAX_RETRIES,
        )
        time.sleep(delay)
    # Should not reach here, but just in case
    error_msg = resp.json().get("message", resp.text)
    raise RuntimeError(f"Notion API error ({resp.status_code}): {error_msg}")


def _extract_title(page_or_db: dict) -> str:
    """Extract title from a Notion page or database object."""
    props = page_or_db.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            parts = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in parts)
    # Fallback for databases
    title_list = page_or_db.get("title", [])
    if title_list:
        return "".join(t.get("plain_text", "") for t in title_list)
    return "(Untitled)"


def _extract_block_text(block: dict) -> str:
    """Extract readable text from a Notion block."""
    btype = block.get("type", "")
    data = block.get(btype, {})
    rich_text = data.get("rich_text", [])
    text = "".join(t.get("plain_text", "") for t in rich_text)

    if btype in ("heading_1", "heading_2", "heading_3"):
        level = btype[-1]
        return f"{'#' * int(level)} {text}"
    elif btype == "to_do":
        checked = "x" if data.get("checked") else " "
        return f"[{checked}] {text}"
    elif btype == "bulleted_list_item":
        return f"• {text}"
    elif btype == "numbered_list_item":
        return f"- {text}"
    elif btype == "code":
        lang = data.get("language", "")
        return f"```{lang}\n{text}\n```"
    elif btype == "quote":
        return f"> {text}"
    elif btype in ("paragraph", "callout", "toggle"):
        return text
    elif btype == "divider":
        return "---"
    return text


def _split_rich_text(text: str) -> list[dict]:
    """Split text into rich_text segments of at most _NOTION_RICH_TEXT_LIMIT chars."""
    if len(text) <= _NOTION_RICH_TEXT_LIMIT:
        return [{"type": "text", "text": {"content": text}}]
    segments = []
    for i in range(0, len(text), _NOTION_RICH_TEXT_LIMIT):
        segments.append({"type": "text", "text": {"content": text[i:i + _NOTION_RICH_TEXT_LIMIT]}})
    return segments


def _text_to_paragraph_blocks(content: str) -> list[dict]:
    """Convert plain text to Notion paragraph blocks (split by newlines)."""
    blocks = []
    for line in content.split("\n"):
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": _split_rich_text(line),
            },
        })
    return blocks


def _append_blocks_chunked(block_id: str, blocks: list[dict]) -> None:
    """Append blocks to a Notion block in chunks of _NOTION_BLOCK_CHUNK_SIZE."""
    for i in range(0, len(blocks), _NOTION_BLOCK_CHUNK_SIZE):
        chunk = blocks[i:i + _NOTION_BLOCK_CHUNK_SIZE]
        _notion_request("PATCH", f"/blocks/{block_id}/children", {"children": chunk})


def _resolve_content(content: str = "", file_path: str = "") -> str:
    """Return content from either the raw string or by reading a file from agent_file_system."""
    if file_path:
        requested = Path(file_path)
        if requested.is_absolute():
            full_path = requested
        else:
            full_path = (_AGENT_FS_BASE / requested).resolve()
        try:
            full_path.relative_to(_AGENT_FS_BASE)
        except ValueError:
            raise RuntimeError(f"Access denied: path '{file_path}' is outside agent_file_system")
        if not full_path.exists():
            raise RuntimeError(f"File not found: {file_path}")
        return full_path.read_text(encoding="utf-8")
    return content


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower().strip())
    return re.sub(r'[\s-]+', '_', slug)[:80] or "untitled"


def search_notion(query: str, filter_type: str = "") -> str:
    body = {"query": query}
    if filter_type in ("page", "database"):
        body["filter"] = {"value": filter_type, "property": "object"}
    data = _notion_request("POST", "/search", body)
    results = data.get("results", [])
    if not results:
        return "No results found in Notion."
    output = []
    for r in results:
        title = _extract_title(r)
        obj_type = r.get("object", "unknown")
        obj_id = r.get("id", "")
        output.append(f"Title: {title}\nType: {obj_type}\nID: {obj_id}")
    return "\n---\n".join(output)


def read_notion_page(page_id: str) -> str:
    def _read_blocks(block_id: str, depth: int = 0) -> list[str]:
        data = _notion_request("GET", f"/blocks/{block_id}/children")
        lines = []
        indent = "  " * depth
        for block in data.get("results", []):
            text = _extract_block_text(block)
            if text:
                lines.append(f"{indent}{text}")
            if block.get("has_children"):
                lines.extend(_read_blocks(block["id"], depth + 1))
        return lines

    # Fetch page metadata for title
    try:
        page_meta = _notion_request("GET", f"/pages/{page_id}")
        title = _extract_title(page_meta)
    except RuntimeError:
        title = page_id

    lines = _read_blocks(page_id)
    if not lines:
        return "(Empty page)"

    full_text = "\n".join(lines)
    total_lines = len(lines)

    # Save to local file
    filename = f"{_slugify(title)}_{page_id[:8]}.md"
    local_path = _NOTION_PAGES_DIR / filename
    local_path.write_text(full_text, encoding="utf-8")
    rel_path = local_path.relative_to(_AGENT_FS_BASE)

    # Return preview + metadata
    preview_lines = lines[:_READ_PREVIEW_LINES]
    preview = "\n".join(preview_lines)
    result = (
        f"Page: {title}\n"
        f"Total lines: {total_lines}\n"
        f"Saved to: {rel_path}\n"
        f"\n--- Preview (first {len(preview_lines)} lines) ---\n"
        f"{preview}"
    )
    if total_lines > _READ_PREVIEW_LINES:
        result += (
            f"\n\n--- Content truncated. Use fs_read_file(path=\"{rel_path}\", offset={_READ_PREVIEW_LINES}, limit=50) "
            f"to read more. ---"
        )
    return result


def create_notion_page(parent_id: str, title: str, content: str = "", file_path: str = "") -> str:
    if content and len(content) > _CONTENT_ARG_MAX_CHARS:
        return (
            f"Error: content exceeds {_CONTENT_ARG_MAX_CHARS} characters ({len(content)} chars). "
            "Write the content to a file in agent_file_system first using fs_write_file, "
            "then pass the file_path instead."
        )
    resolved = _resolve_content(content, file_path)
    all_children = _text_to_paragraph_blocks(resolved) if resolved else []
    # Send at most the first chunk inline; append the rest after creation
    first_chunk = all_children[:_NOTION_BLOCK_CHUNK_SIZE]
    remaining = all_children[_NOTION_BLOCK_CHUNK_SIZE:]
    # Try to detect if parent is a database by querying it
    body = {
        "parent": {"page_id": parent_id},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}]
        },
        "children": first_chunk,
    }
    try:
        page = _notion_request("POST", "/pages", body)
    except RuntimeError:
        # Parent might be a database — retry with database_id
        body["parent"] = {"database_id": parent_id}
        body["properties"] = {
            "Name": {"title": [{"type": "text", "text": {"content": title}}]},
        }
        page = _notion_request("POST", "/pages", body)
    if remaining:
        _append_blocks_chunked(page["id"], remaining)
    return f"Page created: {title}\nID: {page['id']}\nURL: {page.get('url', 'N/A')}"


def update_notion_page(page_id: str, content: str = "", file_path: str = "") -> str:
    if content and len(content) > _CONTENT_ARG_MAX_CHARS:
        return (
            f"Error: content exceeds {_CONTENT_ARG_MAX_CHARS} characters ({len(content)} chars). "
            "Write the content to a file in agent_file_system first using fs_write_file, "
            "then pass the file_path instead."
        )
    resolved = _resolve_content(content, file_path)
    if not resolved:
        return "Error: either content or file_path must be provided."
    blocks = _text_to_paragraph_blocks(resolved)
    _append_blocks_chunked(page_id, blocks)
    return f"Content appended to page {page_id}."


def query_notion_database(database_id: str, filter_json: str = "", sorts_json: str = "") -> str:
    body = {}
    if filter_json:
        body["filter"] = json.loads(filter_json)
    if sorts_json:
        body["sorts"] = json.loads(sorts_json)
    data = _notion_request("POST", f"/databases/{database_id}/query", body)
    results = data.get("results", [])
    if not results:
        return "No entries found in this database."
    output = []
    for entry in results:
        props = entry.get("properties", {})
        parts = [f"ID: {entry['id']}"]
        for prop_name, prop_val in props.items():
            ptype = prop_val.get("type", "")
            if ptype == "title":
                val = "".join(t.get("plain_text", "") for t in prop_val.get("title", []))
            elif ptype == "rich_text":
                val = "".join(t.get("plain_text", "") for t in prop_val.get("rich_text", []))
            elif ptype == "number":
                val = str(prop_val.get("number", ""))
            elif ptype == "select":
                sel = prop_val.get("select")
                val = sel.get("name", "") if sel else ""
            elif ptype == "multi_select":
                val = ", ".join(s.get("name", "") for s in prop_val.get("multi_select", []))
            elif ptype == "date":
                d = prop_val.get("date")
                val = d.get("start", "") if d else ""
            elif ptype == "checkbox":
                val = str(prop_val.get("checkbox", False))
            elif ptype == "url":
                val = prop_val.get("url", "") or ""
            elif ptype == "status":
                st = prop_val.get("status")
                val = st.get("name", "") if st else ""
            else:
                val = f"({ptype})"
            parts.append(f"{prop_name}: {val}")
        output.append("\n".join(parts))
    return "\n---\n".join(output)


def create_database_entry(database_id: str, properties_json: str) -> str:
    properties = json.loads(properties_json)
    body = {
        "parent": {"database_id": database_id},
        "properties": properties,
    }
    page = _notion_request("POST", "/pages", body)
    return f"Entry created\nID: {page['id']}\nURL: {page.get('url', 'N/A')}"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_notion",
            "description": "Search the user's Notion workspace for pages and databases by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find pages or databases in Notion.",
                    },
                    "filter_type": {
                        "type": "string",
                        "description": "Optional: 'page' or 'database' to narrow results to one type.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notion_page",
            "description": "Read a Notion page. The full content is saved to a local file in agent_file_system/notion_pages/ and a preview of the first 20 lines is returned. Use fs_read_file with offset/limit to read more content from the saved file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The ID of the Notion page to read.",
                    },
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_notion_page",
            "description": "Create a new page in Notion. IMPORTANT: The 'content' arg is limited to 200 characters (for short descriptions only). For anything longer, first write content to a file using fs_write_file, then pass the file_path here — the tool reads the file directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "The ID of the parent page or database.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the new page.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Short plain-text content (max 200 chars). For longer content, use file_path instead.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a file in agent_file_system whose contents will be used as the page body. REQUIRED for any content over 200 characters.",
                    },
                },
                "required": ["parent_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_notion_page",
            "description": "Append content to an existing Notion page. IMPORTANT: The 'content' arg is limited to 200 characters (for short text only). For anything longer, first write content to a file using fs_write_file, then pass the file_path here — the tool reads the file directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The ID of the Notion page to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Short plain-text content to append (max 200 chars). For longer content, use file_path instead.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a file in agent_file_system whose contents will be appended to the page. REQUIRED for any content over 200 characters.",
                    },
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_notion_database",
            "description": "Query a Notion database to list its entries. Supports optional filters and sorts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the Notion database to query.",
                    },
                    "filter_json": {
                        "type": "string",
                        "description": "Optional JSON string of a Notion filter object.",
                    },
                    "sorts_json": {
                        "type": "string",
                        "description": "Optional JSON string of a Notion sorts array.",
                    },
                },
                "required": ["database_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_database_entry",
            "description": "Create a new entry (row) in a Notion database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "The ID of the Notion database.",
                    },
                    "properties_json": {
                        "type": "string",
                        "description": "JSON string of Notion property values for the new entry (e.g. '{\"Name\": {\"title\": [{\"text\": {\"content\": \"My Task\"}}]}}').",
                    },
                },
                "required": ["database_id", "properties_json"],
            },
        },
    },
]
