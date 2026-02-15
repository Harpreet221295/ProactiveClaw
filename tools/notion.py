import json
import os

import requests

_NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
_NOTION_BASE_URL = "https://api.notion.com/v1"
_NOTION_HEADERS = {
    "Authorization": f"Bearer {_NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def _notion_request(method: str, endpoint: str, body: dict = None) -> dict:
    url = f"{_NOTION_BASE_URL}{endpoint}"
    resp = requests.request(method, url, headers=_NOTION_HEADERS, json=body)
    if not resp.ok:
        error_msg = resp.json().get("message", resp.text)
        raise RuntimeError(f"Notion API error ({resp.status_code}): {error_msg}")
    return resp.json()


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


def _text_to_paragraph_blocks(content: str) -> list[dict]:
    """Convert plain text to Notion paragraph blocks (split by newlines)."""
    blocks = []
    for line in content.split("\n"):
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": line}}]
            },
        })
    return blocks


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

    lines = _read_blocks(page_id)
    return "\n".join(lines) if lines else "(Empty page)"


def create_notion_page(parent_id: str, title: str, content: str = "") -> str:
    children = _text_to_paragraph_blocks(content) if content else []
    # Try to detect if parent is a database by querying it
    body = {
        "parent": {"page_id": parent_id},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}]
        },
        "children": children,
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
    return f"Page created: {title}\nID: {page['id']}\nURL: {page.get('url', 'N/A')}"


def update_notion_page(page_id: str, content: str) -> str:
    blocks = _text_to_paragraph_blocks(content)
    _notion_request("PATCH", f"/blocks/{page_id}/children", {"children": blocks})
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
            "description": "Read the full content of a Notion page by its page ID.",
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
            "description": "Create a new page in Notion under a parent page or database.",
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
                        "description": "Optional plain-text content for the page body.",
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
            "description": "Append content to an existing Notion page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "The ID of the Notion page to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Plain-text content to append to the page.",
                    },
                },
                "required": ["page_id", "content"],
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
