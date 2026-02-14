import base64
import json
import os
from email.mime.text import MIMEText
from tavily import TavilyClient
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

QUEUE_FILE = os.path.join(os.path.dirname(__file__), "queue.json")
_TOKEN_FILE = os.path.join(os.path.dirname(__file__), "token.json")
_CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")
_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _get_google_creds():
    creds = None
    if os.path.exists(_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(_TOKEN_FILE, _GOOGLE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_FILE, _GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


def _get_calendar_service():
    return build("calendar", "v3", credentials=_get_google_creds())


def _get_gmail_service():
    return build("gmail", "v1", credentials=_get_google_creds())

import requests
from pathlib import Path

_NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
_NOTION_BASE_URL = "https://api.notion.com/v1"
_NOTION_HEADERS = {
    "Authorization": f"Bearer {_NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# Agent file system base directory
_AGENT_FS_BASE = Path(__file__).parent / "agent_file_system"
_AGENT_FS_BASE.mkdir(exist_ok=True)

_current_session_id: str | None = None


def set_current_session_id(session_id: str) -> None:
    global _current_session_id
    _current_session_id = session_id


def tavily_search(query: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=5)
    output = []
    for r in results.get("results", []):
        output.append(f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n")
    return "\n---\n".join(output) if output else "No results found."


def schedule_notifications(notifications: list[dict]) -> str:
    if len(notifications) > 5:
        return "Error: maximum 5 notifications allowed."

    # Load existing queue
    queue = []
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []

    for entry in notifications:
        queue.append({
            "timestamp": entry["timestamp"],
            "message": entry["message"],
            "session_id": _current_session_id,
        })

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

    return f"Scheduled {len(notifications)} notification(s)."


def get_current_datetime() -> str:
    from datetime import datetime
    now = datetime.now().astimezone()
    return now.isoformat()


def list_calendar_events(time_min: str, time_max: str, max_results: int = 10) -> str:
    service = _get_calendar_service()
    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    events = events_result.get("items", [])
    if not events:
        return "No events found in the given time range."
    output = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))
        desc = event.get("description", "")
        output.append(
            f"Title: {event.get('summary', '(No title)')}\n"
            f"Start: {start}\nEnd: {end}\n"
            f"Description: {desc}\nID: {event['id']}"
        )
    return "\n---\n".join(output)


def create_calendar_event(title: str, start_time: str, end_time: str, description: str = "", location: str = "") -> str:
    service = _get_calendar_service()
    body = {
        "summary": title,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    event = service.events().insert(calendarId="primary", body=body).execute()
    return f"Event created: {event.get('summary')} (ID: {event['id']})\nLink: {event.get('htmlLink')}"


def update_calendar_event(event_id: str, title: str = None, start_time: str = None, end_time: str = None, description: str = None, location: str = None) -> str:
    service = _get_calendar_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    if title is not None:
        event["summary"] = title
    if start_time is not None:
        event["start"] = {"dateTime": start_time}
    if end_time is not None:
        event["end"] = {"dateTime": end_time}
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    start = updated["start"].get("dateTime", updated["start"].get("date"))
    return f"Updated: {updated.get('summary')} | Start: {start} | ID: {updated['id']}"


def delete_calendar_event(event_id: str) -> str:
    service = _get_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return f"Event {event_id} deleted successfully."


# ── Gmail tools ──────────────────────────────────────────────────────

def _parse_email_headers(headers: list[dict], *names: str) -> dict[str, str]:
    result = {}
    lower_names = {n.lower(): n for n in names}
    for h in headers:
        key = h["name"].lower()
        if key in lower_names:
            result[lower_names[key]] = h["value"]
    return result


def list_emails(query: str = "", max_results: int = 10) -> str:
    service = _get_gmail_service()
    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    messages = resp.get("messages", [])
    if not messages:
        return "No emails found."
    output = []
    for msg_stub in messages:
        msg = service.users().messages().get(
            userId="me", id=msg_stub["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        hdrs = _parse_email_headers(msg.get("payload", {}).get("headers", []),
                                     "From", "Subject", "Date")
        output.append(
            f"From: {hdrs.get('From', '?')}\n"
            f"Subject: {hdrs.get('Subject', '(no subject)')}\n"
            f"Date: {hdrs.get('Date', '?')}\n"
            f"Snippet: {msg.get('snippet', '')}\n"
            f"ID: {msg['id']}"
        )
    return "\n---\n".join(output)


def read_email(message_id: str) -> str:
    service = _get_gmail_service()
    msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()
    hdrs = _parse_email_headers(msg.get("payload", {}).get("headers", []),
                                 "From", "To", "Subject", "Date")
    # Extract plain-text body
    body = ""
    payload = msg.get("payload", {})
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    else:
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                break
    if not body:
        body = msg.get("snippet", "(could not extract body)")
    return (
        f"From: {hdrs.get('From', '?')}\n"
        f"To: {hdrs.get('To', '?')}\n"
        f"Subject: {hdrs.get('Subject', '(no subject)')}\n"
        f"Date: {hdrs.get('Date', '?')}\n\n"
        f"{body}"
    )


def send_email(to: str, subject: str, body: str) -> str:
    service = _get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return f"Email sent successfully (ID: {sent['id']})"


def reply_to_email(message_id: str, body: str) -> str:
    service = _get_gmail_service()
    original = service.users().messages().get(
        userId="me", id=message_id, format="metadata",
        metadataHeaders=["From", "Subject", "Message-ID"],
    ).execute()
    hdrs = _parse_email_headers(original.get("payload", {}).get("headers", []),
                                 "From", "Subject", "Message-ID")
    subject = hdrs.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    message = MIMEText(body)
    message["to"] = hdrs.get("From", "")
    message["subject"] = subject
    message["In-Reply-To"] = hdrs.get("Message-ID", "")
    message["References"] = hdrs.get("Message-ID", "")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": original.get("threadId")},
    ).execute()
    return f"Reply sent successfully (ID: {sent['id']})"


# ── Notion tools ─────────────────────────────────────────────────────

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


# ── File System tools ────────────────────────────────────────────────

def _validate_agent_path(path_str: str) -> Path:
    """Ensure the path is within agent_file_system and return absolute Path."""
    requested = Path(path_str)
    if requested.is_absolute():
        full_path = requested
    else:
        full_path = (_AGENT_FS_BASE / requested).resolve()

    # Security check: ensure it's within the base directory
    try:
        full_path.relative_to(_AGENT_FS_BASE)
    except ValueError:
        raise RuntimeError(f"Access denied: path '{path_str}' is outside agent_file_system")

    return full_path


def fs_list_files(path: str = ".") -> str:
    """List files and directories in the given path within agent_file_system."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"Path does not exist: {path}"
    if not target.is_dir():
        return f"Not a directory: {path}"

    items = []
    for item in sorted(target.iterdir()):
        rel_path = item.relative_to(_AGENT_FS_BASE)
        item_type = "DIR" if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else "-"
        items.append(f"[{item_type}] {rel_path} ({size} bytes)" if size != "-" else f"[{item_type}] {rel_path}")

    return "\n".join(items) if items else "(empty directory)"


def fs_read_file(path: str) -> str:
    """Read the contents of a file in agent_file_system."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"

    try:
        content = target.read_text(encoding="utf-8")
        return content
    except UnicodeDecodeError:
        return f"Error: {path} is not a text file (binary data)"
    except Exception as e:
        return f"Error reading file: {e}"


def fs_write_file(path: str, content: str, append: bool = False) -> str:
    """Write or append content to a file in agent_file_system."""
    target = _validate_agent_path(path)

    # Create parent directories if needed
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if append:
            target.write_text(target.read_text(encoding="utf-8") + content, encoding="utf-8")
            return f"Content appended to {path}"
        else:
            target.write_text(content, encoding="utf-8")
            return f"File written: {path} ({len(content)} characters)"
    except Exception as e:
        return f"Error writing file: {e}"


def fs_create_directory(path: str) -> str:
    """Create a new directory in agent_file_system."""
    target = _validate_agent_path(path)

    if target.exists():
        return f"Path already exists: {path}"

    try:
        target.mkdir(parents=True, exist_ok=True)
        return f"Directory created: {path}"
    except Exception as e:
        return f"Error creating directory: {e}"


def fs_delete(path: str) -> str:
    """Delete a file or empty directory in agent_file_system."""
    target = _validate_agent_path(path)

    if not target.exists():
        return f"Path does not exist: {path}"

    try:
        if target.is_file():
            target.unlink()
            return f"File deleted: {path}"
        elif target.is_dir():
            if any(target.iterdir()):
                return f"Error: directory is not empty: {path} (delete contents first)"
            target.rmdir()
            return f"Directory deleted: {path}"
    except Exception as e:
        return f"Error deleting: {e}"


def fs_move(source: str, destination: str) -> str:
    """Move or rename a file/directory within agent_file_system."""
    src = _validate_agent_path(source)
    dst = _validate_agent_path(destination)

    if not src.exists():
        return f"Source does not exist: {source}"
    if dst.exists():
        return f"Destination already exists: {destination}"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Moved {source} → {destination}"
    except Exception as e:
        return f"Error moving: {e}"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web for current information on a given query. Use this when you need up-to-date facts, news, or information not in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_notifications",
            "description": "Schedule follow-up notifications to re-engage the user later. Use this when the conversation is ending and there are pending topics, reminders, or follow-ups the user would benefit from. Each notification has a timestamp (ISO 8601) and a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notifications": {
                        "type": "array",
                        "description": "List of notifications to schedule (max 5).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {
                                    "type": "string",
                                    "description": "ISO 8601 datetime when the notification should fire (e.g. '2025-01-15T14:30:00').",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "The notification message to show the user.",
                                },
                            },
                            "required": ["timestamp", "message"],
                        },
                    }
                },
                "required": ["notifications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time with timezone. Use this whenever you need to know the exact current time, e.g. for scheduling events or checking relative times.",
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
            "name": "list_calendar_events",
            "description": "List events from the user's Google Calendar within a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {
                        "type": "string",
                        "description": "Start of time range in ISO 8601 format (e.g. '2025-06-15T00:00:00-08:00').",
                    },
                    "time_max": {
                        "type": "string",
                        "description": "End of time range in ISO 8601 format.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default 10).",
                    },
                },
                "required": ["time_min", "time_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on the user's Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title/summary of the event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Event start time in ISO 8601 format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Event end time in ISO 8601 format.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional event description.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional event location.",
                    },
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update an existing Google Calendar event. Only provided fields will be changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The ID of the event to update.",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title for the event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "New start time in ISO 8601 format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "New end time in ISO 8601 format.",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description for the event.",
                    },
                    "location": {
                        "type": "string",
                        "description": "New location for the event.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete an event from the user's Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The ID of the event to delete.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": "List emails from the user's Gmail inbox. Supports Gmail search queries (e.g. 'is:unread', 'from:someone@example.com', 'subject:meeting').",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query to filter emails (e.g. 'is:unread', 'from:boss@company.com'). Empty string returns recent emails.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read the full content of a specific email by its message ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the email to read.",
                    },
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send a new email from the user's Gmail account. Only use this when the user explicitly asks to send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_email",
            "description": "Reply to an existing email thread. Only use this when the user explicitly asks to reply to an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the email to reply to.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The reply body text.",
                    },
                },
                "required": ["message_id", "body"],
            },
        },
    },
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
    {
        "type": "function",
        "function": {
            "name": "fs_list_files",
            "description": "List files and directories in agent_file_system. Use '.' for root or provide a relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path within agent_file_system to list (default: '.' for root).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read the contents of a text file in agent_file_system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file within agent_file_system.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_file",
            "description": "Write or append content to a file in agent_file_system. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file within agent_file_system.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing file instead of overwriting (default: false).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_create_directory",
            "description": "Create a new directory in agent_file_system. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to create within agent_file_system.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_delete",
            "description": "Delete a file or empty directory in agent_file_system. Directories must be empty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory to delete.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_move",
            "description": "Move or rename a file or directory within agent_file_system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Current path of the file/directory.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "New path for the file/directory.",
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "tavily_search": tavily_search,
    "schedule_notifications": schedule_notifications,
    "get_current_datetime": get_current_datetime,
    "list_calendar_events": list_calendar_events,
    "create_calendar_event": create_calendar_event,
    "update_calendar_event": update_calendar_event,
    "delete_calendar_event": delete_calendar_event,
    "list_emails": list_emails,
    "read_email": read_email,
    "send_email": send_email,
    "reply_to_email": reply_to_email,
    "search_notion": search_notion,
    "read_notion_page": read_notion_page,
    "create_notion_page": create_notion_page,
    "update_notion_page": update_notion_page,
    "query_notion_database": query_notion_database,
    "create_database_entry": create_database_entry,
    "fs_list_files": fs_list_files,
    "fs_read_file": fs_read_file,
    "fs_write_file": fs_write_file,
    "fs_create_directory": fs_create_directory,
    "fs_delete": fs_delete,
    "fs_move": fs_move,
}


def dispatch_tool_call(name: str, args: str) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    parsed_args = json.loads(args)
    return func(**parsed_args)
