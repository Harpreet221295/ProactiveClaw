import json

from ._state import QUEUE_FILE, set_current_session_id
from .web import tavily_search, SCHEMA as _web_schema
from .scheduling import schedule_notifications, get_current_datetime, SCHEMA as _scheduling_schema
from .calendar import list_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event, SCHEMA as _calendar_schema
from .gmail import list_emails, read_email, send_email, reply_to_email, SCHEMA as _gmail_schema
from .notion import search_notion, read_notion_page, create_notion_page, update_notion_page, query_notion_database, create_database_entry, SCHEMA as _notion_schema
from .filesystem import fs_list_files, fs_read_file, fs_write_file, fs_create_directory, fs_delete, fs_move, fs_search_file, SCHEMA as _filesystem_schema
from .data import fs_read_json, fs_write_json, fs_read_csv, fs_write_csv, SCHEMA as _data_schema
from .charts import generate_chart, SCHEMA as _charts_schema

TOOLS_SCHEMA = (
    _web_schema
    + _scheduling_schema
    + _calendar_schema
    + _gmail_schema
    + _notion_schema
    + _filesystem_schema
    + _data_schema
    + _charts_schema
)

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
    "fs_search_file": fs_search_file,
    "fs_read_json": fs_read_json,
    "fs_write_json": fs_write_json,
    "fs_read_csv": fs_read_csv,
    "fs_write_csv": fs_write_csv,
    "generate_chart": generate_chart,
}


def dispatch_tool_call(name: str, args: str) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    parsed_args = json.loads(args)
    return func(**parsed_args)
