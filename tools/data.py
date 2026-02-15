import csv
import io
import json

from .filesystem import _validate_agent_path


def fs_read_json(path: str) -> str:
    """Read and pretty-print a JSON file from agent_file_system."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"Invalid JSON in {path}: {e}"
    except Exception as e:
        return f"Error reading JSON file: {e}"


def fs_write_json(path: str, data: str) -> str:
    """Write a JSON value to a file in agent_file_system. `data` is a JSON string that will be validated and pretty-printed before writing."""
    target = _validate_agent_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        parsed = json.loads(data)
        target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"JSON written: {path}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON data: {e}"
    except Exception as e:
        return f"Error writing JSON file: {e}"


def fs_read_csv(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a CSV file and return its contents as a JSON array of objects (using header row as keys). Supports row-based pagination."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"
    try:
        text = target.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        total = len(rows)
        if offset or limit:
            end = offset + limit if limit else total
            rows = rows[offset:end]
            header = f"[Rows {offset + 1}-{offset + len(rows)} of {total}]\n"
            return header + json.dumps(rows, indent=2, ensure_ascii=False)
        return json.dumps(rows, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error reading CSV file: {e}"


def fs_write_csv(path: str, data: str) -> str:
    """Write a CSV file from a JSON array of objects. `data` is a JSON string like '[{"col1": "val1", "col2": "val2"}, ...]'. Keys of the first object become the header row."""
    target = _validate_agent_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = json.loads(data)
        if not isinstance(rows, list) or not rows:
            return "Error: data must be a non-empty JSON array of objects"
        if not isinstance(rows[0], dict):
            return "Error: each row must be a JSON object"
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        target.write_text(output.getvalue(), encoding="utf-8")
        return f"CSV written: {path} ({len(rows)} rows)"
    except json.JSONDecodeError as e:
        return f"Invalid JSON data: {e}"
    except Exception as e:
        return f"Error writing CSV file: {e}"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_json",
            "description": "Read and parse a JSON file from agent_file_system. Returns pretty-printed JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the JSON file within agent_file_system.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_json",
            "description": "Write a JSON value to a file in agent_file_system. The data is validated and pretty-printed before writing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the JSON file within agent_file_system.",
                    },
                    "data": {
                        "type": "string",
                        "description": "The JSON data to write (as a JSON string, e.g. '{\"key\": \"value\"}' or '[1, 2, 3]').",
                    },
                },
                "required": ["path", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read_csv",
            "description": "Read a CSV file and return its contents as a JSON array of objects, using the header row as keys. Supports row-based pagination with offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CSV file within agent_file_system.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Row number to start reading from (0-based, excludes header). Default: 0 (first data row).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return. Default: 0 (all rows).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_csv",
            "description": "Write a CSV file from structured data. Accepts a JSON array of objects — keys of the first object become the header row.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CSV file within agent_file_system.",
                    },
                    "data": {
                        "type": "string",
                        "description": "JSON array of objects, e.g. '[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]'.",
                    },
                },
                "required": ["path", "data"],
            },
        },
    },
]
