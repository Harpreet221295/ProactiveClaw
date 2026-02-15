import re
from pathlib import Path

from ._state import _AGENT_FS_BASE


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


def fs_read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read the contents of a file in agent_file_system. Supports line-based pagination."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"

    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)
        if offset or limit:
            end = offset + limit if limit else total
            lines = lines[offset:end]
            header = f"[Lines {offset + 1}-{min(offset + len(lines), total)} of {total}]\n"
            return header + "".join(lines)
        return "".join(lines)
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


def fs_search_file(path: str, query: str, is_regex: bool = False, max_results: int = 50) -> str:
    """Search for lines matching a query in a file within agent_file_system."""
    target = _validate_agent_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"Error: {path} is not a text file (binary data)"

    if is_regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"
        test = lambda line: pattern.search(line)
    else:
        query_lower = query.lower()
        test = lambda line: query_lower in line.lower()

    matches = []
    for i, line in enumerate(lines, start=1):
        if test(line):
            matches.append(f"{i}: {line}")
            if len(matches) >= max_results:
                break

    if not matches:
        return f"No matches found for '{query}' in {path}"

    header = f"[{len(matches)} match(es) in {path}, {len(lines)} total lines]\n"
    return header + "\n".join(matches)


SCHEMA = [
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
            "description": "Read the raw contents of any text-based file (txt, md, log, etc.) in agent_file_system. Supports line-based pagination with offset/limit for large files. For structured formats, prefer fs_read_json or fs_read_csv instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file within agent_file_system.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based). Default: 0 (start of file).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Default: 0 (all lines).",
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
            "description": "Write or append raw text content to a file (txt, md, log, etc.) in agent_file_system. Creates parent directories if needed. For structured formats, prefer fs_write_json or fs_write_csv instead.",
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
    {
        "type": "function",
        "function": {
            "name": "fs_search_file",
            "description": "Search for lines matching a text query or regex pattern in a file within agent_file_system. Returns matching lines with line numbers. Use this to find specific content in large files without reading the entire file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to search within agent_file_system.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search string or regex pattern to match against each line.",
                    },
                    "is_regex": {
                        "type": "boolean",
                        "description": "If true, treat query as a regex pattern. Default: false (plain text, case-insensitive).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return. Default: 50.",
                    },
                },
                "required": ["path", "query"],
            },
        },
    },
]
