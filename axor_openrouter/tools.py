from __future__ import annotations

from axor_core.contracts.envelope import ExecutionEnvelope

TOOL_SCHEMAS: dict[str, dict] = {
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                },
                "required": ["path"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "edit": {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Replace an exact string in an existing file. "
                "Prefer this over write when modifying existing files — "
                "only the changed portion needs to be generated. "
                "old_string must match exactly (including whitespace/indentation). "
                "If old_string appears multiple times, set replace_all=true or add more context to make it unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string", "description": "File path to edit."},
                    "old_string":  {"type": "string", "description": "Exact string to find and replace."},
                    "new_string":  {"type": "string", "description": "Replacement string."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false).", "default": False},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for files by name or grep for content inside files. "
                "Uses ripgrep (rg) when available — fast, symlink-aware, respects .gitignore. "
                "Falls back to grep -R (also follows symlinks). "
                "File search uses find -L (follows symlinks)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":     {"type": "string", "description": "Search pattern (regex for content, glob for file names)."},
                    "path":        {"type": "string", "description": "Directory to search.", "default": "."},
                    "type":        {"type": "string", "enum": ["file", "content", "both"], "default": "both",
                                    "description": "What to search: file names, file contents, or both."},
                    "include":     {"type": "string", "description": "Limit content search to files matching this glob, e.g. '*.py' or '*.{ts,tsx}'."},
                    "context":     {"type": "integer", "description": "Lines of context around each content match (0–10).", "default": 0},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive search.", "default": False},
                },
                "required": ["pattern"],
            },
        },
    },
    "glob": {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
                    "cwd":     {"type": "string", "description": "Base directory.", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    "fetch": {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": (
                "Fetch the content of an HTTP/HTTPS URL. "
                "Returns the response body (capped at max_bytes). "
                "Useful for reading documentation, APIs, or any web resource."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string", "description": "HTTP/HTTPS URL to fetch."},
                    "max_bytes": {"type": "integer", "description": "Max bytes to return (default: 65536).", "default": 65536},
                    "timeout":   {"type": "integer", "description": "Request timeout in seconds (default: 15).", "default": 15},
                },
                "required": ["url"],
            },
        },
    },
    "spawn_child": {
        "type": "function",
        "function": {
            "name": "spawn_child",
            "description": "Delegate a subtask to a child agent node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task":         {"type": "string", "description": "Subtask description."},
                    "context_hint": {"type": "string", "description": "Optional context hint.", "default": ""},
                },
                "required": ["task"],
            },
        },
    },
    "todo_write": {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Replace the session todo list. Use this to track tasks and progress.\n"
                "Always pass the complete list — this is a full replace, not a patch.\n"
                "Statuses: pending | in_progress | completed\n"
                "Priorities: high | medium | low\n"
                "Call todo_read first if you need the current list before modifying it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Complete replacement todo list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":       {"type": "string", "description": "Stable identifier, e.g. '1', '2'."},
                                "content":  {"type": "string", "description": "Task description."},
                                "status":   {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["id", "content", "status", "priority"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    "todo_read": {
        "type": "function",
        "function": {
            "name": "todo_read",
            "description": "Return the current session todo list.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
}


def build_tools_with_extensions(
    envelope: ExecutionEnvelope,
    extension_tool_schemas: list[dict] | None = None,
) -> list[dict]:
    tools: list[dict] = []
    for name in sorted(envelope.capabilities.allowed_tools):
        schema = TOOL_SCHEMAS.get(name)
        if schema is not None:
            tools.append(schema)

    if extension_tool_schemas:
        existing_names = {t["function"]["name"] for t in tools if "function" in t}
        for schema in extension_tool_schemas:
            fn_name = schema.get("function", {}).get("name", "")
            if fn_name and fn_name not in existing_names:
                tools.append(schema)
    return tools
