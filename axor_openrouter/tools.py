from __future__ import annotations

from axor_core.contracts.envelope import ExecutionEnvelope

# Standard axor tool schemas in OpenAI function-calling format.
# Adapters may also receive ExtensionTool entries from the envelope;
# those are handled separately in build_tools().
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
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30).", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files by name pattern or grep for content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern."},
                    "path":    {"type": "string", "description": "Directory to search (default: .).", "default": "."},
                    "type":    {"type": "string", "enum": ["file", "content", "both"], "default": "both"},
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
                    "cwd":     {"type": "string", "description": "Base directory (default: .)."},
                },
                "required": ["pattern"],
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
}


def build_tools(envelope: ExecutionEnvelope) -> list[dict]:
    """
    Build the OpenAI-compatible tools list for a chat completion request.

    Sources (in order):
    1. Standard tool schemas for tools in envelope.capabilities.allowed_tools
    2. ExtensionTool entries from the envelope (registered by plugins)
    """
    tools: list[dict] = []

    for name in sorted(envelope.capabilities.allowed_tools):
        schema = TOOL_SCHEMAS.get(name)
        if schema is not None:
            tools.append(schema)
        # Unknown tool names (from extensions) are handled below

    # Extension tools — convert ExtensionTool → OpenAI function schema
    # The envelope carries extension_tools indirectly via capabilities;
    # adapters that need raw ExtensionTool access should read them from the
    # ExtensionBundle. Here we handle the common case where the name is in
    # allowed_tools but has no entry in TOOL_SCHEMAS.
    # (Extension tool schemas are passed in during session build.)

    return tools


def build_tools_with_extensions(
    envelope: ExecutionEnvelope,
    extension_tool_schemas: list[dict] | None = None,
) -> list[dict]:
    """
    Like build_tools() but also accepts pre-converted extension tool schemas.
    Extension tools not in TOOL_SCHEMAS but in allowed_tools are appended.
    """
    tools = build_tools(envelope)
    if extension_tool_schemas:
        # avoid duplicates
        existing_names = {t["function"]["name"] for t in tools if "function" in t}
        for schema in extension_tool_schemas:
            fn_name = schema.get("function", {}).get("name", "")
            if fn_name and fn_name not in existing_names:
                tools.append(schema)
    return tools
