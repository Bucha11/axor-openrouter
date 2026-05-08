from __future__ import annotations

import json
from typing import Any

from axor_core.contracts.envelope import ExecutionEnvelope


# ── Message construction ────────────────────────────────────────────────────────

def build_messages(
    envelope: ExecutionEnvelope,
    cache_hints: dict | None = None,
) -> list[dict]:
    """
    Convert an ExecutionEnvelope into an OpenAI-compatible messages list.

    Layout:
        1. [system]  skill/personality fragments + active constraints
        2. [user]    session context (working_summary + memory/parent_export)
           [assistant] acknowledgement stub (only if context is non-trivial)
        3. [user]    the actual task

    cache_hints overrides envelope.cache_hints when provided.
    When cache_hints.blocks contains "system", the system message content
    is structured as a list of text blocks with cache_control.
    """
    hints = cache_hints if cache_hints is not None else (envelope.cache_hints or {})
    messages: list[dict] = []

    # ─ 1. System message ──────────────────────────────────────────────────────────
    skill_parts: list[str] = [
        f.content
        for f in envelope.context.visible_fragments
        if f.kind == "skill"
    ]
    if envelope.context.active_constraints:
        constraints_text = "\n".join(
            f"- {c}" for c in envelope.context.active_constraints
        )
        skill_parts.append(f"Active execution constraints:\n{constraints_text}")

    if skill_parts:
        combined_system = "\n\n---\n\n".join(skill_parts)
        if "system" in hints.get("blocks", []):
            ttl = _ttl_to_seconds(hints.get("ttl", "5m"))
            system_content: Any = [
                {
                    "type": "text",
                    "text": combined_system,
                    "cache_control": {"type": "ephemeral", "ttl": ttl},
                }
            ]
        else:
            system_content = combined_system
        messages.append({"role": "system", "content": system_content})

    # ─ 2. Context message (history + memory) ────────────────────────────────────
    ctx_parts: list[str] = []

    summary = envelope.context.working_summary or ""
    if summary.strip() and summary.strip() != envelope.task.strip():
        ctx_parts.append(f"<session_context>\n{summary}\n</session_context>")

    for f in envelope.context.visible_fragments:
        if f.kind == "memory":
            ctx_parts.append(f"<memory>\n{f.content}\n</memory>")
        elif f.kind == "parent_export":
            ctx_parts.append(f"<parent_result>\n{f.content}\n</parent_result>")
        elif f.kind == "fact":
            ctx_parts.append(f.content)

    if ctx_parts:
        ctx_text = "\n\n".join(ctx_parts)
        if "context_top_k" in hints.get("blocks", []):
            ttl = _ttl_to_seconds(hints.get("ttl", "5m"))
            ctx_content: Any = [
                {
                    "type": "text",
                    "text": ctx_text,
                    "cache_control": {"type": "ephemeral", "ttl": ttl},
                }
            ]
        else:
            ctx_content = ctx_text
        messages.append({"role": "user", "content": ctx_content})
        messages.append({"role": "assistant", "content": "Understood."})

    # ─ 3. Task ───────────────────────────────────────────────────────────────────
    messages.append({"role": "user", "content": envelope.task})
    return messages


def build_tool_request_messages(
    messages: list[dict],
    assistant_content: str | None,
    tool_calls: list[dict],
) -> list[dict]:
    """
    Append the assistant’s tool_calls turn to the conversation.
    Returns a new list (does not mutate the input).
    """
    new = list(messages)
    assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
    if assistant_content:
        assistant_msg["content"] = assistant_content
    new.append(assistant_msg)
    return new


def append_tool_result(
    messages: list[dict],
    tool_call_id: str,
    result: Any,
) -> None:
    """
    Append a tool result message in-place.
    Converts the result to a string suitable for the OpenAI tool role.
    """
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _format_tool_result(result),
    })


def _format_tool_result(result: Any) -> str:
    """Serialise a tool result for inclusion in a tool-role message."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and result.get("error") == "tool_denied":
        return f"Tool denied: {result.get('reason', 'policy violation')}"
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


# ── Helpers ────────────────────────────────────────────────────────────────────────

def _ttl_to_seconds(ttl: str) -> int:
    """Convert TTL string to seconds. Supports '5m', '1h', raw integer strings."""
    ttl = str(ttl).strip().lower()
    if ttl.endswith("h"):
        return int(ttl[:-1]) * 3600
    if ttl.endswith("m"):
        return int(ttl[:-1]) * 60
    if ttl.endswith("s"):
        return int(ttl[:-1])
    try:
        return int(ttl)
    except ValueError:
        return 300  # fallback: 5 minutes
