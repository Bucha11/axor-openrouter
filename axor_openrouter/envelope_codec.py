from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from axor_core.contracts.envelope import ExecutionEnvelope

if TYPE_CHECKING:
    from axor_core.contracts.envelope import CacheHints


def build_messages(
    envelope: ExecutionEnvelope,
    cache_hints: "CacheHints | None" = None,
) -> list[dict]:
    messages: list[dict] = []

    # System message: skill/personality fragments + constraints
    skill_parts: list[str] = [
        f.content
        for f in envelope.context.visible_fragments
        if f.kind == "skill"
    ]
    if envelope.context.active_constraints:
        constraints_text = "\n".join(f"- {c}" for c in envelope.context.active_constraints)
        skill_parts.append(f"Active execution constraints:\n{constraints_text}")

    if skill_parts:
        messages.append({
            "role": "system",
            "content": "\n\n---\n\n".join(skill_parts),
            # cache_control: OpenRouter forwards this to Anthropic prompt caching.
            # The system message is stable across turns (skills don't change mid-session)
            # so it's the highest-value caching target.
            "cache_control": {"type": "ephemeral"},
        })

    # Context message: working summary + memory/parent exports
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
        messages.append({
            "role": "user",
            "content": "\n\n".join(ctx_parts),
            "cache_control": {"type": "ephemeral"},
        })
        messages.append({"role": "assistant", "content": "Understood."})

    # envelope.task may be a plain string or a content array (multimodal).
    # Pass it through unchanged — OpenRouter accepts both.
    messages.append({"role": "user", "content": envelope.task})
    return messages


def build_tool_request_messages(
    messages: list[dict],
    assistant_content: str | None,
    tool_calls: list[dict],
) -> list[dict]:
    new = list(messages)
    assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
    if assistant_content:
        assistant_msg["content"] = assistant_content
    new.append(assistant_msg)
    return new


def append_tool_result(messages: list[dict], tool_call_id: str, result: Any) -> None:
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _format_tool_result(result),
    })


def _format_tool_result(result: Any) -> str:
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
