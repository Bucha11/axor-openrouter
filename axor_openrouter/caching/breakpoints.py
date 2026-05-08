from __future__ import annotations


def apply_cache_control(
    messages: list[dict],
    tools: list[dict],
    blocks: list[str],
    ttl_seconds: int,
) -> tuple[list[dict], list[dict]]:
    """
    Place cache_control breakpoints on the designated blocks.

    OpenRouter passes cache_control through to providers that support it
    (Anthropic models for prompt caching, OpenAI for some endpoints).
    Providers that do not support it silently ignore the field.

    Args:
        messages:    current messages list (not mutated; returns a new list)
        tools:       current tools list (not mutated; returns a new list)
        blocks:      which blocks to mark: "system", "tools", "context_top_k"
        ttl_seconds: TTL in seconds for ephemeral cache entries

    Returns:
        (messages, tools) with cache_control applied
    """
    cc = {"type": "ephemeral", "ttl": ttl_seconds}
    messages = list(messages)  # shallow copy
    tools    = list(tools)

    if "system" in blocks:
        messages = _apply_to_system(messages, cc)

    if "tools" in blocks and tools:
        # cache_control on the last tool causes caching up to and including it
        tools = list(tools)  # another shallow copy
        last  = dict(tools[-1])
        last["cache_control"] = cc
        tools[-1] = last

    # context_top_k is handled at message-build time in envelope_codec.py
    # (the context message is already structured with cache_control).
    # No additional processing needed here.

    return messages, tools


def _apply_to_system(messages: list[dict], cc: dict) -> list[dict]:
    """Ensure the system message has cache_control on its last content block."""
    new = list(messages)
    for i, msg in enumerate(new):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                # Promote to block list
                new[i] = {
                    **msg,
                    "content": [{"type": "text", "text": content, "cache_control": cc}],
                }
            elif isinstance(content, list) and content:
                # Add / overwrite cache_control on last block
                blocks = list(content)
                last_block = dict(blocks[-1])
                last_block["cache_control"] = cc
                blocks[-1] = last_block
                new[i] = {**msg, "content": blocks}
            break  # only one system message
    return new
