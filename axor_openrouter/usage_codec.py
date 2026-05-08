from __future__ import annotations

from axor_core.contracts.result import TokenUsage


def decode_usage(usage: dict, context_tokens: int = 0) -> TokenUsage:
    """Convert OpenRouter usage dict to axor TokenUsage."""
    if not usage:
        return TokenUsage(
            input_tokens=0,
            output_tokens=0,
            tool_tokens=0,
            context_tokens=context_tokens,
        )
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        tool_tokens=0,
        context_tokens=context_tokens,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
    )
