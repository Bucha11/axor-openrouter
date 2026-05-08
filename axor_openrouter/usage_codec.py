from __future__ import annotations

from axor_core.contracts.result import TokenUsage


def decode_usage(
    or_usage: dict,
    context_tokens: int = 0,
) -> TokenUsage:
    """
    Convert an OpenRouter /chat/completions usage dict to axor TokenUsage.

    OpenRouter returns OpenAI-compatible usage:
        prompt_tokens:     non-cached input + cache writes
        completion_tokens: output
        total_tokens:      prompt + completion

    Extended fields (provider-dependent, may be absent):
        cache_read_input_tokens:   tokens served from cache (billed at ~0.1x)
        cache_write_tokens:        tokens written to cache (billed at ~1.25x)
        input_cache_read_tokens:   alias used by some providers

    For providers that report cache separately, OpenRouter may subtract cache
    tokens from prompt_tokens. We re-separate them here so axor’s cost maths
    stays accurate.
    """
    prompt     = or_usage.get("prompt_tokens", 0)
    completion = or_usage.get("completion_tokens", 0)

    # cache read — various field names across providers
    cache_read = (
        or_usage.get("cache_read_input_tokens")
        or or_usage.get("input_cache_read_tokens")
        or or_usage.get("prompt_cache_hit_tokens", 0)
    )
    # cache write / creation
    cache_write = (
        or_usage.get("cache_write_tokens")
        or or_usage.get("cache_creation_input_tokens")
        or or_usage.get("prompt_cache_miss_tokens", 0)
    )

    # OpenRouter reports prompt_tokens as the TOTAL including cache tokens
    # for some providers, and as ONLY uncached for others.
    # We derive uncached_input conservatively:
    uncached_input = max(0, prompt - cache_read - cache_write)

    return TokenUsage(
        input_tokens=uncached_input,
        output_tokens=completion,
        tool_tokens=0,
        context_tokens=context_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )
