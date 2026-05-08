from __future__ import annotations

from axor_core.contracts.envelope import ExecutionEnvelope


def should_cache_response(envelope: ExecutionEnvelope) -> bool:
    """
    Return True when the executor should request response-level caching.

    Response caching is safe only for deterministic executions (temperature=0
    or equivalent). It is NOT the same as prompt prefix caching.

    Enabled when:
    - envelope.deterministic is True
    - AND the task is purely read-only (no write/bash tools in capabilities)

    Disabled by default — the user must opt in via --response-cache flag
    or explicit policy configuration.
    """
    if not envelope.deterministic:
        return False
    # extra guard: response caching on mutation tasks is semantically unsafe
    mutating = {"write", "bash"}
    if mutating & envelope.capabilities.allowed_tools:
        return False
    return True
