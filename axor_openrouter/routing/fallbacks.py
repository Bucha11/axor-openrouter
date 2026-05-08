from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FallbackChain:
    """
    Per-tier fallback model chain sent as the `models` field.

    OpenRouter tries models in order, skipping unavailable/erroring ones.
    When present, the primary `model` field acts as the first entry;
    `models` adds alternatives.

    Example:
        chain = FallbackChain(tier=0, models=[
            "anthropic/claude-opus-4-7",
            "anthropic/claude-sonnet-4-6",
        ])
        body["models"] = chain.models  # OpenRouter will try in order
    """
    tier: int
    models: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.models


# Default fallback chains per tier
DEFAULT_FALLBACKS: dict[int, list[str]] = {
    0: [
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
    ],
    1: [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ],
    2: [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-flash-1.5",
    ],
    3: [
        "openai/gpt-4o-mini",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    4: [
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.3-70b-instruct",
        "openrouter/auto",
    ],
}


def get_fallbacks(tier: int) -> list[str]:
    """Return the fallback chain for a given tier."""
    return list(DEFAULT_FALLBACKS.get(tier, DEFAULT_FALLBACKS[4]))
