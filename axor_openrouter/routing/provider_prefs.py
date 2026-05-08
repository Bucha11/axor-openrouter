from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderPrefs:
    """
    OpenRouter provider preferences sent in the `provider` field of the request.

    Docs: https://openrouter.ai/docs/api/reference/parameters

    sort:            "price" | "throughput" | "latency" (OpenRouter sorts providers)
    max_price:       per-token price ceiling in $/M tokens
    order:           explicit provider order list (e.g. ["Anthropic", "OpenAI"])
    allow_fallbacks: if False, fail if primary provider unavailable (default True)
    quantizations:   filter by quantisation level (e.g. ["fp16", "bf16"])
    """
    sort: str | None = None             # "price" | "throughput" | "latency"
    max_prompt_price: float | None = None    # $/M input tokens ceiling
    max_completion_price: float | None = None  # $/M output tokens ceiling
    order: list[str] = field(default_factory=list)  # explicit provider order
    allow_fallbacks: bool = True
    quantizations: list[str] = field(default_factory=list)
    # BYOK: pass through API key to specific provider
    byok_provider: str | None = None
    byok_api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the OpenRouter provider object."""
        d: dict[str, Any] = {}
        if self.sort:
            d["sort"] = self.sort
        if self.max_prompt_price is not None or self.max_completion_price is not None:
            price: dict = {}
            if self.max_prompt_price is not None:
                price["prompt"] = str(self.max_prompt_price / 1_000_000)
            if self.max_completion_price is not None:
                price["completion"] = str(self.max_completion_price / 1_000_000)
            d["max_price"] = price
        if self.order:
            d["order"] = self.order
        if not self.allow_fallbacks:
            d["allow_fallbacks"] = False
        if self.quantizations:
            d["quantizations"] = self.quantizations
        if self.byok_provider and self.byok_api_key:
            d["api_key_or_org_ids"] = {self.byok_provider: self.byok_api_key}
        return d


def build_provider_prefs(
    sort: str = "price",
    max_prompt_price: float | None = None,
    max_completion_price: float | None = None,
    order: list[str] | None = None,
    allow_fallbacks: bool = True,
    quantizations: list[str] | None = None,
    byok_provider: str | None = None,
    byok_api_key: str | None = None,
) -> ProviderPrefs:
    return ProviderPrefs(
        sort=sort,
        max_prompt_price=max_prompt_price,
        max_completion_price=max_completion_price,
        order=order or [],
        allow_fallbacks=allow_fallbacks,
        quantizations=quantizations or [],
        byok_provider=byok_provider,
        byok_api_key=byok_api_key,
    )
