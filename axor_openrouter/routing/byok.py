from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BYOKConfig:
    """
    Bring-Your-Own-Key: pass a provider API key to OpenRouter so it routes
    requests through your own account at that provider.

    OpenRouter docs: https://openrouter.ai/docs/features/byok

    provider:  Provider name as OpenRouter understands it (e.g. "Anthropic", "OpenAI")
    api_key:   Provider API key (not the OpenRouter key)
    """
    provider: str
    api_key: str

    def to_provider_dict(self) -> dict:
        """Inject into provider.api_key_or_org_ids."""
        return {"api_key_or_org_ids": {self.provider: self.api_key}}
