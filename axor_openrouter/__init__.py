"""axor-openrouter: OpenRouter adapter for axor-core.

Usage:
    from axor_openrouter import make_session

    session = make_session(
        api_key="sk-or-...",
        model="anthropic/claude-sonnet-4-6",
    )
    result = await session.run("explain this code")
"""
from __future__ import annotations

from typing import Any

from axor_core.worker.session import GovernedSession

from axor_openrouter._version import __version__
from axor_openrouter.executor import OpenRouterExecutor
from axor_openrouter.transport import OpenRouterTransport
from axor_openrouter.cascade.tiers import DEFAULT_TIERS, TierMapper
from axor_openrouter.routing.provider_prefs import ProviderPrefs
from axor_openrouter.tool_handlers import make_capability_executor

__all__ = ["make_session", "OpenRouterExecutor", "__version__"]

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def make_session(
    *,
    api_key: str,
    tools: tuple[str, ...] = ("read", "write", "bash", "search", "glob"),
    model: str | None = None,
    load_skills: bool = True,
    load_plugins: bool = True,
    soft_token_limit: int | None = None,
    system_prompt: str | None = None,
    telemetry: Any = None,
    # Routing overrides
    sort: str = "quality",
    max_prompt_price: float | None = None,
    allow_fallbacks: bool = True,
    **kwargs: Any,
) -> GovernedSession:
    """Build a GovernedSession backed by OpenRouter. Called by axor-cli."""
    effective_model = model or _DEFAULT_MODEL

    cap_executor = make_capability_executor(tools)

    tier_mapper = TierMapper(DEFAULT_TIERS)
    provider_prefs = ProviderPrefs(
        sort=sort,
        max_prompt_price=max_prompt_price,
        allow_fallbacks=allow_fallbacks,
    )
    transport = OpenRouterTransport()

    executor = OpenRouterExecutor(
        api_key=api_key,
        model=effective_model,
        transport=transport,
        tier_mapper=tier_mapper,
        provider_prefs=provider_prefs,
    )

    return GovernedSession(
        executor=executor,
        capability_executor=cap_executor,
        soft_token_limit=soft_token_limit,
        telemetry=telemetry,
    )
