"""axor-openrouter: OpenRouter adapter for axor-core.

Usage:
    from axor_openrouter import make_session

    session = make_session(
        api_key="sk-or-...",
        model="anthropic/claude-sonnet-4-6",
    )
    result = await session.run("explain this code")

    # With automatic smart cascade (default):
    session = make_session(api_key="sk-or-...")
    # depth=0 → model chosen by task complexity
    # depth≥1 → automatically cheaper models
    # depth≥3 → free models

    # With explicit tier override:
    from axor_openrouter.cascade.tiers import TierSpec, TierMapper
    session = make_session(
        api_key="sk-or-...",
        tiers=[
            TierSpec(min_depth=0, max_depth=0,    model="anthropic/claude-sonnet-4-6", tier_index=0),
            TierSpec(min_depth=1, max_depth=None, model="openai/gpt-4o-mini",          tier_index=1),
        ],
    )
"""
from __future__ import annotations

from typing import Any

from axor_core.worker.session import GovernedSession

from axor_openrouter._version import __version__
from axor_openrouter.executor import OpenRouterExecutor
from axor_openrouter.transport import OpenRouterTransport
from axor_openrouter.cascade.tiers import TierSpec, TierMapper
from axor_openrouter.routing.provider_prefs import ProviderPrefs
from axor_openrouter.routing.model_selector import SmartModelSelector
from axor_openrouter.tool_handlers import make_capability_executor

__all__ = [
    "make_session",
    "OpenRouterExecutor",
    "SmartModelSelector",
    "TierSpec",
    "TierMapper",
    "__version__",
]

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def make_session(
    *,
    api_key: str,
    tools: tuple[str, ...] = ("read", "write", "bash", "search", "glob"),
    model: str | None = None,
    # Smart cascade (default) — automatic model selection by task + depth
    smart_cascade: bool = True,
    max_cost_in: float | None = None,
    prefer_free_at_depth: int = 3,
    # Explicit tier override — disables smart_cascade
    tiers: list[TierSpec] | None = None,
    # Provider options
    max_prompt_price: float | None = None,
    allow_fallbacks: bool = True,
    # axor-cli compat kwargs (ignored)
    load_skills: bool = True,
    load_plugins: bool = True,
    soft_token_limit: int | None = None,
    system_prompt: str | None = None,
    telemetry: Any = None,
    sort: str = "quality",
    **kwargs: Any,
) -> GovernedSession:
    """
    Build a GovernedSession backed by OpenRouter.

    Model selection (priority order):
      1. tiers       — explicit TierSpec list overrides everything
      2. smart_cascade=True (default) — SmartModelSelector picks model per task
      3. model       — single model for all nodes (no cascade)
    """
    effective_model = model or _DEFAULT_MODEL

    cap_executor = make_capability_executor(tools)
    provider_prefs = ProviderPrefs(
        sort=sort,
        max_prompt_price=max_prompt_price,
        allow_fallbacks=allow_fallbacks,
    )
    transport = OpenRouterTransport()

    # Resolve which selection strategy to use
    tier_mapper: TierMapper | None = None
    model_selector: SmartModelSelector | None = None

    if tiers is not None:
        tier_mapper = TierMapper(tiers)
    elif smart_cascade:
        model_selector = SmartModelSelector(
            max_cost_in=max_cost_in,
            prefer_free_at_depth=prefer_free_at_depth,
            pinned_root_model=effective_model if model else None,
        )

    executor = OpenRouterExecutor(
        api_key=api_key,
        model=effective_model,
        transport=transport,
        tier_mapper=tier_mapper,
        model_selector=model_selector,
        provider_prefs=provider_prefs,
    )

    return GovernedSession(
        executor=executor,
        capability_executor=cap_executor,
        soft_token_limit=soft_token_limit,
        telemetry=telemetry,
    )
