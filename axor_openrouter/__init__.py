"""axor-openrouter: OpenRouter backend adapter for axor-core.

Quick-start
-----------
    from axor_openrouter import make_session

    session = make_session(
        api_key="sk-or-...",
        task="explain this code",
        context_text="def foo(): ...",
    )
    async for event in session.stream():
        ...

Advanced
--------
Pass keyword arguments to override defaults:

    make_session(
        api_key=...,
        task=...,
        tier_config="axor.openrouter.toml",   # custom cascade
        sort="throughput",
        max_prompt_price=5.0,
        enable_cache=True,
        response_cache=False,
    )
"""
from __future__ import annotations

from typing import Any

from axor_core.contracts.envelope import CacheHints, ExecutionEnvelope
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.worker.session import GovernedSession

from .cascade.tiers import DEFAULT_TIERS, TierMapper, load_cascade_config_if_exists
from .executor import OpenRouterExecutor
from .routing.provider_prefs import ProviderPrefs
from .routing.fallbacks import DEFAULT_FALLBACKS
from .routing.byok import BYOKConfig

__all__ = ["make_session", "OpenRouterExecutor", "CacheHints"]


def make_session(
    *,
    api_key: str,
    task: str,
    context_text: str = "",
    depth: int = 0,
    parent_node_id: str | None = None,
    cache_hints: CacheHints | None = None,
    deterministic: bool = False,
    # Routing / cascade
    tier_config: str | None = None,
    sort: str = "quality",
    max_prompt_price: float | None = None,
    fallback: bool = True,
    byok: BYOKConfig | None = None,
    # Cache
    enable_cache: bool = True,
    response_cache: bool = False,
    # Policy
    policy: ExecutionPolicy | None = None,
    # Extra executor kwargs forwarded verbatim
    **executor_kwargs: Any,
) -> GovernedSession:
    """Build a fully-wired GovernedSession backed by OpenRouter."""
    tiers = (
        load_cascade_config_if_exists(tier_config)
        if tier_config
        else DEFAULT_TIERS
    )
    tier_mapper = TierMapper(tiers)

    provider_prefs = ProviderPrefs(
        sort=sort,
        max_prompt_price=max_prompt_price,
        allow_fallbacks=fallback,
    )

    executor = OpenRouterExecutor(
        api_key=api_key,
        tier_mapper=tier_mapper,
        provider_prefs=provider_prefs,
        fallbacks=DEFAULT_FALLBACKS,
        byok=byok,
        enable_cache=enable_cache,
        response_cache=response_cache,
        **executor_kwargs,
    )

    envelope = ExecutionEnvelope(
        task=task,
        context_text=context_text,
        depth=depth,
        parent_node_id=parent_node_id,
        cache_hints=cache_hints,
        deterministic=deterministic,
    )

    return GovernedSession(
        executor=executor,
        envelope=envelope,
        policy=policy or ExecutionPolicy(),
    )
