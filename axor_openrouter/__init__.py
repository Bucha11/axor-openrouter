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

from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.worker.session import GovernedSession

from .cascade.tiers import DEFAULT_TIERS, TierMapper, load_cascade_config_if_exists
from .executor import OpenRouterExecutor
from .routing.provider_prefs import ProviderPrefs
from .routing.fallbacks import DEFAULT_FALLBACKS
from .routing.byok import BYOKConfig
from .security import SecurityObservers, build_observers

__all__ = [
    "make_session",
    "OpenRouterExecutor",
    "SecurityObservers",
    "build_observers",
]


def make_session(
    *,
    api_key: str,
    task: str,
    context_text: str = "",
    depth: int = 0,
    parent_node_id: str | None = None,
    cache_hints: dict[str, Any] | None = None,
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
    # Optional security observers (P-34: behind the [security] extra)
    probe_pipeline=None,
    enable_sentinel: bool = False,
    # Extra executor kwargs forwarded verbatim
    **executor_kwargs: Any,
) -> GovernedSession:
    """Build a fully-wired GovernedSession backed by OpenRouter.

    Optional security observers are wired here — the adapter is the composition
    root. Passing ``probe_pipeline`` enables an axor-probe context tap; setting
    ``enable_sentinel=True`` enables an axor-sentinel session sink. Both require
    the ``[security]`` extra (``pip install axor-openrouter[security]``) and are
    lazily imported, so the base install never depends on probe/sentinel.
    """
    # Composition root: pull any caller-supplied taps/sinks out of the forwarded
    # kwargs (they belong to GovernedSession, not the executor) so the optional
    # security observers can be merged in alongside them.
    context_taps = list(executor_kwargs.pop("context_taps", None) or [])
    session_sinks = list(executor_kwargs.pop("session_sinks", None) or [])

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

    observers = build_observers(
        probe_pipeline=probe_pipeline,
        enable_sentinel=enable_sentinel,
    )
    if observers.context_tap is not None:
        context_taps.append(observers.context_tap)
    if observers.session_sink is not None:
        session_sinks.append(observers.session_sink)

    session = GovernedSession(
        executor=executor,
        envelope=envelope,
        policy=policy or ExecutionPolicy(),
        context_taps=context_taps or None,
        session_sinks=session_sinks or None,
    )
    session.axor_security = observers
    return session
