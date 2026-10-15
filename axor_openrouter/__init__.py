"""axor-openrouter: OpenRouter backend adapter for axor-core.

Quick-start
-----------
    from axor_openrouter import make_session

    session = make_session(
        api_key="sk-or-...",
        handlers=[MyReadHandler(), MySendHandler()],   # axor_core ToolHandler
        governance="governance.yaml",                  # the kernel gate's config
    )
    result = await session.run("explain this code")

Every tool call the model makes goes through the session's loop and then the
current kernel gate (``gate.Gate`` around a ``ToolCallGovernor``) before its
handler runs; with ``daemon_socket`` the daemon runs the handlers and the gate.

Advanced
--------
    make_session(
        api_key=...,
        tier_config="axor.openrouter.toml",   # custom cascade
        sort="throughput",
        max_prompt_price=5.0,
        tool_schemas=[...],                   # OpenAI function schemas of the handlers
    )
"""
from __future__ import annotations

from typing import Any

from axor_core.contracts.policy import ExecutionPolicy

from .cascade.tiers import DEFAULT_TIERS, TierMapper, load_cascade_config_if_exists
from .executor import OpenRouterExecutor
from .routing.provider_prefs import ProviderPrefs
from .security import SecurityObservers, build_observers
from .gate import Gate
from .capability_gate import GatedCapabilityExecutor, GatedSession, GovernanceRefused

__all__ = [
    "make_session",
    "OpenRouterExecutor",
    "SecurityObservers",
    "build_observers",
    "Gate",
    "GatedCapabilityExecutor",
    "GatedSession",
    "GovernanceRefused",
]


def make_session(
    *,
    api_key: str,
    handlers: "list[Any] | tuple[Any, ...]" = (),
    tool_schemas: "list[dict] | None" = None,
    governance: Any = None,
    confirmer: Any = None,
    structured: Any = None,
    daemon_socket: "str | None" = None,
    model: "str | None" = None,
    # Routing / cascade
    tier_config: "str | None" = None,
    sort: str = "quality",
    max_prompt_price: "float | None" = None,
    fallback: bool = True,
    transport: Any = None,
    # Policy
    policy: "ExecutionPolicy | None" = None,
    # Optional security observers (P-34: behind the [security] extra)
    probe_pipeline=None,
    enable_sentinel: bool = False,
    **session_kwargs: Any,
) -> GatedSession:
    """Build a ``GovernedSession`` backed by OpenRouter whose tools run behind the kernel gate.

    ``handlers`` are ``axor_core`` ``ToolHandler``s; ``tool_schemas`` their OpenAI
    function schemas (the built-in read/write/bash/search/glob schemas need none).
    ``governance`` is a ``GovernanceConfig``, its YAML path or its mapping (``None``:
    an empty configuration — the gate still runs); ``confirmer`` the operator's L4
    channel; ``structured`` the platform's parser of a description source's output
    (MODEL.md 9.15). With ``daemon_socket`` the daemon's own ``--governance`` decides.

    Optional security observers: ``probe_pipeline`` wires an axor-probe context tap,
    ``enable_sentinel=True`` an axor-sentinel session sink (both need ``[security]``).
    """
    from .transport import OpenRouterTransport

    tiers = load_cascade_config_if_exists(tier_config) if tier_config else DEFAULT_TIERS
    tier_mapper = TierMapper(tiers)
    provider_prefs = ProviderPrefs(sort=sort, max_prompt_price=max_prompt_price,
                                   allow_fallbacks=fallback)
    executor = OpenRouterExecutor(
        api_key=api_key,
        model=model or tier_mapper.resolve(0),
        transport=transport or OpenRouterTransport(),
        tier_mapper=tier_mapper,
        provider_prefs=provider_prefs,
        tool_schemas=tool_schemas,
    )

    if daemon_socket is not None:
        if governance is not None or confirmer is not None or handlers:
            raise ValueError("with daemon_socket the daemon runs the handlers and its "
                             "--governance decides; pass them to the daemon, not here")
        from axor_core.capability.daemon_client import DaemonCapabilityClient
        cap: Any = DaemonCapabilityClient(socket_path=daemon_socket)
    else:
        cap = GatedCapabilityExecutor(
            Gate.from_config(governance or {}, confirmer=confirmer, structured=structured))
        for h in handlers:
            cap.register(h)

    context_taps = list(session_kwargs.pop("context_taps", None) or [])
    observers = build_observers(probe_pipeline=probe_pipeline, enable_sentinel=enable_sentinel)
    if observers.context_tap is not None:
        context_taps.append(observers.context_tap)
    if observers.session_sink is not None and "session_sink" not in session_kwargs:
        session_kwargs["session_sink"] = observers.session_sink

    session = GatedSession(
        executor=executor,
        capability_executor=cap,
        default_policy=policy,
        context_taps=context_taps or None,
        **session_kwargs,
    )
    session.axor_security = observers
    return session
