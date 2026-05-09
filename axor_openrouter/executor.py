from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

if TYPE_CHECKING:
    from axor_core.contracts.envelope import ExecutionEnvelope
    from axor_openrouter.cascade.tiers import TierMapper
    from axor_openrouter.routing.provider_prefs import ProviderPrefs
    from axor_openrouter.governance.budget_subscriber import BudgetSubscriber
    from axor_openrouter.governance.adaptive_router import AdaptiveRouter
    from axor_openrouter.governance.cache_health import CacheHealthMonitor

from axor_openrouter.bus import ToolResultBus
from axor_openrouter.transport import OpenRouterTransport, StreamAccumulator, TransportError
from axor_openrouter.tools import build_tools_with_extensions
from axor_openrouter.envelope_codec import (
    build_messages,
    build_tool_request_messages,
    append_tool_result,
)
from axor_openrouter.usage_codec import decode_usage

log = logging.getLogger("axor.openrouter.executor")


class OpenRouterExecutor(Invokable):
    """
    axor-core Invokable backed by the OpenRouter API.

    Model selection priority:
      1. envelope.routing_tier  — explicit tier override (e.g. force opus for a critical leaf)
      2. envelope.depth         — structural depth → tier lookup via TierMapper
      3. AdaptiveRouter shift   — budget-driven up/downgrade applied on top
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        transport: OpenRouterTransport,
        tier_mapper: "TierMapper",
        provider_prefs: "ProviderPrefs | None" = None,
        budget_subscriber: "BudgetSubscriber | None" = None,
        adaptive_router: "AdaptiveRouter | None" = None,
        cache_monitor: "CacheHealthMonitor | None" = None,
    ) -> None:
        self._api_key          = api_key
        self._default_model    = model
        self._transport        = transport
        self._tier_mapper      = tier_mapper
        self._provider_prefs   = provider_prefs
        self._budget_subscriber = budget_subscriber
        self._adaptive_router  = adaptive_router
        self._cache_monitor    = cache_monitor
        self._bus              = ToolResultBus()

    def get_bus(self) -> ToolResultBus:
        """Called by wrapper.py to register tool result injection."""
        return self._bus

    async def stream(
        self,
        envelope: "ExecutionEnvelope",
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Stream governed execution.

        Yields TEXT events during streaming, TOOL_USE when tools are needed,
        and a final STOP event with usage statistics.
        """
        depth = envelope.depth or (
            envelope.lineage.depth if envelope.lineage else 0
        )
        model = self._resolve_model(depth, envelope)
        messages = build_messages(envelope)
        tools = build_tools_with_extensions(envelope)

        yield ExecutorEvent(
            kind=ExecutorEventKind.TEXT,
            payload={"_routing": {"model": model, "depth": depth}},
            node_id=envelope.node_id,
        )

        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 50:
                log.warning("executor: tool loop exceeded 50 iterations, stopping")
                break

            accumulator = StreamAccumulator()
            request_body = self._build_request_body(model, messages, tools, envelope)

            try:
                async for chunk in self._transport.stream(self._api_key, request_body):
                    accumulator.feed(chunk)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta_text = choices[0].get("delta", {}).get("content", "")
                        if delta_text:
                            yield ExecutorEvent(
                                kind=ExecutorEventKind.TEXT,
                                payload={"text": delta_text},
                                node_id=envelope.node_id,
                            )
            except TransportError as exc:
                log.error("OpenRouter transport error: %s", exc)
                yield ExecutorEvent(
                    kind=ExecutorEventKind.ERROR,
                    payload={"error": str(exc), "status": exc.status},
                    node_id=envelope.node_id,
                )
                return

            finish = accumulator.finish_reason

            if finish == "tool_calls" or (not finish and accumulator.has_tool_calls()):
                messages = build_tool_request_messages(
                    messages,
                    assistant_content=accumulator.text or None,
                    tool_calls=accumulator.tool_calls,
                )

                for tc in accumulator.tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    yield ExecutorEvent(
                        kind=ExecutorEventKind.TOOL_USE,
                        payload={
                            "tool":        tool_name,
                            "tool_use_id": tc["id"],
                            "args":        args,
                        },
                        node_id=envelope.node_id,
                    )
                    _, result = await self._bus.wait()
                    append_tool_result(messages, tc["id"], result)

                    if self._cache_monitor is not None:
                        self._cache_monitor.record_tool_call()

                continue

            else:
                usage = accumulator.usage
                token_usage = decode_usage(usage, context_tokens=envelope.context.token_count)

                if self._cache_monitor is not None:
                    self._cache_monitor.record_usage(
                        cache_read=token_usage.cache_read_input_tokens,
                        total_input=(token_usage.input_tokens + token_usage.cache_read_input_tokens),
                    )

                yield ExecutorEvent(
                    kind=ExecutorEventKind.STOP,
                    payload={
                        "usage": {
                            "input_tokens":                 token_usage.input_tokens,
                            "output_tokens":                token_usage.output_tokens,
                            "tool_tokens":                  0,
                            "context_tokens":               envelope.context.token_count,
                            "cache_creation_input_tokens":  token_usage.cache_creation_input_tokens,
                            "cache_read_input_tokens":       token_usage.cache_read_input_tokens,
                        }
                    },
                    node_id=envelope.node_id,
                )
                break

    # ── Private ──────────────────────────────────────────────────────────────────────────

    def _resolve_model(self, depth: int, envelope: "ExecutionEnvelope") -> str:
        """Pick model using routing_tier override, depth lookup, then budget shift."""
        shift = 0
        if self._adaptive_router is not None:
            shift = self._adaptive_router.current_shift()
        return (
            self._tier_mapper.resolve(
                depth,
                tier_shift=shift,
                routing_tier=getattr(envelope, "routing_tier", None),
            )
            or self._default_model
        )

    def _build_request_body(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        envelope: "ExecutionEnvelope",
    ) -> dict:
        body: dict = {
            "model": model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        if envelope.deterministic:
            body["temperature"] = 0

        if self._provider_prefs is not None:
            provider_dict = self._provider_prefs.to_dict()
            if provider_dict:
                body["provider"] = provider_dict

        return body
