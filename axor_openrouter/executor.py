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

    Uses ToolResultBus: wrapper.py detects get_bus() and registers a push
    callback so intent_loop can feed tool results back to this executor.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        transport: OpenRouterTransport,
        tier_mapper: "TierMapper",
        provider_prefs: "ProviderPrefs | None" = None,
    ) -> None:
        self._api_key        = api_key
        self._default_model  = model
        self._transport      = transport
        self._tier_mapper    = tier_mapper
        self._provider_prefs = provider_prefs
        self._bus            = ToolResultBus()
        self._ledger: list[dict] = []  # [{model, depth, in_tokens, out_tokens}]

    def get_bus(self) -> ToolResultBus:
        """Called by wrapper.py to register the tool result injection callback."""
        return self._bus

    def call_ledger(self) -> list[dict]:
        """Return per-API-call cost records for cost attribution."""
        return list(self._ledger)

    def reset_ledger(self) -> None:
        self._ledger.clear()

    async def stream(self, envelope: "ExecutionEnvelope") -> AsyncIterator[ExecutorEvent]:
        depth = envelope.depth or (envelope.lineage.depth if envelope.lineage else 0)
        model = self._tier_mapper.resolve(depth) if self._tier_mapper else self._default_model
        messages = build_messages(envelope)
        tools = build_tools_with_extensions(envelope)

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

            # Record this API call in the ledger for external cost attribution
            if accumulator.usage:
                u = accumulator.usage
                self._ledger.append({
                    "model":     model,
                    "depth":     depth,
                    "in_tokens": u.get("prompt_tokens", 0),
                    "out_tokens": u.get("completion_tokens", 0),
                })

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
                continue

            else:
                token_usage = decode_usage(
                    accumulator.usage,
                    context_tokens=envelope.context.token_count,
                )
                yield ExecutorEvent(
                    kind=ExecutorEventKind.STOP,
                    payload={
                        "usage": {
                            "input_tokens":                token_usage.input_tokens,
                            "output_tokens":               token_usage.output_tokens,
                            "tool_tokens":                 0,
                            "context_tokens":              envelope.context.token_count,
                            "cache_creation_input_tokens": token_usage.cache_creation_input_tokens,
                            "cache_read_input_tokens":     token_usage.cache_read_input_tokens,
                        }
                    },
                    node_id=envelope.node_id,
                )
                break

    def _build_request_body(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        envelope: "ExecutionEnvelope",
    ) -> dict:
        body: dict = {"model": model, "messages": messages}
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
