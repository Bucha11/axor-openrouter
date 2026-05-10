from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.policy import CompressionMode, TaskNature, TaskComplexity

if TYPE_CHECKING:
    from axor_core.contracts.envelope import ExecutionEnvelope
    from axor_openrouter.cascade.tiers import TierMapper
    from axor_openrouter.routing.provider_prefs import ProviderPrefs
    from axor_openrouter.routing.model_selector import SmartModelSelector

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

# Trigger compression once estimated token count exceeds this threshold.
# Rough estimate: 1 token ≈ 4 chars.
_COMPRESS_TRIGGER_TOKENS = 4000

# Max chars kept per old tool-result message, keyed by CompressionMode.
# Head (system+context+task) and the last two round-trips are never touched.
_TOOL_RESULT_MAX_CHARS: dict[CompressionMode, int] = {
    CompressionMode.AGGRESSIVE: 800,
    CompressionMode.BALANCED:   2000,
    CompressionMode.LIGHT:      6000,
}

# How many recent messages (from the tail) to preserve untouched.
# 6 = assistant(tool_calls) + tool_result × 2 round-trips.
_KEEP_TAIL = 6

# Multiplier applied to export_contract.max_export_tokens → max_tokens in API call.
# Generate up to 1.5× the export limit so the model finishes naturally before
# the export truncation kicks in (avoids mid-word cuts in exported output).
_EXPORT_TO_API_TOKENS_FACTOR = 1.5

_BREVITY_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY the requested artifact. "
    "No preamble, no explanation, no commentary after the code."
)


class OpenRouterExecutor(Invokable):
    """
    axor-core Invokable backed by the OpenRouter API.

    Model selection priority (highest wins):
      1. tier_mapper    — explicit depth→model map (manual override)
      2. model_selector — SmartModelSelector (task complexity + depth)
      3. default_model  — single model for all nodes

    Message history compression:
      After each tool round-trip the growing messages list is compressed
      using the compression_mode from envelope.policy (axor-core).
      Old tool-result payloads in the middle of history are truncated;
      head (system + context + task) and recent round-trips are preserved.

    Uses ToolResultBus: wrapper.py detects get_bus() and registers a push
    callback so intent_loop can feed tool results back to this executor.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        transport: OpenRouterTransport,
        tier_mapper: "TierMapper | None" = None,
        model_selector: "SmartModelSelector | None" = None,
        provider_prefs: "ProviderPrefs | None" = None,
        thinking_budget: int | None = None,
    ) -> None:
        self._api_key        = api_key
        self._default_model  = model
        self._transport      = transport
        self._tier_mapper    = tier_mapper
        self._model_selector = model_selector
        self._provider_prefs = provider_prefs
        self._thinking_budget = thinking_budget
        self._bus            = ToolResultBus()
        self._ledger: list[dict] = []
        self._dead_models: set[str] = set()  # 404'd models, skipped on next selection
        self._text_callback: Callable[[str], None] | None = None
        self._tool_start_callback: Callable[[str, dict], None] | None = None
        self._tool_end_callback: Callable[[str, dict, Any], None] | None = None
        self._approval_callback: Callable[[str, dict], Awaitable[bool]] | None = None

    def set_text_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked with each streaming text chunk (used by axor-cli)."""
        self._text_callback = callback

    def set_tool_callbacks(
        self,
        on_start: Callable[[str, dict], None],
        on_end: Callable[[str, dict, Any], None],
    ) -> None:
        """Register callbacks for tool call start and completion (used by axor-cli)."""
        self._tool_start_callback = on_start
        self._tool_end_callback = on_end

    def set_approval_callback(
        self,
        callback: Callable[[str, dict], Awaitable[bool]],
    ) -> None:
        """
        Register an async callback called before each tool executes.
        Return True to approve, False to deny (model sees structured denial).
        """
        self._approval_callback = callback

    def get_bus(self) -> ToolResultBus:
        """Called by wrapper.py to register the tool result injection callback."""
        return self._bus

    def call_ledger(self) -> list[dict]:
        """Return per-API-call cost records for cost attribution."""
        return list(self._ledger)

    def reset_ledger(self) -> None:
        self._ledger.clear()
        self._dead_models.clear()

    # ── Model resolution ───────────────────────────────────────────────────────

    def _resolve_model(self, envelope: "ExecutionEnvelope", skip: frozenset[str] = frozenset()) -> str:
        depth = envelope.depth or (envelope.lineage.depth if envelope.lineage else 0)
        if self._tier_mapper is not None:
            return self._tier_mapper.resolve(depth)
        if self._model_selector is not None:
            task_signal = getattr(envelope, "task_signal", None)
            complexity = task_signal.complexity if task_signal else None
            return self._model_selector.select(
                depth=depth,
                task_signal=complexity,
                task=envelope.task,
                skip=skip,
            )
        return self._default_model

    # ── Message history compression ────────────────────────────────────────────

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total = sum(len(m.get("content") or "") for m in messages)
        # tool_calls field adds tokens too
        for m in messages:
            if m.get("tool_calls"):
                total += len(json.dumps(m["tool_calls"]))
        return total // 4

    def _compress_history(
        self,
        messages: list[dict],
        mode: CompressionMode,
    ) -> list[dict]:
        """
        Truncate old tool-result payloads in message history.

        Uses compression_mode from envelope.policy — the same mode that
        axor-core's ContextCompressor uses for fragment compression.

        Invariants:
          - system message (role=system) always preserved in full
          - first user message (the task) always preserved in full
          - last _KEEP_TAIL messages always preserved in full
          - tool_call/tool_result pairs kept together (API requirement)
          - only role=tool content is truncated in the middle section
        """
        if self._estimate_tokens(messages) < _COMPRESS_TRIGGER_TOKENS:
            return messages

        max_chars = _TOOL_RESULT_MAX_CHARS[mode]

        # Identify head: system + non-tool messages at the start
        head_end = 0
        for i, m in enumerate(messages):
            if m.get("role") in ("system", "user", "assistant") and not m.get("tool_calls"):
                head_end = i + 1
            else:
                break

        if len(messages) <= head_end + _KEEP_TAIL:
            return messages

        head   = messages[:head_end]
        tail   = messages[len(messages) - _KEEP_TAIL:]
        middle = messages[head_end: len(messages) - _KEEP_TAIL]

        compressed_middle: list[dict] = []
        for msg in middle:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > max_chars:
                    kept = content[:max_chars]
                    # cut at last newline to avoid broken JSON/mid-word
                    nl = kept.rfind("\n")
                    if nl > max_chars // 2:
                        kept = kept[:nl]
                    omitted = len(content) - len(kept)
                    msg = {
                        **msg,
                        "content": kept + f"\n[... {omitted} chars removed by {mode.value} compression]",
                    }
            compressed_middle.append(msg)

        result = head + compressed_middle + tail
        saved = self._estimate_tokens(messages) - self._estimate_tokens(result)
        if saved > 0:
            log.debug(
                "history compression (%s): ~%d tokens saved, %d→%d messages",
                mode.value, saved, len(messages), len(result),
            )
        return result

    # ── Main stream loop ───────────────────────────────────────────────────────

    async def stream(self, envelope: "ExecutionEnvelope") -> AsyncIterator[ExecutorEvent]:
        depth = envelope.depth or (envelope.lineage.depth if envelope.lineage else 0)
        model = self._resolve_model(envelope, skip=frozenset(self._dead_models))
        compression_mode = envelope.policy.compression_mode

        messages = build_messages(envelope)
        tools = build_tools_with_extensions(envelope)

        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 50:
                log.warning("executor: tool loop exceeded 50 iterations, stopping")
                break

            # Compress history before each API call (no-op if under threshold)
            messages = self._compress_history(messages, compression_mode)

            accumulator = StreamAccumulator()
            request_body = self._build_request_body(model, messages, tools, envelope)

            try:
                async for chunk in self._transport.stream(self._api_key, request_body):
                    accumulator.feed(chunk)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta_text = choices[0].get("delta", {}).get("content", "")
                        if delta_text:
                            if self._text_callback is not None:
                                self._text_callback(delta_text)
                            yield ExecutorEvent(
                                kind=ExecutorEventKind.TEXT,
                                payload={"text": delta_text},
                                node_id=envelope.node_id,
                            )
            except TransportError as exc:
                if exc.status == 404:
                    # Model not available on OpenRouter — blacklist and retry with next
                    self._dead_models.add(model)
                    log.warning("executor: model %s unavailable (404), blacklisting", model)
                    model = self._resolve_model(envelope, skip=frozenset(self._dead_models))
                    log.warning("executor: retrying with %s", model)
                    continue
                log.error("OpenRouter transport error: %s", exc)
                yield ExecutorEvent(
                    kind=ExecutorEventKind.ERROR,
                    payload={"error": str(exc), "status": exc.status},
                    node_id=envelope.node_id,
                )
                return

            if accumulator.usage:
                u = accumulator.usage
                self._ledger.append({
                    "model":      model,
                    "depth":      depth,
                    "in_tokens":  u.get("prompt_tokens", 0),
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

                    if self._tool_start_callback is not None:
                        self._tool_start_callback(tool_name, args)

                    # Interactive approval: check before executing.
                    # Denial is injected directly as a tool result so the model
                    # can respond gracefully — intent_loop never sees a denied call.
                    if self._approval_callback is not None:
                        user_approved = await self._approval_callback(tool_name, args)
                        if not user_approved:
                            denial = {"error": "tool_denied", "reason": "denied by user"}
                            if self._tool_end_callback is not None:
                                self._tool_end_callback(tool_name, args, denial)
                            append_tool_result(messages, tc["id"], denial)
                            continue

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

                    if self._tool_end_callback is not None:
                        self._tool_end_callback(tool_name, args, result)

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
        depth = envelope.depth or (envelope.lineage.depth if envelope.lineage else 0)
        task_signal = getattr(envelope, "task_signal", None)

        # For GENERATIVE child nodes: reinforce brevity via system instruction.
        # Cheap models (qwen, llama) tend to add prose regardless of task instructions.
        if depth > 0 and task_signal is not None and task_signal.nature == TaskNature.GENERATIVE:
            messages = self._inject_brevity(messages)

        # Derive max_tokens from axor-core's ExportContract rather than hardcoding.
        # If the policy caps exports at N tokens, no point generating more than
        # N × factor — the ExportFilter will truncate the rest anyway.
        export_limit = envelope.export_contract.max_export_tokens
        cap = int(export_limit * _EXPORT_TO_API_TOKENS_FACTOR) if export_limit else None

        body: dict = {"model": model, "messages": messages}
        if cap is not None:
            body["max_tokens"] = cap
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if envelope.deterministic:
            body["temperature"] = 0
        if self._provider_prefs is not None:
            provider_dict = self._provider_prefs.to_dict()
            if provider_dict:
                body["provider"] = provider_dict
        if self._thinking_budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
        return body

    def _inject_brevity(self, messages: list[dict]) -> list[dict]:
        """Append brevity instruction to system message (or add one) for GENERATIVE tasks."""
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + _BREVITY_SUFFIX}
        else:
            msgs.insert(0, {"role": "system", "content": _BREVITY_SUFFIX.strip()})
        return msgs
