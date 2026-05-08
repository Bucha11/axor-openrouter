from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("axor.openrouter.transport")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_USER_AGENT = "axor-openrouter/0.1.0"


class TransportError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"OpenRouter HTTP {status}: {body[:200]}")


class OpenRouterTransport:
    def __init__(
        self,
        http_referer: str = "https://github.com/Bucha11/axor-openrouter",
        x_title: str = "axor-openrouter",
        timeout: float = 120.0,
    ) -> None:
        self._http_referer = http_referer
        self._x_title = x_title
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                http2=True,
            )
        return self._client

    async def stream(self, api_key: str, body: dict) -> AsyncIterator[dict]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._http_referer,
            "X-Title": self._x_title,
            "User-Agent": _USER_AGENT,
        }
        payload = dict(body)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        client = await self._get_client()
        try:
            async with client.stream("POST", BASE_URL, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body_text = await resp.aread()
                    raise TransportError(resp.status_code, body_text.decode())

                async for line in resp.aiter_lines():
                    if not line or line.startswith(":") or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        log.debug("unparseable SSE chunk: %r", data[:100])
        except httpx.TimeoutException as exc:
            raise TransportError(0, f"timeout: {exc}") from exc
        except httpx.RequestError as exc:
            raise TransportError(0, f"request error: {exc}") from exc

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class StreamAccumulator:
    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self._finish_reason: str = ""
        self._usage: dict = {}

    def feed(self, chunk: dict) -> None:
        if "usage" in chunk:
            self._usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        if finish := choice.get("finish_reason"):
            self._finish_reason = finish

        delta = choice.get("delta", {})
        if text := delta.get("content"):
            self._text_parts.append(text)

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            if idx not in self._tool_calls:
                self._tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
            entry = self._tool_calls[idx]
            if tc_id := tc_delta.get("id"):
                entry["id"] = tc_id
            fn = tc_delta.get("function", {})
            if name := fn.get("name"):
                entry["name"] += name
            if args := fn.get("arguments"):
                entry["arguments"] += args

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    @property
    def finish_reason(self) -> str:
        return self._finish_reason

    @property
    def usage(self) -> dict:
        return self._usage

    @property
    def tool_calls(self) -> list[dict]:
        result = []
        for idx in sorted(self._tool_calls):
            entry = self._tool_calls[idx]
            result.append({
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            })
        return result

    def has_tool_calls(self) -> bool:
        return bool(self._tool_calls)
