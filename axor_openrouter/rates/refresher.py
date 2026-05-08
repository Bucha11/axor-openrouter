"""Background asyncio task that refreshes the rates catalog every 24 h."""
from __future__ import annotations

import asyncio

from .catalog import get_rates_catalog


class RatesRefresher:
    """Starts a long-lived background task to keep the catalog fresh."""

    def __init__(self, api_key: str, interval: float = 86400.0) -> None:
        self._api_key = api_key
        self._interval = interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                await get_rates_catalog(self._api_key)
            except Exception:
                pass
            await asyncio.sleep(self._interval)
