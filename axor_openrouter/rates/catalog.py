"""In-memory catalog of per-model TokenCostRates fetched from OpenRouter."""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx

from axor_core.budget.tracker import TokenCostRates

BASE_URL = "https://openrouter.ai/api/v1"
_CATALOG_LOCK = asyncio.Lock()
_catalog_instance: Optional["RatesCatalog"] = None


class RatesCatalog:
    """Lazy-fetched map of model_id -> TokenCostRates."""

    def __init__(self) -> None:
        self._rates: dict[str, TokenCostRates] = {}
        self._fetched_at: float = 0.0
        self._ttl: float = 86400.0  # 24 h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, api_key: str) -> None:
        """(Re)populate the catalog from OpenRouter /api/v1/models."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        self._rates = {}
        for entry in data.get("data", []):
            model_id = entry.get("id", "")
            pricing = entry.get("pricing", {})
            # Only the price parsing may legitimately fail on malformed data —
            # keep the guard tight around it so a genuine construction error in
            # TokenCostRates surfaces instead of silently emptying the catalog.
            try:
                prompt_cost = float(pricing.get("prompt", 0)) * 1_000_000
                completion_cost = float(pricing.get("completion", 0)) * 1_000_000
            except (TypeError, ValueError):
                continue
            self._rates[model_id] = TokenCostRates(
                input_per_m=prompt_cost,
                output_per_m=completion_cost,
            )
        self._fetched_at = time.monotonic()

    def get(self, model_id: str) -> TokenCostRates | None:
        return self._rates.get(model_id)

    def is_stale(self) -> bool:
        # Never-fetched catalog (_fetched_at sentinel 0.0) is always stale,
        # independent of the host's monotonic-clock origin.
        if self._fetched_at == 0.0:
            return True
        return (time.monotonic() - self._fetched_at) > self._ttl

    def all_models(self) -> list[str]:
        return list(self._rates.keys())


async def get_rates_catalog(api_key: str) -> RatesCatalog:
    """Return the process-wide singleton, fetching once if empty/stale."""
    global _catalog_instance
    async with _CATALOG_LOCK:
        if _catalog_instance is None:
            _catalog_instance = RatesCatalog()
        if _catalog_instance.is_stale():
            try:
                await _catalog_instance.fetch(api_key)
            except Exception:
                # Non-fatal — operate without fresh rates.
                pass
    return _catalog_instance
