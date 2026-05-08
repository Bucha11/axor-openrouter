"""Tests for rates.catalog (mocked httpx)."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axor_openrouter.rates.catalog import RatesCatalog


FAKE_MODELS = {
    "data": [
        {
            "id": "anthropic/claude-opus-4-7",
            "pricing": {"prompt": "0.000015", "completion": "0.000075"},
        },
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "bad-model",
            "pricing": {"prompt": "NOT_A_NUMBER"},
        },
    ]
}


@pytest.fixture()
def catalog():
    return RatesCatalog()


def test_new_catalog_is_stale(catalog):
    assert catalog.is_stale()


def test_get_unknown_model_returns_none(catalog):
    assert catalog.get("unknown/model") is None


@pytest.mark.asyncio
async def test_fetch_populates_rates(catalog):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=FAKE_MODELS)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=client_cm)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    client_cm.get = AsyncMock(return_value=resp)

    with patch("axor_openrouter.rates.catalog.httpx.AsyncClient", return_value=client_cm):
        await catalog.fetch("sk-or-test")

    assert catalog.get("anthropic/claude-opus-4-7") is not None
    assert catalog.get("openai/gpt-4o-mini") is not None
    assert catalog.get("bad-model") is None  # parse error skipped


@pytest.mark.asyncio
async def test_fetch_not_stale_immediately_after(catalog):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": []})

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=client_cm)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    client_cm.get = AsyncMock(return_value=resp)

    with patch("axor_openrouter.rates.catalog.httpx.AsyncClient", return_value=client_cm):
        await catalog.fetch("sk-or-test")

    assert not catalog.is_stale()


@pytest.mark.asyncio
async def test_fetch_rates_correct_values(catalog):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=FAKE_MODELS)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=client_cm)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    client_cm.get = AsyncMock(return_value=resp)

    with patch("axor_openrouter.rates.catalog.httpx.AsyncClient", return_value=client_cm):
        await catalog.fetch("sk-or-test")

    rates = catalog.get("anthropic/claude-opus-4-7")
    assert rates is not None
    # 0.000015 * 1_000_000 = 15.0 $/M
    assert abs(rates.input_cost_per_million - 15.0) < 0.01
