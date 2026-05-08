from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderPrefs:
    sort: str = "quality"
    max_prompt_price: float | None = None
    allow_fallbacks: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.max_prompt_price is not None:
            d["max_price"] = {"prompt": self.max_prompt_price}
        if not self.allow_fallbacks:
            d["allow_fallbacks"] = False
        return d
