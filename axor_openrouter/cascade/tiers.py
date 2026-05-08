from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierSpec:
    min_depth: int
    max_depth: int | None
    model: str
    tier_index: int = 0

    def matches(self, depth: int) -> bool:
        if depth < self.min_depth:
            return False
        if self.max_depth is not None and depth > self.max_depth:
            return False
        return True


DEFAULT_TIERS: list[TierSpec] = [
    TierSpec(min_depth=0, max_depth=0,    model="anthropic/claude-opus-4-7",         tier_index=0),
    TierSpec(min_depth=1, max_depth=2,    model="anthropic/claude-sonnet-4-6",       tier_index=1),
    TierSpec(min_depth=3, max_depth=5,    model="openai/gpt-4o-mini",                tier_index=2),
    TierSpec(min_depth=6, max_depth=None, model="meta-llama/llama-3.3-70b-instruct", tier_index=3),
]


class TierMapper:
    def __init__(self, tiers: list[TierSpec] | None = None) -> None:
        self._tiers = sorted(
            tiers if tiers is not None else DEFAULT_TIERS,
            key=lambda t: t.min_depth,
        )

    def resolve(self, depth: int, tier_shift: int = 0) -> str:
        spec = self._find_spec(depth)
        if spec is None:
            spec = self._tiers[-1] if self._tiers else None
        if spec is None:
            return "openai/gpt-4o"

        if tier_shift == 0:
            return spec.model

        all_indices = sorted({s.tier_index for s in self._tiers})
        current_idx = spec.tier_index
        pos = all_indices.index(current_idx) if current_idx in all_indices else 0
        new_pos = max(0, min(len(all_indices) - 1, pos - tier_shift))
        target_index = all_indices[new_pos]
        for s in self._tiers:
            if s.tier_index == target_index:
                return s.model
        return spec.model

    def _find_spec(self, depth: int) -> TierSpec | None:
        for spec in self._tiers:
            if spec.matches(depth):
                return spec
        return None
