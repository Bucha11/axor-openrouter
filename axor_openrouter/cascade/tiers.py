from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierSpec:
    """
    Maps a range of node depths to a model.

    max_depth=None means “this depth and beyond”.
    """
    min_depth: int
    max_depth: int | None
    model: str
    tier_index: int = 0    # 0 = best/most capable, higher = cheaper

    def matches(self, depth: int) -> bool:
        if depth < self.min_depth:
            return False
        if self.max_depth is not None and depth > self.max_depth:
            return False
        return True


# Default cascade: best model at root, progressively cheaper for child nodes
DEFAULT_TIERS: list[TierSpec] = [
    TierSpec(min_depth=0, max_depth=0,    model="anthropic/claude-opus-4-7",            tier_index=0),
    TierSpec(min_depth=1, max_depth=2,    model="anthropic/claude-sonnet-4-6",          tier_index=1),
    TierSpec(min_depth=3, max_depth=5,    model="openai/gpt-4o-mini",                   tier_index=2),
    TierSpec(min_depth=6, max_depth=None, model="meta-llama/llama-3.3-70b-instruct",    tier_index=3),
]

# Tiers sorted cheapest → most capable for shift arithmetic
_TIER_MODELS_ASCENDING_COST = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-flash-1.5",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-7",
]


class TierMapper:
    """
    Resolves an execution depth to a model string.

    Supports an integer `tier_shift` from the adaptive router:
      +1  → one tier higher (more capable)
       0  → nominal tier
      -1  → one tier lower (cheaper)
    Shifts are clamped to valid tier bounds.
    """

    def __init__(self, tiers: list[TierSpec] | None = None) -> None:
        self._tiers = sorted(
            tiers if tiers is not None else DEFAULT_TIERS,
            key=lambda t: t.min_depth,
        )

    def resolve(self, depth: int, tier_shift: int = 0) -> str:
        """Return the model string for the given depth and shift."""
        spec = self._find_spec(depth)
        if spec is None:
            # fallback to the deepest tier
            spec = self._tiers[-1] if self._tiers else None
        if spec is None:
            return "openai/gpt-4o"  # ultimate fallback

        if tier_shift == 0:
            return spec.model

        # Apply shift: find the spec in tier order and move
        all_indices = sorted({s.tier_index for s in self._tiers})
        current_idx = spec.tier_index
        pos = all_indices.index(current_idx) if current_idx in all_indices else 0
        new_pos = max(0, min(len(all_indices) - 1, pos - tier_shift))  # higher index = cheaper
        target_index = all_indices[new_pos]

        for s in self._tiers:
            if s.tier_index == target_index:
                return s.model

        return spec.model

    def tier_label(self, depth: int) -> str:
        """Return a human-readable tier label for tracing/budgeting."""
        spec = self._find_spec(depth)
        idx  = spec.tier_index if spec else len(self._tiers)
        return f"tier_{idx}"

    def _find_spec(self, depth: int) -> TierSpec | None:
        for spec in self._tiers:
            if spec.matches(depth):
                return spec
        return None


def load_cascade_config_if_exists(config_path: str | None = None) -> list[TierSpec]:
    """Return the cascade tiers from ``config_path`` if it loads, else defaults.

    Thin convenience wrapper over ``cascade.config.load_cascade_config`` that
    yields a ``list[TierSpec]`` (the shape ``TierMapper`` is constructed from)
    rather than a ready-made mapper, so the session factory can wrap it itself.
    Falls back to ``DEFAULT_TIERS`` whenever the config is missing or unreadable.
    """
    # Imported lazily to avoid a tiers <-> config import cycle.
    from axor_openrouter.cascade.config import load_cascade_config

    mapper = load_cascade_config(config_path)
    tiers = list(getattr(mapper, "_tiers", []))
    return tiers or list(DEFAULT_TIERS)
