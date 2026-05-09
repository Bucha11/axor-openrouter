from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_core.contracts.policy import TaskComplexity


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    id: str
    tier: int          # 0=flagship, 1=strong, 2=fast, 3=cheap, 4=free
    cost_in: float     # USD per 1M input tokens
    cost_out: float    # USD per 1M output tokens
    context_k: int     # context window in thousands of tokens

    @property
    def is_free(self) -> bool:
        return self.cost_in == 0.0 and self.cost_out == 0.0


# Ordered by tier (0 = most capable / expensive, 4 = free)
MODEL_REGISTRY: list[ModelEntry] = [
    # ── Tier 0: Flagship ──────────────────────────────────────────────────────
    ModelEntry("anthropic/claude-opus-4-7",                  0, 15.00, 75.00, 200),
    ModelEntry("openai/gpt-4o",                              0,  2.50, 10.00, 128),
    ModelEntry("google/gemini-2.5-pro",                      0,  1.25, 10.00, 1000),

    # ── Tier 1: Strong ────────────────────────────────────────────────────────
    ModelEntry("anthropic/claude-sonnet-4-6",                1,  3.00, 15.00, 200),
    ModelEntry("moonshotai/kimi-k2",                         1,  0.14,  0.56, 128),
    ModelEntry("deepseek/deepseek-r1",                       1,  0.55,  2.19, 128),
    ModelEntry("google/gemini-2.5-flash",                    1,  0.15,  0.60, 1000),

    # ── Tier 2: Fast ──────────────────────────────────────────────────────────
    ModelEntry("openai/gpt-4o-mini",                         2,  0.15,  0.60, 128),
    ModelEntry("anthropic/claude-haiku-4-5",                 2,  0.80,  4.00, 200),
    ModelEntry("meta-llama/llama-3.3-70b-instruct",          2,  0.12,  0.30, 128),
    ModelEntry("mistralai/mistral-small-3.2-24b-instruct",   2,  0.10,  0.30,  32),

    # ── Tier 3: Cheap ─────────────────────────────────────────────────────────
    ModelEntry("qwen/qwen3-8b",                              3,  0.06,  0.12,  32),

    # ── Tier 4: Free ──────────────────────────────────────────────────────────
    ModelEntry("meta-llama/llama-3.3-70b-instruct:free",     4,  0.00,  0.00, 128),
    ModelEntry("google/gemma-3-27b-it:free",                 4,  0.00,  0.00,  96),
    ModelEntry("qwen/qwen3-8b:free",                         4,  0.00,  0.00,  32),
]

_BY_ID: dict[str, ModelEntry] = {m.id: m for m in MODEL_REGISTRY}

# TaskComplexity → base tier (before depth penalty)
_COMPLEXITY_TO_TIER: dict[str, int] = {
    "expansive": 0,   # EXPANSIVE → flagship (complex orchestration)
    "moderate":  1,   # MODERATE  → strong
    "focused":   2,   # FOCUSED   → fast is enough
}


# ── Smart model selector ──────────────────────────────────────────────────────

@dataclass
class SmartModelSelector:
    """
    Selects the best model for each node using axor-core's TaskSignal.

    Selection logic:
      base_tier      = COMPLEXITY_TO_TIER[envelope.task_signal.complexity]
      depth_penalty  = min(depth, 3)
      effective_tier = clamp(base_tier + depth_penalty, 0, 4)
      → cheapest model at effective_tier

    Falls back to task text length heuristic when task_signal is None
    (e.g. override_policy path where TaskAnalyzer is skipped).

    Parameters
    ----------
    max_cost_in : float | None
        Hard ceiling on input token price (USD/1M). Models above excluded.
    prefer_free_at_depth : int
        At this depth and beyond, always prefer a free model if available.
    pinned_root_model : str | None
        If set, depth=0 always uses this model regardless of complexity.
    """

    max_cost_in: float | None = None
    prefer_free_at_depth: int = 3
    pinned_root_model: str | None = None
    _registry: list[ModelEntry] = field(default_factory=lambda: list(MODEL_REGISTRY))

    def select(
        self,
        depth: int,
        task_signal: "TaskComplexity | None" = None,
        task: str = "",
        skip: frozenset[str] = frozenset(),
    ) -> str:
        if depth == 0 and self.pinned_root_model and self.pinned_root_model not in skip:
            return self.pinned_root_model

        if depth >= self.prefer_free_at_depth:
            free = [m for m in self._registry if m.is_free and m.id not in skip]
            if free:
                return free[0].id

        base_tier = self._base_tier(task_signal, task)
        effective_tier = min(base_tier + min(depth, 3), 4)

        candidates = self._candidates(effective_tier, skip=skip)
        if not candidates:
            candidates = sorted(
                [m for m in self._registry if m.id not in skip],
                key=lambda m: m.cost_in,
            )

        return candidates[0].id if candidates else self._default_fallback(skip)

    def _base_tier(self, complexity: "TaskComplexity | None", task: str) -> int:
        if complexity is not None:
            return _COMPLEXITY_TO_TIER.get(complexity.value, 2)
        # fallback: word-count heuristic when signal unavailable
        n = len(task.split())
        if n > 200:
            return 0
        if n > 50:
            return 1
        return 2

    def _candidates(self, target_tier: int, skip: frozenset[str] = frozenset()) -> list[ModelEntry]:
        pool = [
            m for m in self._registry
            if m.tier >= target_tier
            and m.id not in skip
            and (self.max_cost_in is None or m.cost_in <= self.max_cost_in)
        ]
        exact  = sorted([m for m in pool if m.tier == target_tier], key=lambda m: m.cost_in)
        higher = sorted([m for m in pool if m.tier >  target_tier], key=lambda m: m.cost_in)
        return exact + higher

    def _default_fallback(self, skip: frozenset[str]) -> str:
        for m in sorted(self._registry, key=lambda m: m.cost_in):
            if m.id not in skip:
                return m.id
        return self._registry[0].id  # should never happen


def get_model_entry(model_id: str) -> ModelEntry | None:
    return _BY_ID.get(model_id)
