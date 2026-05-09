from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    id: str
    tier: int          # 0=flagship, 1=strong, 2=fast, 3=cheap, 4=free
    cost_in: float     # USD per 1M input tokens
    cost_out: float    # USD per 1M output tokens
    context_k: int     # context window in thousands of tokens
    coding: bool = True
    reasoning: bool = False

    @property
    def is_free(self) -> bool:
        return self.cost_in == 0.0 and self.cost_out == 0.0


# Ordered by tier (0 = most capable / expensive, 4 = free)
MODEL_REGISTRY: list[ModelEntry] = [
    # ── Tier 0: Flagship ──────────────────────────────────────────────────────
    ModelEntry("anthropic/claude-opus-4-7",                  0, 15.00, 75.00, 200),
    ModelEntry("openai/gpt-4o",                              0,  2.50, 10.00, 128),
    ModelEntry("google/gemini-2.5-pro",                      0,  1.25,  10.00, 1000),

    # ── Tier 1: Strong ────────────────────────────────────────────────────────
    ModelEntry("anthropic/claude-sonnet-4-6",                1,  3.00, 15.00, 200),
    ModelEntry("moonshotai/kimi-k2",                         1,  0.14,  0.56, 128, reasoning=True),
    ModelEntry("deepseek/deepseek-r1",                       1,  0.55,  2.19, 128, reasoning=True),
    ModelEntry("google/gemini-2.5-flash",                    1,  0.15,  0.60, 1000),

    # ── Tier 2: Fast ──────────────────────────────────────────────────────────
    ModelEntry("openai/gpt-4o-mini",                         2,  0.15,  0.60, 128),
    ModelEntry("anthropic/claude-haiku-4-5",                 2,  0.80,  4.00, 200),
    ModelEntry("meta-llama/llama-3.3-70b-instruct",          2,  0.12,  0.30, 128),
    ModelEntry("mistralai/mistral-small-3.2-24b-instruct",   2,  0.10,  0.30,  32),

    # ── Tier 3: Cheap ─────────────────────────────────────────────────────────
    ModelEntry("mistralai/mistral-7b-instruct",              3,  0.055, 0.055, 32),
    ModelEntry("qwen/qwen3-8b",                              3,  0.06,  0.12,  32),

    # ── Tier 4: Free ──────────────────────────────────────────────────────────
    ModelEntry("meta-llama/llama-3.3-70b-instruct:free",     4,  0.00,  0.00, 128),
    ModelEntry("google/gemma-3-27b-it:free",                 4,  0.00,  0.00, 96),
    ModelEntry("qwen/qwen3-8b:free",                         4,  0.00,  0.00, 32),
]

_BY_ID: dict[str, ModelEntry] = {m.id: m for m in MODEL_REGISTRY}


# ── Complexity estimation ─────────────────────────────────────────────────────

# Keywords that push complexity up or down
_HIGH_TOKENS = re.compile(
    r"\b(architect|design|analyze|analyse|critique|evaluate|optimize|"
    r"comprehensive|multi.component|system|orchestrat|strategiz|reason|"
    r"trade.?off|security|scalab|distributed|concurrent|algorithm)\b",
    re.IGNORECASE,
)
_MED_TOKENS = re.compile(
    r"\b(implement|build|create|develop|refactor|write|generate|"
    r"endpoint|api|function|class|module|test|route|service)\b",
    re.IGNORECASE,
)
_LOW_TOKENS = re.compile(
    r"\b(return only|write only|only the (complete|file|python)|"
    r"translate|convert|format|rename|summarize|restate|copy)\b",
    re.IGNORECASE,
)


def estimate_complexity(task: str) -> int:
    """
    Return a complexity tier hint: 0 (complex) → 2 (trivial).

    Used together with depth to select the final model tier.
    """
    words = task.split()
    n = len(words)

    high_hits = len(_HIGH_TOKENS.findall(task))
    med_hits  = len(_MED_TOKENS.findall(task))
    low_hits  = len(_LOW_TOKENS.findall(task))

    score = high_hits * 2 + med_hits - low_hits * 3

    if score >= 4 or n > 300:
        return 0   # complex — needs flagship/strong
    if score <= -2 or (n < 20 and med_hits == 0 and high_hits == 0):
        return 2   # trivial — fast/cheap is fine
    if score >= 1 or n >= 30:
        return 1   # medium
    return 2       # short with no strong signal → trivial


# ── Smart model selector ──────────────────────────────────────────────────────

@dataclass
class SmartModelSelector:
    """
    Selects the best model for each node based on task complexity and depth.

    Selection logic:
      effective_tier = clamp(complexity_tier + depth_penalty, min_tier, 4)
      → pick cheapest model at effective_tier that is available

    depth_penalty:
      depth 0 → +0  (use complexity as-is)
      depth 1 → +1  (one tier cheaper)
      depth 2 → +2  (two tiers cheaper)
      depth 3+ → +3 (cheapest / free)

    Parameters
    ----------
    max_cost_in : float | None
        Hard ceiling on input token price (USD/1M). Models above this are excluded.
    prefer_free_at_depth : int
        At this depth and beyond, always use a free model if available.
    pinned_root_model : str | None
        If set, depth=0 always uses this model regardless of complexity.
    """

    max_cost_in: float | None = None
    prefer_free_at_depth: int = 3
    pinned_root_model: str | None = None
    _registry: list[ModelEntry] = field(default_factory=lambda: list(MODEL_REGISTRY))

    def select(self, task: str, depth: int) -> str:
        # depth=0 can be pinned
        if depth == 0 and self.pinned_root_model:
            return self.pinned_root_model

        # always free at deep levels
        if depth >= self.prefer_free_at_depth:
            free = [m for m in self._registry if m.is_free]
            if free:
                return free[0].id

        complexity = estimate_complexity(task)
        depth_penalty = min(depth, 3)
        effective_tier = min(complexity + depth_penalty, 4)

        candidates = self._candidates(effective_tier)
        if not candidates:
            # fallback: just pick the cheapest available
            candidates = sorted(self._registry, key=lambda m: m.cost_in)

        return candidates[0].id

    def _candidates(self, target_tier: int) -> list[ModelEntry]:
        pool = [
            m for m in self._registry
            if m.tier >= target_tier
            and (self.max_cost_in is None or m.cost_in <= self.max_cost_in)
        ]
        # prefer exact tier match, then sort by cost ascending
        exact  = sorted([m for m in pool if m.tier == target_tier], key=lambda m: m.cost_in)
        higher = sorted([m for m in pool if m.tier >  target_tier], key=lambda m: m.cost_in)
        return exact + higher


def get_model_entry(model_id: str) -> ModelEntry | None:
    return _BY_ID.get(model_id)
