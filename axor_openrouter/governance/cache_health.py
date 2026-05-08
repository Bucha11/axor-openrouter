"""Rolling-window cache-hit-rate monitor.

If the hit-rate falls below `threshold` we tell TtlChooser to downgrade
to short TTLs so we stop paying for cache writes that never pay off.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_openrouter.caching.ttl_chooser import TtlChooser


class CacheHealthMonitor:
    def __init__(
        self,
        ttl_chooser: "TtlChooser",
        window: int = 50,
        threshold: float = 0.20,
    ) -> None:
        self._chooser = ttl_chooser
        self._threshold = threshold
        self._hits: deque[bool] = deque(maxlen=window)
        self._downgraded = False

    # ------------------------------------------------------------------
    # Public API  (called by executor after each OpenRouter response)
    # ------------------------------------------------------------------

    def record_hit(self) -> None:
        self._hits.append(True)
        self._evaluate()

    def record_miss(self) -> None:
        self._hits.append(False)
        self._evaluate()

    def hit_rate(self) -> float:
        if not self._hits:
            return 1.0
        return sum(self._hits) / len(self._hits)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        rate = self.hit_rate()
        if rate < self._threshold and not self._downgraded:
            self._chooser.downgrade_to_short()
            self._downgraded = True
        elif rate >= self._threshold and self._downgraded:
            # Recover: allow full TTLs again.
            self._downgraded = False
