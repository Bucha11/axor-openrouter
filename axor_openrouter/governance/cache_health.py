"""Rolling-window cache-hit-rate monitor."""
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
    # Public API
    # ------------------------------------------------------------------

    def record_hit(self) -> None:
        self._hits.append(True)
        self._evaluate()

    def record_miss(self) -> None:
        self._hits.append(False)
        self._evaluate()

    def record_tool_call(self) -> None:
        """No-op: tool-call turns don't emit cache usage stats."""
        pass

    def record_usage(self, *, cache_read: int, total_input: int) -> None:
        """Record an API response; infer hit/miss from token counts."""
        if total_input > 0 and cache_read > 0:
            self.record_hit()
        else:
            self.record_miss()

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
            self._chooser.restore_session()
            self._downgraded = False
