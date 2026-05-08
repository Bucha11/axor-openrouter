"""Adaptive tier-shift governor driven by budget feedback."""
from __future__ import annotations

import threading


class AdaptiveRouter:
    """Thread-safe holder of the current tier shift (-1 / 0 / +1).

    The BudgetSubscriber calls `apply_shift()` whenever the policy engine
    recommends moving up or down a tier.  The executor calls `current_shift()`
    on every model-resolution call so all subsequent requests respect the
    latest governance decision.
    """

    def __init__(self, min_shift: int = -2, max_shift: int = 2) -> None:
        self._shift = 0
        self._min = min_shift
        self._max = max_shift
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_shift(self) -> int:
        with self._lock:
            return self._shift

    def apply_shift(self, delta: int) -> None:
        """Clamp-apply a delta to the current shift."""
        with self._lock:
            self._shift = max(self._min, min(self._max, self._shift + delta))

    def reset(self) -> None:
        with self._lock:
            self._shift = 0
