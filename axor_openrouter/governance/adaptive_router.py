"""Adaptive tier-shift governor driven by budget feedback."""
from __future__ import annotations

import threading


class AdaptiveRouter:
    """Thread-safe holder of the current tier shift (-1 / 0 / +1).

    The BudgetSubscriber calls `set_shift()` whenever the policy engine
    recommends a tier level — this is idempotent and does not accumulate.
    The executor calls `current_shift()` on every model-resolution call.
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

    def set_shift(self, value: int) -> None:
        """Clamp-set the shift to an absolute value (idempotent, no accumulation)."""
        with self._lock:
            self._shift = max(self._min, min(self._max, value))

    def apply_shift(self, delta: int) -> None:
        """Clamp-apply a delta to the current shift (accumulating)."""
        with self._lock:
            self._shift = max(self._min, min(self._max, self._shift + delta))

    def reset(self) -> None:
        with self._lock:
            self._shift = 0
