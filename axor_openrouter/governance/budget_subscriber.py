"""Wires BudgetTracker.subscribe() to AdaptiveRouter + BudgetPolicyEngine."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_core.budget import BudgetTracker, BudgetPolicyEngine
    from .adaptive_router import AdaptiveRouter


class BudgetSubscriber:
    """Subscribes to a BudgetTracker and drives AdaptiveRouter tier shifts.

    On every `record()` call the tracker fires our callback.  We ask the
    policy engine for the current tier shift and set it directly on the
    AdaptiveRouter — using set_shift() so the value never accumulates.
    """

    def __init__(
        self,
        tracker: "BudgetTracker",
        policy: "BudgetPolicyEngine",
        router: "AdaptiveRouter",
    ) -> None:
        self._policy = policy
        self._router = router
        self._unsub = tracker.subscribe(self._on_record)

    def _on_record(self, event: dict) -> None:
        """Called synchronously by BudgetTracker after every record()."""
        shift = self._policy.suggest_tier_shift()
        self._router.set_shift(shift)

    def detach(self) -> None:
        """Remove the subscription from the tracker."""
        self._unsub()
