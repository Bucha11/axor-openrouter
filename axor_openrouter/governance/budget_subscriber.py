"""Wires BudgetTracker.subscribe() to AdaptiveRouter + BudgetPolicyEngine."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_core.budget import BudgetTracker, BudgetPolicyEngine
    from .adaptive_router import AdaptiveRouter


class BudgetSubscriber:
    """Subscribes to a BudgetTracker and drives AdaptiveRouter tier shifts.

    On every `record()` call the tracker fires our callback with a
    `RecordEvent` dict.  We ask the policy engine whether a tier shift is
    warranted and forward the recommendation to the AdaptiveRouter.
    """

    def __init__(
        self,
        tracker: "BudgetTracker",
        policy: "BudgetPolicyEngine",
        router: "AdaptiveRouter",
    ) -> None:
        self._policy = policy
        self._router = router
        # Keep the unsubscribe callable so callers can detach cleanly.
        self._unsub = tracker.subscribe(self._on_record)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_record(self, event: dict) -> None:
        """Called synchronously by BudgetTracker after every record()."""
        delta = self._policy.suggest_tier_shift()
        if delta != 0:
            self._router.apply_shift(delta)

    def detach(self) -> None:
        """Remove the subscription from the tracker."""
        self._unsub()
