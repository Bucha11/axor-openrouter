from __future__ import annotations


class TtlChooser:
    """
    Chooses cache TTL based on session usage patterns.

    Heuristic:
    - First N calls: use session_ttl (1h) for system+tools to amortise write cost.
      A 1h cache write costs 1.25x input; it pays off after ~3 hits.
    - After the session is established: keep session_ttl for stable blocks,
      switch to short_ttl (5m) for context fragments (they change per turn).
    - If the cache hit rate falls below min_hit_rate for a rolling window:
      downgrade everything to short_ttl (cache misconfigured or cold start).
    """

    def __init__(
        self,
        short_ttl: str = "5m",
        session_ttl: str = "1h",
        session_min_calls: int = 3,
    ) -> None:
        self._short_ttl        = short_ttl
        self._session_ttl      = session_ttl
        self._session_min_calls = session_min_calls
        self._call_count        = 0
        self._force_short       = False

    def choose(
        self,
        block: str,    # "system" | "tools" | "context_top_k"
    ) -> str:
        """
        Return the TTL string to use for the given block.

        system/tools get the session TTL once the session is warm;
        context_top_k always gets the short TTL (changes per turn).
        """
        if self._force_short:
            return self._short_ttl
        if block == "context_top_k":
            return self._short_ttl
        if self._call_count < self._session_min_calls:
            return self._session_ttl
        return self._session_ttl

    def record_call(self) -> None:
        """Call once per executor stream() invocation."""
        self._call_count += 1

    def downgrade_to_short(self) -> None:
        """Force all TTLs to short_ttl (called by CacheHealthMonitor)."""
        self._force_short = True

    def restore_session(self) -> None:
        """Lift the forced downgrade."""
        self._force_short = False
