"""Exponential backoff with jitter for transient provider errors."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.2
    """Fractional spread applied to each delay, so concurrent clients do not retry in lockstep."""

    def delay_s(self, attempt: int, rng: random.Random) -> float:
        """Delay after the ``attempt``-th failure (1-based)."""
        base = min(self.max_delay_s, self.base_delay_s * 2.0 ** (attempt - 1))
        return max(0.0, base * (1 + rng.uniform(-self.jitter, self.jitter)))
