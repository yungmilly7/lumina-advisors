"""A tiny thread-safe token bucket, shared by data clients that talk to an
API with a hard per-minute rate limit (Finnhub's free tier: 60 calls/min).

The ingestion pipeline fetches many tickers concurrently (see pipeline.py's
thread pool), so the limiter has to be shared and lock-protected -- each
worker thread calls `acquire()` before making a request, and it blocks just
long enough to stay under the configured rate, however many threads are
calling it at once. This is what lets us safely max out a free-tier budget
across a large universe without ever tripping the provider's own 429s.
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float | None = None):
        self.rate_per_sec = rate_per_sec
        self.capacity = capacity if capacity is not None else max(rate_per_sec, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, cost: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                wait = (cost - self._tokens) / self.rate_per_sec
            time.sleep(wait)
