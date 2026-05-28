# SPDX-License-Identifier: Apache-2.0
"""Idempotency guard — task_id dedup with TTL eviction (parity Block E, #340).

Port of iicp-adapter `services/idempotency.py` (ADR-010). Prevents duplicate task
execution when a proxy retries a CALL after a transient failure. Distinct from the nonce
replay cache: nonce protects a signed request from replay; this dedups on `task_id` so a
retried-but-already-running task isn't executed twice.

In-memory with lazy eviction (5-minute TTL matching the proxy retry window). Cross-restart
dedup is intentionally not provided — on restart the proxy would time out the original CALL
and generate a fresh task_id for the retry, so a persistent store isn't required (ADR-010).
"""

from __future__ import annotations

import threading
import time

_TTL_S = 300.0  # 5-minute window — matches ADR-010 §3 and the nonce cache.


class IdempotencyGuard:
    def __init__(self, ttl_s: float = _TTL_S) -> None:
        self._ttl = ttl_s
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_register(self, task_id: str | None) -> bool:
        """Return True if task_id is new (first time seen), False if a duplicate.

        Empty/None task_ids are always treated as new (the caller didn't opt into
        idempotency). Expired entries are evicted lazily on each call.
        """
        if not task_id:
            return True
        now = time.monotonic()
        with self._lock:
            expired = [k for k, exp in self._seen.items() if exp <= now]
            for k in expired:
                del self._seen[k]
            if task_id in self._seen:
                return False
            self._seen[task_id] = now + self._ttl
        return True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._seen)
