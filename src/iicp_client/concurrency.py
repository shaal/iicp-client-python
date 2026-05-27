# SPDX-License-Identifier: Apache-2.0
"""Unified concurrency gate for the hybrid client.

Port of iicp-adapter `services/concurrency.py` (iter-1410) into the
iicp-client-python SDK as part of the adapter→hybrid-client migration
(tracker iicp.network#340 Tier 2 Item 5).

Caps simultaneous inference tasks at `limits.max_concurrent` from the
node's registration envelope (spec/iicp-dir.md §register). When the gate
is full, the SDK returns:

  - HTTP transport (`IicpNode.serve`)         → 429 IICP-E021 +
                                                  Retry-After: 2
  - Native IICP transport (`IicpTcpServer`)   → RESPONSE frame with
                                                  error_code=429,
                                                  error_message="IICP-E021..."

Both surfaces share this single primitive so the directory's load score
(ADR-008) and the proxy's fall-back routing see the same back-pressure
signal regardless of which transport carried the CALL.

WHY asyncio.Semaphore non-blocking acquire rather than a queue: a queue
would silently buffer tasks, masking overload from the proxy. The proxy
must know immediately whether a node can accept a task so it can route
elsewhere; a 429-equivalent is more useful than hidden queuing
(adapter doc verbatim).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class CapacityExceededError(Exception):
    """Raised by [`ConcurrencyGate.acquire`] when the gate is at capacity.

    Callers MUST translate to IICP-E021 on the transport they're answering:
      - HTTP `/v1/task` → 429 with `Retry-After` header
      - IICP TCP CALL  → RESPONSE frame with error_code=429
    """

    def __init__(self, max_concurrent: int) -> None:
        super().__init__(f"max_concurrent ({max_concurrent}) reached")
        self.max_concurrent = max_concurrent


class ConcurrencyGate:
    """Cap simultaneous inference tasks at `max_concurrent`.

    Usage::

        gate = ConcurrencyGate(max_concurrent=4)
        try:
            async with gate.acquire():
                result = await run_task(...)
        except CapacityExceededError as exc:
            return {
                "error_code": 429,
                "error_message": f"IICP-E021: max_concurrent={exc.max_concurrent} reached",
            }

    `active_jobs` and `load` are read-only views useful for heartbeat
    payloads — the directory's NodeScorer uses load to down-rank busy
    nodes for future requests (ADR-008).
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._active = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a slot or raise [`CapacityExceededError`].

        Non-blocking — the gate raises immediately when at capacity rather
        than queuing. Callers catch the exception and respond with
        IICP-E021 on whichever transport they're serving.
        """
        if self._sem.locked() and self._active >= self._max:
            raise CapacityExceededError(self._max)
        async with self._sem:
            self._active += 1
            try:
                yield
            finally:
                self._active -= 1

    @property
    def active_jobs(self) -> int:
        return self._active

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def load(self) -> float:
        """Current load fraction in [0.0, 1.0]. Reported in heartbeats so the
        directory's NodeScorer can down-rank busy nodes (ADR-008)."""
        if self._max == 0:
            return 1.0
        return self._active / self._max
