# SPDX-License-Identifier: Apache-2.0
"""QoS-aware admission policy for the provider serve path (parity Block C, #340).

Port of the QoS *contract* from iicp-adapter `scheduling/queue.py`. The adapter runs a
full asyncio PriorityQueue dispatcher; the SDK serve gate is deliberately fail-fast
(see `concurrency.py` — queuing would hide overload from the proxy). To close the cat-8
parity gap without contradicting that design, the SDK applies **QoS-aware admission**:

  - realtime / interactive  → queue-eligible: wait briefly (`QUEUE_WAIT_S`) for a slot.
    These callers have deadlines but expect to wait a moment, not be dropped.
  - batch / best-effort / unspecified → fail fast with IICP-E021 so the proxy sees
    back-pressure immediately and routes elsewhere.

Priority ordering (lower = higher priority) is exposed for metrics/telemetry parity with
the adapter. This module is intentionally a small, shared, testable classifier rather
than a queue.

Spec: spec/iicp-semantics.md §QoS (qos field semantics and tier definitions).
ADR: ADR-006 (QoS scheduling contract).
"""

from __future__ import annotations

# Lower value = higher priority. Both hyphen and underscore spellings are accepted
# because the adapter uses "best-effort" (hyphen) and the SDK uses "best_effort".
QOS_PRIORITY: dict[str | None, int] = {
    "realtime": 0,
    "interactive": 1,
    "batch": 2,
    "best_effort": 3,
    "best-effort": 3,
    None: 3,
}

# Tiers that wait briefly for a slot rather than failing fast at capacity.
QUEUE_ELIGIBLE: frozenset[str] = frozenset({"realtime", "interactive"})

# Bounded wait for queue-eligible tiers (seconds). Kept short so a busy node still
# surfaces back-pressure quickly rather than buffering indefinitely.
QUEUE_WAIT_S: float = 2.0


def qos_priority(qos: str | None) -> int:
    """Priority rank for a QoS class (lower = higher priority; unknown → 3)."""
    return QOS_PRIORITY.get(qos, 3)


def is_queue_eligible(qos: str | None) -> bool:
    """True if a task of this QoS class should wait briefly for a slot at capacity."""
    return qos in QUEUE_ELIGIBLE
