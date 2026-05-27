# SPDX-License-Identifier: Apache-2.0
"""CIP-W01/CIP-W02 provider-side policy gate — S.12, ADR-012.

Port of iicp-adapter `services/cip_policy.py` (iter-1410) into the
iicp-client-python SDK as part of the adapter→hybrid-client migration
(tracker iicp.network#340 Tier 2 Item 2).

The Cooperative Inference Profile (S.12) gives nodes three independent
operator-controlled roles:

  - Consumer  — dispatches CIP tasks to other nodes (always allowed; just
                a discover + call pattern)
  - Coordinator — fans a single CIP task out to N worker replicas, scores
                  their results (S.12 §3 redundancy / consensus)
  - Worker — accepts CIP sub-tasks from a coordinator on behalf of another
             node's session

Safe Phase-4 defaults: all three are OFF until the operator opts in. The
gate also bounds concurrent worker tasks per S.12 §2.2 — when at capacity
the provider MUST respond with IICP-E021 (`capacity_exhausted`) rather
than silently queue or delay.

ADR-014 safety boundary: remote shell, file access, browser automation,
credential sharing, and private memory access are explicitly prohibited.
The gate enforces "no by default" — operators opt-in field by field.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class CooperativeInferencePolicy:
    """Provider-side CIP policy with safe-by-default flags.

    All booleans default to False. Operators construct this explicitly,
    typically from their own config loading; the SDK never enables CIP
    silently. Mirrors the adapter's CooperativeInferencePolicy contract
    so wire behaviour stays identical between adapter and hybrid clients.
    """

    def __init__(
        self,
        enabled: bool = False,
        allow_coordinator: bool = False,
        allow_worker: bool = False,
        max_replicas: int = 3,
        max_worker_timeout_ms: int = 30_000,
        max_concurrent_remote: int = 2,
    ) -> None:
        self.enabled = enabled
        self.allow_coordinator = allow_coordinator
        self.allow_worker = allow_worker
        self.max_replicas = max(1, max_replicas)
        # Bound max_worker_timeout_ms to a sane range. 60s upper limit matches
        # the adapter's gate so worker requests can't tie up provider slots
        # indefinitely.
        self.max_worker_timeout_ms = max(1, min(60_000, max_worker_timeout_ms))
        self.max_concurrent_remote = max(1, max_concurrent_remote)
        self._remote_sem = threading.BoundedSemaphore(self.max_concurrent_remote)

    # ── Gate predicates ──────────────────────────────────────────────────

    def check_coordinator(self) -> bool:
        """CIP-W01: returns True if this node may act as a CIP coordinator."""
        if not self.enabled or not self.allow_coordinator:
            logger.debug(
                "cip_policy: coordinator denied (enabled=%s, allow_coordinator=%s)",
                self.enabled, self.allow_coordinator,
            )
            return False
        logger.debug(
            "cip_policy: coordinator accepted (max_replicas=%d)", self.max_replicas
        )
        return True

    def check_worker(self) -> bool:
        """CIP-W02: returns True if this node may accept CIP worker tasks."""
        if not self.enabled or not self.allow_worker:
            logger.debug(
                "cip_policy: worker denied (enabled=%s, allow_worker=%s)",
                self.enabled, self.allow_worker,
            )
            return False
        logger.debug(
            "cip_policy: worker accepted (max_timeout_ms=%d)",
            self.max_worker_timeout_ms,
        )
        return True

    # ── Capacity gate (S.12 §2.2) ───────────────────────────────────────

    def try_acquire_cip_slot(self) -> bool:
        """CIP-A1-GATE-06: try to acquire a worker concurrency slot.

        Returns True on success — caller MUST call release_cip_slot() when
        the task completes. Returns False when at capacity, in which case
        the caller MUST respond with IICP-E021 `capacity_exhausted` rather
        than queue or delay (S.12 §2.2 explicit non-silent-queue rule).
        """
        acquired = self._remote_sem.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "cip_policy: CIP worker slot denied (max_concurrent_remote=%d)",
                self.max_concurrent_remote,
            )
        return acquired

    def release_cip_slot(self) -> None:
        """Release a CIP worker slot acquired via try_acquire_cip_slot()."""
        self._remote_sem.release()

    # ── Register-payload shape ──────────────────────────────────────────

    def as_register_policy_block(self) -> dict[str, object]:
        """Build the `policy` sub-object the directory expects in /v1/register.

        Maps the SDK's three CIP flags to the directory's CIP-D1 policy keys
        (spec/iicp-dir.md §3.1 + S.12 §2.1). The directory uses these to
        compute `cip_conformance_level` (CIP-None / CIP-Provider / etc.) and
        return them in /v1/discover responses.

        Only emits keys when CIP is enabled — operators with CIP off shouldn't
        clutter the register payload with `allow_*: false` for every flag.
        """
        if not self.enabled:
            return {}
        return {
            "allow_remote_inference": self.allow_worker,
            # The directory's CIP-D1 currently only flags worker-role acceptance.
            # Coordinator role is coordinator-internal — proxies don't need to
            # know which nodes coordinate; they just discover available workers.
        }


# ── Module-level default policy ──────────────────────────────────────────

_policy: CooperativeInferencePolicy = CooperativeInferencePolicy()


def get_policy() -> CooperativeInferencePolicy:
    """Return the active CIP policy (safe defaults until configure_policy is called)."""
    return _policy


def configure_policy(
    *,
    enabled: bool = False,
    allow_coordinator: bool = False,
    allow_worker: bool = False,
    max_replicas: int = 3,
    max_worker_timeout_ms: int = 30_000,
    max_concurrent_remote: int = 2,
) -> CooperativeInferencePolicy:
    """Replace the module-level CIP policy. Returns the new policy for chaining."""
    global _policy
    _policy = CooperativeInferencePolicy(
        enabled=enabled,
        allow_coordinator=allow_coordinator,
        allow_worker=allow_worker,
        max_replicas=max_replicas,
        max_worker_timeout_ms=max_worker_timeout_ms,
        max_concurrent_remote=max_concurrent_remote,
    )
    return _policy
