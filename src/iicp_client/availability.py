# SPDX-License-Identifier: Apache-2.0
"""Time-based availability windows — operator capacity shaping by time-of-day.

Port of iicp-adapter `scheduling/availability.py` (parity Block D, #340). Lets an operator
dedicate different fractions of `max_concurrent` to IICP tasks at different times — e.g.
full capacity overnight, 30% during business hours when the machine has other work.

Window semantics:
  - start/end: "HH:MM" strings in local system time (not UTC).
  - share: fraction of max_concurrent to dedicate (0.0 = no tasks, 1.0 = full capacity).
  - Outside all windows: 0.5 (available but not primary) so idle periods aren't dead zones.
  - No windows configured: always 1.0 (most operators, most of the time).

WHY operator-controlled windows rather than directory-controlled scheduling: IICP gives
operators sovereignty over their own capacity (ADR-001). The directory learns live load via
heartbeats and scores accordingly — it doesn't push scheduling decisions to nodes.

Spec: spec/iicp-dir.md §register `availability` field (optional, Phase 3+).
"""

from __future__ import annotations

import datetime
from typing import TypedDict


class Window(TypedDict):
    start: str  # "HH:MM"
    end: str
    share: float  # fraction of max_concurrent to dedicate (0.0–1.0)


class AvailabilityEvaluator:
    """Evaluates time-based availability windows. Local time; no windows → always 1.0."""

    def __init__(self, windows: list[Window] | None = None) -> None:
        self._windows = windows or []

    def current_share(self, now: datetime.time | None = None) -> float:
        """Return the capacity share [0,1] for the current time of day."""
        if not self._windows:
            return 1.0

        t = now or datetime.datetime.now().time()
        current = t.strftime("%H:%M")

        for w in self._windows:
            if w["start"] <= w["end"]:
                # Normal window (e.g. 08:00–22:00)
                if w["start"] <= current <= w["end"]:
                    return float(w["share"])
            else:
                # Midnight-spanning window (e.g. 22:00–06:00)
                if current >= w["start"] or current <= w["end"]:
                    return float(w["share"])

        # Outside all windows — half capacity (available but not primary)
        return 0.5

    def effective_max_concurrent(self, base_max: int, now: datetime.time | None = None) -> int:
        """Scale base max_concurrent by the current share (floor 1 when share > 0).

        A base of 0 (operator explicitly disabled) stays 0 — the floor only protects a
        working node from being shaped down to 0 by a fractional share.
        """
        if base_max <= 0:
            return 0
        share = self.current_share(now)
        if share <= 0.0:
            return 0
        return max(1, int(base_max * share))

    def is_within_window(self, now: datetime.time | None = None) -> bool:
        if not self._windows:
            return True
        return self.current_share(now) > 0.0
