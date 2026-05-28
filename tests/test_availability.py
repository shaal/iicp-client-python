# ADR-016: IICP client SDK conformance
"""Unit tests for time-based availability windows (parity Block D)."""

from __future__ import annotations

import datetime

from iicp_client.availability import AvailabilityEvaluator


def _t(hhmm: str) -> datetime.time:
    h, m = hhmm.split(":")
    return datetime.time(int(h), int(m))


class TestNoWindows:
    def test_no_windows_always_full(self):
        ev = AvailabilityEvaluator()
        assert ev.current_share(_t("03:00")) == 1.0
        assert ev.current_share(_t("14:00")) == 1.0

    def test_no_windows_effective_equals_base(self):
        ev = AvailabilityEvaluator()
        assert ev.effective_max_concurrent(4, _t("09:00")) == 4


class TestNormalWindow:
    def test_inside_window_uses_share(self):
        ev = AvailabilityEvaluator([{"start": "08:00", "end": "22:00", "share": 0.5}])
        assert ev.current_share(_t("12:00")) == 0.5

    def test_outside_window_half_default(self):
        ev = AvailabilityEvaluator([{"start": "08:00", "end": "22:00", "share": 0.5}])
        # 02:00 is outside the single window → 0.5 (available but not primary)
        assert ev.current_share(_t("02:00")) == 0.5

    def test_effective_scales_and_floors_at_one(self):
        ev = AvailabilityEvaluator([{"start": "08:00", "end": "22:00", "share": 0.1}])
        # 4 * 0.1 = 0.4 → floored to 1 (share > 0)
        assert ev.effective_max_concurrent(4, _t("10:00")) == 1


class TestMidnightSpanningWindow:
    def test_overnight_window_matches_after_midnight(self):
        ev = AvailabilityEvaluator([{"start": "22:00", "end": "06:00", "share": 1.0}])
        assert ev.current_share(_t("23:30")) == 1.0
        assert ev.current_share(_t("02:00")) == 1.0

    def test_overnight_window_outside_is_half(self):
        ev = AvailabilityEvaluator([{"start": "22:00", "end": "06:00", "share": 1.0}])
        assert ev.current_share(_t("12:00")) == 0.5


class TestClosedWindow:
    def test_zero_share_gives_zero_capacity(self):
        ev = AvailabilityEvaluator([{"start": "00:00", "end": "23:59", "share": 0.0}])
        assert ev.effective_max_concurrent(4, _t("10:00")) == 0
        assert ev.is_within_window(_t("10:00")) is False
