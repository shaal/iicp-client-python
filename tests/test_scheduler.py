# ADR-016: IICP client SDK conformance
"""Unit tests for the QoS-aware admission classifier (parity Block C)."""

from __future__ import annotations

from iicp_client.scheduler import (
    QOS_PRIORITY,
    QUEUE_ELIGIBLE,
    is_queue_eligible,
    qos_priority,
)


class TestQosPriority:
    def test_ordering_realtime_highest(self):
        assert qos_priority("realtime") < qos_priority("interactive")
        assert qos_priority("interactive") < qos_priority("batch")
        assert qos_priority("batch") <= qos_priority("best_effort")

    def test_hyphen_and_underscore_spellings_equal(self):
        assert qos_priority("best-effort") == qos_priority("best_effort")

    def test_unknown_defaults_to_lowest(self):
        assert qos_priority("nonsense") == 3
        assert qos_priority(None) == 3


class TestQueueEligibility:
    def test_high_priority_is_eligible(self):
        assert is_queue_eligible("realtime")
        assert is_queue_eligible("interactive")

    def test_low_priority_fails_fast(self):
        assert not is_queue_eligible("batch")
        assert not is_queue_eligible("best_effort")
        assert not is_queue_eligible("best-effort")
        assert not is_queue_eligible(None)
        assert not is_queue_eligible("unknown")

    def test_eligible_set_is_exactly_realtime_interactive(self):
        assert QUEUE_ELIGIBLE == frozenset({"realtime", "interactive"})

    def test_priority_map_covers_all_named_tiers(self):
        for tier in ("realtime", "interactive", "batch", "best_effort", "best-effort"):
            assert tier in QOS_PRIORITY
