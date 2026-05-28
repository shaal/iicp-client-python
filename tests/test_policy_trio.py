# ADR-016: IICP client SDK conformance
"""Unit tests for the policy trio: token_validator, idempotency, trust_auditor
(parity Block E)."""

from __future__ import annotations

import time

from iicp_client.idempotency import IdempotencyGuard
from iicp_client.token_validator import TokenValidator
from iicp_client.trust_auditor import models_diverge


class TestTokenValidator:
    def test_empty_expected_rejects_all(self):
        v = TokenValidator("")
        assert v.is_valid("anything") is False

    def test_matching_token_accepted(self):
        v = TokenValidator("secret-123")
        assert v.is_valid("secret-123") is True

    def test_mismatched_token_rejected(self):
        v = TokenValidator("secret-123")
        assert v.is_valid("secret-456") is False

    def test_none_presented_rejected(self):
        v = TokenValidator("secret-123")
        assert v.is_valid(None) is False

    def test_update_token_after_registration(self):
        v = TokenValidator("old")
        v.update_token("new-from-directory")
        assert v.is_valid("new-from-directory") is True
        assert v.is_valid("old") is False


class TestIdempotencyGuard:
    def test_first_seen_is_new(self):
        g = IdempotencyGuard()
        assert g.check_and_register("task-1") is True

    def test_duplicate_rejected(self):
        g = IdempotencyGuard()
        g.check_and_register("task-1")
        assert g.check_and_register("task-1") is False

    def test_distinct_ids_both_new(self):
        g = IdempotencyGuard()
        assert g.check_and_register("task-1") is True
        assert g.check_and_register("task-2") is True

    def test_empty_task_id_always_new(self):
        g = IdempotencyGuard()
        assert g.check_and_register("") is True
        assert g.check_and_register(None) is True

    def test_ttl_expiry_allows_reuse(self):
        g = IdempotencyGuard(ttl_s=0.05)
        assert g.check_and_register("task-1") is True
        assert g.check_and_register("task-1") is False
        time.sleep(0.06)
        assert g.check_and_register("task-1") is True

    def test_size_reflects_entries(self):
        g = IdempotencyGuard()
        g.check_and_register("a")
        g.check_and_register("b")
        assert g.size == 2


class TestTrustAuditorCore:
    def test_no_divergence_when_health_superset(self):
        assert models_diverge(["a", "b"], ["a", "b", "c"]) == set()

    def test_missing_model_is_divergence(self):
        assert models_diverge(["a", "b"], ["a"]) == {"b"}

    def test_empty_registered_never_diverges(self):
        assert models_diverge([], ["a"]) == set()
