"""Unit tests for cip_policy — S.12 §2.2 worker-role gate."""

from __future__ import annotations

import asyncio
import json

import httpx
import respx

from iicp_client import IicpNode, NodeConfig
from iicp_client.cip_policy import (
    CooperativeInferencePolicy,
    configure_policy,
    get_policy,
)

# ── Gate predicates ───────────────────────────────────────────────────────


class TestGatePredicates:
    def test_default_is_all_off(self):
        p = CooperativeInferencePolicy()
        assert p.enabled is False
        assert p.allow_coordinator is False
        assert p.allow_worker is False
        assert p.check_coordinator() is False
        assert p.check_worker() is False

    def test_enabled_alone_does_not_open_gates(self):
        # Both `enabled` AND the role flag must be on
        p = CooperativeInferencePolicy(enabled=True)
        assert p.check_coordinator() is False
        assert p.check_worker() is False

    def test_role_flag_alone_does_not_open_gates(self):
        # Symmetric — the global enable must also be on
        p = CooperativeInferencePolicy(allow_coordinator=True, allow_worker=True)
        assert p.check_coordinator() is False
        assert p.check_worker() is False

    def test_enabled_plus_coordinator_opens_coordinator_only(self):
        p = CooperativeInferencePolicy(enabled=True, allow_coordinator=True)
        assert p.check_coordinator() is True
        assert p.check_worker() is False

    def test_enabled_plus_worker_opens_worker_only(self):
        p = CooperativeInferencePolicy(enabled=True, allow_worker=True)
        assert p.check_coordinator() is False
        assert p.check_worker() is True


# ── Capacity gate (S.12 §2.2) ─────────────────────────────────────────────


class TestCapacityGate:
    def test_max_concurrent_remote_lower_bound_enforced(self):
        p = CooperativeInferencePolicy(max_concurrent_remote=0)
        assert p.max_concurrent_remote == 1  # clamped to >=1

    def test_max_worker_timeout_upper_bound_enforced(self):
        p = CooperativeInferencePolicy(max_worker_timeout_ms=999_999)
        assert p.max_worker_timeout_ms == 60_000

    def test_slot_acquire_and_release(self):
        p = CooperativeInferencePolicy(max_concurrent_remote=2)
        assert p.try_acquire_cip_slot() is True
        assert p.try_acquire_cip_slot() is True
        # At capacity now — S.12 §2.2 says we MUST return IICP-E021 not queue
        assert p.try_acquire_cip_slot() is False
        # Release one → next acquire succeeds
        p.release_cip_slot()
        assert p.try_acquire_cip_slot() is True


# ── Register-payload integration ──────────────────────────────────────────


class TestRegisterPayloadIntegration:
    @respx.mock
    def test_cip_disabled_emits_no_policy_block(self):
        """Default-off policy should NOT clutter the register payload."""
        route = respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={"node_token": "tok", "node_id": "n"})
        )
        node = IicpNode(
            NodeConfig(
                node_id="n",
                endpoint="https://provider.example:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                model="q",
                directory_url="https://iicp.test",
                cip_policy=CooperativeInferencePolicy(),  # default OFF
            )
        )
        asyncio.run(node.register())
        body = json.loads(route.calls[0].request.content)
        assert "policy" not in body

    @respx.mock
    def test_cip_worker_enabled_emits_allow_remote_inference_true(self):
        """When CIP worker is enabled, register payload MUST include
        `policy.allow_remote_inference = true` so the directory ranks the
        node as CIP-Provider in /v1/discover."""
        route = respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={"node_token": "tok", "node_id": "n"})
        )
        node = IicpNode(
            NodeConfig(
                node_id="n",
                endpoint="https://provider.example:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                model="q",
                directory_url="https://iicp.test",
                cip_policy=CooperativeInferencePolicy(enabled=True, allow_worker=True),
            )
        )
        asyncio.run(node.register())
        body = json.loads(route.calls[0].request.content)
        assert body["policy"]["allow_remote_inference"] is True

    @respx.mock
    def test_module_level_policy_used_when_node_config_unset(self):
        """When NodeConfig.cip_policy is None, register() falls back to
        the module-level cip_policy.get_policy() — operators can configure
        once and have it apply to all nodes."""
        configure_policy(enabled=True, allow_worker=True)
        try:
            route = respx.post("https://iicp.test/v1/register").mock(
                return_value=httpx.Response(201, json={"node_token": "tok", "node_id": "n"})
            )
            node = IicpNode(
                NodeConfig(
                    node_id="n",
                    endpoint="https://provider.example:8080",
                    intent="urn:iicp:intent:llm:chat:v1",
                    model="q",
                    directory_url="https://iicp.test",
                    # cip_policy not set → falls back to module-level
                )
            )
            asyncio.run(node.register())
            body = json.loads(route.calls[0].request.content)
            assert body.get("policy", {}).get("allow_remote_inference") is True
        finally:
            # Reset module-level policy to safe defaults so we don't leak into
            # other tests (especially test_cip_disabled_emits_no_policy_block).
            configure_policy()


# ── Module-level configuration ─────────────────────────────────────────────


class TestModuleLevelPolicy:
    def test_get_policy_returns_default_when_unconfigured(self):
        # Reset first in case earlier tests left state
        configure_policy()
        p = get_policy()
        assert p.enabled is False
        assert p.allow_worker is False

    def test_configure_policy_replaces_global(self):
        configure_policy(enabled=True, allow_worker=True, max_concurrent_remote=5)
        p = get_policy()
        assert p.enabled is True
        assert p.allow_worker is True
        assert p.max_concurrent_remote == 5
        # Reset
        configure_policy()
