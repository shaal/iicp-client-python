# ADR-016: IICP client SDK conformance
"""Unit tests for the mesh PeerManager: merge, prune, relay, HMAC verify
(parity Block F)."""

from __future__ import annotations

import json
import time

from iicp_client.peer_manager import PeerManager
from iicp_client.pricing import sign_body


def _pm(token: str = "") -> PeerManager:
    pm = PeerManager(directory_url="https://dir.example/api", node_token=token)
    pm._own_id = "self"
    return pm


class TestMerge:
    def test_merge_adds_new_peers(self):
        pm = _pm()
        added = pm.merge_peers(
            [
                {"node_id": "a", "endpoint": "http://a"},
                {"node_id": "b", "endpoint": "http://b"},
            ]
        )
        assert added == 2
        assert {p["node_id"] for p in pm.get_peers()} == {"a", "b"}

    def test_merge_skips_self_and_dedups(self):
        pm = _pm()
        pm.merge_peers([{"node_id": "a", "endpoint": "http://a"}])
        added = pm.merge_peers(
            [
                {"node_id": "a", "endpoint": "http://a2"},  # update, not new
                {"node_id": "self", "endpoint": "http://self"},  # skipped
            ]
        )
        assert added == 0
        assert len(pm.get_peers()) == 1


class TestPruneAndRelay:
    def test_prune_drops_stale(self):
        pm = _pm()
        pm.merge_peers([{"node_id": "a", "endpoint": "http://a"}])
        # Force last_contact well past the expiry window. Must be relative to
        # time.monotonic() (NOT a hardcoded 0.0) — on a freshly-booted host
        # monotonic() can be < the 90s window, making 0.0 look "fresh".
        pm._peers["a"]["last_contact"] = time.monotonic() - 10_000
        pruned = pm.prune()
        assert pruned == 1
        assert pm.get_peers() == []

    def test_relay_target_lookup(self):
        pm = _pm()
        pm.merge_peers([{"node_id": "a", "endpoint": "http://a"}])
        assert pm.relay_target("a")["endpoint"] == "http://a"
        assert pm.relay_target("missing") is None


class TestVerifyExchange:
    def test_no_token_accepts(self):
        pm = _pm(token="")
        assert pm.verify_exchange(b"{}", None) is True

    def test_valid_signature_accepted(self):
        pm = _pm(token="secret")
        body = json.dumps({"known_peers": []}).encode()
        sig = sign_body(body, "secret")
        assert pm.verify_exchange(body, sig) is True

    def test_invalid_signature_rejected(self):
        pm = _pm(token="secret")
        body = json.dumps({"known_peers": []}).encode()
        assert pm.verify_exchange(body, "deadbeef") is False
        assert pm.verify_exchange(body, None) is False


class TestRelayElection:
    """R3: deterministic relay election (#341)."""

    def _pm_with_relays(self) -> PeerManager:
        pm = _pm()
        pm.merge_peers([
            {"node_id": "relay-a", "endpoint": "http://relay-a:8020",
             "relay_capable": True, "relay_accept_port": 9485, "relay_load": 0.2},
            {"node_id": "relay-b", "endpoint": "http://relay-b:8020",
             "relay_capable": True, "relay_accept_port": 9486, "relay_load": 0.1},
            {"node_id": "non-relay", "endpoint": "http://nr:8020", "relay_capable": False},
        ])
        return pm

    def test_elect_relay_returns_candidate(self):
        pm = self._pm_with_relays()
        elected = pm.elect_relay("worker-001")
        assert elected is not None
        assert elected.get("relay_capable") is True

    def test_elect_relay_prefers_lower_load(self):
        pm = self._pm_with_relays()
        elected = pm.elect_relay("worker-001")
        assert elected is not None
        # relay-b has lower load (0.1 vs 0.2) → should be elected when hash tiebreak doesn't override
        # (relay-b always wins because it has strictly lower load)
        assert elected["node_id"] == "relay-b"

    def test_elect_relay_is_deterministic(self):
        pm = self._pm_with_relays()
        e1 = pm.elect_relay("worker-xyz")
        e2 = pm.elect_relay("worker-xyz")
        assert e1 is not None and e2 is not None
        assert e1["node_id"] == e2["node_id"]

    def test_elect_relay_derives_host_port(self):
        pm = self._pm_with_relays()
        elected = pm.elect_relay("worker-001")
        assert elected is not None
        assert "_relay_host" in elected
        assert "_relay_port" in elected
        assert isinstance(elected["_relay_port"], int)

    def test_elect_relay_returns_none_when_no_relays(self):
        pm = _pm()
        pm.merge_peers([{"node_id": "nr", "endpoint": "http://nr:8020", "relay_capable": False}])
        assert pm.elect_relay("worker") is None

    def test_get_relay_candidates_excludes_non_relay(self):
        pm = self._pm_with_relays()
        candidates = pm.get_relay_candidates()
        ids = {c["node_id"] for c in candidates}
        assert "non-relay" not in ids
        assert ids == {"relay-a", "relay-b"}

    def test_relay_capable_stored_in_merge(self):
        pm = _pm()
        pm.merge_peers([{"node_id": "r", "endpoint": "http://r:8020", "relay_capable": True, "relay_accept_port": 9485}])
        peer = pm.relay_target("r")
        assert peer is not None
        assert peer.get("relay_capable") is True
        assert peer.get("relay_accept_port") == 9485
