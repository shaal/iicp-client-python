# ADR-016: IICP client SDK conformance
"""Unit tests for the mesh PeerManager: merge, prune, relay, HMAC verify
(parity Block F)."""

from __future__ import annotations

import json

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
        # Force the peer's last_contact far into the past.
        pm._peers["a"]["last_contact"] = 0.0
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
