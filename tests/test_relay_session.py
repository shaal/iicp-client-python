# ADR-016: IICP client SDK conformance — ADR-041 tier-3 / #341 relay R1
"""Unit tests for RelaySessionRegistry and encode helpers (relay_session.py + iicp_tcp.py)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from iicp_client.iicp_tcp import (
    MsgType,
    decode_relay_bind,
    decode_relay_response,
    encode_relay_ack,
    encode_relay_bind,
    encode_relay_call,
)
from iicp_client.relay_session import RelaySessionRegistry, RelayWorkerSession

# ── encode/decode helpers ──────────────────────────────────────────────────────


class TestRelayFrameHelpers:
    def test_encode_decode_relay_bind_roundtrip(self):
        raw = encode_relay_bind(
            "w-001", "urn:iicp:intent:llm:chat:v1", ["qwen2.5:0.5b", "phi3:mini"]
        )
        worker_id, intent, models = decode_relay_bind(raw)
        assert worker_id == "w-001"
        assert intent == "urn:iicp:intent:llm:chat:v1"
        assert models == ["qwen2.5:0.5b", "phi3:mini"]

    def test_encode_relay_ack_is_cbor_map(self):
        raw = encode_relay_ack("w-001")
        import cbor2
        body = cbor2.loads(raw)
        assert body[1] == "ok"
        assert body[2] == "w-001"

    def test_encode_relay_call_contains_call_id_and_payload(self):
        raw = encode_relay_call("call-abc", {"messages": []})
        import json

        import cbor2
        body = cbor2.loads(raw)
        assert body[15] == "call-abc"
        payload = json.loads(body[5])
        assert "messages" in payload

    def test_decode_relay_response_extracts_call_id_and_result(self):
        import json

        import cbor2
        result = {"choices": [{"message": {"content": "hi"}}]}
        raw = cbor2.dumps({15: "call-abc", 5: json.dumps(result).encode()}, canonical=True)
        call_id, decoded = decode_relay_response(raw)
        assert call_id == "call-abc"
        assert decoded["choices"][0]["message"]["content"] == "hi"

    def test_decode_relay_bind_empty_models(self):
        raw = encode_relay_bind("w-002", "urn:x", [])
        _, _, models = decode_relay_bind(raw)
        assert models == []

    def test_relay_bind_and_ack_have_correct_msg_types(self):
        assert MsgType.RELAY_BIND == 0x0B
        assert MsgType.RELAY_ACK == 0x0C


# ── RelaySessionRegistry ─────────────────────────────────────────────────────


class TestRelaySessionRegistry:
    def test_bind_and_get(self):
        reg = RelaySessionRegistry()
        writer = MagicMock()
        session = RelayWorkerSession("w-001", writer)
        reg.bind("w-001", session)
        assert reg.get("w-001") is session

    def test_get_missing_returns_none(self):
        reg = RelaySessionRegistry()
        assert reg.get("nobody") is None

    def test_unbind_removes_entry(self):
        reg = RelaySessionRegistry()
        writer = MagicMock()
        session = RelayWorkerSession("w-001", writer)
        reg.bind("w-001", session)
        reg.unbind("w-001")
        assert reg.get("w-001") is None

    def test_is_bound_reflects_state(self):
        reg = RelaySessionRegistry()
        writer = MagicMock()
        session = RelayWorkerSession("w-001", writer)
        assert not reg.is_bound("w-001")
        reg.bind("w-001", session)
        assert reg.is_bound("w-001")
        reg.unbind("w-001")
        assert not reg.is_bound("w-001")

    def test_bound_worker_ids_lists_bound(self):
        reg = RelaySessionRegistry()
        writer = MagicMock()
        reg.bind("a", RelayWorkerSession("a", writer))
        reg.bind("b", RelayWorkerSession("b", writer))
        ids = reg.bound_worker_ids()
        assert set(ids) == {"a", "b"}


# ── RelayWorkerSession.on_response ───────────────────────────────────────────


class TestRelayWorkerSessionOnResponse:
    def test_on_response_resolves_pending_future(self):
        loop = asyncio.new_event_loop()

        async def _run():
            writer = MagicMock()
            writer.drain = AsyncMock()
            writer.write = MagicMock()
            session = RelayWorkerSession("w-001", writer)
            # Manually register a future
            fut = loop.create_future()
            session._pending["call-xyz"] = fut
            session.on_response("call-xyz", {"result": "ok"})
            result = await asyncio.wait_for(fut, timeout=1.0)
            assert result == {"result": "ok"}

        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_on_response_ignores_unknown_call_id(self):
        writer = MagicMock()
        session = RelayWorkerSession("w-001", writer)
        # Should not raise
        session.on_response("unknown-call", {"result": "ok"})
