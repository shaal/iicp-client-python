# ADR-041 tier-3 / #341 — relay-as-last-resort R2 (worker lifecycle)
"""Unit tests for RelayWorkerClient frame helpers and reconnect logic."""

from __future__ import annotations

import asyncio
import struct

import pytest

from iicp_client.relay_worker_client import (
    _HEADER_LEN,
    _IICP_MAGIC,
    _MT_PING,
    _MT_RESPONSE,
    RelayWorkerClient,
    _make_frame,
    _read_frame,
)

# ── frame encoding helpers ────────────────────────────────────────────────────


class TestMakeFrame:
    def test_magic_and_type(self):
        frame = _make_frame(0x09, b"")
        assert frame[:4] == _IICP_MAGIC
        assert frame[5] == 0x09
        assert frame[8:12] == b"\x00\x00\x00\x00"

    def test_payload_length_encoded(self):
        payload = b"hello"
        frame = _make_frame(0x05, payload)
        length = struct.unpack("!I", frame[8:12])[0]
        assert length == 5
        assert frame[_HEADER_LEN:] == payload


class TestReadFrame:
    @pytest.mark.asyncio
    async def test_reads_complete_frame(self):
        payload = b"test-payload"
        raw = _make_frame(_MT_PING, payload)
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        result = await _read_frame(reader)
        assert result is not None
        mt, p = result
        assert mt == _MT_PING
        assert p == payload

    @pytest.mark.asyncio
    async def test_returns_none_on_eof(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()
        result = await _read_frame(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_bad_magic(self):
        reader = asyncio.StreamReader()
        bad = b"XXXX\x01\x09\x00\x00\x00\x00\x00\x00"
        reader.feed_data(bad)
        reader.feed_eof()
        result = await _read_frame(reader)
        assert result is None


# ── RelayWorkerClient construction ───────────────────────────────────────────


class TestRelayWorkerClientInit:
    def test_stores_worker_id_and_intent(self):
        async def handler(task):
            return {}

        client = RelayWorkerClient(
            worker_id="w-001",
            intent="urn:iicp:intent:llm:chat:v1",
            relay_host="relay.example.com",
            relay_port=9485,
            task_handler=handler,
            models=["qwen2.5:0.5b"],
        )
        assert client._worker_id == "w-001"
        assert client._intent == "urn:iicp:intent:llm:chat:v1"
        assert client._relay_host == "relay.example.com"
        assert client._relay_port == 9485
        assert client._models == ["qwen2.5:0.5b"]

    def test_default_models_is_empty(self):
        client = RelayWorkerClient(
            worker_id="w",
            intent="urn:x",
            relay_host="h",
            relay_port=9485,
            task_handler=None,
        )
        assert client._models == []


# ── handle_call: handler invoked and RESPONSE sent ───────────────────────────


class TestHandleCall:
    @pytest.mark.asyncio
    async def test_handler_result_encoded_in_response(self):
        import json as _json

        import cbor2

        received_responses = []

        class FakeWriter:
            def write(self, data):
                received_responses.append(data)
            async def drain(self):
                pass

        async def my_handler(task):
            return {"answer": 42}

        client = RelayWorkerClient(
            worker_id="w",
            intent="urn:x",
            relay_host="h",
            relay_port=9,
            task_handler=my_handler,
        )
        call_payload = cbor2.dumps(
            {15: "call-abc", 5: _json.dumps({"question": "??"}).encode()}, canonical=True
        )
        writer = FakeWriter()
        await client._handle_call(call_payload, writer)
        assert len(received_responses) == 1
        frame = received_responses[0]
        # Decode response frame
        mt = frame[5]
        assert mt == _MT_RESPONSE
        plen = struct.unpack("!I", frame[8:12])[0]
        resp_cbor = frame[_HEADER_LEN:_HEADER_LEN + plen]
        resp_body = cbor2.loads(resp_cbor)
        assert resp_body[15] == "call-abc"
        result = _json.loads(resp_body[5])
        assert result["answer"] == 42
