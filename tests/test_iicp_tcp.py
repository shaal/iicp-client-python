"""IicpTcpServer integration tests (spec/iicp-framing.md)

Boots the native IICP TCP server on a free local port, runs the same protocol
matrix as the adapter's /tmp/iicp_test_client.py: INIT/ACK, PING-with-echo,
empty-PING, DISCOVER, CALL via handler, CLOSE, and bad-magic.

Verifies the iter-1410 framing fix is correctly ported — pre-fix the session
loop closed on every payload-bearing frame.
"""

from __future__ import annotations

import asyncio
import socket
import struct

import cbor2
import pytest

from iicp_client.iicp_tcp import (
    FRAME_HEADER_LEN,
    FRAMING_VERSION,
    IICP_MAGIC,
    IicpTcpClient,
    IicpTcpClientError,
    IicpTcpServer,
    MsgType,
)

_HEADER = struct.Struct("!4sBBBBI")
TIMEOUT = 5.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _frame(msg_type: int, payload: bytes = b"") -> bytes:
    return _HEADER.pack(IICP_MAGIC, FRAMING_VERSION, msg_type, 0, 0, len(payload)) + payload


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    head = await asyncio.wait_for(reader.readexactly(FRAME_HEADER_LEN), timeout=TIMEOUT)
    magic, _ver, mt, _flags, _res, length = _HEADER.unpack(head)
    assert magic == IICP_MAGIC, f"bad magic in response: {magic!r}"
    payload = await asyncio.wait_for(reader.readexactly(length), timeout=TIMEOUT) if length else b""
    return mt, payload


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
async def server_port():
    """Boot a server on a free port; yield the port; clean shutdown after."""
    port = _free_port()

    async def task_handler(task: dict) -> dict:
        # Echo handler: returns the payload back under "echo" key
        return {"result": {"echo": task.get("payload", {})}}

    async def discover_lookup(intent: str) -> list[dict]:
        return [
            {"node_id": "fake-1", "endpoint": "http://fake.example:8080", "intent": intent},
            {"node_id": "fake-2", "endpoint": "http://fake.example:8080", "intent": intent},
        ]

    server = IicpTcpServer(
        host="127.0.0.1",
        port=port,
        node_id="test-node-id",
        handler=task_handler,
        discover_lookup=discover_lookup,
    )
    await server.start()
    try:
        yield port
    finally:
        await server.stop()


# ── tests ───────────────────────────────────────────────────────────────────


async def test_init_returns_ack_with_node_id(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        mt, payload = await _read_frame(reader)
        assert mt == MsgType.ACK
        body = cbor2.loads(payload)
        assert body[1] == FRAMING_VERSION
        assert body[2] == "test-node-id"  # node_id echoed back
    finally:
        writer.close()
        await writer.wait_closed()


async def test_ping_with_echo_returns_pong(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        await _read_frame(reader)  # consume ACK

        echo_bytes = b"sdk-tcp-roundtrip-2026"
        writer.write(_frame(MsgType.PING, cbor2.dumps({1: echo_bytes}, canonical=True)))
        await writer.drain()
        mt, payload = await _read_frame(reader)
        assert mt == MsgType.PONG
        body = cbor2.loads(payload)
        assert body[1] == echo_bytes
    finally:
        writer.close()
        await writer.wait_closed()


async def test_empty_ping_returns_pong_with_no_echo(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        await _read_frame(reader)

        writer.write(_frame(MsgType.PING, cbor2.dumps({}, canonical=True)))
        await writer.drain()
        mt, payload = await _read_frame(reader)
        assert mt == MsgType.PONG
        body = cbor2.loads(payload) if payload else {}
        assert 1 not in body
    finally:
        writer.close()
        await writer.wait_closed()


async def test_discover_invokes_lookup_returns_nodes(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        await _read_frame(reader)

        intent = "urn:iicp:intent:llm:chat:v1"
        writer.write(
            _frame(MsgType.DISCOVER, cbor2.dumps({2: "sess-1", 3: intent}, canonical=True))
        )
        await writer.drain()
        mt, payload = await _read_frame(reader)
        assert mt == MsgType.RESPONSE
        body = cbor2.loads(payload)
        assert body[2] == "sess-1"
        assert body[3] == intent
        assert isinstance(body[20], list)
        assert len(body[20]) == 2
        assert body[20][0]["node_id"] == "fake-1"
    finally:
        writer.close()
        await writer.wait_closed()


async def test_call_invokes_handler_returns_result(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        await _read_frame(reader)

        import json

        call_payload = {
            2: "sess-c1",
            3: "urn:iicp:intent:llm:chat:v1",
            15: "call-0001",
            5: json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"),
        }
        writer.write(_frame(MsgType.CALL, cbor2.dumps(call_payload, canonical=True)))
        await writer.drain()
        mt, payload = await _read_frame(reader)
        assert mt == MsgType.RESPONSE
        body = cbor2.loads(payload)
        assert body[2] == "sess-c1"
        assert body[15] == "call-0001"
        assert 100 not in body, f"unexpected error: {body.get(100)}={body.get(101)!r}"
        # Result is CBOR-encoded bytes — decode and check echo handler shape
        result = cbor2.loads(body[5])
        assert "echo" in result
        assert result["echo"]["messages"][0]["content"] == "hi"
    finally:
        writer.close()
        await writer.wait_closed()


async def test_close_results_in_clean_hangup(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
        await writer.drain()
        await _read_frame(reader)

        writer.write(_frame(MsgType.CLOSE, b""))
        await writer.drain()
        # Server should hang up cleanly — read returns b""
        leftover = await asyncio.wait_for(reader.read(1), timeout=TIMEOUT)
        assert leftover == b""
    finally:
        writer.close()
        await writer.wait_closed()


async def test_bad_magic_closes_connection(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        writer.write(b"XXXX" + b"\x00" * (FRAME_HEADER_LEN - 4))
        await writer.drain()
        # Server should close — read returns b""
        leftover = await asyncio.wait_for(reader.read(1), timeout=TIMEOUT)
        assert leftover == b""
    finally:
        writer.close()
        await writer.wait_closed()


async def test_payload_bearing_frame_does_not_close_session(server_port):
    """Regression guard for the iter-1410 adapter bug — pre-fix the session loop
    closed on every frame with a non-empty CBOR payload because IicpFrame.decode
    requires header + payload and the loop only waited for the header."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    try:
        # INIT (3-byte payload) + PING (3-byte payload) back-to-back. Pre-fix
        # the server would close after INIT because decode would error on
        # missing payload bytes that were sitting in the kernel buffer.
        init = _frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True))
        ping = _frame(MsgType.PING, cbor2.dumps({1: b"x"}, canonical=True))
        writer.write(init + ping)
        await writer.drain()
        # Both responses must arrive — ACK then PONG
        mt1, _ = await _read_frame(reader)
        mt2, _ = await _read_frame(reader)
        assert mt1 == MsgType.ACK
        assert mt2 == MsgType.PONG
    finally:
        writer.close()
        await writer.wait_closed()


# ── IicpTcpClient tests — true round-trip against the server fixture ────────


async def test_client_context_manager_handshake_and_close(server_port):
    """iter-1417: connecting via context manager performs handshake on enter and
    sends CLOSE on exit. peer_node_id is populated from the ACK."""
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        assert client.framing_version == FRAMING_VERSION
        assert client.peer_node_id == "test-node-id"


async def test_client_ping_with_echo(server_port):
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        echo = b"client-ping-2026"
        got = await client.ping(echo)
        assert got == echo


async def test_client_ping_empty(server_port):
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        got = await client.ping()
        assert got is None


async def test_client_discover_returns_nodes(server_port):
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        nodes = await client.discover("urn:iicp:intent:llm:chat:v1")
        assert len(nodes) == 2
        assert nodes[0]["node_id"] == "fake-1"


async def test_client_call_returns_handler_result(server_port):
    """End-to-end: client.call sends JSON payload, server invokes handler, client
    decodes the CBOR-encoded result. Same handler echo shape used in the server
    fixture — `{"echo": <payload>}`."""
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        payload = {"messages": [{"role": "user", "content": "hi from client"}]}
        result = await client.call("urn:iicp:intent:llm:chat:v1", payload, call_id="call-abc")
        assert "echo" in result
        assert result["echo"]["messages"][0]["content"] == "hi from client"


async def test_client_call_raises_on_server_error():
    """When the server has no handler configured, CALL returns error_code 503 →
    client raises IicpTcpClientError."""
    port = _free_port()
    server = IicpTcpServer(host="127.0.0.1", port=port, node_id="no-handler-node")
    await server.start()
    try:
        async with IicpTcpClient("127.0.0.1", port) as client:
            await client.handshake()
            try:
                await client.call("urn:iicp:intent:llm:chat:v1", {})
            except IicpTcpClientError as exc:
                assert "503" in str(exc)
                return
            raise AssertionError("expected IicpTcpClientError")
    finally:
        await server.stop()


async def test_client_roundtrip_init_ping_discover_call_close(server_port):
    """Single session exercises the full protocol matrix in order."""
    async with IicpTcpClient("127.0.0.1", server_port) as client:
        await client.handshake()
        assert (await client.ping(b"x")) == b"x"
        assert len(await client.discover("urn:iicp:intent:llm:chat:v1")) == 2
        result = await client.call("urn:iicp:intent:llm:chat:v1", {"k": "v"}, call_id="c1")
        assert result["echo"]["k"] == "v"
