"""Unit tests for ConcurrencyGate + its wiring into IicpTcpServer CALLs."""
from __future__ import annotations

import asyncio
import json
import socket
import struct

import cbor2
import pytest

from iicp_client.concurrency import CapacityExceededError, ConcurrencyGate
from iicp_client.iicp_tcp import (
    FRAME_HEADER_LEN,
    FRAMING_VERSION,
    IICP_MAGIC,
    IicpTcpServer,
    MsgType,
)

_HEADER = struct.Struct("!4sBBBBI")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _frame(msg_type: int, payload: bytes = b"") -> bytes:
    return _HEADER.pack(IICP_MAGIC, FRAMING_VERSION, msg_type, 0, 0, len(payload)) + payload


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    head = await asyncio.wait_for(reader.readexactly(FRAME_HEADER_LEN), timeout=5)
    _magic, _ver, mt, _flags, _res, length = _HEADER.unpack(head)
    payload = await asyncio.wait_for(reader.readexactly(length), timeout=5) if length else b""
    return mt, payload


# ── ConcurrencyGate primitive ──────────────────────────────────────────────

class TestConcurrencyGate:
    def test_rejects_max_concurrent_zero(self):
        with pytest.raises(ValueError):
            ConcurrencyGate(max_concurrent=0)

    async def test_active_jobs_and_load_track_acquisitions(self):
        gate = ConcurrencyGate(max_concurrent=2)
        assert gate.active_jobs == 0
        assert gate.load == 0.0
        async with gate.acquire():
            assert gate.active_jobs == 1
            assert gate.load == 0.5
            async with gate.acquire():
                assert gate.active_jobs == 2
                assert gate.load == 1.0
            assert gate.active_jobs == 1
        assert gate.active_jobs == 0

    async def test_capacity_exceeded_raises_when_full(self):
        """When all slots are held, the third acquire MUST raise rather than
        queue. Use an internal helper that holds the slots so the third
        acquire sees a locked semaphore + active count == max."""
        gate = ConcurrencyGate(max_concurrent=2)
        # Manually deplete the semaphore + active count to simulate full
        await gate._sem.acquire()
        await gate._sem.acquire()
        gate._active = 2
        try:
            with pytest.raises(CapacityExceededError) as excinfo:
                async with gate.acquire():
                    pass  # unreachable
            assert excinfo.value.max_concurrent == 2
        finally:
            gate._sem.release()
            gate._sem.release()
            gate._active = 0

    async def test_capacity_error_message_includes_max(self):
        try:
            raise CapacityExceededError(max_concurrent=7)
        except CapacityExceededError as exc:
            assert exc.max_concurrent == 7
            assert "7" in str(exc)


# ── IicpTcpServer integration ──────────────────────────────────────────────

class TestTcpServerGateIntegration:
    @pytest.fixture
    async def server_with_gate(self):
        port = _free_port()
        gate = ConcurrencyGate(max_concurrent=2)
        # Handler that holds the slot until released so we can deterministically
        # exhaust capacity in the test.
        hold_event = asyncio.Event()

        async def slow_handler(task: dict) -> dict:
            await hold_event.wait()
            return {"result": {"ok": True}}

        server = IicpTcpServer(
            host="127.0.0.1",
            port=port,
            node_id="gated-node",
            handler=slow_handler,
            concurrency_gate=gate,
        )
        await server.start()
        try:
            yield port, gate, hold_event
        finally:
            hold_event.set()
            await server.stop()

    async def _do_call(self, port: int, call_id: str) -> tuple[int, bytes]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_frame(MsgType.INIT, cbor2.dumps({1: FRAMING_VERSION}, canonical=True)))
            await writer.drain()
            await _read_frame(reader)  # consume ACK
            call_payload = {
                2: "sess",
                3: "urn:iicp:intent:llm:chat:v1",
                15: call_id,
                5: json.dumps({"messages": []}).encode("utf-8"),
            }
            writer.write(_frame(MsgType.CALL, cbor2.dumps(call_payload, canonical=True)))
            await writer.drain()
            mt, payload = await _read_frame(reader)
            return mt, payload
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_call_under_capacity_passes_through(self, server_with_gate):
        port, _gate, hold = server_with_gate
        hold.set()  # immediate completion
        mt, payload = await self._do_call(port, "c1")
        assert mt == MsgType.RESPONSE
        body = cbor2.loads(payload)
        # No error code → handler ran successfully
        assert 100 not in body, f"unexpected error: {body.get(100)} {body.get(101)}"

    async def test_call_at_capacity_returns_429_iicp_e021(self, server_with_gate):
        """Two concurrent CALLs occupy both slots; the third gets
        IICP-E021 with error_code=429 in the RESPONSE frame rather than
        being silently queued."""
        port, gate, hold = server_with_gate
        # Start two long-running calls to fill the gate
        c1_task = asyncio.create_task(self._do_call(port, "c1"))
        c2_task = asyncio.create_task(self._do_call(port, "c2"))
        # Give them a moment to acquire slots
        for _ in range(50):
            if gate.active_jobs >= 2:
                break
            await asyncio.sleep(0.01)
        assert gate.active_jobs == 2, "slots should be full before third call"
        # Third CALL should hit capacity gate
        mt, payload = await self._do_call(port, "c3")
        body = cbor2.loads(payload)
        assert mt == MsgType.RESPONSE
        assert body.get(100) == 429, f"expected error_code=429, got body={body}"
        assert "IICP-E021" in str(body.get(101, ""))
        # Cleanup — release the held handlers
        hold.set()
        await asyncio.wait_for(c1_task, timeout=5)
        await asyncio.wait_for(c2_task, timeout=5)

    async def test_no_gate_skips_capacity_check(self):
        """When concurrency_gate is None, server accepts unlimited concurrent CALLs."""
        port = _free_port()
        server = IicpTcpServer(
            host="127.0.0.1",
            port=port,
            node_id="ungated-node",
            handler=lambda task: asyncio.sleep(0.001, {"result": {"ok": True}}),
            concurrency_gate=None,
        )
        # Use a simpler handler that returns a dict directly
        async def h(task: dict) -> dict:
            return {"result": {"ok": True}}
        server.handler = h
        await server.start()
        try:
            mt, payload = await self._do_call(port, "c1")
            assert mt == MsgType.RESPONSE
            body = cbor2.loads(payload)
            assert 100 not in body
        finally:
            await server.stop()
