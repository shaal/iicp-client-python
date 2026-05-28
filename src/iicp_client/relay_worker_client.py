# SPDX-License-Identifier: Apache-2.0
"""Relay worker client — ADR-041 tier-3, Part 3 R2 (#341).

A CGNAT-or-otherwise-unreachable node uses this to hold an outbound TCP
connection to a relay node. The relay pushes CALL frames down the session;
the worker handles them and sends RESPONSE back. The connection is kept
alive with PING/PONG and auto-reconnects on drop.

Usage (from serve() or standalone)::

    worker = RelayWorkerClient(
        worker_id="my-node-001",
        intent="urn:iicp:intent:llm:chat:v1",
        relay_host="relay.example.com",
        relay_port=9485,
        task_handler=my_async_handler,
    )
    asyncio.create_task(worker.run())  # runs until cancelled

The worker:
1. Connects to the relay's TCP port (default 9485).
2. Sends INIT → receives ACK.
3. Sends RELAY_BIND{worker_id, intent, models}.
4. Receives RELAY_ACK.
5. Enters the session loop: PING → PONG keepalive + CALL → RESPONSE.
6. On disconnect, waits and reconnects (exponential backoff, cap 60s).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_IICP_MAGIC = b"IICP"
_FRAMING_VERSION = 0x01
_HEADER_LEN = 12
_HEADER_STRUCT = struct.Struct("!4sBBBBI")
_PING_INTERVAL_S = 30.0
_MAX_RECONNECT_DELAY_S = 60.0

# Frame type constants (mirrors MsgType in iicp_tcp.py)
_MT_INIT = 0x01
_MT_ACK = 0x02
_MT_CALL = 0x05
_MT_RESPONSE = 0x06
_MT_CLOSE = 0x07
_MT_PING = 0x09
_MT_PONG = 0x0A
_MT_RELAY_BIND = 0x0B
_MT_RELAY_ACK = 0x0C

TaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _cbor2() -> Any:
    try:
        import cbor2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "cbor2 is required for relay worker. "
            "Install with: pip install 'iicp-client[iicp-tcp]'"
        ) from exc
    return cbor2


def _enc(obj: dict) -> bytes:
    return _cbor2().dumps(obj, canonical=True)


def _dec(data: bytes) -> dict:
    return _cbor2().loads(data)


def _make_frame(msg_type: int, payload: bytes) -> bytes:
    header = _HEADER_STRUCT.pack(_IICP_MAGIC, _FRAMING_VERSION, msg_type, 0, 0, len(payload))
    return header + payload


async def _read_frame(
    reader: asyncio.StreamReader,
) -> tuple[int, bytes] | None:
    try:
        header = await reader.readexactly(_HEADER_LEN)
    except (asyncio.IncompleteReadError, ConnectionResetError, EOFError):
        return None
    magic = header[:4]
    if magic != _IICP_MAGIC:
        logger.warning("Relay worker: bad magic %r", magic)
        return None
    msg_type = header[5]
    payload_len = _HEADER_STRUCT.unpack(header)[5]
    if payload_len > 16 * 1024 * 1024:
        return None
    try:
        payload = await reader.readexactly(payload_len) if payload_len else b""
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    return msg_type, payload


class RelayWorkerClient:
    """Outbound relay-worker client.

    Maintains a persistent TCP session to a relay node so inbound tasks
    can be received even when the worker has no inbound-routable endpoint.
    """

    def __init__(
        self,
        worker_id: str,
        intent: str,
        relay_host: str,
        relay_port: int,
        task_handler: TaskHandler,
        models: list[str] | None = None,
        on_bind: Callable[[str, str, int], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            worker_id: This node's ID (used in RELAY_BIND).
            intent: The intent URN this worker handles.
            relay_host: Relay node's hostname / public IP.
            relay_port: Relay node's accept port (default 9485).
            task_handler: Async function called for each CALL frame;
                returns the result dict.
            models: Model names advertised in RELAY_BIND (optional).
            on_bind: Called after a successful RELAY_ACK with
                (relay_host, relay_port, session_id). Useful to update
                the directory registration with the relay endpoint.
        """
        self._worker_id = worker_id
        self._intent = intent
        self._relay_host = relay_host
        self._relay_port = relay_port
        self._handler = task_handler
        self._models = models or []
        self._on_bind = on_bind

    async def run(self) -> None:
        """Connect-and-run loop. Reconnects with exponential backoff on drop."""
        delay = 2.0
        while True:
            try:
                await self._session()
                delay = 2.0  # reset on clean exit
            except asyncio.CancelledError:
                logger.info("Relay worker %s cancelled", self._worker_id)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Relay worker %s: session error: %s — reconnecting in %.0fs",
                    self._worker_id, exc, delay,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_RECONNECT_DELAY_S)

    async def _session(self) -> None:
        reader, writer = await asyncio.open_connection(self._relay_host, self._relay_port)
        logger.debug(
            "Relay worker %s: connected to %s:%d",
            self._worker_id, self._relay_host, self._relay_port,
        )
        try:
            await self._handshake(reader, writer)
        except Exception:
            writer.close()
            await writer.wait_closed()
            raise

    async def _handshake(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # Step 1: INIT → ACK
            writer.write(_make_frame(_MT_INIT, _enc({1: _FRAMING_VERSION})))
            await writer.drain()
            frame = await _read_frame(reader)
            if frame is None or frame[0] != _MT_ACK:
                raise ConnectionError(f"Expected ACK, got {frame[0] if frame else 'EOF'}")

            # Step 2: RELAY_BIND → RELAY_ACK
            writer.write(_make_frame(_MT_RELAY_BIND, _enc({
                1: self._worker_id,
                2: self._intent,
                3: self._models,
            })))
            await writer.drain()
            frame = await _read_frame(reader)
            if frame is None or frame[0] != _MT_RELAY_ACK:
                raise ConnectionError(f"Expected RELAY_ACK, got {frame[0] if frame else 'EOF'}")
            ack_body = _dec(frame[1]) if frame[1] else {}
            if ack_body.get(1) != "ok":
                raise ConnectionError(f"RELAY_ACK not ok: {ack_body}")

            logger.info(
                "Relay worker %s: bound to relay %s:%d",
                self._worker_id, self._relay_host, self._relay_port,
            )
            if self._on_bind:
                try:
                    await self._on_bind(self._relay_host, self._relay_port, self._worker_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("on_bind callback failed: %s", exc)

            # Step 3: session loop
            await self._worker_loop(reader, writer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _worker_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming CALL frames and PING keepalive."""
        ping_task = asyncio.create_task(self._ping_loop(writer))
        try:
            while True:
                frame = await _read_frame(reader)
                if frame is None:
                    return
                mt, payload = frame

                if mt == _MT_CALL:
                    asyncio.create_task(self._handle_call(payload, writer))
                elif mt == _MT_PONG:
                    pass  # keepalive acknowledged
                elif mt == _MT_CLOSE:
                    return
                else:
                    logger.debug("Relay worker loop: unhandled frame 0x%02x", mt)
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    async def _ping_loop(self, writer: asyncio.StreamWriter) -> None:
        while True:
            await asyncio.sleep(_PING_INTERVAL_S)
            try:
                writer.write(_make_frame(_MT_PING, _enc({1: b""})))
                await writer.drain()
            except Exception:  # noqa: BLE001
                return

    async def _handle_call(self, payload: bytes, writer: asyncio.StreamWriter) -> None:
        """Decode CALL frame, invoke the task handler, send RESPONSE."""
        try:
            body = _dec(payload) if payload else {}
            call_id = str(body.get(15, ""))
            raw5 = body.get(5, b"")
            if isinstance(raw5, bytes):
                task_dict = json.loads(raw5)
            else:
                task_dict = {}
            result = await self._handler(task_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relay worker CALL handler error: %s", exc)
            result = {"error": str(exc)}

        try:
            response_payload = _enc({
                15: call_id,
                5: json.dumps(result).encode(),
            })
            writer.write(_make_frame(_MT_RESPONSE, response_payload))
            await writer.drain()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relay worker failed to send RESPONSE: %s", exc)
