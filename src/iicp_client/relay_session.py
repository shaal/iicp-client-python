# SPDX-License-Identifier: Apache-2.0
"""Relay-as-last-resort — ADR-041 tier-3, Part 3 R1 (#341).

Allows operators behind CGNAT to stay reachable: the worker holds an
**outbound** IICP-TCP connection to a publicly-reachable relay node.
The relay forwards inbound tasks down the persistent session, and the
worker's RESPONSE comes back through the same channel.

Three components
────────────────
RelaySessionRegistry — thread-safe table: worker_id → RelayWorkerSession.
RelayWorkerSession   — holds the asyncio writer + pending-request futures.
RelayAcceptServer    — asyncio TCP server (port 9485 by default) that
                       accepts RELAY_BIND connections from workers and runs
                       the relay-worker frame loop.

Protocol
────────
  Worker → Relay: IICP_MAGIC + INIT
  Relay  → Worker: ACK
  Worker → Relay: RELAY_BIND{worker_id, intent, models}
  Relay  → Worker: RELAY_ACK
  [persistent session; PING/PONG keepalive]
  Relay  → Worker: CALL{call_id, task_json}    (pushed by /v1/relay HTTP handler)
  Worker → Relay: RESPONSE{call_id, result}
  Relay resolves the pending future → HTTP /v1/relay returns the result.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_FRAME_HEADER_LEN = 12
_IICP_MAGIC = b"IICP"
_FRAMING_VERSION = 0x01
_READ_CHUNK = 4096
_HEADER_STRUCT = struct.Struct("!4sBBBBI")

# Frame type constants (matches MsgType in iicp_tcp.py)
_MT_INIT = 0x01
_MT_ACK = 0x02
_MT_CALL = 0x05
_MT_RESPONSE = 0x06
_MT_CLOSE = 0x07
_MT_PING = 0x09
_MT_PONG = 0x0A
_MT_RELAY_BIND = 0x0B
_MT_RELAY_ACK = 0x0C


def _make_frame(msg_type: int, payload: bytes) -> bytes:
    header = _HEADER_STRUCT.pack(_IICP_MAGIC, _FRAMING_VERSION, msg_type, 0, 0, len(payload))
    return header + payload


def _cbor2() -> Any:
    try:
        import cbor2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "cbor2 is required for relay sessions. "
            "Install with: pip install 'iicp-client[iicp-tcp]'"
        ) from exc
    return cbor2


def _enc(obj: object) -> bytes:
    return _cbor2().dumps(obj, canonical=True)


def _dec(data: bytes) -> object:
    return _cbor2().loads(data)


class RelayWorkerSession:
    """One bound relay-worker TCP session.

    Thread-safe: the relay's HTTP /v1/relay handler (a ThreadingHTTPServer
    worker thread) calls ``forward_task()`` via asyncio.run_coroutine_threadsafe,
    which submits a CALL frame and waits on a per-request Future.
    """

    def __init__(self, worker_id: str, writer: asyncio.StreamWriter) -> None:
        self.worker_id = worker_id
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict]] = {}

    async def forward_task(self, task: dict, timeout: float = 120.0) -> dict:
        """Push a task CALL to the worker and await the RESPONSE."""
        call_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[call_id] = fut
        try:
            payload = _enc({15: call_id, 5: __import__("json").dumps(task).encode()})
            async with self._write_lock:
                self._writer.write(_make_frame(_MT_CALL, payload))
                await self._writer.drain()
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(call_id, None)

    def is_alive(self) -> bool:
        """Whether the underlying worker transport is still alive/writable.

        #510 interim hardening: an alive bound session must not be displaced
        by a new RELAY_BIND for the same worker_id (unauthenticated bind).
        """
        return not self._writer.is_closing()

    def on_response(self, call_id: str, result: dict) -> None:
        """Called by the relay-worker loop when a RESPONSE arrives."""
        fut = self._pending.get(call_id)
        if fut is not None and not fut.done():
            fut.set_result(result)


class HttpPollWorkerSession:
    """One bound HTTP long-poll relay-worker session (#450 browser workers).

    Duck-type compatible with RelayWorkerSession (forward_task / is_alive /
    on_response) so RelaySessionRegistry and the relay handlers treat both
    transports identically. Instead of pushing CALL frames down a TCP writer,
    ``forward_task()`` puts the call on an asyncio queue that the worker
    drains via ``GET /v1/relay/pull``; the worker posts the result back via
    ``POST /v1/relay/result``, which resolves the pending future.

    Auth: ``session_token`` is issued at bind and must be presented as a
    Bearer token on pull/result/unbind — stronger than the unauthenticated
    TCP RELAY_BIND (#510), applied to the new transport from day one.

    Liveness = the worker pulled within ``liveness_window`` seconds. A dead
    session is displaceable by a fresh bind (same #510 interim-C semantics
    as the TCP transport: an *alive* session is never displaced).
    """

    def __init__(
        self,
        worker_id: str,
        intent: str = "",
        models: list[str] | None = None,
        liveness_window: float = 90.0,
    ) -> None:
        self.worker_id = worker_id
        self.intent = intent
        self.models = models or []
        self.session_token = uuid.uuid4().hex
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._last_pull = time.monotonic()
        self._liveness_window = liveness_window
        self._closed = False

    async def forward_task(self, task: dict, timeout: float = 120.0) -> dict:
        """Queue a CALL for the polling worker and await its RESPONSE."""
        call_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[call_id] = fut
        await self._queue.put({"call_id": call_id, "task": task})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(call_id, None)

    async def next_call(self, timeout: float = 25.0) -> dict | None:
        """Long-poll: next queued CALL, or None when the window elapses."""
        self._last_pull = time.monotonic()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._last_pull = time.monotonic()

    def is_alive(self) -> bool:
        return (not self._closed) and (time.monotonic() - self._last_pull) < self._liveness_window

    def on_response(self, call_id: str, result: dict) -> None:
        """Resolve the pending consumer future. Must run on the loop thread."""
        fut = self._pending.get(call_id)
        if fut is not None and not fut.done():
            fut.set_result(result)

    def close(self) -> None:
        self._closed = True


RelaySession = RelayWorkerSession | HttpPollWorkerSession


class RelaySessionRegistry:
    """Thread-safe mapping worker_id → relay session (TCP or HTTP-poll)."""

    def __init__(self) -> None:
        self._sessions: dict[str, RelaySession] = {}
        self._lock = threading.Lock()

    def bind(self, worker_id: str, session: RelaySession) -> None:
        with self._lock:
            self._sessions[worker_id] = session

    def unbind(self, worker_id: str) -> None:
        with self._lock:
            self._sessions.pop(worker_id, None)

    def get(self, worker_id: str) -> RelaySession | None:
        with self._lock:
            return self._sessions.get(worker_id)

    def get_by_token(self, token: str) -> HttpPollWorkerSession | None:
        """Find an HTTP-poll session by its bearer token (pull/result auth)."""
        if not token:
            return None
        with self._lock:
            for sess in self._sessions.values():
                if isinstance(sess, HttpPollWorkerSession) and sess.session_token == token:
                    return sess
        return None

    def is_bound(self, worker_id: str) -> bool:
        with self._lock:
            return worker_id in self._sessions

    def bound_worker_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


class RelayAcceptServer:
    """Asyncio TCP server that accepts RELAY_BIND sessions from workers.

    Runs on a separate port from the IICP-TCP task server (default 9485) so
    workers can connect outbound through their NAT without needing inbound
    reachability. After the handshake the session stays open and the relay
    forwards tasks down it on demand.
    """

    def __init__(
        self,
        registry: RelaySessionRegistry,
        *,
        host: str = "0.0.0.0",
        port: int = 9485,
        http_port: int = 9484,
    ) -> None:
        self.registry = registry
        self.host = host
        self.port = port
        # The relay's public HTTP task port — advertised in RELAY_ACK (field 4)
        # so workers can register the correct {relay}/v1/relay-for/<wid> endpoint.
        self.http_port = http_port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        _cbor2()  # validate import early
        self._server = await asyncio.start_server(
            self._handle_connection, host=self.host, port=self.port
        )
        logger.info("Relay accept server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        if not self._server:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("Relay accept: connection from %s", peer)
        try:
            await self._session(reader, writer)
        except (ConnectionResetError, asyncio.IncompleteReadError, EOFError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relay accept session error from %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _session(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handshake + relay-worker frame loop."""
        # ── Step 1: INIT/ACK ──────────────────────────────────────────────────
        magic = await reader.readexactly(4)
        if magic != _IICP_MAGIC:
            logger.warning("Relay accept: bad magic %r — dropping", magic)
            return
        rest = await reader.readexactly(_FRAME_HEADER_LEN - 4)
        header_bytes = magic + rest
        _, ver, msg_type, flags, _res, payload_len = _HEADER_STRUCT.unpack(header_bytes)
        if msg_type != _MT_INIT:
            logger.warning("Relay accept: expected INIT, got 0x%02x", msg_type)
            return
        if payload_len:
            await reader.readexactly(payload_len)  # discard INIT payload
        ack_payload = _enc({1: _FRAMING_VERSION})
        writer.write(_make_frame(_MT_ACK, ack_payload))
        await writer.drain()

        # ── Step 2: RELAY_BIND ────────────────────────────────────────────────
        frame_bytes = await self._read_frame(reader)
        if frame_bytes is None:
            return
        _, _, ft, _, _, plen = _HEADER_STRUCT.unpack(frame_bytes[:_FRAME_HEADER_LEN])
        frame_payload = frame_bytes[_FRAME_HEADER_LEN:]
        if ft != _MT_RELAY_BIND:
            logger.warning("Relay accept: expected RELAY_BIND, got 0x%02x", ft)
            return
        try:
            body = _dec(frame_payload)
            if not isinstance(body, dict):
                raise ValueError("RELAY_BIND must be CBOR map")
            worker_id = str(body.get(1, ""))
            intent = str(body.get(2, ""))
            models = [str(m) for m in (body.get(3) or [])]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relay accept: RELAY_BIND decode error: %s", exc)
            return
        if not worker_id:
            logger.warning("Relay accept: RELAY_BIND missing worker_id")
            return

        # #510 interim hardening: RELAY_BIND is unauthenticated, so refuse to
        # displace an existing session whose socket is still alive (mid-session
        # hijack). Rebind after socket death (legitimate reconnect) still works.
        existing = self.registry.get(worker_id)
        if existing is not None and existing.is_alive():
            peer = writer.get_extra_info("peername")
            logger.warning(
                "Relay accept: rejected RELAY_BIND for worker=%s from %s: "
                "worker_id already bound to an alive session (#510)",
                worker_id,
                peer,
            )
            writer.write(
                _make_frame(
                    _MT_RELAY_ACK,
                    _enc(
                        {
                            1: "error",
                            2: worker_id,
                            3: "worker_id already bound to an alive session",
                        }
                    ),
                )
            )
            await writer.drain()
            return

        session = RelayWorkerSession(worker_id=worker_id, writer=writer)
        self.registry.bind(worker_id, session)
        logger.info("Relay: worker=%s bound (intent=%s models=%s)", worker_id, intent, models)

        # Field 4 (additive): the relay's HTTP task port, so the worker can
        # register {relay_host}:{http_port}/v1/relay-for/{worker_id} with the
        # directory. Old workers ignore unknown CBOR keys.
        writer.write(_make_frame(_MT_RELAY_ACK, _enc({1: "ok", 2: worker_id, 4: self.http_port})))
        await writer.drain()

        # ── Step 3: relay-worker loop ─────────────────────────────────────────
        try:
            await self._relay_worker_loop(session, reader, writer)
        finally:
            # Only remove the registry entry if it is still ours — a legitimate
            # reconnect may already have bound a newer session for this worker_id.
            if self.registry.get(worker_id) is session:
                self.registry.unbind(worker_id)
            logger.info("Relay: session ended for worker=%s", worker_id)

    async def _relay_worker_loop(
        self,
        session: RelayWorkerSession,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Read PING/RESPONSE/CLOSE frames from the bound worker."""
        while True:
            frame_bytes = await self._read_frame(reader)
            if frame_bytes is None:
                return
            _, _, ft, _, _, plen = _HEADER_STRUCT.unpack(frame_bytes[:_FRAME_HEADER_LEN])
            payload = frame_bytes[_FRAME_HEADER_LEN:]

            if ft == _MT_PING:
                echo = b""
                try:
                    pb = _dec(payload)
                    echo = pb.get(1, b"") if isinstance(pb, dict) else b""  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass
                writer.write(_make_frame(_MT_PONG, _enc({1: echo} if echo else {})))
                await writer.drain()

            elif ft == _MT_RESPONSE:
                try:
                    rb = _dec(payload)
                    if isinstance(rb, dict):
                        call_id = str(rb.get(15, ""))
                        raw5 = rb.get(5, b"")
                        import json as _json
                        if isinstance(raw5, (bytes, bytearray)):
                            result = _json.loads(raw5)
                        elif isinstance(raw5, str):
                            result = _json.loads(raw5)
                        else:
                            result = {}
                        session.on_response(call_id, result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Relay RESPONSE decode error: %s", exc)

            elif ft == _MT_CLOSE:
                return

            else:
                logger.debug("Relay worker loop: unhandled frame type 0x%02x", ft)

    async def _read_frame(self, reader: asyncio.StreamReader) -> bytes | None:
        """Read one complete IICP frame; return raw bytes or None on EOF/error."""
        try:
            header = await reader.readexactly(_FRAME_HEADER_LEN)
        except (asyncio.IncompleteReadError, EOFError, ConnectionResetError):
            return None
        _, _, _, _, _, payload_len = _HEADER_STRUCT.unpack(header)
        if payload_len > 16 * 1024 * 1024:
            return None
        try:
            payload = await reader.readexactly(payload_len) if payload_len else b""
        except (asyncio.IncompleteReadError, EOFError, ConnectionResetError):
            return None
        return header + payload
