# SPDX-License-Identifier: Apache-2.0
"""Native IICP binary transport (port 9484) — server + framing + cbor payloads.

Implements the wire side of spec/iicp-framing.md so a hybrid-client SDK node
can answer task CALLs over the native binary transport instead of (or in
addition to) the HTTP `/v1/task` endpoint exposed by IicpNode.serve().

This is a port from `iicp-adapter/server/tcp.py` + `framing/iicp_frame.py`
+ `framing/cbor_payloads.py`. The frame layout (12-byte header + payload),
magic bytes `IICP`, framing version `0x01`, message types (INIT/ACK/PING/
PONG/DISCOVER/CALL/RESPONSE/CLOSE/FEEDBACK), and integer-keyed CBOR
payloads (RFC 8949 canonical) are all preserved exactly so SDK nodes are
wire-compatible with adapter nodes and the REACH framing probes
(FRAME-PING-01, FRAME-INIT-01).

Two pre-existing adapter server bugs are fixed in this port:
  - Session loop now reads the announced payload before calling decode
    (the adapter version closed on every frame with a non-empty payload
    until iter-1410 / iicp.network commit 444aced).
  - The CALL handler decodes key 5 as a JSON dict before invoking the
    user task handler — same contract mismatch that broke adapter CALLs.

cbor2 is an optional dependency installed via the `[iicp-tcp]` extra.
If absent, IicpTcpServer.start() raises ImportError with the install hint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

IICP_MAGIC: bytes = b"IICP"
FRAMING_VERSION: int = 0x01
FRAME_HEADER_LEN: int = 12  # magic(4) + ver(1) + type(1) + flags(1) + reserved(1) + length(4)

_HEADER_STRUCT = struct.Struct("!4sBBBBI")
_READ_CHUNK = 4096
_MAX_PAYLOAD = 16 * 1024 * 1024  # 16 MiB


class MsgType(IntEnum):
    """spec/iicp-framing.md §3 — core message types 0x01–0x0E."""

    INIT = 0x01
    ACK = 0x02
    DISCOVER = 0x03
    SUB_PROTOCOL = 0x04
    CALL = 0x05
    RESPONSE = 0x06
    CLOSE = 0x07
    FEEDBACK = 0x08
    PING = 0x09
    PONG = 0x0A


# ── Frame ─────────────────────────────────────────────────────────────────────


@dataclass
class IicpFrame:
    version: int
    msg_type: int
    flags: int
    payload: bytes

    def encode(self) -> bytes:
        header = _HEADER_STRUCT.pack(
            IICP_MAGIC,
            self.version,
            self.msg_type,
            self.flags,
            0,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def decode(cls, data: bytes) -> tuple[IicpFrame, int]:
        if len(data) < FRAME_HEADER_LEN:
            raise ValueError(f"IICP frame too short: {len(data)} < {FRAME_HEADER_LEN}")
        magic, version, msg_type, flags, _res, payload_len = _HEADER_STRUCT.unpack_from(data)
        if magic != IICP_MAGIC:
            raise ValueError(f"Invalid IICP magic: {magic!r}")
        total = FRAME_HEADER_LEN + payload_len
        if len(data) < total:
            raise ValueError(f"IICP payload truncated: need {total}, have {len(data)}")
        payload = bytes(data[FRAME_HEADER_LEN:total])
        return cls(version=version, msg_type=msg_type, flags=flags, payload=payload), total

    @classmethod
    def make(cls, msg_type: int, payload: bytes, flags: int = 0) -> IicpFrame:
        return cls(version=FRAMING_VERSION, msg_type=msg_type, flags=flags, payload=payload)


# ── CBOR payload helpers (lazy cbor2 import) ─────────────────────────────────


def _cbor2() -> Any:
    """Lazy import — keeps cbor2 truly optional until first use."""
    try:
        import cbor2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "cbor2 is required for the native IICP transport. "
            "Install with: pip install 'iicp-client[iicp-tcp]'"
        ) from exc
    return cbor2


def encode_cbor(obj: object) -> bytes:
    return _cbor2().dumps(obj, canonical=True)


def decode_cbor(data: bytes) -> object:
    return _cbor2().loads(data)


def encode_ack(framing_version: int = FRAMING_VERSION, node_id: str | None = None) -> bytes:
    payload: dict[int, object] = {1: framing_version}
    if node_id is not None:
        payload[2] = node_id
    return encode_cbor(payload)


def encode_pong(echo: bytes | None = None) -> bytes:
    payload: dict[int, object] = {}
    if echo:
        payload[1] = echo
    return encode_cbor(payload)


def encode_response(
    session_id: str,
    call_id: str | None = None,
    result: bytes | str | None = None,
    error_code: int | None = None,
    error_message: str | None = None,
) -> bytes:
    payload: dict[int, object] = {2: session_id}
    if call_id is not None:
        payload[15] = call_id
    if result is not None:
        payload[5] = result if isinstance(result, bytes) else result.encode()
    if error_code is not None:
        payload[100] = error_code
    if error_message is not None:
        payload[101] = error_message
    return encode_cbor(payload)


def encode_discover_response(session_id: str, intent: str, nodes: list[dict[str, object]]) -> bytes:
    return encode_cbor({2: session_id, 3: intent, 20: nodes})


# ── Server ────────────────────────────────────────────────────────────────────

# A user-supplied task handler shape — mirrors NodeConfig task handler:
#   def handler(task: dict) -> dict
# task dict has keys {task_id, intent, payload, ...}. Handler returns
# {"result": ...} or {"error_code": int, "error_message": str}.
TcpTaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Discover lookup — given an intent URN, return a list of node descriptors.
# Typically delegated back to the IicpClient.discover() against the directory.
DiscoverLookup = Callable[[str], Awaitable[list[dict[str, object]]]]


class IicpTcpServer:
    """Asyncio TCP server speaking native IICP binary framing on `port` (default 9484).

    Runs alongside the HTTP server exposed by IicpNode.serve(). Both can be
    started in the same process to give the node dual-transport reachability.

    Usage::

        server = IicpTcpServer(
            host="0.0.0.0",
            port=9484,
            node_id="my-node-id",
            handler=my_async_task_handler,
            discover_lookup=my_async_discover,
        )
        await server.start()
        ...
        await server.stop()
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 9484,
        node_id: str | None = None,
        handler: TcpTaskHandler | None = None,
        discover_lookup: DiscoverLookup | None = None,
        concurrency_gate: object | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self.handler = handler
        self.discover_lookup = discover_lookup
        # Optional ConcurrencyGate (iicp_client.concurrency.ConcurrencyGate).
        # When set, every CALL frame acquires a slot before invoking the
        # user handler; CapacityExceededError → RESPONSE with error_code=429
        # (IICP-E021). Same primitive the HTTP /v1/task path uses, so the
        # directory's NodeScorer sees back-pressure from either transport.
        self.concurrency_gate = concurrency_gate
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        # Validate cbor2 is importable before opening the socket so we fail fast.
        _cbor2()
        self._server = await asyncio.start_server(
            self._handle_connection, host=self.host, port=self.port
        )
        logger.info("IICP TCP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("IICP TCP server stopped")

    async def serve_forever(self) -> None:
        """Block until the server is shut down. Useful for stand-alone scripts."""
        if not self._server:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    # ── connection handling ──────────────────────────────────────────────────

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("IICP TCP connection from %s", peer)
        buf = bytearray()
        try:
            await self._session(reader, writer, buf)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("IICP TCP session error from %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _session(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        buf: bytearray,
    ) -> None:
        # Magic byte validation (spec §1.2)
        magic = await reader.readexactly(4)
        if magic != IICP_MAGIC:
            logger.warning("Invalid magic from %s — closing", writer.get_extra_info("peername"))
            return
        rest = await reader.readexactly(FRAME_HEADER_LEN - 4)
        buf += magic + rest

        while True:
            # Stage 1: ensure header is complete
            while len(buf) < FRAME_HEADER_LEN:
                chunk = await reader.read(_READ_CHUNK)
                if not chunk:
                    return
                buf += chunk

            # Stage 2: peek payload_len, wait for full frame before decoding.
            # This was the iter-1410 adapter fix — pre-fix the session loop
            # closed on every frame with a non-empty CBOR payload because
            # decode requires header + payload and only the header had arrived.
            magic_bytes, _ver, _mt, _flags, _res, payload_len = _HEADER_STRUCT.unpack_from(buf)
            if magic_bytes != IICP_MAGIC:
                logger.warning("Mid-stream magic drift — closing")
                return
            if payload_len + FRAME_HEADER_LEN > _MAX_PAYLOAD:
                logger.warning("IICP frame payload exceeds limit — closing")
                return
            total_len = FRAME_HEADER_LEN + payload_len
            while len(buf) < total_len:
                chunk = await reader.read(_READ_CHUNK)
                if not chunk:
                    return
                buf += chunk

            try:
                frame, consumed = IicpFrame.decode(bytes(buf))
            except ValueError as exc:
                logger.warning("Frame decode error: %s", exc)
                return
            del buf[:consumed]

            keep_open = await self._dispatch(frame, writer)
            if not keep_open:
                return

    async def _dispatch(self, frame: IicpFrame, writer: asyncio.StreamWriter) -> bool:
        try:
            mt = MsgType(frame.msg_type)
        except ValueError:
            logger.debug("Unknown msg_type 0x%02x — ignoring", frame.msg_type)
            return True

        if mt == MsgType.INIT:
            return await self._on_init(writer)
        if mt == MsgType.PING:
            return await self._on_ping(frame, writer)
        if mt == MsgType.DISCOVER:
            return await self._on_discover(frame, writer)
        if mt == MsgType.CALL:
            return await self._on_call(frame, writer)
        if mt == MsgType.CLOSE:
            return False  # graceful shutdown requested by peer
        if mt == MsgType.FEEDBACK:
            return True
        logger.debug("Unhandled msg_type %s — ignoring", mt.name)
        return True

    # ── handlers ─────────────────────────────────────────────────────────────

    async def _on_init(self, writer: asyncio.StreamWriter) -> bool:
        ack = IicpFrame.make(
            MsgType.ACK, encode_ack(framing_version=FRAMING_VERSION, node_id=self.node_id)
        )
        writer.write(ack.encode())
        await writer.drain()
        return True

    async def _on_ping(self, frame: IicpFrame, writer: asyncio.StreamWriter) -> bool:
        echo: bytes | None = None
        if frame.payload:
            try:
                body = decode_cbor(frame.payload)
                echo = body.get(1) if isinstance(body, dict) else None
            except Exception:  # noqa: BLE001
                pass
        pong = IicpFrame.make(MsgType.PONG, encode_pong(echo=echo))
        writer.write(pong.encode())
        await writer.drain()
        return True

    async def _on_discover(self, frame: IicpFrame, writer: asyncio.StreamWriter) -> bool:
        session_id = "unknown"
        intent = ""
        try:
            body = decode_cbor(frame.payload)
            if isinstance(body, dict):
                session_id = str(body.get(2, "unknown"))
                intent = str(body.get(3, ""))
        except Exception:  # noqa: BLE001
            pass

        nodes: list[dict[str, object]] = []
        if self.discover_lookup is not None and intent:
            try:
                nodes = await self.discover_lookup(intent)
            except Exception as exc:  # noqa: BLE001
                logger.warning("discover_lookup raised: %s", exc)

        resp = IicpFrame.make(
            MsgType.RESPONSE,
            encode_discover_response(session_id=session_id, intent=intent, nodes=nodes),
        )
        writer.write(resp.encode())
        await writer.drain()
        return True

    async def _on_call(self, frame: IicpFrame, writer: asyncio.StreamWriter) -> bool:
        session_id = "unknown"
        call_id: str | None = None
        intent = ""
        task_id = ""
        payload_obj: dict[str, Any] = {}

        try:
            body = decode_cbor(frame.payload)
            if isinstance(body, dict):
                session_id = str(body.get(2, "unknown"))
                intent = str(body.get(3, ""))
                call_id = body.get(15)
                raw5 = body.get(5, b"")
                # Mirror the adapter call_pipeline contract: key 5 is the task
                # body as either a CBOR dict OR a UTF-8 JSON byte string. Decode
                # to a dict before passing to the user handler.
                if isinstance(raw5, dict):
                    payload_obj = raw5
                else:
                    raw5_str = (
                        raw5.decode("utf-8", errors="replace")
                        if isinstance(raw5, bytes)
                        else str(raw5)
                    )
                    if raw5_str:
                        try:
                            decoded = json.loads(raw5_str)
                            if isinstance(decoded, dict):
                                payload_obj = decoded
                        except json.JSONDecodeError:
                            pass
                task_id = str(call_id or session_id)
        except Exception:  # noqa: BLE001
            pass

        result: bytes | str | None = None
        error_code: int | None = None
        error_message: str | None = None

        if self.handler is None:
            error_code, error_message = 503, "no handler configured"
        else:
            task = {
                "task_id": task_id,
                "intent": intent,
                "payload": payload_obj,
            }
            # Tier 2 Item 5 (#340): if a ConcurrencyGate is configured, acquire
            # a slot first. CapacityExceededError → 429 IICP-E021 so the
            # directory's NodeScorer down-ranks busy nodes consistently
            # across HTTP and native IICP transports.
            from iicp_client.concurrency import CapacityExceededError, ConcurrencyGate

            gate = (
                self.concurrency_gate
                if isinstance(self.concurrency_gate, ConcurrencyGate)
                else None
            )

            async def _run_handler() -> None:
                nonlocal result, error_code, error_message
                try:
                    handler_result = await self.handler(task)
                    if isinstance(handler_result, dict):
                        if "error_code" in handler_result:
                            error_code = int(handler_result["error_code"])
                            error_message = str(
                                handler_result.get("error_message", "handler error")
                            )
                        else:
                            result = encode_cbor(handler_result.get("result", handler_result))
                    else:
                        result = encode_cbor({"result": handler_result})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TCP CALL handler raised: %s", exc)
                    error_code, error_message = 500, "handler raised exception"

            if gate is None:
                await _run_handler()
            else:
                try:
                    async with gate.acquire():
                        await _run_handler()
                except CapacityExceededError as exc:
                    error_code = 429
                    error_message = f"IICP-E021: max_concurrent={exc.max_concurrent} reached"

        resp = IicpFrame.make(
            MsgType.RESPONSE,
            encode_response(
                session_id=session_id,
                call_id=call_id,
                result=result,
                error_code=error_code,
                error_message=error_message,
            ),
        )
        writer.write(resp.encode())
        await writer.drain()
        return True


# ── Client ────────────────────────────────────────────────────────────────────


class IicpTcpClientError(RuntimeError):
    """Raised when an IICP TCP RPC fails (wrong response type, server error, timeout)."""


class IicpTcpClient:
    """Asyncio TCP client speaking native IICP binary framing.

    Symmetric counterpart to IicpTcpServer: consumers connect to a node's
    port 9484, do INIT/ACK handshake, then issue PING/DISCOVER/CALL requests.

    Usage::

        async with IicpTcpClient("203.0.113.5", 9484) as client:
            await client.handshake()
            nodes = await client.discover("urn:iicp:intent:llm:chat:v1")
            result = await client.call(
                intent="urn:iicp:intent:llm:chat:v1",
                payload={"messages": [{"role":"user","content":"hi"}]},
            )

    The async-context-manager auto-handles connect + CLOSE + disconnect.
    """

    def __init__(self, host: str, port: int = 9484, *, timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # node_id advertised by the server in the ACK payload — populated by handshake().
        self.peer_node_id: str | None = None
        # framing_version negotiated in INIT/ACK — populated by handshake().
        self.framing_version: int | None = None

    async def __aenter__(self) -> IicpTcpClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._writer is not None and not self._writer.is_closing():
                await self.close()
        finally:
            await self.disconnect()

    async def connect(self) -> None:
        # Eagerly check cbor2 is importable so we fail fast.
        _cbor2()
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.timeout_s,
        )

    async def disconnect(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        self._reader = None
        self._writer = None

    async def handshake(self) -> None:
        """Send INIT, await ACK, populate peer_node_id and framing_version."""
        assert self._reader is not None and self._writer is not None, "not connected"
        init_payload = encode_cbor({1: FRAMING_VERSION})
        self._writer.write(IicpFrame.make(MsgType.INIT, init_payload).encode())
        await self._writer.drain()

        mt, payload = await self._read_frame()
        if mt != MsgType.ACK:
            raise IicpTcpClientError(f"expected ACK (0x02), got 0x{mt:02x}")
        body = decode_cbor(payload) if payload else {}
        if isinstance(body, dict):
            self.framing_version = body.get(1)
            v = body.get(2)
            self.peer_node_id = v if isinstance(v, str) else None

    async def ping(self, echo: bytes | None = None) -> bytes | None:
        """Send PING; return the echoed bytes from the PONG (or None if not echoed)."""
        assert self._writer is not None
        payload = encode_cbor({1: echo}) if echo else encode_cbor({})
        self._writer.write(IicpFrame.make(MsgType.PING, payload).encode())
        await self._writer.drain()
        mt, body_bytes = await self._read_frame()
        if mt != MsgType.PONG:
            raise IicpTcpClientError(f"expected PONG (0x0a), got 0x{mt:02x}")
        body = decode_cbor(body_bytes) if body_bytes else {}
        return body.get(1) if isinstance(body, dict) else None

    async def discover(self, intent: str, *, session_id: str = "discover-1") -> list[dict]:
        """Send DISCOVER for `intent`; return the nodes list from the RESPONSE."""
        assert self._writer is not None
        payload = encode_cbor({2: session_id, 3: intent})
        self._writer.write(IicpFrame.make(MsgType.DISCOVER, payload).encode())
        await self._writer.drain()
        mt, body_bytes = await self._read_frame()
        if mt != MsgType.RESPONSE:
            raise IicpTcpClientError(f"expected RESPONSE (0x06), got 0x{mt:02x}")
        body = decode_cbor(body_bytes)
        if not isinstance(body, dict):
            raise IicpTcpClientError(f"DISCOVER response body not a CBOR map: {body!r}")
        return body.get(20) or []

    async def call(
        self,
        intent: str,
        payload: dict[str, Any],
        *,
        session_id: str = "call-1",
        call_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send CALL with the given JSON payload; return the CBOR-decoded result dict.

        Raises IicpTcpClientError if the server replies with an error code.
        """
        assert self._writer is not None
        body: dict[int, object] = {
            2: session_id,
            3: intent,
            5: json.dumps(payload).encode("utf-8"),
        }
        if call_id is not None:
            body[15] = call_id
        self._writer.write(IicpFrame.make(MsgType.CALL, encode_cbor(body)).encode())
        await self._writer.drain()
        mt, body_bytes = await self._read_frame(timeout_s=timeout_s)
        if mt != MsgType.RESPONSE:
            raise IicpTcpClientError(f"expected RESPONSE (0x06), got 0x{mt:02x}")
        resp = decode_cbor(body_bytes) if body_bytes else {}
        if not isinstance(resp, dict):
            raise IicpTcpClientError(f"CALL response body not a CBOR map: {resp!r}")
        if 100 in resp:
            raise IicpTcpClientError(f"server error {resp[100]}: {resp.get(101, '')!r}")
        result_bytes = resp.get(5)
        if result_bytes is None:
            return {}
        if isinstance(result_bytes, (bytes, bytearray)):
            decoded = decode_cbor(bytes(result_bytes))
            return decoded if isinstance(decoded, dict) else {"value": decoded}
        return result_bytes if isinstance(result_bytes, dict) else {"value": result_bytes}

    async def close(self) -> None:
        """Send CLOSE (graceful teardown). Server hangs up; caller should disconnect."""
        if self._writer is None or self._writer.is_closing():
            return
        self._writer.write(IicpFrame.make(MsgType.CLOSE, b"").encode())
        try:
            await self._writer.drain()
        except Exception:  # noqa: BLE001
            pass

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _read_frame(self, timeout_s: float | None = None) -> tuple[int, bytes]:
        """Read exactly one IICP frame from the wire. Returns (msg_type, payload)."""
        assert self._reader is not None
        t = timeout_s if timeout_s is not None else self.timeout_s
        head = await asyncio.wait_for(self._reader.readexactly(FRAME_HEADER_LEN), timeout=t)
        magic, _ver, mt, _flags, _res, payload_len = _HEADER_STRUCT.unpack_from(head)
        if magic != IICP_MAGIC:
            raise IicpTcpClientError(f"bad magic in response: {magic!r}")
        payload = (
            await asyncio.wait_for(self._reader.readexactly(payload_len), timeout=t)
            if payload_len
            else b""
        )
        return mt, payload
