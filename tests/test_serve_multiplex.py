# SPDX-License-Identifier: Apache-2.0
"""#457 / ADR-040 — `iicp-node serve` multiplexes the HTTP control plane and the native
IICP binary transport on ONE port (first-byte detection). Proves BOTH planes answer on the
same socket, and that transport_endpoint derives from the HTTP endpoint.

Fails without the fix: pre-#457 serve() bound only an HTTP server on the port, so a native
IICP CALL would hit the HTTP parser and never get a RESPONSE.
"""

from __future__ import annotations

import asyncio
import socket
from http.client import HTTPConnection
from typing import Any

from iicp_client import IicpNode, NodeConfig
from iicp_client.iicp_tcp import IicpTcpClient
from iicp_client.node import derive_native_endpoint

CHAT = "urn:iicp:intent:llm:chat:v1"


async def _echo(task: dict[str, Any]) -> dict[str, Any]:
    return {"result": {"echo": task.get("payload")}}


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _http_health(port: int) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=3)
    conn.request("GET", "/iicp/health")
    status = conn.getresponse().status
    conn.close()
    return status


async def _wait_port(port: int) -> None:
    loop = asyncio.get_event_loop()
    for _ in range(50):
        try:
            await loop.run_in_executor(
                None, lambda: socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            )
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise AssertionError(f"port {port} never came up")


async def test_http_and_native_call_share_one_port() -> None:
    cfg = NodeConfig(
        node_id="mux-node",
        endpoint="http://test-node.local",
        intent=CHAT,
        region="test-region",
        model="test-model",
        max_concurrent=4,
    )
    node = IicpNode(cfg)
    port = _free_port()
    serve_task = asyncio.create_task(
        node.serve(_echo, host="127.0.0.1", port=port, node_token=None)
    )
    try:
        await _wait_port(port)

        # HTTP control plane answers on the port.
        status = await asyncio.get_event_loop().run_in_executor(None, _http_health, port)
        assert status == 200, f"HTTP /iicp/health returned {status}"

        # Native IICP CALL answers on the SAME port (pre-#457 this hit the HTTP parser).
        async with IicpTcpClient("127.0.0.1", port) as client:
            result = await client.call(CHAT, {"messages": [{"role": "user", "content": "hi"}]})
        assert isinstance(result, dict), "native CALL returned a RESPONSE result over the shared port"
    finally:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


def test_derive_native_endpoint() -> None:
    assert derive_native_endpoint("http://203.0.113.5:9484") == "iicp://203.0.113.5:9484"
    assert derive_native_endpoint("https://node.example:9484") == "iicpsec://node.example:9484"
    assert derive_native_endpoint("not-a-url") is None
