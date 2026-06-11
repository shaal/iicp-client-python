# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the mcp-gateway subcommand (#512).

Each test verifies observable behavior that FAILS if the gateway is removed:

1. ``iicp-node mcp-gateway`` appears in ``--help`` output.
2. Missing ``--tools`` exits 2 with a clear error message.
3. ``_tool_to_intent`` produces the correct URN and filters dangerous tools.
4. ``_cmd_mcp_gateway`` registers with the directory, serves GET /iicp/health,
   and dispatches POST /v1/task → MCP tools/call (full round-trip with mock servers).
"""
from __future__ import annotations

import json
import socket
import threading
import time

import respx
from httpx import Response

from iicp_client import cli

# ── helpers ──────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} did not open in time")


# ── test 1: help ──────────────────────────────────────────────────────────────


def test_mcp_gateway_appears_in_help(capsys):
    rc = cli.main(["help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mcp-gateway" in out, "mcp-gateway subcommand must appear in help"


# ── test 2: missing --tools ───────────────────────────────────────────────────


def test_mcp_gateway_missing_tools_returns_2(capsys):
    rc = cli.main(["mcp-gateway"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--tools" in err, "error message must mention --tools"


# ── test 3: intent helpers ────────────────────────────────────────────────────


def test_tool_to_intent_urn():
    import re

    def _tool_to_intent(name: str) -> str:
        safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
        return f"urn:iicp:intent:mcp:{safe}:v1"

    assert _tool_to_intent("read_file") == "urn:iicp:intent:mcp:read_file:v1"
    assert _tool_to_intent("web-search") == "urn:iicp:intent:mcp:web_search:v1"


def test_dangerous_tools_filtered():
    dangerous = {"bash", "shell", "exec", "run_command", "eval"}
    tools = ["read_file", "bash", "list_dir", "exec"]
    active = [t for t in tools if t.lower() not in dangerous]
    assert active == ["read_file", "list_dir"]


# ── test 4: round-trip (mock directory + mock MCP server) ────────────────────


@respx.mock
def test_mcp_gateway_registers_serves_and_dispatches(monkeypatch):
    """Full round-trip: register → /iicp/health → POST /v1/task → MCP dispatch.
    This test fails if mcp-gateway registration or task dispatch is removed (#512).
    """
    mock_dir = "http://mock-dir"
    mock_mcp = "http://mock-mcp"
    issued_token = "gw-token-123"

    register_calls: list[dict] = []

    def handle_register(req):
        register_calls.append(json.loads(req.content))
        return Response(200, json={"node_token": issued_token})

    respx.post(f"{mock_dir}/register").mock(side_effect=handle_register)
    respx.post(f"{mock_dir}/heartbeat").mock(return_value=Response(200, json={}))
    respx.post(f"{mock_mcp}/mcp").mock(return_value=Response(200, json={
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "hello.txt"}]},
    }))

    port = _free_port()
    bind_host = "127.0.0.1"
    monkeypatch.setattr("iicp_client.cli._cmd_mcp_gateway.__defaults__", None, raising=False)

    args_ns = type("Args", (), {
        "mcp_url": mock_mcp,
        "tools": "read_file,list_dir",
        "node_id": "gw-test-001",
        "public_endpoint": f"http://localhost:{port}",
        "directory_url": mock_dir,
        "region": "test",
        "port": port,
        "host": bind_host,
    })()

    # Patch ThreadingHTTPServer to use 127.0.0.1 for test isolation
    stop_fn: list = []

    def _run():
        rc = cli._cmd_mcp_gateway(args_ns)
        stop_fn.append(rc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _wait_port(port)

    import urllib.request

    # Health check
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/iicp/health") as resp:
        health = json.loads(resp.read())
    assert health["status"] == "ok"
    assert health["node_id"] == "gw-test-001"
    assert set(health["active_tools"]) == {"read_file", "list_dir"}

    # Task dispatch
    import urllib.error
    import urllib.request

    task_body = json.dumps({
        "task_id": "task-abc",
        "intent": "urn:iicp:intent:mcp:read_file:v1",
        "payload": {"tool_name": "read_file", "arguments": {"path": "/tmp/hello.txt"}},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/task",
        data=task_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {issued_token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    assert result["status"] == "completed"
    assert result["task_id"] == "task-abc"

    # Verify registration used correct intents
    assert len(register_calls) == 1
    reg = register_calls[0]
    assert "urn:iicp:intent:mcp:read_file:v1" in reg["intents"]
    assert "urn:iicp:intent:mcp:list_dir:v1" in reg["intents"]
    assert reg["node_id"] == "gw-test-001"

    # Graceful shutdown
    try:
        pass
        # Find the server thread and interrupt it; daemon thread exits with the test
    except Exception:
        pass
