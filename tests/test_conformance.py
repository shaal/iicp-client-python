"""Unit tests for the four CONF self-conformance probes."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import respx

from iicp_client import IicpNode, NodeConfig
from iicp_client.conformance import (
    ConformanceReport,
    _check_discover_self,
    _check_health_schema,
    _check_reachability,
    _check_registered,
    run_conformance_checks,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── CONF-REG-01 ────────────────────────────────────────────────────────────


class TestConfReg01:
    async def test_passes_when_node_id_and_token_set(self):
        node = IicpNode(
            NodeConfig(
                node_id="abcdef-1234-…",
                endpoint="https://example.com:8080",
                intent="urn:iicp:intent:llm:chat:v1",
            )
        )
        # Stash token via the documented mechanism
        node._last_token = "tok-abc"
        r = await _check_registered(node)
        assert r.passed is True
        assert "Registered" in r.message

    async def test_passes_with_node_id_only_when_token_not_tracked(self):
        node = IicpNode(
            NodeConfig(
                node_id="abc-id",
                endpoint="https://example.com:8080",
                intent="urn:iicp:intent:llm:chat:v1",
            )
        )
        r = await _check_registered(node)
        assert r.passed is True
        assert "not tracked" in r.message

    async def test_fails_when_node_id_unset(self):
        node = IicpNode(
            NodeConfig(
                node_id="",
                endpoint="https://example.com:8080",
                intent="urn:iicp:intent:llm:chat:v1",
            )
        )
        r = await _check_registered(node)
        assert r.passed is False


# ── CONF-HEALTH-01 ─────────────────────────────────────────────────────────


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal /iicp/health server for the health-probe test."""

    response_body: bytes = b'{"status":"ok","node_id":"n","region":"eu","load":0.1,"models":["m"]}'
    response_status: int = 200

    def do_GET(self):  # noqa: N802 — stdlib API
        if self.path == "/iicp/health":
            self.send_response(self.response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.response_body)))
            self.end_headers()
            self.wfile.write(self.response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kwargs):  # silence stdlib console spam
        pass


class TestConfHealth01:
    async def test_returns_pass_on_complete_schema(self):
        port = _free_port()
        _HealthHandler.response_body = (
            b'{"status":"ok","node_id":"n","region":"eu","load":0.1,"models":["m"]}'
        )
        _HealthHandler.response_status = 200
        server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            r = await _check_health_schema(port)
            assert r.passed is True, r.message
        finally:
            server.shutdown()

    async def test_returns_fail_when_required_field_missing(self):
        port = _free_port()
        # Drop the "models" field
        _HealthHandler.response_body = b'{"status":"ok","node_id":"n","region":"eu","load":0.1}'
        _HealthHandler.response_status = 200
        server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            r = await _check_health_schema(port)
            assert r.passed is False
            assert "models" in r.message
        finally:
            server.shutdown()

    async def test_returns_fail_when_health_endpoint_returns_500(self):
        port = _free_port()
        _HealthHandler.response_body = b"oops"
        _HealthHandler.response_status = 500
        server = ThreadingHTTPServer(("127.0.0.1", port), _HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            r = await _check_health_schema(port)
            assert r.passed is False
            assert "500" in r.message
        finally:
            server.shutdown()

    async def test_returns_fail_when_health_endpoint_absent(self):
        # No server running on this port
        r = await _check_health_schema(_free_port())
        assert r.passed is False


# ── CONF-REACH-01 ──────────────────────────────────────────────────────────


class TestConfReach01:
    async def test_skips_for_non_routable_endpoint(self):
        node = IicpNode(
            NodeConfig(
                node_id="n",
                endpoint="http://localhost:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        r = await _check_reachability(node)
        assert r.passed is False
        assert "non-routable" in r.message

    @respx.mock
    async def test_returns_pass_when_directory_probe_reports_reachable(self):
        respx.get("https://iicp.test/api/v1/probe").mock(
            return_value=httpx.Response(200, json={"reachable": True, "latency_ms": 25})
        )
        node = IicpNode(
            NodeConfig(
                node_id="n",
                endpoint="https://node.iicpnet.test-host.org:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        r = await _check_reachability(node)
        assert r.passed is True

    @respx.mock
    async def test_returns_fail_when_directory_probe_reports_unreachable(self):
        respx.get("https://iicp.test/api/v1/probe").mock(
            return_value=httpx.Response(200, json={"reachable": False, "error": "timeout"})
        )
        node = IicpNode(
            NodeConfig(
                node_id="n",
                endpoint="https://node.iicpnet.test-host.org:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        r = await _check_reachability(node)
        assert r.passed is False
        assert "timeout" in r.message


# ── CONF-DISC-01 ───────────────────────────────────────────────────────────


class TestConfDisc01:
    async def test_fails_when_node_id_unset(self):
        node = IicpNode(
            NodeConfig(
                node_id="",
                endpoint="https://example.com:8080",
                intent="urn:iicp:intent:llm:chat:v1",
            )
        )
        r = await _check_discover_self(node)
        assert r.passed is False

    @respx.mock
    async def test_returns_pass_when_node_id_appears_in_nodelist(self):
        respx.get("https://iicp.test/api/v1/discover").mock(
            return_value=httpx.Response(
                200,
                json={
                    "nodes": [
                        {"node_id": "other-1", "endpoint": "..."},
                        {"node_id": "self-id", "endpoint": "..."},
                    ]
                },
            )
        )
        node = IicpNode(
            NodeConfig(
                node_id="self-id",
                endpoint="https://node.iicpnet.test-host.org:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        r = await _check_discover_self(node)
        assert r.passed is True
        assert "Found" in r.message

    @respx.mock
    async def test_returns_fail_when_node_id_absent_from_nodelist(self):
        respx.get("https://iicp.test/api/v1/discover").mock(
            return_value=httpx.Response(200, json={"nodes": [{"node_id": "other-1"}]})
        )
        node = IicpNode(
            NodeConfig(
                node_id="self-id",
                endpoint="https://node.iicpnet.test-host.org:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        r = await _check_discover_self(node)
        assert r.passed is False
        assert "absent" in r.message


# ── run_conformance_checks orchestrator ────────────────────────────────────


class TestOrchestrator:
    @respx.mock
    async def test_runs_all_four_concurrently_and_counts_results(self):
        """Mock ALL routes including local 127.0.0.1 health — respx's default
        assert_all_mocked semantics block any unmatched HTTP request."""
        port = _free_port()  # need a port for the URL match but no real server
        respx.get(f"http://127.0.0.1:{port}/iicp/health").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "node_id": "n-test",
                    "region": "eu",
                    "load": 0.1,
                    "models": ["m"],
                },
            )
        )
        respx.get("https://iicp.test/api/v1/probe").mock(
            return_value=httpx.Response(200, json={"reachable": True})
        )
        respx.get("https://iicp.test/api/v1/discover").mock(
            return_value=httpx.Response(200, json={"nodes": [{"node_id": "n-test"}]})
        )
        node = IicpNode(
            NodeConfig(
                node_id="n-test",
                endpoint="https://node.iicpnet.test-host.org:8080",
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        report = await run_conformance_checks(node, local_port=port, node_token="tok")
        assert isinstance(report, ConformanceReport)
        assert report.pass_count == 4, (
            f"failures: {[(r.test_id, r.message) for r in report.tests if not r.passed]}"
        )
        assert report.fail_count == 0
        assert {r.test_id for r in report.tests} == {
            "CONF-REG-01",
            "CONF-HEALTH-01",
            "CONF-REACH-01",
            "CONF-DISC-01",
        }

    async def test_report_as_dict_is_serializable(self):
        node = IicpNode(
            NodeConfig(
                node_id="",  # forces CONF-REG-01 fail
                endpoint="http://localhost:8080",  # forces CONF-REACH-01 skip
                intent="urn:iicp:intent:llm:chat:v1",
                directory_url="https://iicp.test/api",
            )
        )
        report = await run_conformance_checks(node, local_port=_free_port())
        d = report.as_dict()
        # Round-trip via JSON
        import json

        assert json.dumps(d)  # doesn't raise
        assert d["pass_count"] + d["fail_count"] == 4
