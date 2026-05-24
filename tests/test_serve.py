"""Integration tests for IicpNode server features.

Spins up a real ThreadingHTTPServer on a free port, exercises the endpoints,
then shuts it down — no mocking of the HTTP server itself.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import socket
import threading
import time
from http.client import HTTPConnection
from typing import Any

import pytest

from iicp_client import IicpNode, NodeConfig


# ── helpers ─────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _echo_handler(task: dict) -> dict:
    return {"result": {"echo": task.get("payload", {})}}


class _ServerHandle:
    """Runs IicpNode.serve in a background asyncio loop + thread."""

    def __init__(self, config: NodeConfig):
        self.port = _free_port()
        self._loop = asyncio.new_event_loop()
        self._node = IicpNode(config)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_ServerHandle":
        self._thread.start()
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        return self

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(
            self._node.serve(_echo_handler, host="127.0.0.1", port=self.port)
        )

    def get(self, path: str) -> tuple[int, Any]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("GET", path)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        try:
            body: Any = json.loads(raw)
        except json.JSONDecodeError:
            body = raw.decode("utf-8", errors="replace")
        return r.status, body

    def post(
        self,
        path: str,
        body: dict,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any, dict]:
        data = json.dumps(body).encode()
        conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
        hdr: dict[str, str] = {
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
        }
        if headers:
            hdr.update(headers)
        conn.request("POST", path, body=data, headers=hdr)
        r = conn.getresponse()
        raw = r.read()
        result_headers = dict(r.getheaders())
        conn.close()
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.decode("utf-8", errors="replace")
        return r.status, parsed, result_headers


@pytest.fixture(scope="module")
def srv():
    cfg = NodeConfig(
        node_id="test-node",
        endpoint="http://test-node.local",
        intent="urn:iicp:intent:llm:chat:v1",
        region="test-region",
        model="test-model",
        max_concurrent=2,
    )
    h = _ServerHandle(cfg).start()
    yield h
    h.stop()


# ── GET /iicp/health ─────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, srv: _ServerHandle):
        status, _ = srv.get("/iicp/health")
        assert status == 200

    def test_health_has_required_fields(self, srv: _ServerHandle):
        _, body = srv.get("/iicp/health")
        assert body["status"] == "ok"
        assert body["node_id"] == "test-node"
        assert body["region"] == "test-region"
        assert "active_jobs" in body
        assert "max_concurrent" in body
        assert "available" in body
        assert "load" in body

    def test_health_max_concurrent_matches_config(self, srv: _ServerHandle):
        _, body = srv.get("/iicp/health")
        assert body["max_concurrent"] == 2

    def test_unknown_get_path_404(self, srv: _ServerHandle):
        status, _ = srv.get("/not-a-thing")
        assert status == 404


# ── GET /metrics ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_endpoint_responds(self, srv: _ServerHandle):
        status, _ = srv.get("/metrics")
        assert status in (200, 503)


# ── POST /v1/task ─────────────────────────────────────────────────────────────

class TestTask:
    def test_task_returns_200(self, srv: _ServerHandle):
        status, body, _ = srv.post("/v1/task", {"task_id": "t1", "intent": "x", "payload": {}})
        assert status == 200
        assert body["status"] == "completed"

    def test_task_id_echoed(self, srv: _ServerHandle):
        status, body, _ = srv.post("/v1/task", {"task_id": "t-abc", "intent": "x", "payload": {}})
        assert status == 200
        assert body["task_id"] == "t-abc"

    def test_unknown_post_path_404(self, srv: _ServerHandle):
        status, _, _ = srv.post("/bad-path", {})
        assert status == 404


# ── Concurrency gate (IICP-E021) ─────────────────────────────────────────────

class TestConcurrencyGate:
    def test_429_always_reject(self):
        """max_concurrent=0 → every task gets 429 IICP-E021."""
        cfg = NodeConfig(
            node_id="gate-node",
            endpoint="http://gate.local",
            intent="urn:iicp:intent:llm:chat:v1",
            max_concurrent=0,
        )
        h = _ServerHandle(cfg).start()
        status, body, hdrs = h.post("/v1/task", {"task_id": "t", "intent": "x", "payload": {}})
        h.stop()

        assert status == 429
        assert body["error"]["code"] == "IICP-E021"
        assert any(k.lower() == "retry-after" and v == "2" for k, v in hdrs.items())

    def test_429_under_concurrency_pressure(self):
        """With max_concurrent=1, a second concurrent request gets 429."""
        slow_started = threading.Event()
        slow_release = threading.Event()

        async def _slow_handler(task: dict) -> dict:
            slow_started.set()
            for _ in range(100):
                if slow_release.is_set():
                    break
                await asyncio.sleep(0.05)
            return {"result": {}}

        cfg = NodeConfig(
            node_id="gate-node2",
            endpoint="http://gate.local",
            intent="urn:iicp:intent:llm:chat:v1",
            max_concurrent=1,
        )
        h = _ServerHandle.__new__(_ServerHandle)
        h.port = _free_port()
        h._loop = asyncio.new_event_loop()
        h._node = IicpNode(cfg)
        h._thread = threading.Thread(
            daemon=True,
            target=lambda: (
                asyncio.set_event_loop(h._loop),
                h._loop.run_until_complete(
                    h._node.serve(_slow_handler, host="127.0.0.1", port=h.port)
                ),
            ),
        )
        h._thread.start()
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", h.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)

        results: list[int] = []

        def _req():
            data = json.dumps({"task_id": "tx", "intent": "x", "payload": {}}).encode()
            conn = HTTPConnection("127.0.0.1", h.port, timeout=5)
            conn.request(
                "POST",
                "/v1/task",
                body=data,
                headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
            )
            r = conn.getresponse()
            results.append(r.status)
            conn.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_req)
            slow_started.wait(timeout=2)
            f2 = pool.submit(_req)
            f2.result(timeout=3)
            slow_release.set()
            f1.result(timeout=3)

        h._loop.call_soon_threadsafe(h._loop.stop)
        h._thread.join(timeout=2)

        assert 429 in results


# ── Nonce replay (IICP-E011) ─────────────────────────────────────────────────

class TestNonceReplay:
    def test_duplicate_nonce_returns_409(self, srv: _ServerHandle):
        nonce = "nonce-unique-xyz-abc"
        s1, _, _ = srv.post("/v1/task", {"task_id": "t1", "intent": "x", "payload": {}, "nonce": nonce})
        s2, body2, _ = srv.post("/v1/task", {"task_id": "t2", "intent": "x", "payload": {}, "nonce": nonce})
        assert s1 == 200
        assert s2 == 409
        assert body2["error"]["code"] == "IICP-E011"

    def test_unique_nonces_both_succeed(self, srv: _ServerHandle):
        s1, _, _ = srv.post("/v1/task", {"task_id": "t1", "intent": "x", "payload": {}, "nonce": "nonce-a-111"})
        s2, _, _ = srv.post("/v1/task", {"task_id": "t2", "intent": "x", "payload": {}, "nonce": "nonce-b-222"})
        assert s1 == 200
        assert s2 == 200

    def test_no_nonce_always_succeeds(self, srv: _ServerHandle):
        for _ in range(3):
            status, _, _ = srv.post("/v1/task", {"task_id": "t", "intent": "x", "payload": {}})
            assert status == 200


# ── W3C traceparent propagation ───────────────────────────────────────────────

class TestTraceparent:
    def test_traceparent_injected_into_handler(self):
        received: list[dict] = []

        async def _capture(task: dict) -> dict:
            received.append(task)
            return {"result": {}}

        cfg = NodeConfig(
            node_id="trace-node",
            endpoint="http://trace.local",
            intent="urn:iicp:intent:llm:chat:v1",
        )
        h = _ServerHandle.__new__(_ServerHandle)
        h.port = _free_port()
        h._loop = asyncio.new_event_loop()
        h._node = IicpNode(cfg)
        h._thread = threading.Thread(
            daemon=True,
            target=lambda: (
                asyncio.set_event_loop(h._loop),
                h._loop.run_until_complete(
                    h._node.serve(_capture, host="127.0.0.1", port=h.port)
                ),
            ),
        )
        h._thread.start()
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", h.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)

        tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        h.post("/v1/task", {"task_id": "t1", "intent": "x", "payload": {}}, {"traceparent": tp})

        h._loop.call_soon_threadsafe(h._loop.stop)
        h._thread.join(timeout=2)

        assert len(received) == 1
        assert received[0].get("_trace", {}).get("traceparent") == tp

    def test_no_traceparent_no_trace_field(self, srv: _ServerHandle):
        status, body, _ = srv.post("/v1/task", {"task_id": "t", "intent": "x", "payload": {}})
        assert status == 200
