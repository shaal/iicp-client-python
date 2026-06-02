"""Integration tests for IicpNode server features — ADR-016 §2/§3.

Spins up a real ThreadingHTTPServer on a free port, exercises the endpoints,
then shuts it down — no mocking of the HTTP server itself.

Conformance: SDK-03 (node serve), SDK-05 (error codes), SDK-06 (traceparent).
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
    """Runs IicpNode.serve in a background asyncio loop + thread.

    Shutdown discipline (iter-1447 fix): teardown CANCELS the serve task
    instead of calling loop.stop(). loop.stop() exits run_until_complete
    without unwinding the coroutine through its `finally:` block, so the
    underlying http.server.serve_forever in run_in_executor never gets
    server.shutdown() called → executor thread leaks. On macOS the daemon
    thread is reaped at process exit so the fixture appears to work; on
    Linux (github-hosted runners) pytest's exit-handler waits indefinitely.

    Cancelling the task triggers the coroutine's finally block → calls
    server.shutdown() → serve_forever exits cleanly → no thread leak.
    """

    def __init__(self, config: NodeConfig):
        self.port = _free_port()
        self._node = IicpNode(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> _ServerHandle:
        self._thread.start()
        # Wait for the loop + task to be initialised so stop() can find them.
        self._ready.wait(timeout=5)
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._loop is None or self._task is None:
            return
        loop = self._loop
        task = self._task
        # Cancel the serve coroutine on its own loop. This unwinds through
        # node.serve()'s `finally: server.shutdown()` which exits serve_forever
        # and lets the executor thread terminate cleanly.
        loop.call_soon_threadsafe(task.cancel)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(
            self._node.serve(_echo_handler, host="127.0.0.1", port=self.port)
        )
        self._ready.set()
        try:
            self._loop.run_until_complete(self._task)
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            pass
        except RuntimeError:
            # run_until_complete on a cancelled task can raise this too.
            pass
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

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
        # #343 — pinhole_state always surfaced (active: False when no pinhole opened)
        assert "pinhole_state" in body
        assert body["pinhole_state"]["active"] is False

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


class TestIdempotency:
    def test_disabled_by_default_allows_resubmit(self):
        cfg = NodeConfig(
            node_id="idem-off",
            endpoint="http://idem.local",
            intent="urn:iicp:intent:llm:chat:v1",
        )
        h = _ServerHandle(cfg).start()
        s1, _, _ = h.post("/v1/task", {"task_id": "dup", "intent": "x", "payload": {}})
        s2, _, _ = h.post("/v1/task", {"task_id": "dup", "intent": "x", "payload": {}})
        h.stop()
        assert s1 == 200
        assert s2 == 200

    def test_enabled_rejects_duplicate_task_id(self):
        cfg = NodeConfig(
            node_id="idem-on",
            endpoint="http://idem.local",
            intent="urn:iicp:intent:llm:chat:v1",
            enable_idempotency=True,
        )
        h = _ServerHandle(cfg).start()
        s1, _, _ = h.post("/v1/task", {"task_id": "dup", "intent": "x", "payload": {}})
        s2, body2, _ = h.post("/v1/task", {"task_id": "dup", "intent": "x", "payload": {}})
        s3, _, _ = h.post("/v1/task", {"task_id": "other", "intent": "x", "payload": {}})
        h.stop()
        assert s1 == 200
        assert s2 == 409
        assert body2["error"]["code"] == "IICP-E010"
        assert s3 == 200

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
        h = _ServerHandle(cfg)
        # Replace the default echo handler with the slow one.
        # _ServerHandle.start() creates loop + task with _echo_handler by
        # default; for this test we need _slow_handler. Patch by overriding
        # the _run method via a custom thread before .start().
        h._task = None
        h._loop = None

        def _run_slow() -> None:
            h._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(h._loop)
            h._task = h._loop.create_task(
                h._node.serve(_slow_handler, host="127.0.0.1", port=h.port)
            )
            h._ready.set()
            try:
                h._loop.run_until_complete(h._task)
            except (asyncio.CancelledError, concurrent.futures.CancelledError, RuntimeError):
                pass
            finally:
                try:
                    h._loop.close()
                except Exception:  # noqa: BLE001
                    pass

        h._thread = threading.Thread(target=_run_slow, daemon=True)
        h._thread.start()
        h._ready.wait(timeout=5)
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

        h.stop()

        assert 429 in results


# ── Nonce replay (IICP-E011) ─────────────────────────────────────────────────


class TestNonceReplay:
    def test_duplicate_nonce_returns_409(self, srv: _ServerHandle):
        nonce = "nonce-unique-xyz-abc"
        body = {"task_id": "t1", "intent": "x", "payload": {}, "nonce": nonce}
        s1, _, _ = srv.post("/v1/task", body)
        body2_req = {"task_id": "t2", "intent": "x", "payload": {}, "nonce": nonce}
        s2, body2, _ = srv.post("/v1/task", body2_req)
        assert s1 == 200
        assert s2 == 409
        assert body2["error"]["code"] == "IICP-E011"

    def test_unique_nonces_both_succeed(self, srv: _ServerHandle):
        t1 = {"task_id": "t1", "intent": "x", "payload": {}, "nonce": "nonce-a"}
        t2 = {"task_id": "t2", "intent": "x", "payload": {}, "nonce": "nonce-b"}
        s1, _, _ = srv.post("/v1/task", t1)
        s2, _, _ = srv.post("/v1/task", t2)
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
        h = _ServerHandle(cfg)
        h._task = None
        h._loop = None

        def _run_capture() -> None:
            h._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(h._loop)
            h._task = h._loop.create_task(
                h._node.serve(_capture, host="127.0.0.1", port=h.port)
            )
            h._ready.set()
            try:
                h._loop.run_until_complete(h._task)
            except (asyncio.CancelledError, concurrent.futures.CancelledError, RuntimeError):
                pass
            finally:
                try:
                    h._loop.close()
                except Exception:  # noqa: BLE001
                    pass

        h._thread = threading.Thread(target=_run_capture, daemon=True)
        h._thread.start()
        h._ready.wait(timeout=5)
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", h.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)

        tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        h.post("/v1/task", {"task_id": "t1", "intent": "x", "payload": {}}, {"traceparent": tp})

        h.stop()

        assert len(received) == 1
        assert received[0].get("_trace", {}).get("traceparent") == tp

    def test_no_traceparent_no_trace_field(self, srv: _ServerHandle):
        status, body, _ = srv.post("/v1/task", {"task_id": "t", "intent": "x", "payload": {}})
        assert status == 200


# ── #399 — heartbeat loop re-registers after the directory drops the node ──────


def test_heartbeat_reregisters_on_404(monkeypatch):
    """A 404/401/410 heartbeat (directory forgot the node) must trigger a
    re-register, and the loop must resume with the fresh token — not heartbeat
    into the void forever (#399)."""
    import contextlib

    import httpx

    from iicp_client import node as node_mod

    cfg = NodeConfig(
        node_id="t",
        endpoint="http://t.local",
        intent="urn:iicp:intent:llm:chat:v1",
        region="r",
        model="m",
        max_concurrent=1,
    )
    n = IicpNode(cfg)
    monkeypatch.setattr(node_mod, "_HEARTBEAT_INTERVAL", 0.01)
    calls = {"hb": 0, "reg": 0, "last_token": None}

    async def fake_hb(tok):
        calls["hb"] += 1
        calls["last_token"] = tok
        if calls["hb"] == 1:
            req = httpx.Request("POST", "http://t.local/v1/heartbeat")
            raise httpx.HTTPStatusError(
                "node not found", request=req, response=httpx.Response(404, request=req)
            )

    async def fake_reg():
        calls["reg"] += 1
        return "fresh-token"

    monkeypatch.setattr(n, "heartbeat", fake_hb)
    monkeypatch.setattr(n, "register", fake_reg)

    async def _run():
        task = asyncio.create_task(n._heartbeat_loop("orig-token"))
        await asyncio.sleep(0.06)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert calls["reg"] >= 1, "should re-register after a 404 heartbeat"
    assert calls["last_token"] == "fresh-token", "loop must resume with the fresh token"


def test_heartbeat_self_heals_from_empty_initial_token(monkeypatch):
    """#404 — when startup registration failed the heartbeat loop is started with
    an empty token; its first heartbeat 401s and it must re-register and recover,
    without a manual restart."""
    import contextlib

    import httpx

    from iicp_client import node as node_mod

    cfg = NodeConfig(
        node_id="t",
        endpoint="http://t.local",
        intent="urn:iicp:intent:llm:chat:v1",
        region="r",
        model="m",
        max_concurrent=1,
    )
    n = IicpNode(cfg)
    monkeypatch.setattr(node_mod, "_HEARTBEAT_INTERVAL", 0.01)
    calls = {"hb": 0, "reg": 0, "last_token": None}

    async def fake_hb(tok):
        calls["hb"] += 1
        calls["last_token"] = tok
        if tok == "":  # empty startup token → directory rejects with 401
            req = httpx.Request("POST", "http://t.local/v1/heartbeat")
            raise httpx.HTTPStatusError(
                "unauthorized", request=req, response=httpx.Response(401, request=req)
            )

    async def fake_reg():
        calls["reg"] += 1
        return "recovered-token"

    monkeypatch.setattr(n, "heartbeat", fake_hb)
    monkeypatch.setattr(n, "register", fake_reg)

    async def _run():
        task = asyncio.create_task(n._heartbeat_loop(""))  # started with empty token
        await asyncio.sleep(0.06)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert calls["reg"] >= 1, "empty-token loop must re-register on the 401"
    assert calls["last_token"] == "recovered-token", "loop must resume with the recovered token"
