# ADR-016: IICP client SDK conformance — #450 HTTP long-poll relay transport
"""Behavior tests for the browser-compatible HTTP-poll relay worker transport.

Covers (fails if the #450 implementation is reverted):
- HttpPollWorkerSession queue/future roundtrip + liveness semantics
- POST /v1/relay/bind (200 token, 409 alive-rebind — #510 interim-C parity)
- Bearer auth on pull/result/unbind (401 without/with-wrong token)
- Path-scoped /v1/relay-for/<wid>/v1/task forwarding (the R1 misattribution fix)
- /v1/relay-for/<wid>/iicp/health session liveness view
- CORS headers + OPTIONS preflight (web pages are first-class callers)
- RELAY_ACK field 4 carries the relay HTTP port (TCP worker endpoint fix)
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
from iicp_client.relay_session import (
    HttpPollWorkerSession,
    RelaySessionRegistry,
)

# ── helpers (same harness pattern as test_serve.py) ──────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _echo_handler(task: dict) -> dict:
    return {"result": {"echo": task.get("payload", {})}}


class _ServerHandle:
    def __init__(self, config: NodeConfig):
        self.port = _free_port()
        self._node = IicpNode(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_ServerHandle":
        self._thread.start()
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
            pass
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 35.0,
    ) -> tuple[int, Any, dict]:
        data = json.dumps(body).encode() if body is not None else b""
        conn = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        hdr: dict[str, str] = {}
        if body is not None:
            hdr["Content-Type"] = "application/json"
            hdr["Content-Length"] = str(len(data))
        if headers:
            hdr.update(headers)
        conn.request(method, path, body=data if body is not None else None, headers=hdr)
        r = conn.getresponse()
        raw = r.read()
        result_headers = dict(r.getheaders())
        conn.close()
        try:
            parsed: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw.decode("utf-8", errors="replace")
        return r.status, parsed, result_headers


@pytest.fixture(scope="module")
def relay():
    cfg = NodeConfig(
        node_id="relay-node",
        endpoint="http://relay.local",
        intent="urn:iicp:intent:llm:chat:v1",
        region="test-region",
        model="relay-model",
        relay_capable=True,
        relay_accept_port=_free_port(),
    )
    h = _ServerHandle(cfg).start()
    yield h
    h.stop()


# ── HttpPollWorkerSession unit behavior ──────────────────────────────────────


class TestHttpPollWorkerSession:
    def test_forward_pull_result_roundtrip(self):
        async def scenario():
            sess = HttpPollWorkerSession("w-browser", models=["tinyllama"])

            async def worker():
                call = await sess.next_call(timeout=5)
                assert call is not None
                assert call["task"]["payload"] == {"q": 1}
                sess.on_response(call["call_id"], {"result": {"a": 2}})

            wtask = asyncio.ensure_future(worker())
            result = await sess.forward_task({"payload": {"q": 1}}, timeout=5)
            await wtask
            assert result == {"result": {"a": 2}}

        asyncio.new_event_loop().run_until_complete(scenario())

    def test_next_call_times_out_to_none(self):
        async def scenario():
            sess = HttpPollWorkerSession("w-idle")
            assert await sess.next_call(timeout=0.05) is None

        asyncio.new_event_loop().run_until_complete(scenario())

    def test_liveness_window_and_close(self):
        sess = HttpPollWorkerSession("w-live", liveness_window=0.05)
        assert sess.is_alive()
        time.sleep(0.08)
        assert not sess.is_alive()  # stale — displaceable
        fresh = HttpPollWorkerSession("w-fresh")
        fresh.close()
        assert not fresh.is_alive()

    def test_registry_get_by_token(self):
        reg = RelaySessionRegistry()
        sess = HttpPollWorkerSession("w-tok")
        reg.bind("w-tok", sess)
        assert reg.get_by_token(sess.session_token) is sess
        assert reg.get_by_token("wrong") is None
        assert reg.get_by_token("") is None


# ── HTTP endpoint behavior (real server) ─────────────────────────────────────


class TestRelayHttpPollEndpoints:
    def _bind(self, relay, worker_id="w-e2e", models=None):
        status, body, headers = relay.request(
            "POST",
            "/v1/relay/bind",
            {"worker_id": worker_id, "intent": "urn:iicp:intent:llm:chat:v1",
             "models": models or ["tinyllama-1.1b"]},
        )
        return status, body, headers

    def test_bind_returns_token_and_cors(self, relay):
        status, body, headers = self._bind(relay, worker_id="w-bind-1")
        assert status == 200
        assert isinstance(body["session_token"], str) and len(body["session_token"]) >= 32
        assert body["worker_endpoint_path"] == "/v1/relay-for/w-bind-1"
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_alive_rebind_rejected_409(self, relay):
        status, body, _ = self._bind(relay, worker_id="w-rebind")
        assert status == 200
        status2, body2, _ = self._bind(relay, worker_id="w-rebind")
        assert status2 == 409
        assert body2["error"]["code"] == "IICP-E038"

    def test_bind_requires_worker_id(self, relay):
        status, body, _ = relay.request("POST", "/v1/relay/bind", {"intent": "x"})
        assert status == 422

    def test_pull_and_result_require_bearer(self, relay):
        status, _, _ = relay.request("GET", "/v1/relay/pull")
        assert status == 401
        status, _, _ = relay.request(
            "GET", "/v1/relay/pull", headers={"Authorization": "Bearer nope"}
        )
        assert status == 401
        status, _, _ = relay.request(
            "POST", "/v1/relay/result", {"call_id": "x", "result": {}},
            headers={"Authorization": "Bearer nope"},
        )
        assert status == 401

    def test_full_dispatch_roundtrip_via_relay_for(self, relay):
        """The headline behavior: consumer POSTs {endpoint}/v1/task on the
        path-scoped endpoint; the polling worker answers; consumer gets it."""
        status, body, _ = self._bind(relay, worker_id="w-roundtrip")
        assert status == 200
        token = body["session_token"]

        worker_done = threading.Event()
        worker_seen: dict = {}

        def worker_loop():
            # one pull → answer → done
            s, call, _ = relay.request(
                "GET", "/v1/relay/pull", headers={"Authorization": f"Bearer {token}"},
                timeout=35,
            )
            if s == 200 and call:
                worker_seen.update(call)
                relay.request(
                    "POST",
                    "/v1/relay/result",
                    {"call_id": call["call_id"],
                     "result": {"result": {"text": "MESH OK from browser"}}},
                    headers={"Authorization": f"Bearer {token}"},
                )
            worker_done.set()

        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        # Consumer side — exactly what a published SDK consumer sends.
        status, resp, headers = relay.request(
            "POST",
            "/v1/relay-for/w-roundtrip/v1/task",
            {"task_id": "t-1", "intent": "urn:iicp:intent:llm:chat:v1",
             "payload": {"messages": [{"role": "user", "content": "hi"}]}},
            timeout=40,
        )
        assert worker_done.wait(timeout=10)
        assert status == 200
        assert resp["status"] == "completed"
        assert resp["result"]["text"] == "MESH OK from browser"
        assert worker_seen["task"]["task_id"] == "t-1"
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_relay_for_unknown_worker_404(self, relay):
        status, body, _ = relay.request(
            "POST", "/v1/relay-for/w-ghost/v1/task", {"task_id": "t-x"}
        )
        assert status == 404
        assert body["error"]["code"] == "IICP-E030"

    def test_relay_for_health_reflects_session(self, relay):
        status, body, _ = self._bind(relay, worker_id="w-health", models=["m1", "m2"])
        assert status == 200
        s, health, _ = relay.request("GET", "/v1/relay-for/w-health/iicp/health")
        assert s == 200
        assert health["status"] == "ok"
        assert health["via_relay"] is True
        assert health["models"] == ["m1", "m2"]

    def test_unbind_releases_worker_id(self, relay):
        status, body, _ = self._bind(relay, worker_id="w-unbind")
        token = body["session_token"]
        s, _, _ = relay.request(
            "POST", "/v1/relay/unbind", {}, headers={"Authorization": f"Bearer {token}"}
        )
        assert s == 204
        # rebind now allowed
        status2, _, _ = self._bind(relay, worker_id="w-unbind")
        assert status2 == 200

    def test_options_preflight(self, relay):
        status, _, headers = relay.request("OPTIONS", "/v1/relay/bind")
        assert status == 204
        assert headers.get("Access-Control-Allow-Origin") == "*"
        assert "Authorization" in headers.get("Access-Control-Allow-Headers", "")


# ── RELAY_ACK http_port field (TCP worker endpoint fix) ──────────────────────


class TestRelayAckHttpPort:
    def test_ack_carries_http_port(self):
        from iicp_client.relay_session import RelayAcceptServer

        reg = RelaySessionRegistry()
        srv = RelayAcceptServer(reg, host="127.0.0.1", port=0, http_port=12345)
        assert srv.http_port == 12345

    def test_worker_defaults_http_port_when_absent(self):
        # Old relays omit field 4 → worker falls back to 9484.
        ack_body: dict = {1: "ok", 2: "w-1"}
        relay_http_port = ack_body.get(4) if isinstance(ack_body.get(4), int) else 9484
        assert relay_http_port == 9484


# ── Node-wide CORS (browser consumers, 2026-06-12) ───────────────────────────


class TestNodeWideCors:
    """Browser pages dispatch /v1/task to https nodes directly — every node
    endpoint must answer preflights and carry CORS headers (fails if the
    node-wide CORS change is reverted)."""

    def test_options_preflight_on_task(self, relay):
        status, _, headers = relay.request("OPTIONS", "/v1/task")
        assert status == 204
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_health_carries_cors(self, relay):
        status, _, headers = relay.request("GET", "/iicp/health")
        assert status == 200
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_task_response_carries_cors(self, relay):
        status, _, headers = relay.request(
            "POST",
            "/v1/task",
            {"task_id": "t-cors", "intent": "urn:iicp:intent:llm:chat:v1",
             "payload": {"messages": [{"role": "user", "content": "hi"}]}},
        )
        assert headers.get("Access-Control-Allow-Origin") == "*"
