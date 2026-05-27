"""IICP provider node — registration, heartbeats, and task serving.

Endpoints served by ``IicpNode.serve()``:

+---------+----------------+----------------------------------------------+
| Method  | Path           | Description                                  |
+=========+================+==============================================+
| POST    | /v1/task       | Handle an inference task (IICP-E021 gate,    |
|         |                | IICP-E011 nonce replay, W3C traceparent)     |
+---------+----------------+----------------------------------------------+
| GET     | /iicp/health   | Liveness / capacity (always 200)             |
+---------+----------------+----------------------------------------------+
| GET     | /metrics       | Prometheus text (503 if client absent)       |
+---------+----------------+----------------------------------------------+
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0
_HEARTBEAT_INTERVAL = 30
_NONCE_TTL = 300
_REGISTER_PATH = "/v1/register"
_HEARTBEAT_PATH = "/api/v1/heartbeat"

# Lazy Prometheus import — None until first call, False when unavailable.
_prom_mod: Any = None
# Singleton: avoids duplicate metric registration across IicpNode instances.
_global_metrics: _Metrics | None = None


def _get_prom() -> Any:
    global _prom_mod
    if _prom_mod is None:
        try:
            import prometheus_client as _p

            _prom_mod = _p
        except ImportError:
            _prom_mod = False
    return _prom_mod if _prom_mod is not False else None


@dataclass
class NodeConfig:
    node_id: str
    endpoint: str
    intent: str
    model: str | None = None
    region: str | None = None
    capabilities: list[str] = field(default_factory=list)
    directory_url: str = "https://iicp.network/api"
    timeout: float = _DEFAULT_TIMEOUT
    max_concurrent: int = 4
    tokens_per_min: int = 10000
    max_tokens: int = 8192
    # spec/iicp-dir.md v0.7.0 — optional native IICP binary endpoint (ADR-040).
    # Scheme MUST be iicp:// (plaintext) or iicpsec:// (TLS); default port 9484.
    # When set, the directory persists it and clients SHOULD prefer it over
    # `endpoint` for task CALLs. Leave None for HTTP-only operation.
    transport_endpoint: str | None = None
    # #331 Phase A.1 / ADR-041 — NAT-traversal observability fields surfaced
    # to the directory in the register payload. Populated automatically by
    # apply_nat_profile() when the operator runs detect_nat() at startup;
    # set manually if the operator already knows their topology.
    #
    # transport_method: one of {direct, upnp_mapped, stun_hole_punch,
    #                   turn_relay, external_tunnel, unknown, unreachable}
    # nat_type:         one of {full_cone, restricted_cone, port_restricted,
    #                   symmetric, unknown} — observability only
    # transport_metadata: forward-compat slot for ADR-041 transport_candidates[]
    #                   + relay_endpoint
    transport_method: str | None = None
    nat_type: str | None = None
    transport_metadata: dict | None = None


TaskHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class _Metrics:
    """Prometheus metrics wrapper — no-ops when prometheus_client is absent."""

    def __init__(self, prom: Any) -> None:
        self._enabled = prom is not None
        if not self._enabled:
            return
        self.tasks_total = prom.Counter(
            "iicp_tasks_total",
            "Total IICP tasks handled",
            ["status", "intent", "qos"],
        )
        self.task_latency_ms = prom.Histogram(
            "iicp_task_latency_ms",
            "IICP task processing latency (ms)",
            ["intent", "qos"],
            buckets=[50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000],
        )
        self.tokens_used_total = prom.Counter(
            "iicp_tokens_used_total",
            "Total tokens consumed",
            ["intent"],
        )

    def observe(
        self,
        status: str,
        intent: str,
        qos: str,
        latency_ms: float,
        tokens: int = 0,
    ) -> None:
        if not self._enabled:
            return
        self.tasks_total.labels(status=status, intent=intent, qos=qos).inc()
        self.task_latency_ms.labels(intent=intent, qos=qos).observe(latency_ms)
        if tokens:
            self.tokens_used_total.labels(intent=intent).inc(tokens)


def _get_metrics() -> _Metrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = _Metrics(_get_prom())
    return _global_metrics


class IicpNode:
    """IICP provider node — registration, heartbeats, and task serving.

    Example::

        async def my_handler(task: dict) -> dict:
            prompt = task["payload"]["messages"][-1]["content"]
            return {"result": {"content": f"Echo: {prompt}"}}

        node = IicpNode(NodeConfig(
            node_id="my-node-001",
            endpoint="https://my-host.example.com",
            intent="urn:iicp:intent:llm:chat:v1",
            max_concurrent=4,
        ))
        token = await node.register()
        await node.serve(my_handler, port=8020, node_token=token)
    """

    def __init__(self, config: NodeConfig) -> None:
        self._cfg = config
        self._http = httpx.AsyncClient(timeout=config.timeout)
        self._sem = threading.Semaphore(config.max_concurrent)
        self._active_jobs = 0
        self._jobs_lock = threading.Lock()
        self._nonces: dict[str, float] = {}
        self._nonces_lock = threading.Lock()
        self._metrics = _get_metrics()

    def apply_nat_profile(self, profile: Any) -> None:
        """Populate transport_endpoint + NAT observability fields from a
        :class:`iicp_client.nat_detection.NatProfile` produced by
        :func:`iicp_client.detect_nat`.

        Operators typically call this right after detect_nat() and before
        register() so the directory receives the discovered public endpoint
        + transport_method/nat_type/transport_metadata in the same payload.

        Idempotent: only overwrites fields the profile actually carries.
        """
        if getattr(profile, "is_reachable", lambda: False)():
            pub = getattr(profile, "public_endpoint", None)
            if pub:
                self._cfg.endpoint = pub
        tep = getattr(profile, "transport_endpoint", None)
        if tep:
            self._cfg.transport_endpoint = tep
        tm = getattr(profile, "transport_method", None)
        if tm and tm != "unreachable":
            self._cfg.transport_method = tm
        self._cfg.nat_type = self._cfg.nat_type or "unknown"
        # Surface a small dict of detection metadata so directory operators can
        # see what tier the SDK landed on without us shipping every detail.
        tier = getattr(profile, "tier", None)
        log = getattr(profile, "detection_log", []) or []
        if tier is not None:
            self._cfg.transport_metadata = {
                "tier": tier,
                "detection_log_tail": log[-1:] if log else [],
            }

    # ── Directory operations ──────────────────────────────────────────────

    async def register(self) -> str:
        """Register this node with the directory and return the node_token.

        Payload conforms to spec/iicp-dir.md §3.1 REGISTER (Phase 1+) plus
        the v0.7.0 dual-endpoint extension (`transport_endpoint`). Earlier
        iicp-client versions used a non-spec flat-`intent` shape that the
        production directory rejects with 422 — fixed in iter-1411.
        """
        # Build the spec-compliant capabilities array. Models defaults to
        # [config.model] when model is set, otherwise empty (directory will
        # reject; that's a configuration error the operator should fix).
        models = [self._cfg.model] if self._cfg.model else []
        if self._cfg.capabilities:
            # Legacy flat capabilities list — interpret each entry as an
            # additional model name for the same intent. Keeps existing
            # callers working without immediate API break.
            models = list({*models, *self._cfg.capabilities})

        payload: dict[str, Any] = {
            "endpoint": self._cfg.endpoint,
            "region": self._cfg.region or "eu-central",
            "capabilities": [{
                "intent": self._cfg.intent,
                "models": models,
                "max_tokens": self._cfg.max_tokens,
            }],
            "limits": {
                "max_concurrent": self._cfg.max_concurrent,
                "tokens_per_min": self._cfg.tokens_per_min,
            },
        }
        # node_id is optional — directory assigns one if absent. Send only when set.
        if self._cfg.node_id:
            payload["node_id"] = self._cfg.node_id
        # spec v0.7.0 — advertise native IICP binary endpoint if configured
        if self._cfg.transport_endpoint:
            payload["transport_endpoint"] = self._cfg.transport_endpoint
        # #331 / ADR-041 — surface NAT-traversal observability when populated
        # (typically via apply_nat_profile() after detect_nat())
        if self._cfg.transport_method:
            payload["transport_method"] = self._cfg.transport_method
        if self._cfg.nat_type:
            payload["nat_type"] = self._cfg.nat_type
        if self._cfg.transport_metadata:
            payload["transport_metadata"] = self._cfg.transport_metadata

        resp = await self._http.post(
            f"{self._cfg.directory_url.rstrip('/')}{_REGISTER_PATH}",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("node_token") or data.get("token")
        if not token:
            raise RuntimeError(f"Directory did not return node_token: {data}")
        logger.info("Registered node %s, token acquired", self._cfg.node_id)
        return str(token)

    async def heartbeat(self, node_token: str) -> None:
        """Send a single heartbeat to the directory."""
        resp = await self._http.post(
            f"{self._cfg.directory_url.rstrip('/')}{_HEARTBEAT_PATH}",
            json={
                "node_id": self._cfg.node_id,
                "node_token": node_token,
                "status": "available",
            },
        )
        resp.raise_for_status()

    async def _heartbeat_loop(self, node_token: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await self.heartbeat(node_token)
                logger.debug("Heartbeat sent for %s", self._cfg.node_id)
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)

    # ── Nonce replay protection ───────────────────────────────────────────

    def _check_nonce(self, nonce: str | None) -> bool:
        """Return True if nonce is fresh (first use within TTL window)."""
        if not nonce:
            return True
        now = time.monotonic()
        with self._nonces_lock:
            expired = [k for k, v in self._nonces.items() if v < now]
            for k in expired:
                del self._nonces[k]
            if nonce in self._nonces:
                return False
            self._nonces[nonce] = now + _NONCE_TTL
        return True

    # ── HTTP server ───────────────────────────────────────────────────────

    async def serve(
        self,
        handler: TaskHandler,
        host: str = "0.0.0.0",
        port: int = 8020,
        node_token: str | None = None,
    ) -> None:
        """Start the task server (blocks until interrupted).

        Args:
            handler:    ``async def handler(task: dict) -> dict``
            host:       Bind address (default ``0.0.0.0``).
            port:       Bind port (default 8020).
            node_token: If provided, starts a background heartbeat loop.
        """
        loop = asyncio.get_event_loop()
        node = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
                logger.debug(fmt, *args)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/iicp/health":
                    self._health()
                elif self.path == "/metrics":
                    self._prometheus()
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/v1/task":
                    self._task()
                else:
                    self.send_error(404)

            # ── GET /iicp/health ──────────────────────────────────────────

            def _health(self) -> None:
                with node._jobs_lock:
                    active = node._active_jobs
                denom = node._cfg.max_concurrent or 1
                body = json.dumps(
                    {
                        "status": "ok",
                        "node_id": node._cfg.node_id,
                        "region": node._cfg.region or "unknown",
                        "load": round(active / denom, 3),
                        "active_jobs": active,
                        "max_concurrent": node._cfg.max_concurrent,
                        "available": active < node._cfg.max_concurrent,
                        "model": node._cfg.model or "",
                        "intent": node._cfg.intent,
                    }
                ).encode()
                self._json_response(200, body)

            # ── GET /metrics ──────────────────────────────────────────────

            def _prometheus(self) -> None:
                prom = _get_prom()
                if prom is None:
                    body = b"prometheus_client not installed"
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = prom.generate_latest()
                self.send_response(200)
                self.send_header("Content-Type", prom.CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            # ── POST /v1/task ─────────────────────────────────────────────

            def _task(self) -> None:
                # Concurrency gate — IICP-E021
                if not node._sem.acquire(blocking=False):
                    err = json.dumps(
                        {
                            "error": {
                                "code": "IICP-E021",
                                "message": "capacity_exceeded",
                                "qos_class": None,
                                "retry_after_ms": 2000,
                            }
                        }
                    ).encode()
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "2")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                with node._jobs_lock:
                    node._active_jobs += 1
                t0 = time.monotonic()
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body: dict[str, Any] = (
                        json.loads(self.rfile.read(length)) if length else {}
                    )

                    # Nonce replay — IICP-E011
                    if not node._check_nonce(body.get("nonce")):
                        err = json.dumps(
                            {"error": {"code": "IICP-E011", "message": "replay_detected"}}
                        ).encode()
                        self.send_response(409)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                        return

                    # W3C traceparent propagation
                    traceparent = self.headers.get("traceparent")
                    if traceparent:
                        body.setdefault("_trace", {})["traceparent"] = traceparent

                    intent = body.get("intent") or node._cfg.intent
                    constraints = body.get("constraints") or {}
                    qos = (
                        constraints.get("qos_class", "best_effort")
                        if isinstance(constraints, dict)
                        else "best_effort"
                    )

                    try:
                        result = asyncio.run_coroutine_threadsafe(
                            handler(body), loop
                        ).result(timeout=60)
                        latency_ms = (time.monotonic() - t0) * 1000
                        usage = result.get("usage") or {}
                        tokens = (
                            usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                        )
                        node._metrics.observe("completed", intent, qos, latency_ms, tokens)
                        resp_body = json.dumps(
                            {
                                "task_id": body.get("task_id", ""),
                                "status": "completed",
                                **result,
                            }
                        ).encode()
                        self._json_response(200, resp_body)
                    except Exception as exc:
                        latency_ms = (time.monotonic() - t0) * 1000
                        node._metrics.observe("error", intent, qos, latency_ms)
                        logger.error("Handler error: %s", exc)
                        self.send_error(500, str(exc))
                finally:
                    node._sem.release()
                    with node._jobs_lock:
                        node._active_jobs -= 1

            # ── helpers ───────────────────────────────────────────────────

            def _json_response(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer((host, port), _Handler)
        logger.info("IICP node %s listening on %s:%d", self._cfg.node_id, host, port)

        bg_tasks: list[asyncio.Task] = []
        if node_token:
            bg_tasks.append(asyncio.create_task(self._heartbeat_loop(node_token)))

        try:
            await loop.run_in_executor(None, server.serve_forever)
        finally:
            server.shutdown()
            for t in bg_tasks:
                t.cancel()
            await self._http.aclose()

    async def __aenter__(self) -> IicpNode:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._http.aclose()
