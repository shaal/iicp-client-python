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
import hashlib
import hmac
import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from iicp_client.availability import AvailabilityEvaluator, Window
from iicp_client.idempotency import IdempotencyGuard
from iicp_client.iicp_tcp import IICP_MAGIC, IicpTcpServer  # #457 single-port multiplexer
from iicp_client.peer_manager import PeerManager
from iicp_client.scheduler import QUEUE_WAIT_S, is_queue_eligible

logger = logging.getLogger(__name__)

_EMBEDDING_INTENT = "urn:iicp:intent:llm:embedding:v1"


async def _post_cip_receipt(
    *,
    directory_url: str,
    token: str,
    hmac_key: str,
    node_id: str,
    task_id: str,
    tokens_used: int,
    result: dict[str, Any],
    querying_node_id: str | None = None,
) -> None:
    """TC-9c: best-effort CIPWorkerReceipt POST to /v1/credits/award.

    Server-side credit award path — the node reports task completion directly
    so the directory credits the provider wallet without proxy forwarding.
    Fire-and-forget: errors are suppressed so they never affect the task response.

    querying_node_id (#488): when set, the directory uses it to detect same-operator
    self-queries and returns excluded=true instead of awarding credits.
    """
    import secrets as _secrets
    from datetime import UTC, datetime, timedelta

    if not hmac_key or not token:
        return

    # response_hash = SHA-256 of canonical JSON of the result.
    result_bytes = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    response_hash = hashlib.sha256(result_bytes).hexdigest()

    nonce = _secrets.token_hex(16)
    expires_at = (datetime.now(UTC) + timedelta(seconds=300)).isoformat()

    # Canonical HMAC message (TC-9c §10.3): task:tokens:parent:session:nonce:hash[:qni]
    # #490 — querying_node_id appended when present to prevent spoofing; directory ≥ v1.10.25
    # verifies the extended canonical; older receipts omit it and use the short form.
    canonical_str = f"{task_id}:{tokens_used}:::{nonce}:{response_hash}"
    if querying_node_id:
        canonical_str += f":{querying_node_id}"
    canonical = canonical_str.encode()
    signature = hmac.new(hmac_key.encode(), canonical, hashlib.sha256).hexdigest()

    # amount = tokens / 1000.0; floor at 0.001 to satisfy directory min:0.0001.
    amount = max(tokens_used, 1) / 1000.0

    body: dict[str, Any] = {
        "node_id": node_id,
        "task_id": task_id,
        "tokens_used": tokens_used,
        "amount": round(amount, 4),
        "nonce": nonce,
        "expires_at": expires_at,
        "signature": signature,
        "response_hash": response_hash,
        "reason": "task_completion",
    }
    # #488: include querying_node_id for self-query neutrality detection at the directory.
    if querying_node_id:
        body["querying_node_id"] = querying_node_id

    url = directory_url.rstrip("/") + "/v1/credits/award"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
    except Exception:  # noqa: BLE001
        pass  # best-effort: never propagate to caller


def derive_native_endpoint(endpoint: str) -> str | None:
    """#457 / ADR-040 — derive the native binary transport_endpoint from the HTTP `endpoint`.

    They share one host:port (serve() multiplexes both planes on one socket via first-byte
    detection), so the native URI is the same authority with the ``iicp`` scheme (``iicpsec``
    for TLS). Returns None if `endpoint` is not a parseable http(s) URL.
    """
    parts = urlsplit(endpoint)
    if parts.scheme == "http" and parts.netloc:
        return f"iicp://{parts.netloc}"
    if parts.scheme == "https" and parts.netloc:
        return f"iicpsec://{parts.netloc}"
    return None


def _listen_family(host: str, port: int) -> int:
    """Resolve the socket address family to bind for ``host``.

    The CLI defaults ``--host``/``IICP_HOST`` to ``"::"`` (IPv6 any). Binding that
    to a hardcoded ``AF_INET`` socket raises ``gaierror: Address family for hostname
    not supported``, so derive the family from the host instead: ``"0.0.0.0"`` →
    ``AF_INET``, ``"::"`` → ``AF_INET6``, ``"127.0.0.1"`` → ``AF_INET``, hostnames →
    whatever they resolve to. ``AI_PASSIVE`` makes an empty/wildcard host resolve to
    a bindable wildcard address. Falls back to ``AF_INET`` if resolution yields nothing.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return socket.AF_INET
    return infos[0][0] if infos else socket.AF_INET


def _intent_for_model(model: str, default_intent: str) -> str:
    """#409 — classify a backend model to the IICP intent it serves.

    Embedding models (name contains "embed") advertise the embedding intent;
    every other model advertises the node's configured/default intent (chat).
    Conservative: only embeddings are split out — the verified real case.
    """
    return _EMBEDDING_INTENT if "embed" in model.lower() else default_intent


def _modalities_for_model(model: str) -> list[str]:
    """#408 / ADR-046 — input modalities a model accepts (B1: audio-in added).

    Vision-language models (name contains vl/vision/llava) accept images; "omni"
    models accept both image and audio; audio models (audio/voxtral/qwen*-audio)
    accept audio; else text-only. Each is a modality of chat, not a separate intent.
    The directory + spec accept text/image/audio/video in input_modalities (v0.10.0).
    """
    m = model.lower()
    has_image = "-vl-" in m or m.endswith("-vl") or "vision" in m or "llava" in m or "omni" in m
    has_audio = "audio" in m or "voxtral" in m or "omni" in m
    mods = ["text"]
    if has_image:
        mods.append("image")
    if has_audio:
        mods.append("audio")
    return mods


def _build_capabilities(models: list[str], default_intent: str, max_tokens: int) -> list[dict[str, Any]]:
    """#409 + #408 — group detected backend models into one capability object per
    (intent, input_modalities): advertise every intent the backend serves (chat +
    embedding) AND distinguish text-only vs image-capable (vision) chat. The
    directory accepts a multi-element capabilities array; clients pick the
    per-(intent,modality) model from discover. Back-compatible: a single text chat
    model yields the same single ["text"] capability. First-seen group leads.
    """
    if not models:
        return [{"intent": default_intent, "models": [], "max_tokens": max_tokens, "input_modalities": ["text"]}]
    order: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for m in models:
        intent = _intent_for_model(m, default_intent)
        modalities = _modalities_for_model(m)
        key = f"{intent}\0{','.join(modalities)}"
        if key not in groups:
            groups[key] = {"intent": intent, "models": [], "max_tokens": max_tokens, "input_modalities": modalities}
            order.append(key)
        if m not in groups[key]["models"]:
            groups[key]["models"].append(m)
    return [groups[key] for key in order]


_DEFAULT_TIMEOUT = 5.0
_HEARTBEAT_INTERVAL = 30
_NONCE_TTL = 300
_REGISTER_PATH = "/v1/register"
# Heartbeat path is /v1/heartbeat (NOT /api/v1/heartbeat) because the
# default directory_url already ends in /api. Previous double-/api/ bug
# made all heartbeats 404 — nodes registered fine but never updated
# `last_seen`, so they aged out of the 90s freshness window and the
# directory's stats endpoint always showed Active nodes: 0.
_HEARTBEAT_PATH = "/v1/heartbeat"

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
    # Detected backend server flavor advertised at register (node-detail field):
    # ollama / lmstudio / vllm / llamacpp / anthropic / custom.
    backend: str | None = None
    # #494 — backend base URL for live model health probing during heartbeat.
    # When set, the heartbeat includes health_models=<current runtime model list>
    # so the directory can filter stale-model nodes from discover results.
    # Empty = no probing (backward compat; health_models field omitted from heartbeat).
    backend_url: str = ""
    backend_api_key: str = ""
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
    # ADR-043 §9 (#344) — 8-category exposure_mode, derived via qualify_service in
    # apply_nat_profile; surfaced to the directory nodes.exposure_mode column.
    exposure_mode: str | None = None
    # S.12 §2.1 — CIP-D1 policy block surfaced to the directory's register
    # payload. When None, the SDK falls back to iicp_client.cip_policy.get_policy().
    # Operators with CIP-Provider mode enabled either pass a CooperativeInferencePolicy
    # here OR call cip_policy.configure_policy() before register().
    cip_policy: object | None = None
    # ADR-019 declarative pricing block surfaced to the directory at register.
    # When None, the SDK does not advertise pricing (directory defaults to 1.0
    # multiplier). When set with `sign_declarations=True` AND a node_hmac_key
    # is provisioned, the SDK signs the pricing body with HMAC-SHA256 so the
    # directory marks `pricing.attested=true` in /v1/discover.
    pricing: object | None = None
    # Operator-provisioned HMAC key for ADR-019 signing. If empty, the SDK
    # falls back to the key the directory returned on register (populated
    # by register() into IicpNode._node_hmac_key for subsequent calls).
    node_hmac_key: str = ""
    # ADR-045 Phase A (#407) — optional operator→node delegation token (built by
    # the operator wallet via delegation.issue_delegation for this node_id). When
    # set, register() attaches it; the directory verifies it offline and records
    # the verified operator identity. Key lifecycle (gen/store/backup) is the
    # wallet's concern (#307/ADR-030), kept out of the SDK transport layer.
    operator_delegation: dict | None = None
    # #463/#464 — operator-identity attributes advertised at register (only bound when the
    # delegation verifies). display_name is the public handle (node detail + leaderboard);
    # created_at + integrity_hash are identity-integrity. NEVER the operator's contact/email
    # or secret key (those stay local in the operator wallet).
    operator_display_name: str | None = None
    operator_created_at: str | None = None
    operator_integrity_hash: str | None = None
    # Phase 3+ availability windows (ADR-006 / spec/iicp-dir.md §register
    # `availability`). Each entry: {"start": "HH:MM", "end": "HH:MM", "share": 0.0-1.0}
    # in local time. Shapes the effective capacity advertised to the directory and
    # gated at serve time. None/empty → always full capacity. See availability.py.
    availability_windows: list[Window] | None = None
    # ADR-010 task_id idempotency. Off by default to preserve the pre-0.6 contract
    # (a task_id may be resubmitted). When True, a duplicate task_id within the
    # 5-minute window is rejected with IICP-E010. See idempotency.py.
    enable_idempotency: bool = False
    # Phase 2 mesh layer (ADR-009/ADR-022). When True, serve() starts the gossip
    # loop and exposes POST /v1/peers (HMAC peer exchange). See peer_manager.py.
    enable_mesh: bool = False
    # When True, serve() exposes POST /v1/relay to forward tasks to unreachable
    # peers learned via gossip (ADR-022). Requires enable_mesh.
    relay_capable: bool = False
    # Port for the RelayAcceptServer (R1 relay-as-last-resort, #341).
    # Workers behind CGNAT connect outbound to this port and send RELAY_BIND.
    relay_accept_port: int = 9485
    # R2: when set, this node acts as a RELAY WORKER — it will connect outbound
    # to the specified relay and advertise tasks through it (for CGNAT operators).
    # Format: "host:port" — e.g. "relay.example.com:9485".
    relay_worker_endpoint: str | None = None
    # Optional path to persist the peer list across restarts.
    peer_persist_path: str | None = None


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
        await node.serve(my_handler, port=9484, node_token=token)
    """

    def __init__(self, config: NodeConfig) -> None:
        self._cfg = config
        self._http = httpx.AsyncClient(timeout=config.timeout)
        self._sem = threading.Semaphore(config.max_concurrent)
        self._active_jobs = 0
        self._jobs_lock = threading.Lock()
        self._availability = AvailabilityEvaluator(config.availability_windows)
        self._idempotency = IdempotencyGuard()
        self._peer_manager = PeerManager(
            directory_url=config.directory_url,
            node_token=config.node_hmac_key,
            persist_path=(Path(config.peer_persist_path) if config.peer_persist_path else None),
            relay_capable=config.relay_capable,
            relay_accept_port=config.relay_accept_port,
        )
        self._nonces: dict[str, float] = {}
        self._nonces_lock = threading.Lock()
        self._metrics = _get_metrics()
        # Incremental task counters drained on each heartbeat for directory reporting.
        self._tasks_success = 0
        self._tasks_failed = 0
        self._task_counters_lock = threading.Lock()
        # R1 relay-as-last-resort (#341): session registry populated when
        # RelayAcceptServer is started by serve(). HTTP /v1/relay checks here
        # first before falling back to HTTP peer forwarding.
        from iicp_client.relay_session import RelaySessionRegistry

        self._relay_sessions = RelaySessionRegistry()
        # ADR-019: HMAC key for signing pricing declarations. Initialized from
        # NodeConfig.node_hmac_key; overwritten from the directory's response
        # on register() so subsequent re-registrations (after expiry) sign
        # with the directory-issued key.
        self._node_hmac_key: str = config.node_hmac_key
        # ADR-047 Part A (#411) — latest liveness nonce from the heartbeat response,
        # answered (HMAC) on the next beat. None until the first response.
        self._liveness_challenge: str | None = None
        # BUG-5: token stashed by register() so deregister()/heartbeat don't need it re-passed.
        self._node_token: str = ""
        # #343 — UPnP IPv6 pinhole tracking. Set by apply_nat_profile() when
        # detect_nat opened a firewall pinhole; consumed by _revoke_pinhole()
        # on graceful shutdown and the renewal loop.
        self._pinhole_uid: int | None = None
        self._pinhole_lease_seconds: int = 3600

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
        # ADR-043 §9 (#344) — derive the canonical 8-category exposure_mode.
        try:
            from iicp_client.qualify import qualify_service

            self._cfg.exposure_mode = qualify_service(profile).exposure_mode
        except Exception:
            pass  # best-effort; exposure_mode stays None if qualification can't run
        # #343 — capture the IPv6 firewall pinhole UID if one was opened so
        # serve()'s finally block can revoke it on shutdown.
        ipv6 = getattr(profile, "ipv6", None)
        if ipv6 is not None:
            if getattr(ipv6, "pinhole_active", False):
                uid = getattr(ipv6, "pinhole_unique_id", None)
                if isinstance(uid, int):
                    self._pinhole_uid = uid
                lease = getattr(ipv6, "pinhole_lease_seconds", None)
                if isinstance(lease, int) and lease > 0:
                    self._pinhole_lease_seconds = lease
            elif getattr(ipv6, "global_v6_available", False):
                # IPv6 is available but AddPinhole failed — advertised IPv6
                # endpoint may not be reachable if router firewall blocks inbound.
                # This is a common FRITZ!Box scenario (error 606 from router).
                pub = getattr(profile, "public_endpoint", "")
                if pub and pub.startswith("http://["):
                    logger.warning(
                        "NAT: IPv6 endpoint %s advertised but firewall pinhole "
                        "could not be opened (router rejected AddPinhole). "
                        "IPv4 clients cannot reach this node; IPv6 clients may work "
                        "if your router allows inbound TCP (FRITZ!Box: Network → "
                        "Firewall → IPv6). Alternatively use IICP_RELAY_WORKER_ENDPOINT "
                        "for relay-as-last-resort fallback.",
                        pub,
                    )
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
        for cap in self._cfg.capabilities:
            # Legacy flat capabilities list — additional model names (GAP-6
            # backend model probe). Preserve order (configured model leads).
            if cap not in models:
                models.append(cap)

        payload: dict[str, Any] = {
            "endpoint": self._cfg.endpoint,
            "region": self._cfg.region or "unknown",
            # #409 — one capability object per intent the backend can serve
            # (e.g. chat + embedding), classified from the detected model set.
            "capabilities": _build_capabilities(models, self._cfg.intent, self._cfg.max_tokens),
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
        if self._cfg.exposure_mode:
            payload["exposure_mode"] = self._cfg.exposure_mode
        if self._cfg.transport_metadata:
            payload["transport_metadata"] = self._cfg.transport_metadata
        # ADR-045 Phase A (#407) — attach the operator→node delegation when the
        # operator/wallet has issued one (built via delegation.issue_delegation for
        # this node_id). The directory verifies it offline and binds the operator.
        if self._cfg.operator_delegation:
            payload["operator_delegation"] = self._cfg.operator_delegation
            # #463/#464 — operator-identity attributes ride with the delegation (the directory
            # only binds them when the delegation verifies). Never contact/secret.
            if self._cfg.operator_display_name:
                payload["operator_display_name"] = self._cfg.operator_display_name
            if self._cfg.operator_created_at:
                payload["operator_created_at"] = self._cfg.operator_created_at
            if self._cfg.operator_integrity_hash:
                payload["operator_integrity_hash"] = self._cfg.operator_integrity_hash

        # SDK self-identification — directory surfaces these on /v1/discover
        # so dashboards can render a language badge. Free-form so future SDKs
        # in other languages can self-tag without a directory change.
        from iicp_client import __version__ as _iicp_client_version

        payload["sdk_language"] = "python"
        payload["sdk_version"] = _iicp_client_version
        if self._cfg.backend:
            payload["backend"] = self._cfg.backend
        if self._cfg.relay_capable:
            payload["relay_capable"] = True
            payload["relay_accept_port"] = self._cfg.relay_accept_port

        # S.12 §2.1 — CIP-D1 policy block. Use the per-config policy if set,
        # otherwise fall back to the module-level cip_policy.get_policy().
        from iicp_client.cip_policy import CooperativeInferencePolicy, get_policy

        policy_obj = self._cfg.cip_policy
        if policy_obj is None:
            policy_obj = get_policy()
        if isinstance(policy_obj, CooperativeInferencePolicy):
            block = policy_obj.as_register_policy_block()
            if block:
                payload["policy"] = block

        # ADR-019 — declarative pricing block. Operator-controlled; when
        # sign_declarations=True AND a HMAC key is present, sign the body so
        # the directory marks pricing.attested=true in /v1/discover.
        from iicp_client.pricing import PricingConfig, build_pricing_block

        if isinstance(self._cfg.pricing, PricingConfig):
            payload["pricing"] = build_pricing_block(self._cfg.pricing, hmac_key=self._node_hmac_key)
        # When the operator pre-provisions a node_hmac_key, surface it so the
        # directory's RegisterController uses it instead of generating one.
        if self._cfg.node_hmac_key:
            payload["node_hmac_key"] = self._cfg.node_hmac_key

        resp = await self._http.post(
            f"{self._cfg.directory_url.rstrip('/')}{_REGISTER_PATH}",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("node_token") or data.get("token")
        if not token:
            raise RuntimeError(f"Directory did not return node_token: {data}")
        # BUG-5: stash the token so deregister()/heartbeat work without the caller
        # re-passing it.
        self._node_token = str(token)
        # ADR-019: capture the directory-issued HMAC key for subsequent
        # pricing signatures. Operator-provisioned key wins when set.
        if not self._node_hmac_key:
            hk = data.get("node_hmac_key", "")
            if hk:
                self._node_hmac_key = str(hk)
        logger.info("Registered node %s, token acquired", self._cfg.node_id)
        return str(token)

    @property
    def node_hmac_key(self) -> str:
        """The HMAC key in use for ADR-019 pricing signatures (empty if
        unregistered AND no operator-provisioned key)."""
        return self._node_hmac_key

    async def heartbeat(self, node_token: str) -> None:
        """Send a single heartbeat to the directory.

        Requires `Authorization: Bearer <node_token>` because the
        directory's `/v1/heartbeat` route is guarded by NodeTokenAuth.
        The token also stays in the JSON body for back-compat with
        older directory builds that read it from the payload.
        """
        # Drain incremental task counters for directory reputation reporting.
        with self._task_counters_lock:
            ok = self._tasks_success
            fail = self._tasks_failed
            self._tasks_success = 0
            self._tasks_failed = 0
        payload: dict = {
            "node_id": self._cfg.node_id,
            "node_token": node_token,
            "status": "available",
            # Explicit availability boolean. The directory reads `available` (not the
            # `status` string) when deciding discover eligibility; sending it lets the
            # directory restore a node that briefly went dormant (host sleep) on the
            # very next beat — robust even against directory builds older than v1.10.17
            # whose heartbeat handler defaulted to the stored (possibly false) value.
            "available": True,
            "max_concurrent": self._availability.effective_max_concurrent(self._cfg.max_concurrent),
        }
        if ok > 0 or fail > 0:
            payload["metrics"] = {"tasks_success": ok, "tasks_failed": fail}
        # #494 — report live model list from backend so directory can filter stale-model
        # nodes from discover. Best-effort: if probe fails, omit health_models (backward compat).
        if self._cfg.backend_url:
            health_models = await self._probe_health_models()
            if health_models is not None:
                payload["health_models"] = health_models
        # ADR-047 Part A (#411) — answer the directory's liveness challenge from the
        # previous beat by HMAC-ing the nonce with node_hmac_key. Proves we control
        # the key (anti-replay) without any dial-back. No-op until we have both a
        # stored challenge and a key (back-compat with pre-challenge directories).
        if self._liveness_challenge and self._node_hmac_key:
            payload["challenge_response"] = hmac.new(
                self._node_hmac_key.encode(), self._liveness_challenge.encode(), hashlib.sha256
            ).hexdigest()
        resp = await self._http.post(
            f"{self._cfg.directory_url.rstrip('/')}{_HEARTBEAT_PATH}",
            headers={"Authorization": f"Bearer {node_token}"},
            json=payload,
        )
        resp.raise_for_status()
        # Capture the fresh nonce to answer on the next beat.
        try:
            self._liveness_challenge = resp.json().get("challenge") or self._liveness_challenge
        except Exception:  # noqa: BLE001
            pass

    async def _probe_health_models(self) -> list[str] | None:
        """Best-effort: return the backend's current model list for health_models reporting.

        Tries Ollama /api/tags then OpenAI /v1/models. Returns None on any error
        so the caller can silently omit health_models from the heartbeat payload
        (backward compat with directories that don't support the field yet).
        """
        base = self._cfg.backend_url.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        headers = {}
        if self._cfg.backend_api_key:
            headers["Authorization"] = f"Bearer {self._cfg.backend_api_key}"
        # Ollama /api/tags → {"models":[{"name":"..."},...]}
        try:
            resp = await self._http.get(f"{root}/api/tags", headers=headers, timeout=2.0)
            if resp.is_success:
                data = resp.json()
                names = sorted({m["name"] for m in data.get("models", []) if m.get("name")})
                return names
        except Exception:  # noqa: BLE001
            pass
        # OpenAI-compat /v1/models → {"data":[{"id":"..."},...]}
        try:
            resp = await self._http.get(f"{root}/v1/models", headers=headers, timeout=2.0)
            if resp.is_success:
                data = resp.json()
                return [m["id"] for m in data.get("data", []) if m.get("id")]
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _heartbeat_loop(self, node_token: str) -> None:
        token = node_token
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await self.heartbeat(token)
                logger.debug("Heartbeat sent for %s", self._cfg.node_id)
            except httpx.HTTPStatusError as exc:
                # #399 — the directory dropped the node (deregistered on a prior
                # shutdown, TTL-expired after a heartbeat gap, or the directory
                # restarted and forgot it). Re-register and resume with the fresh
                # token instead of heartbeating into the void forever.
                if exc.response.status_code in (401, 404, 410):
                    logger.warning(
                        "Heartbeat rejected (%s) — node unknown to directory; re-registering",
                        exc.response.status_code,
                    )
                    try:
                        token = await self.register()
                        logger.info("Re-registered %s after heartbeat rejection", self._cfg.node_id)
                    except Exception as reg_exc:  # noqa: BLE001
                        logger.warning("Re-registration failed: %s", reg_exc)
                else:
                    logger.warning("Heartbeat failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
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
        port: int = 9484,
        node_token: str | None = None,
    ) -> None:
        """Start the task server (blocks until interrupted).

        Args:
            handler:    ``async def handler(task: dict) -> dict``
            host:       Bind address (default ``0.0.0.0``).
            port:       Bind port (default 9484).
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
                elif self.path == "/v1/peers" and node._cfg.enable_mesh:
                    self._peers()
                elif self.path == "/v1/relay" and node._cfg.relay_capable:
                    self._relay()
                else:
                    self.send_error(404)

            # ── POST /v1/peers (ADR-009 gossip exchange) ──────────────────

            def _peers(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                sig = self.headers.get("X-IICP-Signature")
                if not node._peer_manager.verify_exchange(raw, sig):
                    self.send_error(401, "invalid peer signature")
                    return
                try:
                    incoming = json.loads(raw).get("known_peers", [])
                except (ValueError, json.JSONDecodeError):
                    self.send_error(400, "invalid JSON body")
                    return
                if isinstance(incoming, list):
                    # Entries may be ids (from gossip) or dicts; merge dict entries only.
                    node._peer_manager.merge_peers([p for p in incoming if isinstance(p, dict)])
                body = json.dumps({"peers": node._peer_manager.get_peers()}).encode()
                self._json_response(200, body)

            # ── POST /v1/relay (ADR-022 mesh relay) ───────────────────────

            def _relay(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length)) if length else {}
                except (ValueError, json.JSONDecodeError):
                    self.send_error(400, "invalid JSON body")
                    return
                target_id = payload.get("target_node_id", "")
                task = payload.get("task", {})
                if not target_id or not task:
                    self.send_error(422, "target_node_id and task are required")
                    return

                # R1: check relay session registry first (CGNAT workers with no
                # inbound endpoint bind here via RelayAcceptServer, not HTTP).
                relay_session = node._relay_sessions.get(target_id)
                if relay_session is not None:
                    try:
                        result = asyncio.run_coroutine_threadsafe(
                            relay_session.forward_task(task, timeout=120.0), loop
                        ).result(timeout=125)
                        resp_body = json.dumps(
                            {"task_id": task.get("task_id", ""), "status": "completed", **result}
                        ).encode()
                        self._json_response(200, resp_body)
                    except Exception as exc:  # noqa: BLE001
                        err = json.dumps(
                            {"error": {"code": "IICP-E031", "message": f"relay session forward failed: {exc}"}}
                        ).encode()
                        self._json_response(502, err)
                    return

                # Fall back to HTTP forwarding for routable peers (ADR-022).
                target = node._peer_manager.relay_target(target_id)
                if target is None:
                    _msg = "target not in peer list and not a bound relay worker"
                    err = json.dumps({"error": {"code": "IICP-E030", "message": _msg}}).encode()
                    self._json_response(404, err)
                    return
                try:
                    resp = httpx.post(f"{target['endpoint'].rstrip('/')}/v1/task", json=task, timeout=120.0)
                    self._json_response(resp.status_code, resp.content)
                except Exception as exc:  # noqa: BLE001
                    err = json.dumps({"error": {"code": "IICP-E031", "message": f"relay failed: {exc}"}}).encode()
                    self._json_response(502, err)

            # ── GET /iicp/health ──────────────────────────────────────────

            def _health(self) -> None:
                with node._jobs_lock:
                    active = node._active_jobs
                denom = node._cfg.max_concurrent or 1
                uid = node._pinhole_uid
                pinhole_state = (
                    {"active": True, "unique_id": uid, "lease_seconds": node._pinhole_lease_seconds}
                    if uid is not None
                    else {"active": False}
                )
                eff_max = node._availability.effective_max_concurrent(node._cfg.max_concurrent)
                body = json.dumps(
                    {
                        "status": "ok",
                        "node_id": node._cfg.node_id,
                        "region": node._cfg.region or "unknown",
                        "load": round(active / denom, 3),
                        "active_jobs": active,
                        "max_concurrent": node._cfg.max_concurrent,
                        "effective_max_concurrent": eff_max,
                        "available": active < eff_max,
                        "model": node._cfg.model or "",
                        "intent": node._cfg.intent,
                        "pinhole_state": pinhole_state,
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
                # Read the body first so QoS-aware admission can see
                # constraints.qos_class before deciding whether to wait for a slot.
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body: dict[str, Any] = json.loads(self.rfile.read(length)) if length else {}
                except (ValueError, json.JSONDecodeError):
                    self.send_error(400, "invalid JSON body")
                    return

                constraints = body.get("constraints") or {}
                qos = constraints.get("qos_class", "best_effort") if isinstance(constraints, dict) else "best_effort"

                # #403 — CIP per-task admission gate (parity with the adapter
                # cip_gate): reject tool-execution-domain intents unless the
                # operator opted in via cip_policy.allow_tool_execution.
                from iicp_client.cip_policy import CooperativeInferencePolicy, get_policy

                _pol = node._cfg.cip_policy
                if not isinstance(_pol, CooperativeInferencePolicy):
                    _pol = get_policy()
                _intent = body.get("intent", "")
                if isinstance(_intent, str) and _intent and not _pol.permits_intent(_intent):
                    err = json.dumps(
                        {
                            "error": {
                                "code": "tool_execution_denied",
                                "message": "Tool-execution intents are not permitted by this node's CIP policy",
                            }
                        }
                    ).encode()
                    self.send_response(403)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                # Availability gate (ADR-006) — reduced-capacity windows cap admissions
                # below max_concurrent. This is a deliberate operator policy, so it
                # rejects immediately (no QoS wait) when the window is full/closed.
                eff_max = node._availability.effective_max_concurrent(node._cfg.max_concurrent)
                with node._jobs_lock:
                    at_window_cap = node._active_jobs >= eff_max
                if at_window_cap:
                    err = json.dumps(
                        {
                            "error": {
                                "code": "IICP-E021",
                                "message": "capacity_exceeded",
                                "qos_class": qos,
                                "reason": "availability_window",
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

                # QoS-aware admission — IICP-E021. realtime/interactive wait briefly
                # for a slot; batch/best-effort/unspecified fail fast so the proxy
                # sees back-pressure immediately (ADR-006; see scheduler.py).
                if is_queue_eligible(qos):
                    acquired = node._sem.acquire(blocking=True, timeout=QUEUE_WAIT_S)
                else:
                    acquired = node._sem.acquire(blocking=False)
                if not acquired:
                    err = json.dumps(
                        {
                            "error": {
                                "code": "IICP-E021",
                                "message": "capacity_exceeded",
                                "qos_class": qos,
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
                    # Nonce replay — IICP-E011
                    if not node._check_nonce(body.get("nonce")):
                        err = json.dumps({"error": {"code": "IICP-E011", "message": "replay_detected"}}).encode()
                        self.send_response(409)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                        return

                    # Idempotency — duplicate task_id within the retry window (ADR-010).
                    # Distinct from nonce replay: dedups a retried CALL of the same task.
                    # Opt-in (NodeConfig.enable_idempotency) to preserve pre-0.6 behaviour.
                    if node._cfg.enable_idempotency and not node._idempotency.check_and_register(body.get("task_id")):
                        err = json.dumps({"error": {"code": "IICP-E010", "message": "duplicate_task"}}).encode()
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

                    task_id = body.get("task_id", "")
                    intent = body.get("intent") or node._cfg.intent

                    from iicp_client.otel_tracer import task_execute_span, task_validate_span

                    with task_validate_span(task_id):
                        pass  # nonce check already completed above; span marks validation done

                    try:
                        # BUG-3 guard: reject incoming tasks if the event loop is
                        # already closed (interpreter shutdown race).
                        if loop.is_closed():
                            self.send_error(503, "node shutting down")
                            return
                        with task_execute_span(task_id, intent):
                            result = asyncio.run_coroutine_threadsafe(handler(body), loop).result(timeout=60)
                        latency_ms = (time.monotonic() - t0) * 1000
                        usage = result.get("usage") or {}
                        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                        node._metrics.observe("completed", intent, qos, latency_ms, tokens)
                        with node._task_counters_lock:
                            node._tasks_success += 1
                        resp_body = json.dumps(
                            {
                                "task_id": task_id,
                                "status": "completed",
                                **result,
                            }
                        ).encode()
                        self._json_response(200, resp_body)
                        # TC-9c: fire best-effort CIPWorkerReceipt to the directory.
                        # Server-side award path: provider reports completion so the
                        # directory credits the wallet without proxy forwarding.
                        if node._node_hmac_key and node._node_token:
                            asyncio.run_coroutine_threadsafe(
                                _post_cip_receipt(
                                    directory_url=node._cfg.directory_url,
                                    token=node._node_token,
                                    hmac_key=node._node_hmac_key,
                                    node_id=node._cfg.node_id,
                                    task_id=task_id,
                                    tokens_used=tokens,
                                    result=result,
                                    # #488: pass requester identity for self-query detection.
                                    querying_node_id=body.get("source_node_id"),
                                ),
                                loop,
                            )
                    except RuntimeError as exc:
                        if "shutdown" in str(exc).lower() or "closed" in str(exc).lower():
                            logger.debug("Handler skipped during node shutdown: %s", exc)
                            self.send_error(503, "node shutting down")
                            return
                        raise
                    except Exception as exc:
                        latency_ms = (time.monotonic() - t0) * 1000
                        node._metrics.observe("error", intent, qos, latency_ms)
                        with node._task_counters_lock:
                            node._tasks_failed += 1
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

        # #457 / ADR-040 — single-port multiplexer: the HTTP control plane and the native
        # IICP binary transport share ONE socket. Each accepted connection's first 4 bytes
        # are peeked (MSG_PEEK, non-consuming): the IICP frame magic "IICP" routes to the
        # native handler (the SAME backend task handler as HTTP), anything else (an HTTP
        # request line) to the BaseHTTPRequestHandler above. One socket ⇒ one pinhole ⇒
        # native is reachable exactly when HTTP is (advertise-when-reachable); a CGNAT node
        # needs no second hole. bind_and_activate=False: we own the listening socket.
        server = ThreadingHTTPServer((host, port), _Handler, bind_and_activate=False)
        native_server = IicpTcpServer(
            host=host, port=port, node_id=self._cfg.node_id, handler=handler
        )
        # Bind to the address family implied by `host` — the CLI defaults host to
        # "::" (IPv6), which a hardcoded AF_INET socket cannot bind (gaierror).
        family = _listen_family(host, port)
        # Keep the HTTP server's notion of the family consistent with the socket we
        # own; server_close()/process_request never re-bind, but this avoids any
        # AF_INET-vs-IPv6 mismatch if the stdlib ever consults address_family.
        server.address_family = family
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            # Dual-stack: a "::" bind should also accept IPv4-mapped clients.
            # Best-effort — some platforms reject toggling V6ONLY.
            try:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        listener.bind((host, port))
        listener.listen(128)
        listener.settimeout(0.5)  # so the accept loop notices shutdown promptly
        mux_stop = threading.Event()

        async def _handle_native_conn(conn: socket.socket) -> None:
            try:
                conn.setblocking(False)
                reader, writer = await asyncio.open_connection(sock=conn)
                await native_server._handle_connection(reader, writer)
            except Exception as exc:  # noqa: BLE001
                logger.debug("native IICP connection error: %s", exc)
                try:
                    conn.close()
                except OSError:
                    pass

        def _route_conn(conn: socket.socket, addr: Any) -> None:
            try:
                conn.settimeout(10.0)
                # Wait for the full 4-byte prefix without consuming it; the chosen consumer
                # then parses from the start. MSG_WAITALL avoids misrouting on a fragmented magic.
                prefix = conn.recv(4, socket.MSG_PEEK | socket.MSG_WAITALL)
            except OSError:
                try:
                    conn.close()
                except OSError:
                    pass
                return
            conn.settimeout(None)
            if prefix == IICP_MAGIC:
                asyncio.run_coroutine_threadsafe(_handle_native_conn(conn), loop)
            else:
                # ThreadingHTTPServer.process_request threads the request; _Handler reads the
                # connection from the start (MSG_PEEK left the bytes in the kernel buffer).
                server.process_request(conn, addr)

        def _accept_loop() -> None:
            while not mux_stop.is_set():
                try:
                    conn, addr = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                # Peek+route off the accept thread so a slow client can't block new connections.
                threading.Thread(target=_route_conn, args=(conn, addr), daemon=True).start()

        logger.info(
            "IICP node %s listening on %s:%d (HTTP + native IICP, single port)",
            self._cfg.node_id,
            host,
            port,
        )

        bg_tasks: list[asyncio.Task] = []
        # #404 — start the heartbeat loop when a token is present OR empty (register
        # failed → loop self-heals via re-register on 401). None = --skip-registration.
        if node_token is not None:
            bg_tasks.append(asyncio.create_task(self._heartbeat_loop(node_token)))
        if self._pinhole_uid is not None:
            bg_tasks.append(asyncio.create_task(self._pinhole_renewal_loop()))
        if self._cfg.enable_mesh:
            # Phase 2 mesh: bootstrap from the directory then gossip every 30s.
            await self._peer_manager.start(self._cfg.node_id, own_endpoint=self._cfg.endpoint)
            bg_tasks.append(asyncio.create_task(self._peer_manager.gossip_loop()))
        # R1: start RelayAcceptServer when this node is relay-capable (#341).
        # Workers behind CGNAT connect here to bind outbound relay sessions.
        relay_accept_srv = None
        if self._cfg.relay_capable:
            from iicp_client.relay_session import RelayAcceptServer

            relay_port = getattr(self._cfg, "relay_accept_port", 9485)
            relay_accept_srv = RelayAcceptServer(self._relay_sessions, host=host, port=relay_port)
            try:
                await relay_accept_srv.start()
                bg_tasks.append(
                    asyncio.create_task(
                        relay_accept_srv._server.serve_forever()  # type: ignore[union-attr]
                    )
                )
                logger.info("Relay accept server started on %s:%d", host, relay_port)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Relay accept server failed to start: %s — relay sessions disabled", exc)

        # R2: if relay_worker_endpoint is configured, connect outbound to a relay.
        # This node acts as a relay worker — its tasks are routed through the relay
        # for operators behind CGNAT who can't receive inbound connections.
        if self._cfg.relay_worker_endpoint:
            relay_worker_ep = self._cfg.relay_worker_endpoint
            _relay_host, _, _relay_port_str = relay_worker_ep.rpartition(":")
            _relay_host = _relay_host or relay_worker_ep
            _relay_port_n = int(_relay_port_str) if _relay_port_str.isdigit() else 9485
            _node_ref = self  # capture for on_bind callback
            _current_token: list[str | None] = [node_token]

            async def _on_relay_bind(rhost: str, rport: int, worker_id: str) -> None:
                """Re-register with the directory advertising the relay as our endpoint.

                After a successful RELAY_BIND the relay becomes our public endpoint.
                We deregister the old (private) endpoint and register a new one with
                transport_method='turn_relay' and endpoint=<relay_host>:<relay_port>.
                This makes the node appear ACTIVE in directory + stats (#358).
                """
                new_endpoint = f"http://{rhost}:{rport}"
                # Update config so register() uses the relay endpoint
                _node_ref._cfg.endpoint = new_endpoint
                _node_ref._cfg.transport_method = "turn_relay"
                _node_ref._cfg.transport_metadata = {
                    "relay_for": worker_id,
                    "relay_host": rhost,
                    "relay_port": rport,
                }
                # Deregister old token, then register with relay endpoint
                old_token = _current_token[0]
                if old_token:
                    try:
                        await _node_ref.deregister(old_token)
                        logger.info("Relay worker: deregistered old endpoint")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Relay worker: deregister failed: %s", exc)
                try:
                    new_token = await _node_ref.register()
                    _current_token[0] = new_token
                    logger.info(
                        "Relay worker: re-registered with relay endpoint %s (token=%s…)",
                        new_endpoint,
                        (new_token or "")[:8],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Relay worker: re-registration failed: %s", exc)

            from iicp_client.relay_worker_client import RelayWorkerClient

            relay_worker = RelayWorkerClient(
                worker_id=self._cfg.node_id,
                intent=self._cfg.intent,
                relay_host=_relay_host,
                relay_port=_relay_port_n,
                task_handler=handler,
                models=[self._cfg.model] if self._cfg.model else [],
                on_bind=_on_relay_bind,
            )
            bg_tasks.append(asyncio.create_task(relay_worker.run()))
            logger.info("Relay worker started → %s:%d", _relay_host, _relay_port_n)

        try:
            # #457 — run the single-port accept/route loop (replaces server.serve_forever;
            # the HTTP server never binds its own socket — we feed it routed connections).
            await loop.run_in_executor(None, _accept_loop)
        finally:
            # BUG-3 fix: cancel background tasks BEFORE teardown so the gossip/heartbeat
            # coroutines stop scheduling futures onto the event loop during interpreter
            # teardown — silences the "cannot schedule new futures after interpreter
            # shutdown" noise on CTRL-C / normal exit.
            for t in bg_tasks:
                t.cancel()
            if bg_tasks:
                await asyncio.gather(*bg_tasks, return_exceptions=True)
            # #457 — stop the multiplexer accept loop + close the listening socket.
            mux_stop.set()
            try:
                listener.close()
            except OSError:
                pass
            server.server_close()
            # #343 — graceful pinhole revoke. Best-effort; failure here is
            # never fatal because router lease auto-expires (3600s default).
            self._revoke_pinhole_sync()
            # Notify directory we're going away so it can flip our entry to
            # dormant + free up the rate-limit slot the operator's IP holds.
            if node_token:
                try:
                    await self.deregister(node_token)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("deregister on shutdown failed: %s", exc)
            await self._http.aclose()

    async def deregister(self, node_token: str | None = None) -> None:
        """Notify the directory this node is shutting down (DELETE /v1/register).

        `node_token` defaults to the token stashed by register() (BUG-5) so callers
        can simply `await node.deregister()`. Pass an explicit token to override.

        Returns silently on success; logs + raises on transport / 4xx errors so
        callers can surface them. Bearer-authed.

        Directory side: marks node status='dormant' + cascades a DEREGISTER
        event to replicas (S.13 §5.1 federated event log).
        """
        token = node_token or self._node_token
        if not token:
            raise RuntimeError("deregister() requires a node_token (none stashed — call register() first)")
        url = self._cfg.directory_url.rstrip("/") + _REGISTER_PATH
        resp = await self._http.request(
            "DELETE",
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"node_id": self._cfg.node_id},
        )
        resp.raise_for_status()
        logger.info("deregistered node %s", self._cfg.node_id)

    async def _pinhole_renewal_loop(self) -> None:
        """Background task: renew the UPnP IPv6 pinhole at lease/2 intervals (#343).

        Fires when serve() detects a tracked pinhole. Best-effort: any renewal
        failure is logged and the loop continues — the IGD lease will eventually
        expire, but we keep trying so brief IGD hiccups don't kill the session.
        """
        from iicp_client.nat_detection import renew_ipv6_pinhole

        while True:
            delay = max(self._pinhole_lease_seconds // 2, 60)
            await asyncio.sleep(delay)
            uid = self._pinhole_uid
            if uid is None:
                return
            ok = await asyncio.get_event_loop().run_in_executor(
                None, renew_ipv6_pinhole, uid, self._pinhole_lease_seconds
            )
            if ok:
                logger.debug(
                    "UPnP IPv6 pinhole uid=%s renewed (lease=%ss)",
                    uid,
                    self._pinhole_lease_seconds,
                )
            else:
                logger.warning(
                    "UPnP IPv6 pinhole uid=%s renewal failed — will retry at next interval",
                    uid,
                )

    def _revoke_pinhole_sync(self) -> None:
        """Close the UPnP IPv6 firewall pinhole if one is tracked (#343).

        Runs synchronously inside serve()'s finally block. Best-effort: any
        failure (router unreachable, UPnP service flapped, etc.) is logged
        and ignored. Leases auto-expire after pinhole_lease_seconds so
        nothing is left "permanently open" even if revoke fails.
        """
        uid = self._pinhole_uid
        if uid is None:
            return
        try:
            from iicp_client.nat_detection import delete_ipv6_pinhole

            ok = delete_ipv6_pinhole(uid)
            if ok:
                logger.info("UPnP IPv6 pinhole uid=%s closed cleanly", uid)
            else:
                logger.info(
                    "UPnP IPv6 pinhole uid=%s revoke attempted but no IGD responded — router lease will auto-expire",
                    uid,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pinhole revoke (uid=%s) failed: %s", uid, exc)
        finally:
            self._pinhole_uid = None

    async def __aenter__(self) -> IicpNode:
        return self

    async def __aexit__(self, *_: Any) -> None:
        # Belt-and-braces: cleanup pinholes here too in case the operator
        # used the context-manager pattern outside of serve().
        self._revoke_pinhole_sync()
        await self._http.aclose()
