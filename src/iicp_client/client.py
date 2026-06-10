"""IicpClient — primary entrypoint for the IICP Python SDK (ADR-016 §1)."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import random
import re
import time
import uuid
from typing import Any

import httpx
from urllib.parse import urlparse

from iicp_client._http import _traceparent, get_json, post_json
from iicp_client.errors import IicpError
from iicp_client.types import (
    ChatChoice,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatUsage,
    ClientConfig,
    DiscoverOptions,
    Node,
    NodeList,
    TaskAuth,
    TaskConstraints,
    TaskMetrics,
    TaskRequest,
    TaskResponse,
)

_INTENT_RE = re.compile(r"^urn:iicp:intent:[a-z0-9_:/-]+$")
_MAX_TIMEOUT_MS = 120_000


def _is_ssrf_safe(url: str) -> bool:
    """Return True if url is safe to connect to as a node endpoint (SSRF guard, #388)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host or host in {"localhost", "0.0.0.0", "::1", "::"}:
        return False
    if any(host.endswith(s) for s in (".local", ".internal", ".lan", ".test", ".invalid", ".localhost")):
        return False
    try:
        addr = ipaddress.ip_address(host)
        # IP address (IPv4 or IPv6): safe unless private/loopback/link-local/reserved
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return False
        return True  # global unicast IPv4/IPv6 — safe
    except ValueError:
        pass
    # Hostname: require at least one dot to block bare Docker service names
    if "." not in host:
        return False
    return True


class IicpClient:
    """Discover → select → submit client for the IICP protocol.

    Implements ADR-016 §1 (SDK-01..SDK-06 conformance rules).
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        self._cfg = config or ClientConfig()
        if self._cfg.timeout_ms > _MAX_TIMEOUT_MS:
            # SDK-04: reject oversized timeouts at construction time
            raise ValueError(f"timeout_ms must be ≤ {_MAX_TIMEOUT_MS}; got {self._cfg.timeout_ms}")
        # IICP_ROUTING_EPSILON overrides config; clamp to [0.0, 1.0]
        _env_eps = os.environ.get("IICP_ROUTING_EPSILON")
        if _env_eps is not None:
            try:
                self._cfg.routing_epsilon = max(0.0, min(1.0, float(_env_eps)))
            except ValueError:
                pass
        # Phase 2 (#496): consumer token cache — (target_node_id, intent) → (token, exp_unix)
        self._ct_cache: dict[tuple[str, str], tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Phase 2 consumer token acquisition (#496)
    # ------------------------------------------------------------------

    async def _acquire_consumer_token(
        self, target_node_id: str, intent: str, timeout_s: float = 5.0
    ) -> str | None:
        node_token = self._cfg.node_token
        if not node_token:
            return None
        cache_key = (target_node_id, intent)
        cached = self._ct_cache.get(cache_key)
        if cached:
            tok, exp = cached
            if time.time() + 30 < exp:
                return tok
        base = self._cfg.directory_url.rstrip("/api").rstrip("/")
        url = f"{base}/api/v1/consumer-token"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(
                    url,
                    json={"target_node_id": target_node_id, "intent": intent},
                    headers={"Authorization": f"Bearer {node_token}"},
                )
                if r.status_code == 201:
                    data = r.json()
                    token: str = data.get("token", "")
                    exp_unix: int = int(data.get("expires_at", 0))
                    if token:
                        self._ct_cache[cache_key] = (token, exp_unix)
                        return token
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def discover_async(
        self,
        intent: str,
        options: DiscoverOptions | None = None,
        *,
        traceparent: str | None = None,
    ) -> NodeList:
        """Discover nodes capable of handling *intent*."""
        opts = options or DiscoverOptions()
        params: dict[str, Any] = {"limit": min(opts.limit, 50)}
        params["intent"] = intent
        if opts.region or self._cfg.region:
            params["region"] = opts.region or self._cfg.region
        if opts.qos:
            params["qos"] = opts.qos
        if opts.min_reputation is not None:
            params["min_reputation"] = opts.min_reputation
        if opts.model:
            params["model"] = opts.model

        import time

        t0 = time.monotonic()
        data = await get_json(
            f"{self._cfg.directory_url}/v1/discover",
            params=params,
            timeout_ms=5_000,
            component="directory",
            tls_verify=self._cfg.tls_verify,
            traceparent=traceparent,
        )
        elapsed = int((time.monotonic() - t0) * 1000)

        raw_nodes = data.get("nodes", [])
        nodes = []
        for n in raw_nodes:
            endpoint = n["endpoint"]
            if not _is_ssrf_safe(endpoint):
                logging.getLogger(__name__).warning(
                    "SDK: skipping node %s — endpoint %s is not publicly routable (SSRF guard)",
                    n.get("node_id", "?")[:8],
                    endpoint,
                )
                continue
            cx_key = n.get("cx_public_key")
            nodes.append(Node(
                node_id=n["node_id"],
                endpoint=endpoint,
                score=float(n.get("score", 0.0)),
                available=bool(n.get("available", True)),
                region=n.get("region", ""),
                latency_estimate_ms=n.get("latency_estimate_ms"),
                reputation_score=n.get("reputation_score"),
                health_label=n.get("health_label"),
                exposure_mode=n.get("exposure_mode"),
                cx_public_key=cx_key if isinstance(cx_key, dict) else None,
                transport=n.get("transport") if isinstance(n.get("transport"), list) else None,
            ))
        return NodeList(nodes=nodes, query_ms=elapsed)

    async def submit_async(self, request: TaskRequest) -> TaskResponse:
        """Discover → select best node → submit task.

        Retries up to max_retries on transient errors (SDK-01).
        A single W3C traceparent is generated per submit call and propagated
        to both the discover request and the node POST (SDK-06).
        """
        self._validate_intent(request.intent)
        tp = _traceparent()  # SDK-06: one trace per operation, shared across calls
        node_list = await self.discover_async(
            request.intent,
            DiscoverOptions(
                region=request.constraints.region or self._cfg.region,
                # Do not filter by qos — qos is a task execution hint, not a
                # node capability filter. Most nodes don't declare qos support
                # in registration, so filtering by qos=interactive returns 0.
            ),
            traceparent=tp,
        )
        if not node_list.nodes:
            raise IicpError(
                code="IICP-E006",
                message=f"No nodes available for intent {request.intent!r}",
                component="directory",
                retryable=True,
            )

        task_id = str(uuid.uuid4())
        # ε-greedy provider selection (R4): with probability ε pick a random node
        # from the full discovered set; otherwise use the directory-sorted top pick.
        # Remaining fallback candidates always come from the directory-sorted list.
        all_nodes = node_list.nodes
        top_n = max(1, self._cfg.max_retries)
        if len(all_nodes) > 1 and random.random() < self._cfg.routing_epsilon:
            explore_node = random.choice(all_nodes)
            rest = [n for n in all_nodes[:top_n] if n.node_id != explore_node.node_id][: top_n - 1]
            candidates = [explore_node] + rest
        else:
            candidates = all_nodes[:top_n]
        last_exc: IicpError | None = None

        for node in candidates:
            body: dict[str, Any] = {
                "task_id": task_id,
                "intent": request.intent,
                "constraints": {
                    "timeout_ms": request.constraints.timeout_ms,
                    "qos": request.constraints.qos,
                },
            }
            if request.auth.node_token:
                body["auth"] = {"node_token": request.auth.node_token}
            # #488: include requester identity for self-query neutrality at the directory.
            if request.source_node_id:
                body["source_node_id"] = request.source_node_id

            # IICP-CX S.16 §5: encrypt payload when use_confidentiality=True and node advertises a key
            if self._cfg.use_confidentiality and node.cx_public_key:
                from iicp_client._confidentiality import encrypt_payload
                body["iicp_conf"] = encrypt_payload(request.payload, node.cx_public_key, task_id, request.intent)
            else:
                body["payload"] = request.payload

            # Phase 2 (#496): acquire consumer token if configured
            node_headers: dict[str, str] = {}
            ct = await self._acquire_consumer_token(node.node_id, request.intent)
            if ct:
                node_headers["X-IICP-Consumer-Token"] = ct

            node_connected = True
            for attempt in range(self._cfg.max_retries):
                try:
                    raw, elapsed = await post_json(
                        f"{node.endpoint}/v1/task",
                        body,
                        timeout_ms=request.constraints.timeout_ms,
                        component="adapter",
                        tls_verify=self._cfg.tls_verify,
                        traceparent=tp,
                        extra_headers=node_headers or None,
                    )
                    return TaskResponse(
                        task_id=raw.get("task_id", task_id),
                        status=raw.get("status", "success"),
                        result=raw.get("result"),
                        metrics=TaskMetrics(
                            latency_ms=elapsed,
                            tokens_used=raw.get("usage", {}).get("total_tokens"),
                            node_id=node.node_id,
                        ),
                    )
                except IicpError as exc:
                    last_exc = exc
                    if not exc.retryable:
                        raise  # hard auth/validation failure — don't retry or fallback
                    # Network/connection error → skip to next node immediately
                    if exc.code in ("IICP-E003", "IICP-E004"):
                        node_connected = False
                        break
                    # Server-side 5xx — retry same node with backoff
                    if attempt < self._cfg.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        break  # exhausted retries on this node → try next
            if not node_connected:
                continue  # this node was unreachable, try next

        raise last_exc  # type: ignore[misc]

    async def chat_async(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        """OpenAI-compatible chat over urn:iicp:intent:llm:chat:v1 (SDK-02)."""
        opts = options or ChatOptions()
        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if opts.model:
            payload["model"] = opts.model
        if opts.max_tokens is not None:
            payload["max_tokens"] = opts.max_tokens
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature

        response = await self.submit_async(
            TaskRequest(
                intent="urn:iicp:intent:llm:chat:v1",
                payload=payload,
                constraints=TaskConstraints(
                    timeout_ms=opts.timeout_ms or self._cfg.timeout_ms,
                    qos=opts.qos,
                ),
                auth=TaskAuth(node_token=opts.node_token),
            )
        )

        result = response.result or {}
        raw_choices = result.get("choices", [])
        choices = [
            ChatChoice(
                message=ChatMessage(
                    role=c.get("message", {}).get("role", "assistant"),
                    content=c.get("message", {}).get("content", ""),
                ),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in raw_choices
        ]
        raw_usage = result.get("usage", {})
        return ChatResponse(
            id=response.task_id,
            choices=choices,
            usage=ChatUsage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            ),
            model=result.get("model", opts.model or ""),
            iicp_node_id=response.metrics.node_id,
        )

    # ------------------------------------------------------------------
    # Sync wrappers (runs asyncio.run internally)
    # ------------------------------------------------------------------

    def discover(self, intent: str, options: DiscoverOptions | None = None) -> NodeList:
        return asyncio.run(self.discover_async(intent, options))

    def submit(self, request: TaskRequest) -> TaskResponse:
        return asyncio.run(self.submit_async(request))

    def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> ChatResponse:
        return asyncio.run(self.chat_async(messages, options))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_intent(self, intent: str) -> None:
        # SDK-03: validate URN format before sending
        if not _INTENT_RE.match(intent):
            raise IicpError(
                code="IICP-E001",
                message=f"Invalid intent URN: {intent!r}. Must match urn:iicp:intent:*",
                component="proxy",
                retryable=False,
            )
