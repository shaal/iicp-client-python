# SPDX-License-Identifier: Apache-2.0
"""IICP Routing Proxy — Client Plane entry point (ADR-001, ADR-005).

This is the FastAPI application that constitutes the Client Plane. It bridges
client requests into IICP CALL messages and routes them to the best available
adapter node discovered through the directory.

Three inbound protocol surfaces are exposed on port 9483 (reserved IICP proxy band):
  - OpenAI-compat:   POST /v1/chat/completions (ChatGPT SDK, LangChain, LlamaIndex, liteLLM)
  - Ollama-compat:   POST /api/chat, POST /api/generate (Open WebUI, Continue.dev, aider, Jan)
  - Anthropic-compat: POST /v1/messages (Anthropic SDK with base_url override)

All three surfaces translate to the same IICP CALL message and route through the same
FallbackChain → CircuitBreaker → RetryManager → NodeClient pipeline.

Key responsibilities:
  1. Expose the three inbound protocol surfaces above (all translate to IICP task shape).
  2. Discover candidate nodes via DirectoryClient → GET /v1/discover (directory trust; proxy
     does NOT re-rank — ADR-008 hard rule: score order preserved from directory).
  3. Select available nodes via NodeSelector (filter-only: removes available=false nodes).
  4. Execute task via FallbackChain → CircuitBreaker → RetryManager → NodeClient.
  5. Wire CIP coordinator/consumer for Phase 5 multi-node cooperative inference.

The proxy binds to 127.0.0.1 only (loopback) and is never exposed to the internet
directly. TLS certificate validation is not bypassable (no verify=False anywhere).

Spec: spec/iicp-core.md §3 (CALL/RESPONSE), spec/iicp-semantics.md (routing).
ADRs: ADR-001 (Client Plane), ADR-005 (Python/FastAPI), ADR-008 (scoring — directory-side),
      ADR-010 (idempotency — adapter enforces; proxy sends same task_id on retry).
Issues: #278 (Ollama-compat), #279 (Anthropic-compat).
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from iicp_client.proxy.address_state import get_address_state
from iicp_client.proxy.anthropic_compat.server import add_anthropic_routes
from iicp_client.proxy.auth.secrets import MissingNodeTokenError, load_node_token
from iicp_client.proxy.clients.directory import DirectoryClient, check_observed_ip_vs_endpoint
from iicp_client.proxy.config import ProxyConfig
from iicp_client.proxy.metrics import metrics_output
from iicp_client.proxy.network import PeerCache
from iicp_client.proxy.ollama_compat.server import add_ollama_routes
from iicp_client.proxy.openai_compat.server import create_compat_app
from iicp_client.proxy.routing.aggregator import ResultAggregator
from iicp_client.proxy.routing.circuit_breaker import CircuitBreaker
from iicp_client.proxy.routing.fallback import FallbackChain
from iicp_client.proxy.routing.retry import RetryManager
from iicp_client.proxy.routing.router import TaskRouter
from iicp_client.proxy.routing.selector import NodeSelector

logger = logging.getLogger(__name__)


def create_app(cfg: ProxyConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            node_token = load_node_token(cfg.node_token_env)
        except MissingNodeTokenError:
            logger.warning("IICP_NODE_TOKEN not set — using empty token (dev mode)")
            node_token = ""

        retry = RetryManager(max_retries=cfg.max_retries, base_ms=cfg.retry_base_ms)
        circuit = CircuitBreaker(
            threshold=cfg.circuit_breaker_threshold,
            reset_s=cfg.circuit_breaker_reset_s,
        )
        router = TaskRouter(node_token=node_token, retry=retry, circuit=circuit)
        selector = NodeSelector(
            preferred_region=cfg.preferred_region,
            min_reputation=cfg.cip_min_reputation,  # §2.2 D4: from iicp_client.proxy.toml [cooperative_inference]
        )
        from iicp_client.proxy.cip.coordinator import ReplayCache
        replay_cache = ReplayCache()
        fallback = FallbackChain(
            router=router,
            replay_cache=replay_cache,
            directory_url=cfg.directory_url,
            node_token=node_token,
        )
        aggregator = ResultAggregator(router=router, fan_out=cfg.redundancy_fan_out)
        directory = DirectoryClient(cfg.directory_url, cfg.directory_timeout_ms)
        peer_cache = PeerCache(
            directory_url=cfg.directory_url,
            ttl_s=cfg.peer_cache_ttl_s,
        )
        cip_config = cfg.to_cip_dispatch_config()  # Phase 5 §2.2 consumer dispatch config

        # §2.2 session_credit_budget (CIP-CALL-03): track cumulative spend for this
        # proxy session. None when no budget is configured (unlimited).
        cip_budget_tracker = None
        if cfg.cip_session_credit_budget is not None:
            from iicp_client.proxy.cip.strategies import SessionBudgetTracker
            cip_budget_tracker = SessionBudgetTracker(
                session_credit_budget=cfg.cip_session_credit_budget
            )

        app.state.config = cfg
        app.state.directory = directory
        app.state.selector = selector
        app.state.fallback_chain = fallback
        app.state.aggregator = aggregator
        app.state.nodes = []
        app.state.peer_cache = peer_cache
        app.state.cip_config = cip_config
        app.state.cip_budget_tracker = cip_budget_tracker
        app.state.node_token = node_token  # WQ-059: §10.1 consumer balance fetch

        # Starlette sets scope["app"]=compat (not app) for mounted sub-app requests,
        # so request.app inside compat handlers is the compat sub-app. Propagate the
        # routing objects to compat.state so handlers can reach them via request.app.state.
        compat.state.directory = directory
        compat.state.selector = selector
        compat.state.fallback_chain = fallback
        compat.state.aggregator = aggregator
        compat.state.peer_cache = peer_cache
        compat.state.cip_config = cip_config
        compat.state.cip_budget_tracker = cip_budget_tracker
        compat.state.node_token = node_token  # WQ-059: §10.1 consumer balance fetch

        await peer_cache.start()

        # Implicit Address Learning — DIR-ADDR-02: fetch observed IP from directory
        if node_token:
            addr_state = get_address_state()
            try:
                me = await directory.me(node_token)
                addr_state.update_from_me(me)
                check_observed_ip_vs_endpoint(
                    me.get("observed_source_ip", ""),
                    me.get("endpoint", cfg.directory_url),
                )
                logger.info(
                    "Directory observes this node at %s",
                    me.get("observed_source_ip"),
                )
            except Exception as exc:
                logger.warning("Could not fetch /v1/me from directory: %s", exc)

        yield

        peer_cache.stop()

    app = FastAPI(title="IICP Proxy", lifespan=lifespan)

    @app.middleware("http")
    async def _identify_as_iicp_proxy(request, call_next):  # type: ignore[no-untyped-def]
        """The proxy self-identifies as `iicp-proxy` on every response (overrides
        uvicorn's default Server header) so clients/tools can tell what answered."""
        response = await call_next(request)
        response.headers["Server"] = "iicp-proxy"
        return response

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """ADR-014: Prometheus metrics endpoint — METRICS-01."""
        data, content_type = metrics_output()
        return Response(content=data, media_type=content_type)

    @app.get("/status", include_in_schema=False)
    async def proxy_status() -> JSONResponse:
        """Returns the proxy's current observed external IP (DIR-ADDR-02)."""
        addr = get_address_state()
        return JSONResponse({
            "node_id":            addr.node_id,
            "observed_source_ip": addr.observed_source_ip,
            "endpoint":           addr.endpoint,
        })

    compat = create_compat_app()
    add_ollama_routes(compat)     # /api/chat, /api/generate, /api/tags, /api/version
    add_anthropic_routes(compat)  # /v1/messages, /v1/models
    app.mount("/", compat)

    @app.exception_handler(Exception)
    async def generic_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled proxy error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "proxy_error", "message": "Internal proxy error"}},
        )

    return app


def run() -> None:
    cfg = ProxyConfig.from_toml()
    app = create_app(cfg)
    # server_header=False so our `Server: iicp-proxy` middleware header is not
    # overridden by uvicorn's default `Server: uvicorn`.
    uvicorn.run(app, host=cfg.host, port=cfg.port, server_header=False)
