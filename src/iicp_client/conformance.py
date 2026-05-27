# SPDX-License-Identifier: Apache-2.0
"""Self-conformance probes — operator-side health verification.

Port of iicp-adapter's `services/conformance_probe.py` (iter-1410, #228)
into the iicp-client-python SDK as part of the adapter→hybrid-client
migration (tracker iicp.network#340 Tier 2 Item 4).

Runs four IICP conformance checks against the operator's own node + the
directory so operators can confirm their node is fully conformant
without running the external REACH daemon:

  CONF-REG-01    — node_id + node_token are set (registration succeeded)
  CONF-HEALTH-01 — GET /iicp/health returns 200 with required schema fields
  CONF-REACH-01  — directory /v1/probe confirms internet reachability
  CONF-DISC-01   — own node_id appears in /v1/discover NODELIST

Usage::

    from iicp_client import IicpNode, NodeConfig
    from iicp_client.conformance import run_conformance_checks

    node = IicpNode(NodeConfig(...))
    token = await node.register()
    await node.serve(handler, port=8080, node_token=token)
    # ...later, in a periodic health check loop:
    report = await run_conformance_checks(node, local_port=8080)
    if report.fail_count:
        for r in report.tests:
            if not r.passed:
                logger.warning("conformance %s: %s", r.test_id, r.message)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_REQUIRED_HEALTH_FIELDS = {"status", "node_id", "region", "load", "models"}
_NON_ROUTABLE = ("localhost", "127.0.0.1", "::1", "example.com", "0.0.0.0")
_DISCOVER_INTENT = "urn:iicp:intent:llm:chat:v1"


# ── Result types ─────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    test_id: str
    passed: bool
    message: str
    latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "passed": self.passed,
            "message": self.message,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ConformanceReport:
    pass_count: int
    fail_count: int
    last_run_at: str
    tests: list[ProbeResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "last_run_at": self.last_run_at,
            "tests": [r.as_dict() for r in self.tests],
        }


# ── Individual probes ─────────────────────────────────────────────────────


async def _check_registered(node: Any) -> ProbeResult:
    """CONF-REG-01: node_id (config) is set and we have a token in hand.

    The SDK's IicpNode doesn't keep the node_token on the instance after
    register() returns it (caller stores it), so this probe accepts an
    optional `node_token` argument via run_conformance_checks. When the
    caller doesn't pass one, the probe checks just that node_id is set.
    """
    node_id = getattr(node, "_cfg", None) and node._cfg.node_id
    token = getattr(node, "_last_token", "") or getattr(node, "node_token", "")
    if node_id and token:
        short = node_id[:8] + "…" if len(node_id) > 8 else node_id
        return ProbeResult("CONF-REG-01", True, f"Registered ({short})")
    if node_id:
        return ProbeResult(
            "CONF-REG-01", True, f"node_id set ({node_id[:8]}…); token not tracked by SDK"
        )
    return ProbeResult("CONF-REG-01", False, "node_id empty — register() not yet called")


async def _check_health_schema(local_port: int) -> ProbeResult:
    """CONF-HEALTH-01: GET /iicp/health → 200 + required schema fields."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"http://127.0.0.1:{local_port}/iicp/health")
        latency = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            return ProbeResult("CONF-HEALTH-01", False, f"HTTP {resp.status_code}", latency)
        missing = _REQUIRED_HEALTH_FIELDS - set(resp.json().keys())
        if missing:
            return ProbeResult(
                "CONF-HEALTH-01", False, f"Missing fields: {sorted(missing)}", latency
            )
        return ProbeResult("CONF-HEALTH-01", True, f"OK ({latency:.0f}ms)", latency)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("CONF-HEALTH-01", False, f"Error: {exc}")


async def _check_reachability(node: Any) -> ProbeResult:
    """CONF-REACH-01: directory /v1/probe confirms public_endpoint is internet-reachable."""
    cfg = node._cfg
    endpoint = cfg.endpoint.rstrip("/") if cfg.endpoint else ""
    if not endpoint or any(p in endpoint for p in _NON_ROUTABLE):
        return ProbeResult(
            "CONF-REACH-01",
            False,
            "endpoint is non-routable — external check skipped; "
            "see https://iicp.network/docs/port-forwarding",
        )

    # Parse host:port out of the endpoint
    without_scheme = endpoint
    for scheme in ("https://", "http://"):
        if endpoint.startswith(scheme):
            without_scheme = endpoint[len(scheme) :]
            break
    authority = without_scheme.split("/")[0]
    if ":" in authority:
        host, port_str = authority.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            port = 443
    else:
        port = 443 if endpoint.startswith("https://") else 80
        host = authority

    directory_url = cfg.directory_url.rstrip("/")
    # SDK directory_url already includes /api; the probe endpoint is /v1/probe
    # under that root so the full URL is e.g. https://iicp.network/api/v1/probe
    probe_url = (
        f"{directory_url}/v1/probe"
        if directory_url.endswith("/api")
        else f"{directory_url}/api/v1/probe"
    )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(probe_url, params={"host": host, "port": port})
        latency = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            body = resp.json()
            if body.get("reachable"):
                return ProbeResult("CONF-REACH-01", True, f"Reachable ({latency:.0f}ms)", latency)
            return ProbeResult(
                "CONF-REACH-01",
                False,
                body.get("error", "not reachable"),
                latency,
            )
        return ProbeResult("CONF-REACH-01", False, f"HTTP {resp.status_code}", latency)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("CONF-REACH-01", False, f"Probe unavailable: {exc}")


async def _check_discover_self(node: Any) -> ProbeResult:
    """CONF-DISC-01: own node_id appears in /v1/discover NODELIST."""
    cfg = node._cfg
    if not cfg.node_id:
        return ProbeResult("CONF-DISC-01", False, "No node_id — register() not yet called")

    directory_url = cfg.directory_url.rstrip("/")
    discover_url = (
        f"{directory_url}/v1/discover"
        if directory_url.endswith("/api")
        else f"{directory_url}/api/v1/discover"
    )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(discover_url, params={"intent": _DISCOVER_INTENT})
        latency = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            return ProbeResult("CONF-DISC-01", False, f"HTTP {resp.status_code}", latency)
        nodes = resp.json().get("nodes", [])
        if any(n.get("node_id") == cfg.node_id for n in nodes):
            return ProbeResult(
                "CONF-DISC-01",
                True,
                f"Found in NODELIST ({len(nodes)} nodes)",
                latency,
            )
        return ProbeResult(
            "CONF-DISC-01",
            False,
            f"node_id absent from NODELIST (got {len(nodes)} nodes)",
            latency,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("CONF-DISC-01", False, f"Discover error: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────


async def run_conformance_checks(
    node: Any, local_port: int = 8080, *, node_token: str | None = None
) -> ConformanceReport:
    """Run the four conformance probes concurrently and return a report.

    Arguments:
        node:       an :class:`IicpNode` instance.
        local_port: the port the operator's HTTP server is listening on.
        node_token: optional — when passed, CONF-REG-01 verifies the token
                    in addition to node_id. When omitted, the probe accepts
                    node_id-set as sufficient (the SDK doesn't track the
                    token on the instance after register() returns it).
    """
    if node_token is not None:
        # Stash on the node so _check_registered can find it without polluting
        # the IicpNode public API.
        node._last_token = node_token

    results: list[ProbeResult] = list(
        await asyncio.gather(
            _check_registered(node),
            _check_health_schema(local_port),
            _check_reachability(node),
            _check_discover_self(node),
        )
    )

    for r in results:
        if r.passed:
            logger.info("Conformance %s PASS — %s", r.test_id, r.message)
        else:
            logger.warning("Conformance %s FAIL — %s", r.test_id, r.message)

    return ConformanceReport(
        pass_count=sum(1 for r in results if r.passed),
        fail_count=sum(1 for r in results if not r.passed),
        last_run_at=datetime.now(UTC).isoformat(),
        tests=results,
    )
