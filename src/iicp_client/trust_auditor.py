# SPDX-License-Identifier: Apache-2.0
"""Trust auditor — cross-node declaration consistency check (parity Block E, #340).

Port of iicp-adapter `services/trust_auditor.py` (#118). Discovers active peers via the
directory, probes each peer's `/iicp/health`, and verifies that the models the directory
has registered for a peer actually appear in that peer's live health response. Missing
models are a "declaration divergence" — reported to the directory's `/v1/audit-report`
endpoint so it can apply the reputation penalty.

This is an opt-in background capability (call `run_audit_pass` on a timer); it is not in
the request hot path. The pure `models_diverge` helper is the unit-testable core.
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

_DISCOVER_INTENT = "urn:iicp:intent:llm:chat:v1"
_PROBE_TIMEOUT_S = 5.0
_DISCOVER_TIMEOUT_S = 8.0
_AUDIT_REPORT_TIMEOUT_S = 5.0


def models_diverge(registered: list[str], health: list[str]) -> set[str]:
    """Return registered models absent from the peer's health response.

    Health may report extra models (fine); registered-but-missing models are the
    divergence. Empty result == consistent.
    """
    return set(registered) - set(health)


@dataclass
class NodeAuditResult:
    node_id: str
    endpoint: str
    health_reachable: bool
    declared_models_match: bool
    registered_models: list[str]
    health_models: list[str]
    latency_ms: float | None = None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.health_reachable and self.declared_models_match

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "passed": self.passed,
            "health_reachable": self.health_reachable,
            "declared_models_match": self.declared_models_match,
            "registered_models": self.registered_models,
            "health_models": self.health_models,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


@dataclass
class AuditReport:
    run_at: str
    nodes_probed: int
    nodes_passed: int
    nodes_failed: int
    results: list[NodeAuditResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at,
            "nodes_probed": self.nodes_probed,
            "nodes_passed": self.nodes_passed,
            "nodes_failed": self.nodes_failed,
            "results": [r.as_dict() for r in self.results],
        }


async def _discover_peers(directory_url: str, own_node_id: str) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=_DISCOVER_TIMEOUT_S) as client:
            resp = await client.get(
                f"{directory_url.rstrip('/')}/v1/discover",
                params={"intent": _DISCOVER_INTENT},
            )
        if resp.status_code != 200:
            logger.warning("Trust audit: discover returned HTTP %s", resp.status_code)
            return []
        nodes = resp.json().get("nodes", [])
        return [n for n in nodes if n.get("node_id") != own_node_id]
    except Exception as exc:
        logger.warning("Trust audit: discover error: %s", exc)
        return []


async def _probe_node(node: dict[str, Any]) -> NodeAuditResult:
    node_id = node.get("node_id", "unknown")
    endpoint = node.get("operator_url") or node.get("endpoint", "")
    registered = node.get("models", [])
    if not endpoint:
        return NodeAuditResult(node_id, "", False, False, registered, [], detail="no endpoint")

    health_url = endpoint.rstrip("/") + "/iicp/health"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            resp = await client.get(health_url)
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            return NodeAuditResult(
                node_id, endpoint, False, False, registered, [], latency_ms,
                f"HTTP {resp.status_code}",
            )
        health_models = resp.json().get("models", [])
        missing = models_diverge(registered, health_models)
        return NodeAuditResult(
            node_id, endpoint, True, not missing, registered, health_models, latency_ms,
            "OK" if not missing else f"registered {sorted(missing)} absent from health",
        )
    except Exception as exc:
        return NodeAuditResult(
            node_id, endpoint, False, False, registered, [],
            (time.monotonic() - t0) * 1000, f"connection error: {exc}",
        )


async def _report_divergence(
    directory_url: str, own_node_id: str, node_token: str, target_node_id: str
) -> None:
    if not own_node_id or not node_token:
        return
    try:
        async with httpx.AsyncClient(timeout=_AUDIT_REPORT_TIMEOUT_S) as client:
            await client.post(
                f"{directory_url.rstrip('/')}/v1/audit-report",
                json={
                    "node_id": own_node_id,
                    "target_node_id": target_node_id,
                    "finding": "declaration_divergence",
                },
                headers={"Authorization": f"Bearer {node_token}"},
            )
    except Exception as exc:
        logger.warning("Trust audit: audit-report failed for %s: %s", target_node_id[:8], exc)


async def run_audit_pass(
    directory_url: str, own_node_id: str, node_token: str = ""
) -> AuditReport:
    """Discover peers, probe each concurrently, report divergences. One pass."""
    nodes = await _discover_peers(directory_url, own_node_id)
    run_at = datetime.now(UTC).isoformat()
    if not nodes:
        return AuditReport(run_at, 0, 0, 0)

    results = list(await asyncio.gather(*(_probe_node(n) for n in nodes)))
    for r in results:
        if r.health_reachable and not r.declared_models_match:
            await _report_divergence(directory_url, own_node_id, node_token, r.node_id)

    return AuditReport(
        run_at=run_at,
        nodes_probed=len(results),
        nodes_passed=sum(1 for r in results if r.passed),
        nodes_failed=sum(1 for r in results if not r.passed),
        results=results,
    )
