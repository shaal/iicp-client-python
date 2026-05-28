# SPDX-License-Identifier: Apache-2.0
"""Phase 2 mesh layer — peer discovery, gossip, and relay support (parity Block F, #340).

Port of iicp-adapter `network/peer_manager.py` + `handlers/{peers,relay}.py` (ADR-009,
ADR-022) into the SDK. Two capabilities:

  1. Bootstrap: GET /v1/bootstrap from the directory on startup to learn an initial peer
     set, persisted to disk so the node can rejoin the mesh after a restart.
  2. Gossip: periodic POST /v1/peers exchange with a random known peer (30s). Each
     exchange is HMAC-SHA256-signed (reusing the node's pricing HMAC key) so a rogue
     peer can't inject false entries (ADR-009).

Peers are pruned after 90s without contact. The relay capability (POST /v1/relay) lets a
relay-enabled node forward a task to an unreachable peer it knows from gossip (ADR-022).

Thread-safe: the gossip coroutine runs on the serve event loop while the HTTP handlers
call from server threads, so the peer store is guarded by a threading.Lock.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path

import httpx

from iicp_client.pricing import sign_body

logger = logging.getLogger(__name__)

_GOSSIP_INTERVAL_S = 30.0
_PEER_EXPIRY_S = 90.0
_BOOTSTRAP_LIMIT = 5


class PeerManager:
    """Tracks mesh peers, gossips with them, and resolves relay targets."""

    def __init__(
        self,
        directory_url: str,
        node_token: str = "",
        persist_path: Path | None = None,
    ) -> None:
        self._directory_url = directory_url.rstrip("/")
        self._node_token = node_token
        self._persist_path = persist_path
        self._peers: dict[str, dict] = {}
        self._own_id = ""
        self._lock = threading.Lock()

    # ── accessors used by the HTTP handlers (sync, thread-safe) ──────────────

    def get_peers(self) -> list[dict]:
        with self._lock:
            return list(self._peers.values())

    def relay_target(self, node_id: str) -> dict | None:
        with self._lock:
            return self._peers.get(node_id)

    def merge_peers(self, incoming: list[dict]) -> int:
        """Merge incoming peer entries. Returns the count of newly added peers."""
        now = time.monotonic()
        added = 0
        with self._lock:
            for p in incoming:
                nid = p.get("node_id")
                if not nid or nid == self._own_id:
                    continue
                if nid not in self._peers:
                    added += 1
                self._peers[nid] = {
                    "node_id": nid,
                    "endpoint": p.get("endpoint", ""),
                    "region": p.get("region", ""),
                    "last_seen": p.get("last_seen", ""),
                    "last_contact": now,
                }
        if added:
            self._persist()
        return added

    def prune(self) -> int:
        """Drop peers not contacted within the expiry window. Returns count pruned."""
        cutoff = time.monotonic() - _PEER_EXPIRY_S
        with self._lock:
            stale = [nid for nid, p in self._peers.items() if p["last_contact"] < cutoff]
            for nid in stale:
                del self._peers[nid]
        return len(stale)

    def verify_exchange(self, body: bytes, signature: str | None) -> bool:
        """Verify an inbound /v1/peers HMAC signature. No token configured → accept."""
        if not self._node_token:
            return True
        if not signature:
            return False
        return sign_body(body, self._node_token) == signature

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self, node_id: str) -> None:
        self._own_id = node_id
        self._load_persisted()
        await self._bootstrap()

    async def gossip_round(self) -> None:
        peers = self.get_peers()
        if not peers:
            await self._bootstrap()
            return
        target = random.choice(peers)
        await self._exchange(target)
        pruned = self.prune()
        if pruned:
            logger.info("Pruned %d stale peers", pruned)

    async def gossip_loop(self) -> None:
        import asyncio

        while True:
            try:
                await self.gossip_round()
            except Exception as exc:  # noqa: BLE001
                logger.debug("gossip round error: %s", exc)
            await asyncio.sleep(_GOSSIP_INTERVAL_S)

    async def _bootstrap(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._directory_url}/v1/bootstrap",
                    params={"limit": _BOOTSTRAP_LIMIT},
                )
            if resp.is_success:
                added = self.merge_peers(resp.json().get("peers", []))
                logger.info("Bootstrap: merged %d peers from directory", added)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bootstrap failed: %s", exc)

    async def _exchange(self, target: dict) -> None:
        with self._lock:
            known = list(self._peers.keys())
        body = json.dumps({"known_peers": known}).encode()
        headers = {"Content-Type": "application/json"}
        if self._node_token:
            headers["X-IICP-Signature"] = sign_body(body, self._node_token)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{target['endpoint'].rstrip('/')}/v1/peers",
                    content=body,
                    headers=headers,
                )
            if resp.is_success:
                self.merge_peers(resp.json().get("peers", []))
                with self._lock:
                    if target["node_id"] in self._peers:
                        self._peers[target["node_id"]]["last_contact"] = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            logger.debug("gossip exchange with %s failed: %s", target["node_id"][:8], exc)
            with self._lock:
                if target["node_id"] in self._peers:
                    self._peers[target["node_id"]]["last_contact"] = 0.0

    def _load_persisted(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text())
            now = time.monotonic()
            with self._lock:
                for p in raw:
                    nid = p.get("node_id")
                    if nid and nid != self._own_id:
                        p["last_contact"] = now
                        self._peers[nid] = p
            logger.info("Loaded %d persisted peers", len(self._peers))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load peers: %s", exc)

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            with self._lock:
                data = list(self._peers.values())
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist peers: %s", exc)
