# SPDX-License-Identifier: Apache-2.0
"""Quick-Tunnel escalation — #520 rung 5 of the NAT ladder.

When every NAT variant fails (no direct endpoint, no UPnP pinhole, no IPv6
GUA, no relay-capable peer in the directory), the node can still become
publicly reachable with ZERO account, domain, or router changes: spawn
``cloudflared tunnel --url http://127.0.0.1:<port>`` and register the issued
``https://*.trycloudflare.com`` URL as the endpoint.

Lifecycle is fully automatic ("automagical", maintainer 2026-06-12):
  setup     — detect the cloudflared binary (never auto-installed; supply-chain
              discipline — one actionable hint when missing)
  initiate  — spawn, parse the public URL from process output (≤20 s)
  supervise — watchdog thread; unexpected death → respawn (bounded) and hand
              the NEW url to the caller for re-registration
  tear down — close() terminates the child; also runs via atexit so a normal
              process exit never leaves an orphaned tunnel

Proven live 2026-06-12: a real /v1/task completed through a Quick Tunnel, and
a browser node became directory-LISTED via a tunnel-exposed relay (#452).
"""

from __future__ import annotations

import atexit
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

# cloudflared usually prints the URL within ~5 s; 20 s covers slow first runs.
TUNNEL_START_TIMEOUT = 20.0
# Bounded self-healing: after this many unexpected deaths, stop respawning
# and surface the failure (operator restart required).
MAX_RESPAWNS = 3

INSTALL_HINT = (
    "cloudflared not found — install it to become reachable without router "
    "changes (zero-account Quick Tunnel): "
    "macOS `brew install cloudflared` · Linux: https://pkg.cloudflare.com · "
    "Windows `winget install Cloudflare.cloudflared`"
)


def cloudflared_path() -> str | None:
    """Locate the cloudflared binary, or None (we never auto-install it)."""
    return shutil.which("cloudflared")


class QuickTunnel:
    """A running Quick Tunnel: public ``url`` → ``http://127.0.0.1:<port>``."""

    def __init__(self, process: subprocess.Popen, url: str, local_port: int, binary: str) -> None:
        self.process = process
        self.url = url
        self.local_port = local_port
        self._binary = binary
        self._closed = False
        self._respawns = 0
        self._watchdog: threading.Thread | None = None
        atexit.register(self.close)

    # ── supervise ────────────────────────────────────────────────────────────

    def watch(self, on_new_url: Callable[[str], None], on_dead: Callable[[], None]) -> None:
        """Start the watchdog: on unexpected exit, respawn (bounded) and call
        ``on_new_url(new_url)`` — Quick Tunnel URLs rotate per process, so the
        caller MUST re-register. After MAX_RESPAWNS, ``on_dead()`` fires once.

        Callbacks run on the watchdog thread; marshal to your loop if needed.
        """

        def _run() -> None:
            while not self._closed:
                self.process.wait()
                if self._closed:
                    return
                self._respawns += 1
                if self._respawns > MAX_RESPAWNS:
                    logger.error(
                        "Quick Tunnel died %d times — giving up. Node is no longer "
                        "publicly reachable; restart `iicp-node serve` to recover.",
                        self._respawns - 1,
                    )
                    on_dead()
                    return
                logger.warning(
                    "Quick Tunnel exited unexpectedly — respawning (%d/%d)…",
                    self._respawns,
                    MAX_RESPAWNS,
                )
                try:
                    fresh = open_quick_tunnel(self.local_port, binary=self._binary)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Quick Tunnel respawn failed: %s", exc)
                    on_dead()
                    return
                self.process = fresh.process
                self.url = fresh.url
                logger.info("Quick Tunnel back up at %s — re-registering.", self.url)
                on_new_url(self.url)

        self._watchdog = threading.Thread(target=_run, name="quick-tunnel-watchdog", daemon=True)
        self._watchdog.start()

    # ── tear down ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Terminate the tunnel child. Idempotent; also registered via atexit."""
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        logger.info("Quick Tunnel closed.")


def open_quick_tunnel(
    local_port: int,
    timeout: float = TUNNEL_START_TIMEOUT,
    binary: str | None = None,
) -> QuickTunnel:
    """Spawn cloudflared and return the running tunnel with its public URL.

    Raises FileNotFoundError when cloudflared is absent (caller prints
    INSTALL_HINT once) and RuntimeError when no URL appears within ``timeout``.
    """
    resolved = binary or cloudflared_path()
    if not resolved:
        raise FileNotFoundError(INSTALL_HINT)
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [resolved, "tunnel", "--url", f"http://127.0.0.1:{local_port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Read on a thread: readline() on the main thread would block forever if
    # the child prints nothing, defeating the deadline. The same thread keeps
    # draining after the URL is found so the child never stalls on a full pipe
    # (cloudflared logs continuously).
    lines: queue.Queue[str | None] = queue.Queue()
    url_found = threading.Event()  # once set, the reader drains without queueing

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not url_found.is_set():
                lines.put(line)
        lines.put(None)  # EOF sentinel

    threading.Thread(target=_reader, name="quick-tunnel-read", daemon=True).start()

    deadline = time.monotonic() + timeout
    url: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            break
        if line is None:
            break  # process exited before printing a URL
        m = _URL_RE.search(line)
        if m:
            url = m.group(0)
            url_found.set()
            break
    if url is None:
        proc.terminate()
        raise RuntimeError(
            f"cloudflared produced no tunnel URL within {timeout:.0f}s "
            f"(exit={proc.poll()})"
        )
    logger.info("Quick Tunnel up: %s → http://127.0.0.1:%d", url, local_port)
    return QuickTunnel(proc, url, local_port, resolved)
