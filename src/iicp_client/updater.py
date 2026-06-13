# SPDX-License-Identifier: Apache-2.0
"""Self-updater P1 — read-only version check (#521 WQ-089).

This phase is deliberately inert: it tells the operator whether a newer
release exists and prints the exact upgrade command. No download, no install,
no restart — those are P2/P3 (opt-in, signed). Zero risk surface; answers the
"a user several versions behind shouldn't have to worry" goal by making the
gap visible at a glance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

PYPI_URL = "https://pypi.org/pypi/iicp-client/json"


def parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version into a comparable tuple. Non-numeric/pre-release
    suffixes (e.g. '1.2.3rc1') truncate at the first non-numeric segment —
    good enough for the stable-channel compare P1 does."""
    out: list[int] = []
    for part in v.strip().lstrip("vV").split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        out.append(int(num))
    return tuple(out)


def is_outdated(current: str, latest: str) -> bool:
    """True when `latest` is strictly newer than `current`."""
    return parse_version(latest) > parse_version(current)


def latest_pypi_version(timeout: float = 5.0) -> str | None:
    """Fetch the latest published version from PyPI, or None on any error."""
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=timeout) as resp:  # noqa: S310 — fixed https URL
            data = json.loads(resp.read().decode())
        v = data.get("info", {}).get("version")
        return str(v) if v else None
    except Exception:  # noqa: BLE001 — offline / registry blip → treat as "unknown"
        return None


def check_update(current: str, latest: str | None) -> dict:
    """Produce a structured update verdict for the CLI to render.

    Returns: {current, latest, outdated, command} — `command` is the exact
    upgrade line for whichever install method the operator used (pip)."""
    outdated = bool(latest) and is_outdated(current, latest)  # type: ignore[arg-type]
    return {
        "current": current,
        "latest": latest,
        "outdated": outdated,
        "command": "pip install -U iicp-client",
    }


# ── P2 — background self-updater (#521) ─────────────────────────────────────────
# A node running `serve` periodically checks the registry and, when a newer
# release is published, upgrades itself and re-execs so it comes back on the new
# version. This removes the dependency on operators manually upgrading downlevel
# clients — once a node reaches the first release that contains this updater, all
# future releases self-propagate. Default-on; opt out with IICP_AUTO_UPDATE=0.
# Loop-safe by construction: after a successful upgrade the running version equals
# `latest`, so the next tick is a no-op.


def perform_self_update(spec: str = "iicp-client", timeout: float = 600.0) -> bool:
    """`pip install --upgrade` the package in a subprocess. True on success."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", spec],
            check=True,
            timeout=timeout,
        )
        return True
    except Exception:  # noqa: BLE001 — any failure → "did not upgrade", retry next tick
        return False


def reexec_cli() -> None:
    """Re-exec the current command so the just-upgraded package is loaded. Replaces
    the process image (all threads); returns only if exec failed."""
    try:
        os.execvp(sys.argv[0], sys.argv)  # noqa: S606 — re-running our own argv
    except Exception:  # noqa: BLE001 — fall back to the module entrypoint
        os.execv(sys.executable, [sys.executable, "-m", "iicp_client.cli", *sys.argv[1:]])


def auto_update_enabled() -> bool:
    """Default-on; IICP_AUTO_UPDATE=0 (or false/no) opts out."""
    return os.environ.get("IICP_AUTO_UPDATE", "1").strip().lower() not in {"0", "false", "no", "off"}


def auto_update_interval_s(default: int = 21600) -> int:
    """Check cadence in seconds (default 6h), floored at 5 min."""
    try:
        return max(300, int(os.environ.get("IICP_AUTO_UPDATE_INTERVAL_S", str(default))))
    except ValueError:
        return default


def auto_update_tick(
    current: str,
    latest: str | None,
    enabled: bool,
    upgrade_fn,
    reexec_fn,
    log_fn,
) -> str:
    """One evaluation of the auto-update rule. Pure orchestration — all I/O is
    injected so the decision is unit-testable. Returns the action taken:
    'disabled' | 'unknown' | 'current' | 'upgraded' | 'upgrade-failed'."""
    if not enabled:
        return "disabled"
    if latest is None:
        return "unknown"
    if not is_outdated(current, latest):
        return "current"
    log_fn(f"auto-update: newer release {latest} available (running {current}) — upgrading…")
    if upgrade_fn():
        log_fn(f"auto-update: upgraded to {latest}; restarting to apply…")
        reexec_fn()  # normally does not return (process replaced)
        return "upgraded"
    log_fn("auto-update: upgrade failed; staying on current version, will retry next check")
    return "upgrade-failed"
