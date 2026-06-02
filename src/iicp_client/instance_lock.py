"""#405 — single-instance lock per node_id.

Two `iicp-node serve` processes for the SAME node_id fight: each registration
rotates the directory-issued token and invalidates the other's, so they enter a
401 -> re-register war that makes the node flap in the directory. This guard
holds a pidfile at ``~/.iicp/run/<node_id>.pid``; a second LIVE process for the
same node_id is refused (unless ``force``). Distinct node_ids are unaffected — a
fleet of N nodes runs fine (each has its own lock).

Fail-open: any filesystem error degrades to a no-op lock — the guard must never
prevent a node from starting.
"""

from __future__ import annotations

import os
from pathlib import Path


def _run_dir() -> Path:
    base = Path(os.environ.get("IICP_HOME") or (Path.home() / ".iicp"))
    return base / "run"


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists. PermissionError means it exists
    (we just may not signal it) — treat as alive to be safe."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class NodeAlreadyServingError(RuntimeError):
    """Raised when another live process already serves this node_id."""


class InstanceLock:
    def __init__(self, path: Path | None) -> None:
        self._path = path

    @classmethod
    def acquire(cls, node_id: str, force: bool = False) -> InstanceLock:
        """Acquire the per-node_id lock. Raises NodeAlreadyServingError if another
        LIVE process holds it and ``force`` is False. Fails open on I/O error."""
        try:
            d = _run_dir()
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{node_id}.pid"
        except OSError:
            return cls(None)  # fail open
        if not force and path.exists():
            try:
                pid = int(path.read_text().strip())
            except (ValueError, OSError):
                pid = None
            if pid is not None and pid != os.getpid() and _pid_alive(pid):
                raise NodeAlreadyServingError(
                    f"node_id {node_id} is already being served by PID {pid}. "
                    f"Stop that process, choose a different --node, or pass --force to take over."
                )
        try:
            path.write_text(str(os.getpid()))
        except OSError:
            return cls(None)
        return cls(path)

    def release(self) -> None:
        if self._path is not None:
            try:
                self._path.unlink()
            except OSError:
                pass
