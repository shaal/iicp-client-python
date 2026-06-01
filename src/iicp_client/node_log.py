"""Persistent node log writer — ~/.iicp/logs/<node_id>.log + events.jsonl.

Provides two outputs:
- A human-readable rotating text log via Python's RotatingFileHandler.
- A structured NDJSON event stream (events.jsonl) for programmatic inspection.

No credentials are written here; callers MUST NOT pass tokens or keys.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

_JSONL_LOCK = threading.Lock()

_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
_DEFAULT_BACKUP_COUNT = 3


def log_dir() -> Path:
    """Resolve the log directory: IICP_LOG_DIR > ~/.iicp/logs/."""
    raw = os.environ.get("IICP_LOG_DIR", "")
    if raw:
        return Path(raw)
    base = Path(os.environ.get("IICP_HOME", Path.home() / ".iicp"))
    return base / "logs"


def setup_node_log(node_id: str, override_dir: str | None = None) -> Path:
    """Attach a RotatingFileHandler to the iicp-node logger.

    Returns the resolved log directory path (created if absent).
    Safe to call multiple times — duplicate handlers are not added.
    """
    d = Path(override_dir) if override_dir else log_dir()
    d.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("iicp-node")
    log_path = d / f"{node_id}.log"

    # Guard against duplicate handlers on repeated calls.
    if any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "baseFilename", None) == str(log_path.resolve())
        for h in logger.handlers
    ):
        return d

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_DEFAULT_MAX_BYTES,
        backupCount=_DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return d


def write_event(
    node_id: str,
    event: str,
    details: str = "",
    log_dir_override: str | None = None,
) -> None:
    """Append one structured event to events.jsonl.

    ``event`` is a snake_case key (e.g. ``register_ok``, ``heartbeat_fail``).
    ``details`` is a plain string — MUST NOT contain credentials.
    """
    d = Path(log_dir_override) if log_dir_override else log_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {"ts": ts, "event": event, "node_id": node_id, "details": details}
    line = json.dumps(record) + "\n"
    events_path = d / "events.jsonl"
    with _JSONL_LOCK:
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(line)
