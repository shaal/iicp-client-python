"""Persistent on-disk identity for the IICP SDK CLI.

Two layers of identity:

- **Operator identity** at ``~/.iicp/operator.json``. One per machine
  account; accumulates credits across every node this operator runs.
  Equivalent of a "user account" in the network. Survives node churn,
  backend changes, model rotations.

- **Node identity** at ``~/.iicp/nodes/<name>.json``. One per provider
  node the operator runs. Carries the stable ``node_id`` (UUIDv4) so
  restarts don't create duplicate directory entries (#215 — same fix
  the deprecated adapter shipped).

The wizard (`iicp-node init`) creates / reads / lists these files
interactively; ``iicp-node serve`` can then load a saved node config
via ``--node <name>`` and run with the persisted identity.

File permissions are tightened to 0600 on creation so other local users
can't read the node tokens or operator identity.
"""
from __future__ import annotations

import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chmod_600(path: Path) -> None:
    """Best-effort: tighten file permissions to user-read/write only.
    Silently skipped on filesystems where chmod is a no-op (Windows + WSL)."""
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def config_dir() -> Path:
    """Return the per-user IICP config directory, creating it if absent.

    Default: ``$IICP_HOME`` if set, else ``~/.iicp``.
    """
    base = os.environ.get("IICP_HOME")
    p = Path(base).expanduser() if base else Path.home() / ".iicp"
    p.mkdir(parents=True, exist_ok=True)
    nodes_dir = p / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class OperatorIdentity:
    """Operator-level identity. One per machine account.

    Credits earned by any node this operator runs accumulate to this
    operator_id at the directory. Treat it as your IICP user account.
    """
    operator_id: str
    created_at: str
    display_name: str = ""
    contact: str = ""

    @classmethod
    def generate(cls, *, display_name: str = "", contact: str = "") -> OperatorIdentity:
        return cls(
            operator_id=f"op-{uuid.uuid4()}",
            created_at=_now_iso(),
            display_name=display_name,
            contact=contact,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def operator_path() -> Path:
    return config_dir() / "operator.json"


def load_operator() -> OperatorIdentity | None:
    """Return the existing operator identity, or None if not yet created."""
    p = operator_path()
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return OperatorIdentity(
        operator_id=data["operator_id"],
        created_at=data["created_at"],
        display_name=data.get("display_name", ""),
        contact=data.get("contact", ""),
    )


def save_operator(op: OperatorIdentity) -> Path:
    p = operator_path()
    p.write_text(json.dumps(op.to_dict(), indent=2) + "\n")
    _chmod_600(p)
    return p


@dataclass
class NodeIdentity:
    """Per-node configuration. Stable node_id survives restarts (#215).

    Wraps just enough to call ``iicp-node serve`` headlessly later —
    the rest (NAT auto-detect, public_endpoint override, etc.) stays
    on the command line where operators iterate on it.
    """
    node_id: str
    operator_id: str
    name: str  # short human label, used as the filename stem
    backend_url: str
    model: str
    intent: str = "urn:iicp:intent:llm:chat:v1"
    region: str = "eu-central"
    directory_url: str = "https://iicp.network/api"
    max_concurrent: int = 4
    port: int = 8020
    host: str = "0.0.0.0"
    public_endpoint: str = ""
    auto_detect_nat: bool = False
    external_ip_probe_url: str = ""
    # #456 — node_token cached after register so `iicp-node credits` can authenticate
    # without re-registering. Bearer credential (not a key); stored in the chmod-600
    # config. None until the node first registers via `serve`.
    node_token: str | None = None
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def generate(cls, *, operator_id: str, name: str, **fields: Any) -> NodeIdentity:
        return cls(
            node_id=str(uuid.uuid4()),
            operator_id=operator_id,
            name=name,
            **fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def _validate_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid node name {name!r} — must match [a-z0-9][a-z0-9._-]{{0,62}}"
        )
    return name


def node_path(name: str) -> Path:
    name = _validate_name(name)
    return config_dir() / "nodes" / f"{name}.json"


def load_node(name: str) -> NodeIdentity | None:
    p = node_path(name)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return NodeIdentity(**data)


def save_node(node: NodeIdentity) -> Path:
    p = node_path(node.name)
    p.write_text(json.dumps(node.to_dict(), indent=2) + "\n")
    _chmod_600(p)
    return p


def list_nodes() -> list[NodeIdentity]:
    """Return all node configs in ~/.iicp/nodes/ sorted by name."""
    nodes_dir = config_dir() / "nodes"
    out: list[NodeIdentity] = []
    if not nodes_dir.exists():
        return out
    for p in sorted(nodes_dir.glob("*.json")):
        try:
            out.append(NodeIdentity(**json.loads(p.read_text())))
        except (ValueError, KeyError, TypeError):
            # Malformed config — skip silently so list keeps working
            continue
    return out
