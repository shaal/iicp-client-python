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

import base64
import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass, field, replace
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

    #464 — the identity is an ed25519 keypair: ``operator_id`` IS the base64
    public key (the same value the directory verifies + stores as
    ``operator_pubkey`` via the ADR-045 delegation), so the operator_id is
    cryptographically verifiable rather than a random UUID. ``operator_secret``
    is the base64 32-byte private key — LOCAL ONLY (0600 file), never sent to
    the directory; password-at-rest encryption is #460. ``operator_integrity_hash``
    binds the immutable fields so the directory can pin-on-first-use + detect
    tampering (the directory's own clock, not ``created_at``, is authoritative
    for founder ordinals). ``display_name`` is the public, mutable handle;
    ``contact`` is private.
    """
    operator_id: str
    created_at: str
    display_name: str = ""
    contact: str = ""
    # #464 — base64 ed25519 private key (32-byte seed). Local-only secret. Empty when the
    # secret is sealed at rest (see operator_secret_enc, #460).
    operator_secret: str = ""
    # #464/#460 — SHA256(operator_id ':' created_at), pinned by the directory on first use.
    operator_integrity_hash: str = ""
    # #460 — AES-256-GCM-sealed seed (PBKDF2-HMAC-SHA256 from a passphrase) when the operator
    # opts into at-rest encryption; None for a plaintext (default/legacy) identity.
    operator_secret_enc: dict[str, Any] | None = None

    @classmethod
    def generate(cls, *, display_name: str = "", contact: str = "") -> OperatorIdentity:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = Ed25519PrivateKey.generate()
        operator_id = base64.b64encode(sk.public_key().public_bytes_raw()).decode()
        operator_secret = base64.b64encode(sk.private_bytes_raw()).decode()
        created_at = _now_iso()
        return cls(
            operator_id=operator_id,
            created_at=created_at,
            display_name=display_name,
            contact=contact,
            operator_secret=operator_secret,
            operator_integrity_hash=cls.compute_integrity_hash(operator_id, created_at),
        )

    @staticmethod
    def compute_integrity_hash(operator_id: str, created_at: str) -> str:
        return hashlib.sha256(f"{operator_id}:{created_at}".encode()).hexdigest()

    def is_encrypted(self) -> bool:
        """True when the seed is sealed at rest (#460) and a passphrase is needed to sign."""
        return self.operator_secret_enc is not None and not self.operator_secret

    def _seed_b64(self, passphrase: str | None) -> str:
        """Resolve the base64 seed: plaintext if present, else decrypt the sealed seed with
        ``passphrase`` (falling back to $IICP_OPERATOR_PASSPHRASE for headless serve)."""
        if self.operator_secret:
            return self.operator_secret
        if self.operator_secret_enc is not None:
            from iicp_client.operator_crypto import decrypt_seed, passphrase_from_env

            pw = passphrase or passphrase_from_env()
            if not pw:
                raise ValueError(
                    "operator secret is encrypted — set $IICP_OPERATOR_PASSPHRASE (or pass a "
                    "passphrase) to unlock it (#460)"
                )
            return decrypt_seed(pw, self.operator_secret_enc, self.operator_id)
        raise ValueError(
            "legacy operator identity has no key (operator_id is a UUID, not a public key) — "
            "regenerate with `iicp-node operator init` (#464)"
        )

    def signing_key(self, passphrase: str | None = None):
        """Return the ed25519 private key for signing delegations / mutations. Decrypts the
        sealed seed when the identity is encrypted (#460). Raises for a legacy keyless identity."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(self._seed_b64(passphrase)))

    def is_key_backed(self) -> bool:
        """True when operator_id is a real ed25519 pubkey (not a legacy op-<uuid>)."""
        return (
            bool(self.operator_secret) or self.operator_secret_enc is not None
        ) and not self.operator_id.startswith("op-")

    def encrypt_at_rest(self, passphrase: str) -> OperatorIdentity:
        """Return a copy with the seed sealed under ``passphrase`` (operator_secret cleared, #460)."""
        from iicp_client.operator_crypto import encrypt_seed

        enc = encrypt_seed(passphrase, self._seed_b64(None), self.operator_id)
        return replace(self, operator_secret="", operator_secret_enc=enc)

    def decrypt_at_rest(self, passphrase: str) -> OperatorIdentity:
        """Return a copy with the plaintext seed restored (operator_secret_enc cleared, #460)."""
        return replace(self, operator_secret=self._seed_b64(passphrase), operator_secret_enc=None)

    def to_dict(self) -> dict[str, Any]:
        """Full dict for the local 0600 file (INCLUDES operator_secret — never send this
        to the directory; the register payload is built explicitly elsewhere)."""
        d = asdict(self)
        if d.get("operator_secret_enc") is None:
            d.pop("operator_secret_enc", None)  # keep plaintext files unchanged
        return d

    def public_dict(self) -> dict[str, Any]:
        """Directory-safe view: operator_id + created_at + display_name + integrity hash.
        NEVER includes operator_secret or contact (private)."""
        return {
            "operator_id": self.operator_id,
            "created_at": self.created_at,
            "display_name": self.display_name,
            "operator_integrity_hash": self.operator_integrity_hash,
        }


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
        # #464 — present on key-backed identities; absent on legacy op-<uuid> files.
        operator_secret=data.get("operator_secret", ""),
        operator_integrity_hash=data.get("operator_integrity_hash", ""),
        # #460 — sealed seed when the operator opted into at-rest encryption.
        operator_secret_enc=data.get("operator_secret_enc"),
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
