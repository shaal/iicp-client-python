"""#460 — at-rest encryption of the operator secret (ed25519 seed) in ``operator.json``.

The operator_secret is the private key behind the operator_id; by default it is stored as
plaintext base64 in a 0600 file. An operator may opt in to passphrase encryption: the seed is
sealed with **AES-256-GCM**, the key derived from the passphrase with **PBKDF2-HMAC-SHA256**
(OWASP-2023 iteration count). Both primitives are available with NO new dependency in every
SDK flavour (Python ``cryptography``, Node built-in ``crypto``, Rust ``aes-gcm`` + ``hmac`` +
``sha2``), so this never trips the third-party due-diligence gate (TC-11).

The encrypted record is a small JSON object whose BYTE-SHAPE is identical across all three
SDKs — a file encrypted by one decrypts in another given the passphrase. This is pinned by a
cross-language known-answer test (KAT). The operator_id is bound in as AES-GCM additional
authenticated data (AAD), so a sealed seed cannot be transplanted onto a different identity.

Unlock is headless-compatible: a serving node reads the passphrase from
``$IICP_OPERATOR_PASSPHRASE`` — never an interactive prompt.
"""
from __future__ import annotations

import base64
import os
from typing import Any

# OWASP 2023 minimum for PBKDF2-HMAC-SHA256. Pinned in the record so it can be raised later
# without breaking existing files (decrypt reads the stored count).
PBKDF2_ITERATIONS = 600_000
_KDF = "pbkdf2-hmac-sha256"
_VERSION = 1
ENV_PASSPHRASE = "IICP_OPERATOR_PASSPHRASE"


def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(algorithm=SHA256(), length=32, salt=salt, iterations=iterations)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_seed(passphrase: str, seed_b64: str, operator_id: str) -> dict[str, Any]:
    """Seal the raw 32-byte ed25519 seed (given as base64) under ``passphrase``. Returns the
    JSON-able encrypted record. operator_id is bound as AAD (tamper/transplant protection)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not passphrase:
        raise ValueError("passphrase must not be empty")
    seed = base64.b64decode(seed_b64)
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_key(passphrase, salt, PBKDF2_ITERATIONS)
    ct = AESGCM(key).encrypt(nonce, seed, operator_id.encode("utf-8"))
    return {
        "v": _VERSION,
        "kdf": _KDF,
        "iter": PBKDF2_ITERATIONS,
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ct": base64.b64encode(ct).decode(),
    }


def decrypt_seed(passphrase: str, enc: dict[str, Any], operator_id: str) -> str:
    """Open an encrypted record and return the base64 seed. Raises on a wrong passphrase,
    tampered ciphertext, mismatched operator_id (AAD), or an unsupported format."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if enc.get("kdf") != _KDF or int(enc.get("v", 0)) != _VERSION:
        raise ValueError(f"unsupported operator_secret_enc format: {enc.get('kdf')} v{enc.get('v')}")
    salt = base64.b64decode(enc["salt"])
    nonce = base64.b64decode(enc["nonce"])
    ct = base64.b64decode(enc["ct"])
    key = _derive_key(passphrase, salt, int(enc["iter"]))
    try:
        seed = AESGCM(key).decrypt(nonce, ct, operator_id.encode("utf-8"))
    except InvalidTag as exc:  # wrong passphrase OR tampered ciphertext OR wrong operator_id
        raise ValueError("operator secret decryption failed (wrong passphrase or corrupt file)") from exc
    return base64.b64encode(seed).decode()


def passphrase_from_env() -> str | None:
    """Headless unlock source — never an interactive prompt for a serving node."""
    return os.environ.get(ENV_PASSPHRASE) or None
