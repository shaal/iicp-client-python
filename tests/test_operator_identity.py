# SPDX-License-Identifier: Apache-2.0
"""#464 — OperatorIdentity is the ed25519 operator key: operator_id is the verifiable
public key (== the directory's operator_pubkey via the ADR-045 delegation), not a random
UUID. Fails without the fix (old operator_id was `op-<uuid>` with no key)."""

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from iicp_client.delegation import issue_delegation, operator_pub_b64, verify_delegation
from iicp_client.identity import OperatorIdentity


def test_operator_id_is_the_base64_ed25519_pubkey():
    op = OperatorIdentity.generate(display_name="Rebel One", contact="me@example.com")
    # operator_id decodes to a 32-byte ed25519 public key (not a UUID).
    assert not op.operator_id.startswith("op-")
    assert len(base64.b64decode(op.operator_id)) == 32
    assert len(base64.b64decode(op.operator_secret)) == 32
    assert op.is_key_backed()


def test_signing_key_public_matches_operator_id():
    op = OperatorIdentity.generate()
    sk = op.signing_key()
    assert isinstance(sk, Ed25519PrivateKey)
    assert operator_pub_b64(sk) == op.operator_id


def test_delegation_uses_the_identity_key_and_verifies():
    op = OperatorIdentity.generate()
    token = issue_delegation(op.signing_key(), "node-123")
    # The delegation's operator_pub IS the identity's operator_id (one key, verifiable).
    assert token["operator_pub"] == op.operator_id
    assert verify_delegation(token, "node-123") is True
    assert verify_delegation(token, "other-node") is False


def test_integrity_hash_binds_operator_id_and_created_at():
    op = OperatorIdentity.generate()
    assert op.operator_integrity_hash == OperatorIdentity.compute_integrity_hash(
        op.operator_id, op.created_at
    )
    # Tampering created_at changes the hash (directory pins the original first-use).
    assert OperatorIdentity.compute_integrity_hash(op.operator_id, "1999-01-01T00:00:00Z") != op.operator_integrity_hash


def test_public_dict_never_leaks_secret_or_contact():
    op = OperatorIdentity.generate(display_name="Pub", contact="secret@example.com")
    pub = op.public_dict()
    assert "operator_secret" not in pub
    assert "contact" not in pub
    assert pub["operator_id"] == op.operator_id
    assert pub["display_name"] == "Pub"


def test_legacy_uuid_identity_is_not_key_backed():
    legacy = OperatorIdentity(operator_id="op-deadbeef", created_at="2026-01-01T00:00:00Z")
    assert not legacy.is_key_backed()
    try:
        legacy.signing_key()
        raise AssertionError("legacy keyless identity must refuse to sign")
    except ValueError:
        pass
