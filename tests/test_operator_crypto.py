# SPDX-License-Identifier: Apache-2.0
"""#460 — at-rest encryption of the operator secret (AES-256-GCM / PBKDF2-HMAC-SHA256).

Pins the cross-language KAT (a record sealed by any SDK must open in any other given the
passphrase), and the identity-level round-trip: encrypt → sign still works (via the unlock
passphrase) → decrypt restores plaintext; legacy plaintext files keep loading; a wrong
passphrase fails cleanly without mutating anything.
"""

import base64

import pytest

from iicp_client.identity import OperatorIdentity, load_operator, save_operator
from iicp_client.operator_crypto import decrypt_seed, encrypt_seed

# Cross-language KAT — MUST decrypt identically in the TS and Rust SDKs (same fixed inputs).
PASSPHRASE = "correct horse battery staple"
OPERATOR_ID = "T3BQdWI="  # AAD
SEED_B64 = "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8="
KAT_RECORD = {
    "v": 1,
    "kdf": "pbkdf2-hmac-sha256",
    "iter": 600000,
    "salt": "AAECAwQFBgcICQoLDA0ODw==",
    "nonce": "EBESExQVFhcYGRob",
    "ct": "LDNf5jTajlDjk7Pj4N5a1SEJqNeyUuCc+wkh0fSEftCq1ypsedl8nLMPuMZQ7Xvl",
}


def test_kat_record_decrypts_to_known_seed():
    # Any SDK must open this exact record — pins KDF params + AEAD + AAD + byte-shape.
    assert decrypt_seed(PASSPHRASE, KAT_RECORD, OPERATOR_ID) == SEED_B64


def test_encrypt_then_decrypt_round_trip():
    enc = encrypt_seed("hunter2", SEED_B64, OPERATOR_ID)
    assert enc["kdf"] == "pbkdf2-hmac-sha256" and enc["v"] == 1
    # Fresh random salt/nonce each time → ciphertext differs from the KAT.
    assert enc["ct"] != KAT_RECORD["ct"]
    assert decrypt_seed("hunter2", enc, OPERATOR_ID) == SEED_B64


def test_wrong_passphrase_fails_cleanly():
    enc = encrypt_seed("right", SEED_B64, OPERATOR_ID)
    with pytest.raises(ValueError):
        decrypt_seed("WRONG", enc, OPERATOR_ID)


def test_aad_binds_operator_id():
    # A seed sealed for one operator_id cannot be opened under another (AES-GCM AAD).
    enc = encrypt_seed("pw", SEED_B64, OPERATOR_ID)
    with pytest.raises(ValueError):
        decrypt_seed("pw", enc, "different-operator-id")


def test_identity_encrypt_sign_decrypt_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("IICP_HOME", str(tmp_path))
    op = OperatorIdentity.generate(display_name="Padme")
    assert not op.is_encrypted()
    plain_pub = op.operator_id

    enc = op.encrypt_at_rest("s3cret")
    assert enc.is_encrypted()
    assert enc.operator_secret == ""  # plaintext seed gone from the at-rest record
    assert enc.is_key_backed()  # still a real key identity
    save_operator(enc)

    # Reloaded encrypted identity can sign once unlocked (headless via env, or explicit arg).
    reloaded = load_operator()
    assert reloaded.is_encrypted()
    monkeypatch.setenv("IICP_OPERATOR_PASSPHRASE", "s3cret")
    sk = reloaded.signing_key()
    assert base64.b64encode(sk.public_key().public_bytes_raw()).decode() == plain_pub

    # Decrypt restores the exact original seed.
    back = enc.decrypt_at_rest("s3cret")
    assert not back.is_encrypted()
    assert back.operator_secret == op.operator_secret


def test_legacy_plaintext_identity_still_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("IICP_HOME", str(tmp_path))
    op = OperatorIdentity.generate(display_name="Han")
    save_operator(op)  # no operator_secret_enc field written for a plaintext identity
    reloaded = load_operator()
    assert not reloaded.is_encrypted()
    assert reloaded.signing_key() is not None
