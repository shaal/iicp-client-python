# SPDX-License-Identifier: Apache-2.0
"""Constant-time bearer-token validation (parity Block E, #340).

Port of iicp-adapter `services/token_validator.py`. Validates a presented bearer token
against the node's configured/issued token using a constant-time comparison so a timing
side-channel can't be used to recover the token byte-by-byte. The expected token is
updated after registration (the directory issues the real token in the register response).
"""

from __future__ import annotations

import hmac


class TokenValidator:
    def __init__(self, expected_token: str = "") -> None:
        self._expected = expected_token

    def is_valid(self, presented: str | None) -> bool:
        if not self._expected or not presented:
            return False
        # Constant-time comparison prevents timing attacks.
        return hmac.compare_digest(self._expected.encode(), presented.encode())

    def update_token(self, new_token: str) -> None:
        """Set the expected token after registration (directory-issued)."""
        self._expected = new_token
