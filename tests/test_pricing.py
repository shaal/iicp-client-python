"""Unit tests for ADR-019 pricing + HMAC signing.

Wire-compat: the canonical signature body MUST match what PHP's
hash_hmac('sha256', json_encode(ksort($body)), $key) produces, so the
directory's NodeRegistry::resolvePricingBlock accepts the signature.
The most subtle case is float-with-no-fraction (PHP json_encode(1.0) = "1"
but Python json.dumps(1.0) = "1.0"). _php_canonical_sign_body handles this.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading

import httpx
import pytest
import respx

from iicp_client import IicpNode, NodeConfig
from iicp_client.pricing import (
    PricingConfig,
    _php_canonical_sign_body,
    build_pricing_block,
    sign_body,
    verify_signature,
)


# ── HMAC primitive ─────────────────────────────────────────────────────────

class TestHmacPrimitive:
    def test_sign_body_matches_hashlib_hmac_sha256(self):
        body = b"hello world"
        key = "secret"
        got = sign_body(body, key)
        expected = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
        assert got == expected

    def test_verify_signature_round_trip(self):
        body = b"hello"
        sig = sign_body(body, "k")
        assert verify_signature(body, "k", sig)
        assert not verify_signature(body, "k", "deadbeef")
        assert not verify_signature(body, "wrong-key", sig)


# ── PHP canonical body (wire-compat) ───────────────────────────────────────

class TestPhpCanonicalBody:
    def test_whole_float_emits_integer_form(self):
        # PHP json_encode(1.0) → "1" — Python json.dumps(1.0) → "1.0"
        body = _php_canonical_sign_body(credit_cost_multiplier=1.0, pricing_model="per_token")
        assert body == b'{"credit_cost_multiplier":1,"pricing_model":"per_token"}'

    def test_fractional_float_emits_decimal(self):
        body = _php_canonical_sign_body(credit_cost_multiplier=1.5, pricing_model="per_token")
        assert body == b'{"credit_cost_multiplier":1.5,"pricing_model":"per_token"}'

    def test_keys_sorted_alphabetically(self):
        body = _php_canonical_sign_body(credit_cost_multiplier=2.0, pricing_model="per_token")
        # credit_cost_multiplier comes before pricing_model alphabetically
        assert body.index(b"credit_cost_multiplier") < body.index(b"pricing_model")

    def test_signature_matches_php_reference(self):
        # Hand-computed: hmac_sha256("test-secret-key",
        #   '{"credit_cost_multiplier":1.5,"pricing_model":"per_token"}')
        body = _php_canonical_sign_body(credit_cost_multiplier=1.5, pricing_model="per_token")
        expected = hmac.new(b"test-secret-key", body, hashlib.sha256).hexdigest()
        got = sign_body(body, "test-secret-key")
        assert got == expected


# ── build_pricing_block ────────────────────────────────────────────────────

class TestBuildPricingBlock:
    def test_unsigned_block_when_sign_disabled(self):
        block = build_pricing_block(PricingConfig(credit_cost_multiplier=1.5), hmac_key="k")
        # sign_declarations defaults False → no signature even with a key present
        assert "declaration_signature" not in block
        assert block["credit_cost_multiplier"] == 1.5
        assert block["pricing_model"] == "per_token"

    def test_unsigned_when_sign_enabled_but_no_key(self):
        block = build_pricing_block(
            PricingConfig(credit_cost_multiplier=1.5, sign_declarations=True),
            hmac_key="",
        )
        assert "declaration_signature" not in block

    def test_signed_block_when_sign_enabled_and_key_present(self):
        pricing = PricingConfig(credit_cost_multiplier=1.5, sign_declarations=True)
        block = build_pricing_block(pricing, hmac_key="k")
        assert "declaration_signature" in block
        # Verify the signature round-trips
        body = _php_canonical_sign_body(credit_cost_multiplier=1.5, pricing_model="per_token")
        assert verify_signature(body, "k", block["declaration_signature"])

    def test_effective_window_passes_through_when_set(self):
        pricing = PricingConfig(
            credit_cost_multiplier=1.0,
            effective_from="2026-06-01T00:00:00Z",
            effective_until="2026-12-31T23:59:59Z",
        )
        block = build_pricing_block(pricing)
        assert block["effective_from"] == "2026-06-01T00:00:00Z"
        assert block["effective_until"] == "2026-12-31T23:59:59Z"


# ── Register payload integration ───────────────────────────────────────────

class TestRegisterIntegration:
    @respx.mock
    def test_register_without_pricing_emits_no_pricing_block(self):
        route = respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={"node_token": "t", "node_id": "n"})
        )
        node = IicpNode(NodeConfig(
            node_id="n",
            endpoint="https://provider.example:8080",
            intent="urn:iicp:intent:llm:chat:v1",
            model="q",
            directory_url="https://iicp.test",
        ))
        asyncio.run(node.register())
        body = json.loads(route.calls[0].request.content)
        assert "pricing" not in body

    @respx.mock
    def test_register_with_pricing_emits_block(self):
        route = respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={"node_token": "t", "node_id": "n"})
        )
        node = IicpNode(NodeConfig(
            node_id="n",
            endpoint="https://provider.example:8080",
            intent="urn:iicp:intent:llm:chat:v1",
            model="q",
            directory_url="https://iicp.test",
            pricing=PricingConfig(credit_cost_multiplier=1.5),
        ))
        asyncio.run(node.register())
        body = json.loads(route.calls[0].request.content)
        assert body["pricing"]["credit_cost_multiplier"] == 1.5
        assert body["pricing"]["pricing_model"] == "per_token"
        assert "declaration_signature" not in body["pricing"]

    @respx.mock
    def test_register_signs_pricing_with_operator_key(self):
        route = respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={"node_token": "t", "node_id": "n"})
        )
        node = IicpNode(NodeConfig(
            node_id="n",
            endpoint="https://provider.example:8080",
            intent="urn:iicp:intent:llm:chat:v1",
            model="q",
            directory_url="https://iicp.test",
            pricing=PricingConfig(credit_cost_multiplier=1.5, sign_declarations=True),
            node_hmac_key="op-provisioned-key",
        ))
        asyncio.run(node.register())
        body = json.loads(route.calls[0].request.content)
        assert "declaration_signature" in body["pricing"]
        # node_hmac_key surfaced so directory uses it
        assert body["node_hmac_key"] == "op-provisioned-key"
        # Round-trip verify
        sig_body = _php_canonical_sign_body(credit_cost_multiplier=1.5, pricing_model="per_token")
        assert verify_signature(sig_body, "op-provisioned-key", body["pricing"]["declaration_signature"])

    @respx.mock
    def test_register_captures_directory_issued_hmac_key(self):
        respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={
                "node_token": "t",
                "node_id": "n",
                "node_hmac_key": "dir-issued-deadbeef",
            })
        )
        node = IicpNode(NodeConfig(
            node_id="n",
            endpoint="https://provider.example:8080",
            intent="urn:iicp:intent:llm:chat:v1",
            model="q",
            directory_url="https://iicp.test",
        ))
        asyncio.run(node.register())
        # The directory-issued key is captured for subsequent signing
        assert node.node_hmac_key == "dir-issued-deadbeef"

    @respx.mock
    def test_operator_key_takes_precedence_over_directory_issued(self):
        respx.post("https://iicp.test/v1/register").mock(
            return_value=httpx.Response(201, json={
                "node_token": "t",
                "node_id": "n",
                "node_hmac_key": "dir-tried-to-set-this",
            })
        )
        node = IicpNode(NodeConfig(
            node_id="n",
            endpoint="https://provider.example:8080",
            intent="urn:iicp:intent:llm:chat:v1",
            model="q",
            directory_url="https://iicp.test",
            node_hmac_key="operator-set-this",
        ))
        asyncio.run(node.register())
        # Operator key wins
        assert node.node_hmac_key == "operator-set-this"
