# Phase 2 (#529/#55): re-register sends current_node_token ownership proof
"""The register payload must include current_node_token when a prior token is
held (re-registration), and omit it on a fresh register."""

from __future__ import annotations

import httpx
import pytest
import respx

from iicp_client import IicpNode, NodeConfig


def _cfg() -> NodeConfig:
    return NodeConfig(
        node_id="n-reg",
        endpoint="https://node.example.com",
        intent="urn:iicp:intent:llm:chat:v1",
        model="llama-3-8b",
        region="eu-central",
        directory_url="https://iicp.test/api",
    )


@respx.mock
@pytest.mark.asyncio
async def test_fresh_register_omits_current_node_token():
    route = respx.post("https://iicp.test/api/v1/register").mock(
        return_value=httpx.Response(201, json={"node_token": "tok-new", "node_id": "n-reg"})
    )
    node = IicpNode(_cfg())
    await node.register()
    body = httpx.Request("POST", "x", content=route.calls[0].request.content).read()
    import json

    assert "current_node_token" not in json.loads(body)


@respx.mock
@pytest.mark.asyncio
async def test_reregister_sends_current_node_token():
    route = respx.post("https://iicp.test/api/v1/register").mock(
        return_value=httpx.Response(201, json={"node_token": "tok-new", "node_id": "n-reg"})
    )
    node = IicpNode(_cfg())
    node._node_token = "tok-prior"  # simulate a cached token from a prior run
    await node.register()
    import json

    payload = json.loads(route.calls[0].request.content)
    assert payload["current_node_token"] == "tok-prior"
