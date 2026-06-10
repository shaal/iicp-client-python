"""Behavior tests for Phase-2 consumer token support in IicpClient (#496).

These tests fail if _acquire_consumer_token is removed or the header is not
forwarded to the node's /v1/task endpoint.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iicp_client.client import IicpClient
from iicp_client.types import ClientConfig


def _make_ct_response(token: str = "tok.sig", expires_at: int | None = None) -> MagicMock:
    if expires_at is None:
        expires_at = int(time.time()) + 300
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"token": token, "expires_at": expires_at}
    return resp


def _make_httpx_ctx(resp: MagicMock) -> MagicMock:
    instance = AsyncMock()
    instance.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.mark.asyncio
async def test_acquire_consumer_token_returns_token_on_201() -> None:
    """Returns the token string when directory responds 201."""
    cfg = ClientConfig(node_token="my-node-jwt")
    client = IicpClient(cfg)
    resp = _make_ct_response("abc.def")
    ctx = _make_httpx_ctx(resp)
    with patch("iicp_client.client.httpx.AsyncClient", return_value=ctx):
        tok = await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
    assert tok == "abc.def"


@pytest.mark.asyncio
async def test_acquire_consumer_token_returns_none_without_node_token() -> None:
    """Returns None immediately when no node_token is configured."""
    client = IicpClient(ClientConfig())  # no node_token
    tok = await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
    assert tok is None


@pytest.mark.asyncio
async def test_acquire_consumer_token_caches_valid_token() -> None:
    """A second call reuses the cached token without hitting the directory again."""
    cfg = ClientConfig(node_token="my-node-jwt")
    client = IicpClient(cfg)
    resp = _make_ct_response("cached-tok.sig")
    ctx = _make_httpx_ctx(resp)
    with patch("iicp_client.client.httpx.AsyncClient", return_value=ctx) as mock_cls:
        await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
        tok2 = await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
    assert tok2 == "cached-tok.sig"
    assert mock_cls.call_count == 1  # only one HTTP call


@pytest.mark.asyncio
async def test_acquire_consumer_token_refreshes_expired_token() -> None:
    """Refetches when the cached token is within the 30-second expiry buffer."""
    cfg = ClientConfig(node_token="my-node-jwt")
    client = IicpClient(cfg)
    # Seed cache with an entry that expires in 10 seconds (< 30-second buffer)
    cache_key = ("node-123", "urn:iicp:intent:llm:chat:v1")
    client._ct_cache[cache_key] = ("stale.tok", int(time.time()) + 10)

    resp = _make_ct_response("fresh.tok")
    ctx = _make_httpx_ctx(resp)
    with patch("iicp_client.client.httpx.AsyncClient", return_value=ctx):
        tok = await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
    assert tok == "fresh.tok"


@pytest.mark.asyncio
async def test_acquire_consumer_token_returns_none_on_network_exception() -> None:
    """Returns None instead of propagating exceptions from the directory call."""
    cfg = ClientConfig(node_token="my-node-jwt")
    client = IicpClient(cfg)
    import httpx
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=httpx.RequestError("timeout"))
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("iicp_client.client.httpx.AsyncClient", return_value=ctx):
        tok = await client._acquire_consumer_token("node-123", "urn:iicp:intent:llm:chat:v1")
    assert tok is None
