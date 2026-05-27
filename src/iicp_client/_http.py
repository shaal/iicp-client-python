"""Internal HTTP helpers — TLS context, timeout normalization."""

from __future__ import annotations

import secrets
import ssl
import time
from typing import Any

import httpx

from iicp_client.errors import IicpError, from_http


def _traceparent() -> str:
    """Generate a W3C traceparent header value (SDK-06).

    Format: 00-<trace-id>-<parent-id>-01
    trace-id  = 16 random bytes as 32 hex chars
    parent-id =  8 random bytes as 16 hex chars
    flags     = 01 (sampled)
    """
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def _tls_context(verify: bool) -> ssl.SSLContext | bool:
    if not verify:
        # SDK-05: tls_verify=False only permitted in debug; prod builds must verify
        return False
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_default_certs()
    return ctx


async def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_ms: int = 5_000,
    component: str = "directory",
    tls_verify: bool = True,
    traceparent: str | None = None,
) -> dict[str, Any]:
    timeout = timeout_ms / 1000.0
    headers = {"traceparent": traceparent or _traceparent()}
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_tls_context(tls_verify)) as client:
            resp = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException:
        raise IicpError(
            code="IICP-E003",
            message=f"Request to {url} timed out after {timeout_ms}ms",
            component=component,
            retryable=True,
        ) from None
    except httpx.RequestError as exc:
        raise IicpError(
            code="IICP-E004",
            message=f"Network error reaching {url}: {exc}",
            component=component,
            retryable=True,
        ) from exc
    if not resp.is_success:
        raise from_http(resp.status_code, _safe_json(resp), component)
    return resp.json()


async def post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout_ms: int = 30_000,
    component: str = "adapter",
    tls_verify: bool = True,
    traceparent: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Returns (response_body, elapsed_ms)."""
    timeout = (timeout_ms / 1000.0) + 2.0
    headers = {"traceparent": traceparent or _traceparent()}
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_tls_context(tls_verify)) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        raise IicpError(
            code="IICP-E003",
            message=f"Request to {url} timed out after {timeout_ms}ms",
            component=component,
            retryable=True,
        ) from None
    except httpx.RequestError as exc:
        raise IicpError(
            code="IICP-E004",
            message=f"Network error reaching {url}: {exc}",
            component=component,
            retryable=True,
        ) from exc
    elapsed = int((time.monotonic() - t0) * 1000)
    if not resp.is_success:
        raise from_http(resp.status_code, _safe_json(resp), component)
    return resp.json(), elapsed


def _safe_json(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"message": resp.text[:200]}
