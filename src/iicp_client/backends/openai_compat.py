# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible backend helper.

Port of iicp-adapter's `backends/openai_compat.py` (iter-1410) into the
iicp-client-python SDK as part of the adapter→hybrid-client migration
(tracker iicp.network#340 Tier 1 Item 5).

Designed as a callable factory that produces a TaskHandler suitable for
both `IicpNode.serve(handler)` and `IicpTcpServer(handler=...)`. The
returned handler:

  - inspects `task["intent"]` to choose the OpenAI-compatible HTTP path
    (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
  - forwards `task["payload"]` as the JSON body (with `model` defaulted
    when the caller didn't set it)
  - returns the upstream JSON as the result dict

Drives Ollama (port 11434), vLLM, LM Studio, llama-cpp-server, or any
other OpenAI-compatible HTTP server. The provider URL is the operator's
choice — by default `http://localhost:11434/v1` for Ollama.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Maps IICP intent URN → OpenAI-compatible HTTP path.
_INTENT_TO_PATH: dict[str, str] = {
    "urn:iicp:intent:llm:chat:v1": "/chat/completions",
    "urn:iicp:intent:llm:completion:v1": "/completions",
    "urn:iicp:intent:llm:embedding:v1": "/embeddings",
}


def openai_compat_handler(
    *,
    base_url: str = "http://localhost:11434/v1",
    model: str | None = None,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build a TaskHandler that proxies CALLs to an OpenAI-compatible server.

    Arguments:
        base_url: provider HTTP root (default Ollama's `http://localhost:11434/v1`).
        model: model name. If None, the handler trusts `task["payload"]["model"]`
            and returns an error_code when neither is set.
        api_key: bearer token for the provider's `/v1/*` routes. Empty for
            local Ollama / vLLM; required for OpenAI/Together/Anyscale.
        timeout_s: per-request HTTP timeout.

    Returns:
        An async callable `(task: dict) -> dict` where `task` has shape
        `{task_id, intent, payload}` and the return dict has shape
        `{result: <upstream JSON>}` on success OR
        `{error_code: int, error_message: str}` on failure.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def handler(task: dict[str, Any]) -> dict[str, Any]:
        intent = str(task.get("intent", ""))
        payload = task.get("payload") or {}
        if not isinstance(payload, dict):
            return {
                "error_code": 400,
                "error_message": (
                    f"openai_compat: task.payload must be a dict, got {type(payload).__name__}"
                ),
            }

        path = _INTENT_TO_PATH.get(intent)
        if path is None:
            return {
                "error_code": 400,
                "error_message": (
                    f"openai_compat: unsupported intent {intent!r}; "
                    f"supported: {sorted(_INTENT_TO_PATH.keys())}"
                ),
            }

        # Merge model: explicit task payload field wins; factory default fills in.
        body = dict(payload)
        body.setdefault("model", model)
        if not body.get("model"):
            return {
                "error_code": 400,
                "error_message": (
                    "openai_compat: no model — either pass `model=...` to "
                    "openai_compat_handler(...) or include `model` in the task payload"
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
                r = await client.post(f"{base}{path}", json=body)
        except httpx.TimeoutException:
            return {"error_code": 408, "error_message": "openai_compat: backend timed out"}
        except httpx.HTTPError as exc:
            return {
                "error_code": 502,
                "error_message": f"openai_compat: HTTP transport error: {exc}",
            }

        if r.status_code >= 400:
            # Surface the upstream error verbatim — operators usually need the
            # original message (rate-limit, model-not-loaded, etc.)
            return {
                "error_code": r.status_code,
                "error_message": f"openai_compat: upstream {r.status_code}: {r.text[:512]}",
            }

        try:
            data = r.json()
        except ValueError as exc:
            return {
                "error_code": 502,
                "error_message": f"openai_compat: upstream returned non-JSON: {exc}",
            }

        return {"result": data}

    return handler
