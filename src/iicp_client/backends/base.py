# SPDX-License-Identifier: Apache-2.0
"""Shared core for OpenAI-dialect backend handlers.

vLLM, llama.cpp, LM Studio and Ollama all speak the OpenAI `/v1/*` HTTP dialect, so
the request/response plumbing is identical — only the default port and the engine
label in error messages differ. This module hosts that shared plumbing so the
per-engine modules (`openai_compat`, `vllm`, `llamacpp`) stay thin and a new engine
is one factory call, not a copy of the whole handler.

Port of iicp-adapter `backends/{base,vllm,llamacpp,openai_compat}.py` into the SDK's
handler-factory style (tracker iicp.network#340; parity Block B).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Maps IICP intent URN → OpenAI-compatible HTTP path.
INTENT_TO_PATH: dict[str, str] = {
    "urn:iicp:intent:llm:chat:v1": "/chat/completions",
    "urn:iicp:intent:llm:completion:v1": "/completions",
    "urn:iicp:intent:llm:embedding:v1": "/embeddings",
}


def build_openai_dialect_handler(
    *,
    engine: str,
    base_url: str,
    model: str | None,
    api_key: str,
    timeout_s: float,
) -> TaskHandler:
    """Build a TaskHandler that proxies CALLs to an OpenAI-dialect server.

    `engine` is the label used in error messages (e.g. "vllm"). All engines share
    this body; the per-engine modules differ only in their default `base_url`.
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
                    f"{engine}: task.payload must be a dict, got {type(payload).__name__}"
                ),
            }

        path = INTENT_TO_PATH.get(intent)
        if path is None:
            return {
                "error_code": 400,
                "error_message": (
                    f"{engine}: unsupported intent {intent!r}; "
                    f"supported: {sorted(INTENT_TO_PATH.keys())}"
                ),
            }

        # Merge model: explicit task payload field wins; factory default fills in.
        body = dict(payload)
        body.setdefault("model", model)
        if not body.get("model"):
            return {
                "error_code": 400,
                "error_message": (
                    f"{engine}: no model — either pass `model=...` to the backend "
                    "factory or include `model` in the task payload"
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
                r = await client.post(f"{base}{path}", json=body)
        except httpx.TimeoutException:
            return {"error_code": 408, "error_message": f"{engine}: backend timed out"}
        except httpx.HTTPError as exc:
            return {
                "error_code": 502,
                "error_message": f"{engine}: HTTP transport error: {exc}",
            }

        if r.status_code >= 400:
            # Surface the upstream error verbatim — operators usually need the
            # original message (rate-limit, model-not-loaded, etc.)
            return {
                "error_code": r.status_code,
                "error_message": f"{engine}: upstream {r.status_code}: {r.text[:512]}",
            }

        try:
            data = r.json()
        except ValueError as exc:
            return {
                "error_code": 502,
                "error_message": f"{engine}: upstream returned non-JSON: {exc}",
            }

        return {"result": data}

    return handler
