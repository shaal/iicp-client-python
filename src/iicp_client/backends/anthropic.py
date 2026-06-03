# SPDX-License-Identifier: Apache-2.0
"""Native Anthropic Messages-API backend helper.

Unlike `openai_compat`/`vllm`/`llamacpp` (which share the OpenAI `/v1/*` dialect via
`base.build_openai_dialect_handler`), Anthropic speaks the **Messages API**
(`POST /v1/messages`): a top-level `system` string instead of a system-role message,
`x-api-key` + `anthropic-version` headers instead of a bearer token, a **required**
`max_tokens`, and `content` blocks instead of OpenAI's `message.content`.

This handler translates an IICP `llm:chat:v1` task (OpenAI chat shape, the dialect the
rest of the SDK already speaks) → an Anthropic Messages request, then translates the
response **back** to the OpenAI chat-completion shape — so a Claude-backed node looks
identical to an Ollama/vLLM node to any IICP client. First-class Claude support
(prompt caching, native content blocks) without the lossy OpenAI-compat shim, which
strips audio and disables caching.

Capability roadmap C1 (reports/capability-gaps-implementation-plan-2026-06-03.md;
research #414). No new dependency — plain `httpx`, already a backend dep.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from iicp_client.backends.base import TaskHandler

logger = logging.getLogger(__name__)

_CHAT_INTENT = "urn:iicp:intent:llm:chat:v1"
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


def _to_anthropic_content(content: Any) -> Any:
    """Translate one OpenAI message `content` into Anthropic content.

    A plain string passes through unchanged. A list of OpenAI content parts is
    mapped block-by-block: `text` → `{type:text}`; `image_url` → an Anthropic image
    block (base64 from a `data:` URL, else a `url` source for remote images).
    Unknown parts are dropped with a debug log rather than failing the request.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # data:<media_type>;base64,<data>
                try:
                    header, b64 = url.split(",", 1)
                    media_type = header.split(";")[0][len("data:") :] or "image/png"
                except ValueError:
                    logger.debug("anthropic: malformed data URL in image_url; skipping")
                    continue
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    }
                )
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            logger.debug("anthropic: dropping unsupported content part %r", ptype)
    return blocks


def _to_anthropic_request(payload: dict[str, Any], model: str | None, default_max_tokens: int) -> dict[str, Any]:
    """Translate an OpenAI chat payload → an Anthropic Messages request body.

    System-role messages are hoisted into the top-level `system` param (Anthropic has
    no system role in `messages`). `max_tokens` is defaulted because Anthropic
    *requires* it (OpenAI treats it as optional). Recognised sampling params are
    carried over; `stop` → `stop_sequences`.
    """
    body: dict[str, Any] = {}
    body["model"] = payload.get("model") or model

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            continue
        messages.append({"role": role, "content": _to_anthropic_content(msg.get("content"))})
    body["messages"] = messages
    if system_parts:
        body["system"] = "\n\n".join(p for p in system_parts if p)

    body["max_tokens"] = int(payload.get("max_tokens") or default_max_tokens)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if payload.get(src) is not None:
            body[dst] = payload[src]
    stop = payload.get("stop")
    if stop is not None:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return body


_STOP_REASON_TO_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _to_openai_response(data: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages response → the OpenAI chat-completion shape.

    Text content blocks are concatenated into a single `message.content` string so
    IICP clients consume the same shape regardless of backend. Usage is renamed
    (`input_tokens`/`output_tokens` → `prompt_tokens`/`completion_tokens`).
    """
    text = "".join(
        b.get("text", "") for b in (data.get("content") or []) if isinstance(b, dict) and b.get("type") == "text"
    )
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "model": data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _STOP_REASON_TO_FINISH.get(data.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def anthropic_handler(
    *,
    base_url: str = "https://api.anthropic.com/v1",
    model: str | None = None,
    api_key: str = "",
    timeout_s: float = 30.0,
    anthropic_version: str = _DEFAULT_ANTHROPIC_VERSION,
    default_max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> TaskHandler:
    """Build a TaskHandler that proxies `llm:chat:v1` CALLs to the Anthropic Messages API.

    Arguments:
        base_url: Anthropic API root (default `https://api.anthropic.com/v1`). Point at
            a llama.cpp server's `/v1` for a local Messages-API backend.
        model: Claude model id (e.g. `claude-opus-4-8`). If None, taken from the task
            payload `model`.
        api_key: Anthropic API key → sent as `x-api-key` (reuses the SDK's backend
            api-key support, #5).
        timeout_s: per-request HTTP timeout.
        anthropic_version: value for the required `anthropic-version` header.
        default_max_tokens: `max_tokens` to send when the task omits it (Anthropic
            requires the field; OpenAI does not).

    Returns:
        An async `(task: dict) -> dict`; success → `{result: <OpenAI-shaped JSON>}`,
        failure → `{error_code, error_message}`. Only `llm:chat:v1` is supported
        (Anthropic has no completion/embedding endpoint).
    """
    base = base_url.rstrip("/")
    headers = {
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key

    async def handler(task: dict[str, Any]) -> dict[str, Any]:
        intent = str(task.get("intent", ""))
        if intent != _CHAT_INTENT:
            return {
                "error_code": 400,
                "error_message": (
                    f"anthropic: unsupported intent {intent!r}; the Messages API serves only {_CHAT_INTENT}"
                ),
            }
        payload = task.get("payload") or {}
        if not isinstance(payload, dict):
            return {
                "error_code": 400,
                "error_message": (f"anthropic: task.payload must be a dict, got {type(payload).__name__}"),
            }

        body = _to_anthropic_request(payload, model, default_max_tokens)
        if not body.get("model"):
            return {
                "error_code": 400,
                "error_message": (
                    "anthropic: no model — pass `model=...` to the backend factory "
                    "or include `model` in the task payload"
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
                r = await client.post(f"{base}/messages", json=body)
        except httpx.TimeoutException:
            return {"error_code": 408, "error_message": "anthropic: backend timed out"}
        except httpx.HTTPError as exc:
            return {
                "error_code": 502,
                "error_message": f"anthropic: HTTP transport error: {exc}",
            }

        if r.status_code >= 400:
            return {
                "error_code": r.status_code,
                "error_message": f"anthropic: upstream {r.status_code}: {r.text[:512]}",
            }

        try:
            data = r.json()
        except ValueError as exc:
            return {
                "error_code": 502,
                "error_message": f"anthropic: upstream returned non-JSON: {exc}",
            }

        return {"result": _to_openai_response(data)}

    return handler
