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

import base64
import binascii
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TaskHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]

# #414 — speech-to-text. Multipart file upload, not a JSON body, so it takes a
# distinct code path below.
AUDIO_TRANSCRIBE_INTENT = "urn:iicp:intent:audio:transcribe:v1"
# #414 — text-to-speech. JSON request but a *binary* audio response, so it also
# takes a distinct path (the audio bytes are base64-encoded into the result).
AUDIO_SPEECH_INTENT = "urn:iicp:intent:audio:speech:v1"

# Maps IICP intent URN → OpenAI-compatible HTTP path.
INTENT_TO_PATH: dict[str, str] = {
    "urn:iicp:intent:llm:chat:v1": "/chat/completions",
    "urn:iicp:intent:llm:completion:v1": "/completions",
    "urn:iicp:intent:llm:embedding:v1": "/embeddings",
    AUDIO_TRANSCRIBE_INTENT: "/audio/transcriptions",
    AUDIO_SPEECH_INTENT: "/audio/speech",
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

        # #414 — audio:transcribe is a multipart file upload (OpenAI
        # /v1/audio/transcriptions). IICP tasks are JSON, so the audio rides as
        # base64 in payload["audio"]; model is OPTIONAL (whisper.cpp ignores it,
        # vLLM/OpenAI use it). Distinct from the JSON flow below.
        if intent == AUDIO_TRANSCRIBE_INTENT:
            audio_b64 = payload.get("audio") or payload.get("audio_b64")
            if not isinstance(audio_b64, str) or not audio_b64:
                return {
                    "error_code": 400,
                    "error_message": (
                        f"{engine}: audio:transcribe requires payload.audio "
                        "(base64-encoded audio bytes)"
                    ),
                }
            try:
                audio_bytes = base64.b64decode(audio_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                return {
                    "error_code": 400,
                    "error_message": f"{engine}: payload.audio is not valid base64: {exc}",
                }
            filename = str(payload.get("filename") or "audio.wav")
            form: dict[str, Any] = {"response_format": "json"}
            req_model = payload.get("model") or model
            if req_model:
                form["model"] = req_model
            for opt in ("language", "response_format", "prompt", "temperature"):
                if payload.get(opt) is not None:
                    form[opt] = payload[opt]
            try:
                async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
                    r = await client.post(
                        f"{base}{path}",
                        files={"file": (filename, audio_bytes)},
                        data=form,
                    )
            except httpx.TimeoutException:
                return {"error_code": 408, "error_message": f"{engine}: backend timed out"}
            except httpx.HTTPError as exc:
                return {
                    "error_code": 502,
                    "error_message": f"{engine}: HTTP transport error: {exc}",
                }
            if r.status_code >= 400:
                return {
                    "error_code": r.status_code,
                    "error_message": f"{engine}: upstream {r.status_code}: {r.text[:512]}",
                }
            try:
                data = r.json()
            except ValueError:
                # response_format=text → plain body; normalise to {"text": ...}
                data = {"text": r.text}
            return {"result": data}

        # #414 — audio:speech (TTS): JSON request, but the response is binary audio.
        # We base64-encode the bytes into the result so it rides the JSON task pipe.
        if intent == AUDIO_SPEECH_INTENT:
            text = payload.get("input")
            if not isinstance(text, str) or not text:
                return {
                    "error_code": 400,
                    "error_message": f"{engine}: audio:speech requires payload.input (text to synthesize)",
                }
            speech_body: dict[str, Any] = {"input": text}
            req_model = payload.get("model") or model
            if req_model:
                speech_body["model"] = req_model
            for opt in ("voice", "response_format", "speed"):
                if payload.get(opt) is not None:
                    speech_body[opt] = payload[opt]
            # OpenAI-dialect servers require a voice; default for back-ends that need
            # one (ignored by engines like espeak-ng).
            speech_body.setdefault("voice", "alloy")
            try:
                async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
                    r = await client.post(f"{base}{path}", json=speech_body)
            except httpx.TimeoutException:
                return {"error_code": 408, "error_message": f"{engine}: backend timed out"}
            except httpx.HTTPError as exc:
                return {
                    "error_code": 502,
                    "error_message": f"{engine}: HTTP transport error: {exc}",
                }
            if r.status_code >= 400:
                return {
                    "error_code": r.status_code,
                    "error_message": f"{engine}: upstream {r.status_code}: {r.text[:512]}",
                }
            content_type = r.headers.get("content-type", "audio/mpeg")
            return {
                "result": {
                    "audio": base64.b64encode(r.content).decode(),
                    "content_type": content_type,
                    "format": speech_body.get("response_format")
                    or content_type.split("/")[-1],
                }
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
