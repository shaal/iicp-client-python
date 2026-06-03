# ADR-016: IICP client SDK conformance
"""Unit tests for the openai_compat backend helper."""

from __future__ import annotations

import base64

import httpx
import respx

from iicp_client.backends import openai_compat_handler

# ── Factory configuration ──────────────────────────────────────────────────


class TestFactoryDefaults:
    def test_returns_callable(self):
        h = openai_compat_handler(base_url="http://localhost:11434/v1", model="qwen2.5:0.5b")
        assert callable(h)


# ── Happy path: chat / completions / embeddings ────────────────────────────


@respx.mock
async def test_chat_completion_happy_path():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG"}}],
            },
        )
    )
    handler = openai_compat_handler(model="qwen2.5:0.5b")
    task = {
        "task_id": "t1",
        "intent": "urn:iicp:intent:llm:chat:v1",
        "payload": {"messages": [{"role": "user", "content": "hi"}]},
    }
    result = await handler(task)
    assert "error_code" not in result
    assert result["result"]["id"] == "chatcmpl-test"
    assert result["result"]["choices"][0]["message"]["content"] == "PONG"


# ── #414 audio:transcribe (STT) — multipart file upload ──────────────────────


@respx.mock
async def test_audio_transcribe_posts_multipart_and_returns_text():
    """audio:transcribe:v1 decodes base64 audio and POSTs it as a multipart file
    upload to /v1/audio/transcriptions (not a JSON body), returning the text.
    Verified end-to-end against whisper.cpp's whisper-server (#414)."""
    route = respx.post("http://localhost:11434/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello world"})
    )
    handler = openai_compat_handler(model="whisper-1")
    task = {
        "task_id": "t-audio",
        "intent": "urn:iicp:intent:audio:transcribe:v1",
        "payload": {
            "audio": base64.b64encode(b"RIFF....fake-wav-bytes").decode(),
            "filename": "clip.wav",
            "language": "en",
        },
    }
    result = await handler(task)
    assert "error_code" not in result, result
    assert result["result"]["text"] == "hello world"
    req = route.calls[0].request
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert b"clip.wav" in req.content  # multipart file part present
    assert b"whisper-1" in req.content  # model sent as a form field


@respx.mock
async def test_audio_transcribe_rejects_invalid_base64():
    handler = openai_compat_handler(model="whisper-1")
    result = await handler(
        {"intent": "urn:iicp:intent:audio:transcribe:v1", "payload": {"audio": "!!not-base64!!"}}
    )
    assert result["error_code"] == 400
    assert "base64" in result["error_message"]


@respx.mock
async def test_audio_transcribe_requires_audio_field():
    handler = openai_compat_handler(model="whisper-1")
    result = await handler(
        {"intent": "urn:iicp:intent:audio:transcribe:v1", "payload": {}}
    )
    assert result["error_code"] == 400
    assert "audio" in result["error_message"]


def test_cli_backend_url_default_is_empty_so_saved_config_applies(monkeypatch):
    """#410 — the --backend-url flag must default to EMPTY (not the Ollama literal)
    so a saved-node config can supply it via `args.backend_url or saved.backend_url`.
    Regression: a non-empty default silently shadowed the saved backend_url."""
    from iicp_client.cli import _build_parser

    monkeypatch.delenv("IICP_BACKEND_URL", raising=False)
    assert _build_parser().parse_args(["serve"]).backend_url == ""
    # explicit flag still wins
    assert _build_parser().parse_args(["serve", "--backend-url", "http://x:1/v1"]).backend_url == "http://x:1/v1"
    # env still honoured
    monkeypatch.setenv("IICP_BACKEND_URL", "http://env:2/v1")
    assert _build_parser().parse_args(["serve"]).backend_url == "http://env:2/v1"


def test_cli_backend_api_key_flag_and_env(monkeypatch):
    """#5 — the serve CLI exposes --backend-api-key, falling back to
    IICP_BACKEND_API_KEY, defaulting to empty (local Ollama needs no key)."""
    from iicp_client.cli import _build_parser

    p = _build_parser()
    monkeypatch.delenv("IICP_BACKEND_API_KEY", raising=False)
    assert p.parse_args(["serve"]).backend_api_key == ""
    assert p.parse_args(["serve", "--backend-api-key", "sk-lm-flag"]).backend_api_key == "sk-lm-flag"
    monkeypatch.setenv("IICP_BACKEND_API_KEY", "sk-lm-env")
    assert _build_parser().parse_args(["serve"]).backend_api_key == "sk-lm-env"


@respx.mock
async def test_backend_api_key_sets_bearer_header():
    """#5 — an auth'd OpenAI-compat backend (LM Studio, hosted) requires a Bearer
    key. When api_key is configured, the request must carry Authorization."""
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = openai_compat_handler(
        base_url="http://localhost:1234/v1",
        model="qwen2.5-coder-14b-instruct-mlx",
        api_key="sk-lm-test",
    )
    await handler(
        {
            "task_id": "t-key",
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
        }
    )
    assert route.called
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-lm-test"


@respx.mock
async def test_no_api_key_sends_no_auth_header():
    """Local Ollama path: no api_key → no Authorization header (back-compat)."""
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = openai_compat_handler(model="qwen2.5:0.5b")
    await handler(
        {
            "task_id": "t-nokey",
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
        }
    )
    assert route.called
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
async def test_factory_model_is_injected_when_payload_missing():
    """Operator instantiates handler with model='qwen2.5:0.5b'; task payload
    doesn't set `model`. Factory default fills in."""
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = openai_compat_handler(model="qwen2.5:0.5b")
    await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    sent_body = route.calls[0].request.read().decode()
    assert "qwen2.5:0.5b" in sent_body


@respx.mock
async def test_task_payload_model_overrides_factory_default():
    """When the task payload sets `model`, the factory default is ignored —
    consumers can override per-call."""
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = openai_compat_handler(model="qwen2.5:0.5b")
    await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": [], "model": "llama-3-8b"},
        }
    )
    sent_body = route.calls[0].request.read().decode()
    assert "llama-3-8b" in sent_body
    assert "qwen2.5:0.5b" not in sent_body


@respx.mock
async def test_completion_intent_routes_to_completions_path():
    route = respx.post("http://localhost:11434/v1/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"text": "PONG"}]})
    )
    handler = openai_compat_handler(model="qwen2.5:0.5b")
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:completion:v1",
            "payload": {"prompt": "ping"},
        }
    )
    assert route.calls.call_count == 1
    assert result["result"]["choices"][0]["text"] == "PONG"


@respx.mock
async def test_embedding_intent_routes_to_embeddings_path():
    respx.post("http://localhost:11434/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
    )
    handler = openai_compat_handler(model="text-embedding-3-small")
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:embedding:v1",
            "payload": {"input": "hello"},
        }
    )
    assert result["result"]["data"][0]["embedding"] == [0.1, 0.2]


# ── Error paths ────────────────────────────────────────────────────────────


async def test_unsupported_intent_returns_400():
    handler = openai_compat_handler(model="q")
    result = await handler({"intent": "urn:iicp:intent:llm:fancy:v1", "payload": {}})
    assert result["error_code"] == 400
    assert "unsupported intent" in result["error_message"]


async def test_no_model_returns_400():
    """No factory default AND no model in task payload → 400."""
    handler = openai_compat_handler(model=None)
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    assert result["error_code"] == 400
    assert "no model" in result["error_message"]


async def test_non_dict_payload_returns_400():
    handler = openai_compat_handler(model="q")
    result = await handler({"intent": "urn:iicp:intent:llm:chat:v1", "payload": "string-not-dict"})
    assert result["error_code"] == 400
    assert "must be a dict" in result["error_message"]


@respx.mock
async def test_upstream_500_is_surfaced():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="model not loaded")
    )
    handler = openai_compat_handler(model="q")
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    assert result["error_code"] == 500
    assert "model not loaded" in result["error_message"]


@respx.mock
async def test_upstream_429_rate_limit_is_surfaced():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="rate limit exceeded")
    )
    handler = openai_compat_handler(model="q")
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    assert result["error_code"] == 429
    assert "rate limit" in result["error_message"]


@respx.mock
async def test_api_key_sets_authorization_header():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(return_value=httpx.Response(200, json={}))
    handler = openai_compat_handler(model="q", api_key="sk-test-1234")
    await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    sent_headers = dict(route.calls[0].request.headers)
    assert sent_headers.get("authorization") == "Bearer sk-test-1234"


@respx.mock
async def test_base_url_trailing_slash_normalized():
    respx.post("http://localhost:11434/v1/chat/completions").mock(return_value=httpx.Response(200, json={}))
    handler = openai_compat_handler(base_url="http://localhost:11434/v1/", model="q")
    result = await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": []},
        }
    )
    assert "error_code" not in result


# ── Dedicated backends (vLLM / llama.cpp) + selector — parity Block B ───────


from iicp_client.backends import (  # noqa: E402
    BACKEND_TYPES,
    get_backend_handler,
    llamacpp_handler,
    vllm_handler,
)


@respx.mock
async def test_vllm_handler_defaults_to_port_8000():
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = vllm_handler(model="mistral-7b")
    result = await handler({"intent": "urn:iicp:intent:llm:chat:v1", "payload": {"messages": []}})
    assert route.calls.call_count == 1
    assert "error_code" not in result


@respx.mock
async def test_llamacpp_handler_defaults_to_port_8080():
    route = respx.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = llamacpp_handler(model="gguf-model")
    result = await handler({"intent": "urn:iicp:intent:llm:chat:v1", "payload": {"messages": []}})
    assert route.calls.call_count == 1
    assert "error_code" not in result


async def test_vllm_error_message_uses_engine_label():
    handler = vllm_handler(model="m")
    result = await handler({"intent": "urn:iicp:intent:bogus:v1", "payload": {}})
    assert result["error_code"] == 400
    assert result["error_message"].startswith("vllm:")


class TestSelector:
    def test_backend_types_lists_all_four(self):
        assert set(BACKEND_TYPES) == {"openai_compat", "vllm", "llamacpp", "anthropic"}

    def test_get_backend_handler_returns_callable(self):
        assert callable(get_backend_handler("vllm", model="m"))
        assert callable(get_backend_handler("llamacpp", model="m"))
        assert callable(get_backend_handler("openai_compat", model="m"))
        assert callable(get_backend_handler("anthropic", model="m"))

    def test_get_backend_handler_unknown_raises(self):
        try:
            get_backend_handler("nope", model="m")
        except ValueError as exc:
            assert "unknown backend_type" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown backend_type")


# ── C1: native Anthropic Messages-API backend (#414) ───────────────────────


from iicp_client.backends import anthropic_handler  # noqa: E402

_ANTHROPIC_OK = {
    "id": "msg_01abc",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [{"type": "text", "text": "PONG"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 11, "output_tokens": 2},
}


@respx.mock
async def test_anthropic_chat_translates_request_and_response():
    """Behavior test (fails without anthropic.py): an llm:chat task must POST to
    /messages with x-api-key + anthropic-version + a defaulted max_tokens, hoist the
    system message to the top-level `system` param, and the Anthropic response must
    come back as the OpenAI chat-completion shape."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_OK)
    )
    handler = anthropic_handler(model="claude-opus-4-8", api_key="sk-ant-test")
    result = await handler(
        {
            "task_id": "t-ant",
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "ping"},
                ]
            },
        }
    )
    assert route.called
    req = route.calls.last.request
    assert req.headers["x-api-key"] == "sk-ant-test"
    assert req.headers["anthropic-version"] == "2023-06-01"
    import json as _json

    body = _json.loads(req.read())
    assert body["system"] == "Be terse."  # system hoisted out of messages
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] == 4096  # defaulted (Anthropic requires it)
    # response mapped to OpenAI chat shape
    out = result["result"]
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "PONG"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}


@respx.mock
async def test_anthropic_maps_openai_image_block_to_anthropic_source():
    """A vision chat (OpenAI image_url data-URL) must become an Anthropic base64
    image block — first-class multimodal Claude, not the audio-stripping shim."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_OK)
    )
    handler = anthropic_handler(model="claude-opus-4-8", api_key="k")
    await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        ],
                    }
                ]
            },
        }
    )
    import json as _json

    blocks = _json.loads(route.calls.last.request.read())["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this?"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


@respx.mock
async def test_anthropic_max_tokens_passthrough_when_set():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_OK)
    )
    handler = anthropic_handler(model="claude-opus-4-8", api_key="k")
    await handler(
        {
            "intent": "urn:iicp:intent:llm:chat:v1",
            "payload": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256, "stop": "END"},
        }
    )
    import json as _json

    body = _json.loads(route.calls.last.request.read())
    assert body["max_tokens"] == 256
    assert body["stop_sequences"] == ["END"]


async def test_anthropic_rejects_non_chat_intent():
    """Anthropic Messages serves only chat — embedding/completion must 400."""
    handler = anthropic_handler(model="claude-opus-4-8", api_key="k")
    result = await handler({"intent": "urn:iicp:intent:llm:embedding:v1", "payload": {"input": "x"}})
    assert result["error_code"] == 400
    assert "only" in result["error_message"]


@respx.mock
async def test_anthropic_upstream_error_surfaced():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(401, text='{"error":{"type":"authentication_error"}}')
    )
    handler = anthropic_handler(model="claude-opus-4-8", api_key="bad")
    result = await handler(
        {"intent": "urn:iicp:intent:llm:chat:v1", "payload": {"messages": [{"role": "user", "content": "hi"}]}}
    )
    assert result["error_code"] == 401
    assert "authentication_error" in result["error_message"]
