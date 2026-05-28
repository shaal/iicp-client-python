# ADR-016: IICP client SDK conformance
"""Unit tests for the openai_compat backend helper."""

from __future__ import annotations

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
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
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
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
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
    result = await handler(
        {"intent": "urn:iicp:intent:llm:chat:v1", "payload": {"messages": []}}
    )
    assert route.calls.call_count == 1
    assert "error_code" not in result


@respx.mock
async def test_llamacpp_handler_defaults_to_port_8080():
    route = respx.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    handler = llamacpp_handler(model="gguf-model")
    result = await handler(
        {"intent": "urn:iicp:intent:llm:chat:v1", "payload": {"messages": []}}
    )
    assert route.calls.call_count == 1
    assert "error_code" not in result


async def test_vllm_error_message_uses_engine_label():
    handler = vllm_handler(model="m")
    result = await handler({"intent": "urn:iicp:intent:bogus:v1", "payload": {}})
    assert result["error_code"] == 400
    assert result["error_message"].startswith("vllm:")


class TestSelector:
    def test_backend_types_lists_all_three(self):
        assert set(BACKEND_TYPES) == {"openai_compat", "vllm", "llamacpp"}

    def test_get_backend_handler_returns_callable(self):
        assert callable(get_backend_handler("vllm", model="m"))
        assert callable(get_backend_handler("llamacpp", model="m"))
        assert callable(get_backend_handler("openai_compat", model="m"))

    def test_get_backend_handler_unknown_raises(self):
        try:
            get_backend_handler("nope", model="m")
        except ValueError as exc:
            assert "unknown backend_type" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown backend_type")
