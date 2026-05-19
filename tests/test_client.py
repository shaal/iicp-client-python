"""Unit tests for IicpClient (ADR-016 SDK-01..SDK-06)."""
from __future__ import annotations

import pytest
import respx
import httpx

from iicp_client import (
    IicpClient,
    IicpError,
    ClientConfig,
    ChatMessage,
    ChatOptions,
    TaskRequest,
    TaskConstraints,
    TaskAuth,
    DiscoverOptions,
)


DIRECTORY = "https://iicp.test"
NODE = "https://node.iicp.test"
DISCOVER_URL = f"{DIRECTORY}/v1/discover"
TASK_URL = f"{NODE}/v1/task"

GOOD_NODES = {
    "nodes": [
        {
            "node_id": "node-abc",
            "endpoint": NODE,
            "score": 0.95,
            "available": True,
            "region": "eu-west",
        }
    ]
}


# ---------------------------------------------------------------------------
# Construction / validation (SDK-03, SDK-04, SDK-05)
# ---------------------------------------------------------------------------


def test_sdk04_rejects_oversized_timeout():
    with pytest.raises(ValueError, match="timeout_ms must be"):
        IicpClient(ClientConfig(timeout_ms=120_001))


def test_sdk03_rejects_invalid_intent_urn():
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    with pytest.raises(IicpError) as exc_info:
        client.submit(
            TaskRequest(intent="bad-intent", payload={})
        )
    assert exc_info.value.code == "IICP-E001"
    assert not exc_info.value.retryable


def test_sdk03_accepts_valid_intent_urn(respx_mock):
    respx_mock.get(DISCOVER_URL).mock(
        return_value=httpx.Response(200, json={"nodes": []})
    )
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    with pytest.raises(IicpError) as exc_info:
        client.submit(
            TaskRequest(intent="urn:iicp:intent:llm:chat:v1", payload={})
        )
    assert exc_info.value.code == "IICP-E006"  # no nodes — URN was valid


# ---------------------------------------------------------------------------
# discover() (happy path + no-node case)
# ---------------------------------------------------------------------------


@respx.mock
def test_discover_returns_node_list():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    result = client.discover("urn:iicp:intent:llm:chat:v1")
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == "node-abc"
    assert result.nodes[0].score == 0.95


@respx.mock
def test_discover_empty_returns_empty_node_list():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json={"nodes": []}))
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    result = client.discover("urn:iicp:intent:llm:chat:v1")
    assert result.nodes == []


# ---------------------------------------------------------------------------
# submit() — SDK-01: retry on transient errors
# ---------------------------------------------------------------------------


@respx.mock
def test_submit_happy_path():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    respx.post(TASK_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "t-123",
                "status": "success",
                "result": {"answer": 42},
                "usage": {"total_tokens": 100},
            },
        )
    )
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    resp = client.submit(
        TaskRequest(
            intent="urn:iicp:intent:llm:chat:v1",
            payload={"messages": []},
        )
    )
    assert resp.status == "success"
    assert resp.result == {"answer": 42}
    assert resp.metrics.node_id == "node-abc"
    assert resp.metrics.tokens_used == 100


@respx.mock
async def test_submit_sdk01_retries_transient(monkeypatch):
    """Transient 503 triggers a retry; second attempt succeeds."""
    async def _noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    respx.post(TASK_URL).mock(
        side_effect=[
            httpx.Response(503, json={"code": "IICP-E005", "message": "overload"}),
            httpx.Response(200, json={"task_id": "t-2", "status": "success", "result": {}, "usage": {}}),
        ]
    )
    client = IicpClient(ClientConfig(directory_url=DIRECTORY, max_retries=3))
    resp = await client.submit_async(
        TaskRequest(intent="urn:iicp:intent:llm:chat:v1", payload={})
    )
    assert resp.status == "success"


@respx.mock
def test_submit_non_retryable_raises_immediately():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    respx.post(TASK_URL).mock(
        return_value=httpx.Response(
            401, json={"code": "IICP-E002", "message": "unauthorized"}
        )
    )
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    with pytest.raises(IicpError) as exc_info:
        client.submit(TaskRequest(intent="urn:iicp:intent:llm:chat:v1", payload={}))
    assert exc_info.value.http_status == 401
    assert not exc_info.value.retryable


# ---------------------------------------------------------------------------
# chat() — SDK-02: OpenAI-compatible output shape
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_sdk02_openai_compat_shape():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    respx.post(TASK_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "t-chat-1",
                "status": "success",
                "result": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    "model": "llama3",
                },
                "usage": {"total_tokens": 8},
            },
        )
    )
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    resp = client.chat(
        [ChatMessage(role="user", content="Hi")],
        ChatOptions(model="llama3"),
    )
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == 8
    assert resp.model == "llama3"
    assert resp.iicp_node_id == "node-abc"


# ---------------------------------------------------------------------------
# SDK-06: node_token must not appear in IicpError message
# ---------------------------------------------------------------------------


@respx.mock
def test_sdk06_node_token_not_in_error():
    respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    respx.post(TASK_URL).mock(
        return_value=httpx.Response(400, json={"message": "bad request"})
    )
    secret = "super-secret-token"
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    with pytest.raises(IicpError) as exc_info:
        client.submit(
            TaskRequest(
                intent="urn:iicp:intent:llm:chat:v1",
                payload={},
                auth=TaskAuth(node_token=secret),
            )
        )
    err = exc_info.value
    assert secret not in err.message
    assert secret not in str(err)
    assert secret not in repr(err)

@respx.mock
def test_discover_passes_min_reputation_and_model():
    """SDK-04 parity: DiscoverOptions.min_reputation + model are sent as query params."""
    route = respx.get(DISCOVER_URL).mock(return_value=httpx.Response(200, json=GOOD_NODES))
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    client.discover(
        "urn:iicp:intent:llm:chat:v1",
        DiscoverOptions(min_reputation=0.7, model="phi3:mini"),
    )
    url = str(route.calls[0].request.url)
    assert "min_reputation=0.7" in url
    assert "model=phi3%3Amini" in url or "model=phi3:mini" in url
