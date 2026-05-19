# iicp-client · Python SDK

[![CI](https://github.com/RobLe3/iicp-client-python/actions/workflows/ci.yml/badge.svg)](https://github.com/RobLe3/iicp-client-python/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/IICP-v1.5-indigo.svg)](https://iicp.network/spec)
[![PyPI](https://img.shields.io/badge/PyPI-iicp--client-blue?logo=pypi&logoColor=white)](https://pypi.org/project/iicp-client/)

Official Python client library for the [IICP protocol](https://iicp.network) — route AI agent tasks by intent across a self-organising mesh of provider nodes. No central broker. No hardcoded endpoints.

```
urn:iicp:intent:llm:chat:v1  →  discover  →  select  →  submit
```

---

## Install

```bash
pip install iicp-client
```

Requires **Python ≥ 3.11** and [`httpx`](https://www.python-httpx.org/).

---

## Quickstart

```python
import asyncio
from iicp_client import IicpClient, ClientConfig

async def main():
    client = IicpClient(ClientConfig(directory_url="https://iicp.network/api"))

    nodes = await client.discover_async("urn:iicp:intent:llm:chat:v1")
    if not nodes.nodes:
        print("No nodes available")
        return

    response = await client.chat_async(
        node=nodes.nodes[0],
        messages=[{"role": "user", "content": "Hello from IICP!"}],
    )
    print(response.choices[0].message["content"])

asyncio.run(main())
```

A synchronous wrapper is available for scripts and notebooks:

```python
from iicp_client import IicpClient

client   = IicpClient()
nodes    = client.discover("urn:iicp:intent:llm:chat:v1")
response = client.chat(node=nodes.nodes[0], messages=[{"role": "user", "content": "Hi"}])
print(response.choices[0].message["content"])
```

---

## Configuration

```python
from iicp_client import ClientConfig

config = ClientConfig(
    directory_url = "https://iicp.network/api",  # IICP directory
    timeout_ms    = 30_000,                       # max 120 000 (SDK-04)
    region        = "eu-central",                 # prefer nodes in region
    node_token    = "your-token",                 # optional auth token
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `directory_url` | `"https://iicp.network/api"` | IICP directory endpoint |
| `timeout_ms` | `30000` | Request timeout — max 120 000 ms |
| `region` | `None` | Preferred node region |
| `node_token` | `None` | Bearer token for authenticated nodes |

---

## Discover options

```python
from iicp_client import DiscoverOptions

nodes = await client.discover_async(
    "urn:iicp:intent:llm:chat:v1",
    DiscoverOptions(
        region         = "eu-central",
        model          = "phi3:mini",
        min_reputation = 0.7,
        limit          = 5,
    )
)
```

---

## Error handling

```python
from iicp_client import IicpError

try:
    response = await client.submit_async(node, request)
except IicpError as e:
    print(f"[{e.code}] {e.message}  (HTTP {e.status_code})")
```

Error codes match the [IICP error reference](https://iicp.network/docs/error-reference) — e.g. `task_timeout`, `capacity_exceeded`, `no_nodes_available`.

---

## SDK conformance

| Rule | Description | Status |
|------|-------------|--------|
| SDK-01 | discover → select → submit pipeline with node retry | ✓ |
| SDK-02 | `task_id` auto-generated (UUID v4) | ✓ |
| SDK-03 | Intent URN pattern validation | ✓ |
| SDK-04 | `timeout_ms` capped at 120 000 ms | ✓ |
| SDK-05 | Retry on 429 / 503 with exponential back-off | ✓ |
| SDK-06 | W3C `traceparent` propagation | ✓ |

Conformance tier: `iicp:sdk:v1` (spec S.14) · [Request a badge](https://iicp.network/conformance)

---

## Development

```bash
pip install -e ".[dev]"   # install with dev deps
pytest tests/ -v          # run 10 unit tests
ruff check src tests       # lint
```

---

## Links

- [Protocol spec](https://iicp.network/spec) — full IICP specification
- [Node setup guide](https://iicp.network/docs/node-setup) — run your own node
- [Error reference](https://iicp.network/docs/error-reference) — all error codes
- [iicp-client-typescript](https://github.com/RobLe3/iicp-client-typescript) — TypeScript SDK
- [iicp-client-rust](https://github.com/RobLe3/iicp-client-rust) — Rust SDK

---

Apache 2.0 · [iicp.network](https://iicp.network)
