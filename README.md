# iicp-client · Python SDK

[![CI](https://github.com/RobLe3/iicp-client-python/actions/workflows/ci.yml/badge.svg)](https://github.com/RobLe3/iicp-client-python/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/IICP-v1.7-indigo.svg)](https://iicp.network/spec)
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

## Architecture — consumer or provider?

This SDK covers **both** sides of the IICP protocol:

| Role | What you do | Class |
|------|-------------|-------|
| **Consumer** | Send AI tasks to the mesh; discover and submit | `IicpClient` |
| **Provider** | Run a node, register with the directory, serve tasks | `IicpNode` |

Consumer and provider can run in the same process. A node that serves requests can also route tasks it can't handle to other mesh nodes (`IicpClient` inside the task handler).

For production provider nodes backed by Ollama/vLLM, the `iicp-node` binary (Rust) and the Python adapter (`pip install iicp-adapter`) provide additional resilience and monitoring. See [iicp.network/docs/node-setup](https://iicp.network/docs/node-setup).

---

## Quickstart

```python
import asyncio
from iicp_client import IicpClient, ChatMessage

async def main():
    client = IicpClient()

    # chat_async discovers, selects best node, and submits in one call
    response = await client.chat_async(
        messages=[ChatMessage(role="user", content="Hello from IICP!")],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

Synchronous wrapper for scripts and notebooks:

```python
from iicp_client import IicpClient, ChatMessage

client   = IicpClient()
response = client.chat([ChatMessage(role="user", content="Hello from IICP!")])
print(response.choices[0].message.content)
```

---

## Configuration

```python
from iicp_client import ClientConfig

config = ClientConfig(
    directory_url = "https://iicp.network",  # IICP directory
    timeout_ms    = 30_000,                  # max 120 000 (SDK-04)
    region        = "eu-central",            # prefer nodes in region
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `directory_url` | `"https://iicp.network"` | IICP directory endpoint |
| `timeout_ms` | `30000` | Request timeout — max 120 000 ms |
| `region` | `None` | Preferred node region |
| `max_retries` | `3` | Retry count for transient errors |

---

## Discover options

```python
from iicp_client import DiscoverOptions

node_list = await client.discover_async(
    "urn:iicp:intent:llm:chat:v1",
    DiscoverOptions(
        region         = "eu-central",
        model          = "phi3:mini",
        min_reputation = 0.7,
        limit          = 5,
    )
)
nodes = node_list.nodes  # list of Node objects
```

---

## Error handling

```python
from iicp_client import IicpClient, IicpError, ChatMessage

client = IicpClient()
try:
    response = client.chat([ChatMessage(role="user", content="hi")])
except IicpError as e:
    print(f"[{e.code}] {e.message}  (HTTP {e.http_status})")
```

Error codes match the [IICP error reference](https://iicp.network/docs/error-reference) — e.g. `task_timeout`, `capacity_exceeded`, `no_nodes_available`.

---

## Serving as a provider node

```python
import asyncio
from iicp_client import IicpNode, NodeConfig

async def my_handler(task):
    return {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}

async def main():
    node = IicpNode(NodeConfig(
        node_id="my-node-001",
        endpoint="http://my.public.host:8020",
        intent="urn:iicp:intent:llm:chat:v1",
        model="llama3:8b",
    ))
    token = await node.register()
    stop = node.serve(my_handler, port=8020, node_token=token)
    try:
        await asyncio.Event().wait()  # run until stopped
    finally:
        stop()

asyncio.run(main())
```

### Listen port — default 9484, auto-increment (v0.7.5+)

The official IICP port **9484** is the default listen port (`IICP_PORT`, `--port`).
The `iicp-node` CLI auto-increments to the next free port when 9484 is already in
use, so you can run several nodes on one host without picking ports by hand — the
first binds 9484, the second 9485, the third 9486, and so on. Each node gets its
own port, hence its own NAT pinhole; multiple models served by one node share that
single port. Auto-increment is skipped when you pass an explicit `--public-endpoint`
(you own the port mapping in that case). `IicpNode.serve(port=…)` uses the port you
give it as-is (no auto-increment at the library level).

---

## NAT traversal — automatic (v0.7.3+)

Since v0.7.3, NAT detection runs automatically on every node startup — no flags needed.
The SDK tries each path in order and picks the best one for your network:

| Tier | When | What happens |
|------|------|-------------|
| **0** | VPS/cloud (public IP on NIC) or `IICP_PUBLIC_ENDPOINT` set | Registers directly with that IP |
| **1a** | Home router with UPnP, no CGNAT | Opens a port-forward via UPnP → registers WAN IP |
| **1b** | CGNAT + IPv6 available + AddPinhole works | Registers IPv6 address with firewall rule |
| **1c** | CGNAT + IPv6 + AddPinhole fails (e.g. FRITZ!Box error 606) | Registers IPv6 GUA anyway + logs guidance |
| **3** | CGNAT + no usable IPv6 | Auto-elects relay from directory → registers via relay |
| **4** | Nothing worked | Serves locally with operator guidance |

### Environment-specific behaviour

**VPS / bare metal** — no action needed. The SDK detects the public IP on the NIC (Tier 0).

**Home router (no CGNAT)** — UPnP opens a port-forward automatically. One pinhole per port,
so three nodes on ports 8020 / 8024 / 8025 open three pinholes.

**CGNAT (carrier-grade NAT, e.g. NetCologne DSLite)** — IPv4 path is blocked by the ISP.
The SDK tries IPv6 instead. If your FRITZ!Box rejects `AddPinhole` with error 606, the SDK
still advertises your IPv6 address (many clients can reach it via stateful firewall) and logs:

```
WARNING: NAT: IPv6 endpoint http://[2a0a:...]:8020 advertised but firewall pinhole
could not be opened. Open manually: FRITZ!Box → Network → Firewall → IPv6.
Alternatively use IICP_RELAY_WORKER_ENDPOINT for relay-as-last-resort fallback.
```

**Docker bridge (`-p 8020:8020`)** — UPnP is skipped (it would reach the Docker NAT, not
your home router). Set `IICP_PUBLIC_ENDPOINT` so the node knows its real address:

```yaml
# docker-compose.yml
environment:
  IICP_PUBLIC_ENDPOINT: "http://your-host-ip:8020"
  IICP_BACKEND_URL: "http://host.docker.internal:11434"
```

Or run with `--network host` to let UPnP work as on bare metal.

**Kubernetes** — set `IICP_PUBLIC_ENDPOINT` to the Service IP or external LoadBalancer:

```yaml
env:
  - name: IICP_PUBLIC_ENDPOINT
    value: "http://$(LOAD_BALANCER_IP):8020"
```

### CGNAT + no IPv6 → automatic relay

When no direct path is possible, the SDK automatically finds a relay:

```
NAT tier=3: no direct or IPv6 endpoint available.
Auto-electing relay from directory...
Auto-elected relay: relay.example.com:9485
```

The node connects outbound to the elected relay, which forwards inbound tasks down the
tunnel. Re-registration happens automatically when the relay bind succeeds.

To use a specific relay instead of auto-electing:
```bash
IICP_RELAY_WORKER_ENDPOINT=relay.example.com:9485 python -m iicp_client.cli serve ...
```

### Running a relay-capable node (relay operators)

```python
node = IicpNode(NodeConfig(
    endpoint="http://relay.example.com:8020",
    intent="urn:iicp:intent:llm:chat:v1",
    relay_capable=True,      # accept RELAY_BIND on TCP port 9485
    relay_accept_port=9485,
    enable_mesh=True,        # advertise relay_capable=True in gossip
))
```

### Opt-out / override

```bash
IICP_AUTO_DETECT_NAT=false   # disable NAT detection entirely
IICP_PUBLIC_ENDPOINT=http://x.x.x.x:8020   # trust this endpoint, skip detection
IICP_EXTERNAL_IP_PROBE_URL=https://api.ipify.org  # WAN IP probe (default)
```

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
pytest tests/ -v          # run 213 unit tests
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
