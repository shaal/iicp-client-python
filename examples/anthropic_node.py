"""Provider-side example — serve an IICP node backed by Claude.

Uses the native `anthropic` backend, which speaks the Anthropic Messages API
(POST /v1/messages) directly and translates responses back to the OpenAI
chat-completion shape — so a Claude-backed node looks identical to an
Ollama/vLLM node to any IICP client.

The CLI is the recommended path:

    iicp-node serve \\
      --backend-type anthropic \\
      --backend-api-key "$ANTHROPIC_API_KEY" \\
      --model claude-opus-4-8

This example shows the building blocks if you need a custom handler.

Prereqs:
  - An Anthropic API key in ANTHROPIC_API_KEY.

Run:

    pip install iicp-client
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_node.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from iicp_client import IicpNode, NodeConfig
from iicp_client.backends import anthropic_handler


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("IICP_BACKEND_MODEL", "claude-opus-4-8")
    port = int(os.environ.get("IICP_PORT", "8020"))
    public_endpoint = os.environ.get("IICP_PUBLIC_ENDPOINT", f"http://localhost:{port}")

    cfg = NodeConfig(
        node_id=f"sdk-claude-{uuid.uuid4().hex[:8]}",
        endpoint=public_endpoint,
        intent="urn:iicp:intent:llm:chat:v1",
        model=model,
        region="local",
        max_concurrent=4,
    )
    node = IicpNode(cfg)

    # anthropic_handler defaults base_url to https://api.anthropic.com/v1.
    handler = anthropic_handler(model=model, api_key=api_key)

    # In production you'd `await node.register()` and pass the returned token
    # to `serve(node_token=...)` so heartbeats fire. Skipped here for the
    # offline example.
    print(f"serving on 127.0.0.1:{port} → Anthropic Messages API ({model})")
    await node.serve(handler, host="127.0.0.1", port=port)


if __name__ == "__main__":
    asyncio.run(main())
