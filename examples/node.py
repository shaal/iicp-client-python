"""Provider-side example — register and serve one IICP node.

Equivalent to running `iicp-node serve --model qwen2.5:0.5b --backend-url
http://localhost:11434` (the CLI is the recommended path; this example
shows the building blocks if you need a custom handler).

Prereqs:
  - An OpenAI-compatible backend at IICP_BACKEND_URL (Ollama, vLLM, ...)
  - Set IICP_BACKEND_URL + IICP_BACKEND_MODEL env vars, or edit the
    constants below.

Run:

    pip install iicp-client
    IICP_BACKEND_URL=http://localhost:11434 \\
      IICP_BACKEND_MODEL=qwen2.5:0.5b \\
      python examples/node.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from iicp_client import IicpNode, NodeConfig
from iicp_client.backends import openai_compat_handler


async def main() -> None:
    backend_url = os.environ.get("IICP_BACKEND_URL", "http://localhost:11434")
    model = os.environ.get("IICP_BACKEND_MODEL", "qwen2.5:0.5b")
    port = int(os.environ.get("IICP_PORT", "8020"))
    public_endpoint = os.environ.get(
        "IICP_PUBLIC_ENDPOINT", f"http://localhost:{port}"
    )

    cfg = NodeConfig(
        node_id=f"sdk-example-{uuid.uuid4().hex[:8]}",
        endpoint=public_endpoint,
        intent="urn:iicp:intent:llm:chat:v1",
        model=model,
        region="local",
        max_concurrent=4,
    )
    node = IicpNode(cfg)
    handler = openai_compat_handler(backend_url=backend_url, model=model)

    # In production you'd `await node.register()` and pass the returned token
    # to `serve(node_token=...)` so heartbeats fire. Skipped here for the
    # offline example.
    print(f"serving on 127.0.0.1:{port} → {backend_url} ({model})")
    await node.serve(handler, host="127.0.0.1", port=port)


if __name__ == "__main__":
    asyncio.run(main())
