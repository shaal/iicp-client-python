"""Consumer-side example — submit one chat task to the IICP mesh.

The IicpClient discovers a node that serves the intent and POSTs the task
via HTTP /v1/task. No protocol details needed at this layer.

Run:

    pip install iicp-client
    python examples/consumer.py
"""

from __future__ import annotations

import asyncio

from iicp_client import IicpClient


async def main() -> None:
    async with IicpClient() as client:
        # `chat()` discovers + selects the best node + submits the task.
        reply = await client.chat(
            [{"role": "user", "content": "Hi! Tell me one IICP fun fact."}]
        )
        print(reply)


if __name__ == "__main__":
    asyncio.run(main())
