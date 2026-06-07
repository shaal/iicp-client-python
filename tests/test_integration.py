# SPDX-License-Identifier: Apache-2.0
"""Live-mesh integration tests (#2) — OPT-IN.

The rest of the suite mocks the directory with respx; nothing exercises a REAL IICP node.
These do, against the live mesh, and are skipped unless explicitly enabled:

    IICP_INTEGRATION_TEST=1   → discover (read-only, safe)
    IICP_INTEGRATION_CHAT=1   → submit a real task to a live operator's node

Override the directory with IICP_DIRECTORY_URL. Unblocked once a node registered a routable
public endpoint (W-011 resolved; an external operator runs https://iicp.shaal.dev).
"""
import os

import pytest

from iicp_client import ChatMessage, ChatOptions, ClientConfig, IicpClient

DIRECTORY = os.getenv("IICP_DIRECTORY_URL", "https://iicp.network/api")
CHAT_INTENT = "urn:iicp:intent:llm:chat:v1"


@pytest.mark.skipif(
    not os.getenv("IICP_INTEGRATION_TEST"),
    reason="set IICP_INTEGRATION_TEST=1 to run against the live IICP mesh",
)
@pytest.mark.asyncio
async def test_live_discover():
    """discover() against the production directory returns at least one routable node."""
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    nodes = await client.discover_async(CHAT_INTENT)
    assert len(nodes.nodes) > 0, "live directory returned no chat nodes"
    assert nodes.nodes[0].endpoint.startswith("http"), (
        f"node endpoint is not routable: {nodes.nodes[0].endpoint!r}"
    )


@pytest.mark.skipif(
    not os.getenv("IICP_INTEGRATION_CHAT"),
    reason="set IICP_INTEGRATION_CHAT=1 to send a real task to a live operator's node",
)
@pytest.mark.asyncio
async def test_live_chat():
    """chat() routes a real task to a live node and gets a non-empty reply."""
    client = IicpClient(ClientConfig(directory_url=DIRECTORY))
    resp = await client.chat_async(
        [ChatMessage(role="user", content="Reply with the single word: OK")],
        ChatOptions(max_tokens=16),
    )
    assert resp.choices, "chat response had no choices"
    assert resp.choices[0].message.content, "chat reply was empty"
