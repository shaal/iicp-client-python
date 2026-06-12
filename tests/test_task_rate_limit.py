# ADR-016 — F4 (#524): per-origin /v1/task rate limit
"""Unit tests for the node's fixed-window task rate limiter."""

from __future__ import annotations

import os

from iicp_client import IicpNode, NodeConfig


def _node(limit: int) -> IicpNode:
    os.environ["IICP_TASK_RATE_LIMIT"] = str(limit)
    try:
        return IicpNode(
            NodeConfig(
                node_id="rl-test",
                endpoint="https://x.example.com",
                intent="urn:iicp:intent:llm:chat:v1",
                model="llama-3-8b",
                region="eu-central",
                directory_url="https://d.example.com",
            )
        )
    finally:
        os.environ.pop("IICP_TASK_RATE_LIMIT", None)


class TestTaskRateLimit:
    def test_allows_under_limit_then_blocks(self):
        node = _node(3)
        assert node._task_rate_allow("origin-a") is True
        assert node._task_rate_allow("origin-a") is True
        assert node._task_rate_allow("origin-a") is True
        assert node._task_rate_allow("origin-a") is False

    def test_origins_are_independent(self):
        node = _node(1)
        assert node._task_rate_allow("origin-a") is True
        assert node._task_rate_allow("origin-b") is True
        assert node._task_rate_allow("origin-a") is False

    def test_window_resets(self):
        node = _node(1)
        assert node._task_rate_allow("k") is True
        assert node._task_rate_allow("k") is False
        ws, _count = node._task_rate_buckets["k"]
        node._task_rate_buckets["k"] = (ws - node._task_rate_window_s - 1, 1)
        assert node._task_rate_allow("k") is True
