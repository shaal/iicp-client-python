"""Public types for iicp-client (ADR-016 §1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iicp_client.errors import IicpError


@dataclass
class ClientConfig:
    directory_url: str = "https://iicp.network"
    region: str | None = None
    timeout_ms: int = 30_000
    max_retries: int = 3
    tls_verify: bool = True


@dataclass
class TaskConstraints:
    timeout_ms: int = 30_000
    qos: str = "interactive"
    region: str | None = None


@dataclass
class TaskAuth:
    node_token: str | None = None


@dataclass
class TaskRequest:
    intent: str
    payload: dict[str, Any]
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    auth: TaskAuth = field(default_factory=TaskAuth)


@dataclass
class TaskMetrics:
    latency_ms: int
    tokens_used: int | None
    node_id: str


@dataclass
class TaskResponse:
    task_id: str
    status: str
    result: dict[str, Any] | None
    metrics: TaskMetrics
    error: IicpError | None = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatOptions:
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_ms: int | None = None
    qos: str = "interactive"
    node_token: str | None = None


@dataclass
class ChatChoice:
    message: ChatMessage
    finish_reason: str


@dataclass
class ChatUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    id: str
    choices: list[ChatChoice]
    usage: ChatUsage
    model: str
    iicp_node_id: str


@dataclass
class DiscoverOptions:
    region: str | None = None
    qos: str | None = None
    min_reputation: float | None = None
    model: str | None = None
    limit: int = 10


@dataclass
class Node:
    node_id: str
    endpoint: str
    score: float
    available: bool
    region: str
    latency_estimate_ms: int | None = None
    reputation_score: float | None = None


@dataclass
class NodeList:
    nodes: list[Node]
    query_ms: int
