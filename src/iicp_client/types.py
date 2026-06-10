"""Public types for iicp-client (ADR-016 §1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iicp_client.errors import IicpError


@dataclass
class ClientConfig:
    directory_url: str = "https://iicp.network/api"
    region: str | None = None
    timeout_ms: int = 30_000
    max_retries: int = 3
    tls_verify: bool = True
    use_confidentiality: bool = False  # IICP-CX S.16: encrypt payloads when node advertises cx_public_key
    routing_epsilon: float = 0.05  # ε-greedy exploration probability (R4); 0.0 disables
    # Phase 2 (#496): caller's JWT from directory registration; used to acquire consumer tokens.
    node_token: str | None = None


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
    # #488 — requester node identity for self-query neutrality at the directory.
    source_node_id: str | None = None


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
    # ADR-044 — composed health label (healthy/degraded/impaired/critical/offline)
    # and ADR-043 8-category network exposure. Both optional: present only when
    # the directory is on v1.10.0+; None against older directories.
    health_label: str | None = None
    exposure_mode: str | None = None
    # IICP-CX S.16 §3.1 — X25519 public key for E2E payload confidentiality.
    # Present only when the node registered with cx_public_key (v1.10.7+).
    cx_public_key: dict[str, str] | None = None
    # #397 — transport protocols the node speaks (e.g. ["https", "iicp-native"]).
    transport: list[str] | None = None


@dataclass
class NodeList:
    nodes: list[Node]
    query_ms: int
