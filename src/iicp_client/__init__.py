"""iicp-client — Official Python client SDK for the IICP protocol."""

from iicp_client.availability import AvailabilityEvaluator, Window
from iicp_client.backends import (
    BACKEND_TYPES,
    get_backend_handler,
    llamacpp_handler,
    openai_compat_handler,
    vllm_handler,
)
from iicp_client.cip_policy import (
    CooperativeInferencePolicy,
)
from iicp_client.cip_policy import (
    configure_policy as configure_cip_policy,
)
from iicp_client.cip_policy import (
    get_policy as get_cip_policy,
)
from iicp_client.client import IicpClient
from iicp_client.concurrency import CapacityExceededError, ConcurrencyGate
from iicp_client.conformance import ConformanceReport, ProbeResult, run_conformance_checks
from iicp_client.errors import IicpError
from iicp_client.idempotency import IdempotencyGuard
from iicp_client.iicp_tcp import IicpTcpClient, IicpTcpClientError, IicpTcpServer, MsgType
from iicp_client.nat_detection import (
    NatProfile,
    delete_ipv6_pinhole,
    detect_nat,
    renew_ipv6_pinhole,
)
from iicp_client.node import IicpNode, NodeConfig
from iicp_client.otel_tracer import task_execute_span, task_validate_span
from iicp_client.peer_manager import PeerManager
from iicp_client.pricing import PricingConfig, build_pricing_block, sign_body, verify_signature
from iicp_client.scheduler import is_queue_eligible, qos_priority
from iicp_client.token_validator import TokenValidator
from iicp_client.trust_auditor import AuditReport, models_diverge, run_audit_pass
from iicp_client.types import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ClientConfig,
    DiscoverOptions,
    NodeList,
    TaskAuth,
    TaskConstraints,
    TaskRequest,
    TaskResponse,
)

__version__ = "0.7.2"
__all__ = [
    "IicpClient",
    "IicpError",
    "IicpNode",
    "IicpTcpClient",
    "IicpTcpClientError",
    "IicpTcpServer",
    "MsgType",
    "NatProfile",
    "NodeConfig",
    "delete_ipv6_pinhole",
    "detect_nat",
    "renew_ipv6_pinhole",
    "openai_compat_handler",
    "vllm_handler",
    "llamacpp_handler",
    "get_backend_handler",
    "BACKEND_TYPES",
    "qos_priority",
    "is_queue_eligible",
    "AvailabilityEvaluator",
    "Window",
    "IdempotencyGuard",
    "TokenValidator",
    "AuditReport",
    "models_diverge",
    "run_audit_pass",
    "PeerManager",
    "ClientConfig",
    "TaskAuth",
    "TaskConstraints",
    "TaskRequest",
    "TaskResponse",
    "ChatMessage",
    "ChatOptions",
    "ChatResponse",
    "CapacityExceededError",
    "ConcurrencyGate",
    "ConformanceReport",
    "CooperativeInferencePolicy",
    "DiscoverOptions",
    "NodeList",
    "PricingConfig",
    "ProbeResult",
    "build_pricing_block",
    "configure_cip_policy",
    "get_cip_policy",
    "run_conformance_checks",
    "sign_body",
    "task_execute_span",
    "task_validate_span",
    "verify_signature",
]
