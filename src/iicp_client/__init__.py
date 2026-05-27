"""iicp-client — Official Python client SDK for the IICP protocol."""

from iicp_client.backends import openai_compat_handler
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
from iicp_client.iicp_tcp import IicpTcpClient, IicpTcpClientError, IicpTcpServer, MsgType
from iicp_client.nat_detection import NatProfile, delete_ipv6_pinhole, detect_nat, renew_ipv6_pinhole
from iicp_client.node import IicpNode, NodeConfig
from iicp_client.pricing import PricingConfig, build_pricing_block, sign_body, verify_signature
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

__version__ = "0.5.6"
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
    "verify_signature",
]
