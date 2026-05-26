"""iicp-client — Official Python client SDK for the IICP protocol."""

from iicp_client.client import IicpClient
from iicp_client.errors import IicpError
from iicp_client.iicp_tcp import IicpTcpClient, IicpTcpClientError, IicpTcpServer, MsgType
from iicp_client.node import IicpNode, NodeConfig
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

__version__ = "0.3.1"
__all__ = [
    "IicpClient",
    "IicpError",
    "IicpNode",
    "IicpTcpClient",
    "IicpTcpClientError",
    "IicpTcpServer",
    "MsgType",
    "NodeConfig",
    "ClientConfig",
    "TaskAuth",
    "TaskConstraints",
    "TaskRequest",
    "TaskResponse",
    "ChatMessage",
    "ChatOptions",
    "ChatResponse",
    "DiscoverOptions",
    "NodeList",
]
