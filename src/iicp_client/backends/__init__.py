"""Drop-in backend handlers for iicp-client.

Each helper returns a TaskHandler callable suitable for passing to
`IicpNode.serve(handler, port=...)` OR `IicpTcpServer(handler=...)`.

Available backends:
  - openai_compat — drives Ollama, LM Studio, or any OpenAI-compatible HTTP server.
                    Maps intent URN → /v1/{chat/completions,completions,embeddings}.
  - vllm          — vLLM OpenAI server (default port 8000).
  - llamacpp      — llama.cpp `llama-server` (default port 8080).

Use `get_backend_handler(backend_type, ...)` to select one by name (e.g. from a CLI
`--backend-type` flag). Adding a new backend: create a module here whose factory
delegates to `base.build_openai_dialect_handler` (or implements a fresh dialect),
then register it in `_FACTORIES` below.
"""

from __future__ import annotations

from iicp_client.backends.anthropic import anthropic_handler
from iicp_client.backends.base import TaskHandler
from iicp_client.backends.llamacpp import llamacpp_handler
from iicp_client.backends.openai_compat import openai_compat_handler
from iicp_client.backends.vllm import vllm_handler

__all__ = [
    "openai_compat_handler",
    "vllm_handler",
    "llamacpp_handler",
    "anthropic_handler",
    "get_backend_handler",
    "BACKEND_TYPES",
]

_FACTORIES = {
    "openai_compat": openai_compat_handler,
    "vllm": vllm_handler,
    "llamacpp": llamacpp_handler,
    "anthropic": anthropic_handler,
}

BACKEND_TYPES = tuple(_FACTORIES.keys())


def get_backend_handler(backend_type: str, **kwargs) -> TaskHandler:
    """Return a backend handler by name.

    `backend_type` is one of `BACKEND_TYPES`. Remaining kwargs (base_url, model,
    api_key, timeout_s) are forwarded to the selected factory. Each factory supplies
    its own engine-appropriate default `base_url` when omitted.
    """
    try:
        factory = _FACTORIES[backend_type]
    except KeyError:
        raise ValueError(
            f"unknown backend_type {backend_type!r}; choose one of {list(BACKEND_TYPES)}"
        ) from None
    return factory(**kwargs)
