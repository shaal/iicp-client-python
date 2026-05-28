# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible backend helper.

Port of iicp-adapter's `backends/openai_compat.py` (iter-1410) into the
iicp-client-python SDK as part of the adapter→hybrid-client migration
(tracker iicp.network#340 Tier 1 Item 5).

Designed as a callable factory that produces a TaskHandler suitable for
both `IicpNode.serve(handler)` and `IicpTcpServer(handler=...)`. The
returned handler:

  - inspects `task["intent"]` to choose the OpenAI-compatible HTTP path
    (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
  - forwards `task["payload"]` as the JSON body (with `model` defaulted
    when the caller didn't set it)
  - returns the upstream JSON as the result dict

Drives Ollama (port 11434), vLLM, LM Studio, llama-cpp-server, or any
other OpenAI-compatible HTTP server. The provider URL is the operator's
choice — by default `http://localhost:11434/v1` for Ollama.
"""

from __future__ import annotations

import logging

from iicp_client.backends.base import TaskHandler, build_openai_dialect_handler

logger = logging.getLogger(__name__)


def openai_compat_handler(
    *,
    base_url: str = "http://localhost:11434/v1",
    model: str | None = None,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> TaskHandler:
    """Build a TaskHandler that proxies CALLs to an OpenAI-compatible server.

    Arguments:
        base_url: provider HTTP root (default Ollama's `http://localhost:11434/v1`).
        model: model name. If None, the handler trusts `task["payload"]["model"]`
            and returns an error_code when neither is set.
        api_key: bearer token for the provider's `/v1/*` routes. Empty for
            local Ollama / vLLM; required for OpenAI/Together/Anyscale.
        timeout_s: per-request HTTP timeout.

    Returns:
        An async callable `(task: dict) -> dict` where `task` has shape
        `{task_id, intent, payload}` and the return dict has shape
        `{result: <upstream JSON>}` on success OR
        `{error_code: int, error_message: str}` on failure.
    """
    return build_openai_dialect_handler(
        engine="openai_compat",
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
    )
