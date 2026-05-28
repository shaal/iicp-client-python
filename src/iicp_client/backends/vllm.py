# SPDX-License-Identifier: Apache-2.0
"""vLLM backend handler.

vLLM's OpenAI-compatible server (`python -m vllm.entrypoints.openai.api_server`) speaks
the standard `/v1/*` dialect, so this is a thin factory over the shared core with vLLM's
default port (8000). Kept as a dedicated module so operators can select `backend_type=vllm`
explicitly and so vLLM-specific behaviour can diverge here later without touching the
generic path. Port of iicp-adapter `backends/vllm.py` (parity Block B, #340).
"""

from __future__ import annotations

from iicp_client.backends.base import TaskHandler, build_openai_dialect_handler


def vllm_handler(
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str | None = None,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> TaskHandler:
    """Build a TaskHandler that proxies CALLs to a vLLM OpenAI-compatible server.

    Defaults to vLLM's standard port 8000. `api_key` is optional — set it only when
    the vLLM server was started with `--api-key`.
    """
    return build_openai_dialect_handler(
        engine="vllm",
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
    )
