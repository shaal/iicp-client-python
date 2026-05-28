# SPDX-License-Identifier: Apache-2.0
"""llama.cpp backend handler.

The `llama-server` binary (llama.cpp) exposes an OpenAI-compatible `/v1/*` API, so this
is a thin factory over the shared core with llama.cpp's default port (8080). Kept as a
dedicated module so operators can select `backend_type=llamacpp` explicitly. Port of
iicp-adapter `backends/llamacpp.py` (parity Block B, #340).
"""

from __future__ import annotations

from iicp_client.backends.base import TaskHandler, build_openai_dialect_handler


def llamacpp_handler(
    *,
    base_url: str = "http://localhost:8080/v1",
    model: str | None = None,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> TaskHandler:
    """Build a TaskHandler that proxies CALLs to a llama.cpp `llama-server`.

    Defaults to llama.cpp's standard port 8080. llama.cpp ignores the `model` field for
    single-model servers, but it is still sent so multi-model builds route correctly.
    """
    return build_openai_dialect_handler(
        engine="llamacpp",
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
    )
