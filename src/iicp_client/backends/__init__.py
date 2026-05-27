"""Drop-in backend handlers for iicp-client.

Each helper returns a TaskHandler callable suitable for passing to
`IicpNode.serve(handler, port=...)` OR `IicpTcpServer(handler=...)`.

Available backends:
  - openai_compat — drives Ollama, vLLM, LM Studio, or any OpenAI-compatible
                    HTTP server. Maps intent URN → /v1/{chat/completions,
                    completions, embeddings} path.

Adding a new backend: create a module here that exports a factory like
`openai_compat_handler` returning `async def(task: dict) -> dict`.
"""

from iicp_client.backends.openai_compat import openai_compat_handler

__all__ = ["openai_compat_handler"]
