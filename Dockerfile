FROM python:3.11-slim AS build
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[metrics]"

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /app/src /app/src
ENV PYTHONPATH=/app/src
EXPOSE 8020
CMD ["python", "-c", "import asyncio; from iicp_client import IicpNode, NodeConfig; node = IicpNode(NodeConfig(node_id='docker-node', endpoint='http://localhost:8020', intent='urn:iicp:intent:llm:chat:v1')); asyncio.run(node.serve(lambda t: __import__(\"asyncio\").coroutine(lambda: {\"result\": {}})()))"]
