"""iicp-node — turn the SDK into a runnable provider node.

Usage:

    # via the installed entry point
    iicp-node serve --model qwen2.5:0.5b --backend-url http://localhost:11434

    # via python -m
    python -m iicp_client.cli serve --model qwen2.5:0.5b ...

    # all flags also read from env vars (IICP_BACKEND_URL, IICP_BACKEND_MODEL,
    # IICP_PUBLIC_ENDPOINT, IICP_DIRECTORY_URL, IICP_REGION,
    # IICP_MAX_CONCURRENT, IICP_NODE_ID, IICP_INTENT, IICP_PORT, IICP_HOST)

Why this exists: before, mesh joiners had to write their own boilerplate
script (see deprecated adapter/sdk_node.py) just to spin up a provider
node. This CLI replaces that with a one-liner that registers, serves
HTTP /v1/task + /iicp/health + /metrics, and forwards each task to the
configured backend.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

from iicp_client import IicpNode, NodeConfig
from iicp_client.backends import openai_compat_handler

logger = logging.getLogger("iicp-node")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iicp-node",
        description="Run an IICP provider node backed by an OpenAI-compatible server.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Register and serve a node.")
    serve.add_argument(
        "--backend-url",
        default=_env("IICP_BACKEND_URL"),
        help="OpenAI-compatible backend URL (Ollama / vLLM / LM Studio). "
        "env: IICP_BACKEND_URL",
    )
    serve.add_argument(
        "--model",
        default=_env("IICP_BACKEND_MODEL"),
        help="Backend model name (e.g. qwen2.5:0.5b). env: IICP_BACKEND_MODEL",
    )
    serve.add_argument(
        "--public-endpoint",
        default=_env("IICP_PUBLIC_ENDPOINT"),
        help="Externally reachable URL of this node. env: IICP_PUBLIC_ENDPOINT. "
        "If unset, the node registers as non-routable (development mode).",
    )
    serve.add_argument(
        "--directory-url",
        default=_env("IICP_DIRECTORY_URL", "https://iicp.network/api"),
        help="IICP directory base URL. env: IICP_DIRECTORY_URL",
    )
    serve.add_argument(
        "--region",
        default=_env("IICP_REGION", "eu-central"),
        help="Region tag. env: IICP_REGION",
    )
    serve.add_argument(
        "--intent",
        default=_env("IICP_INTENT", "urn:iicp:intent:llm:chat:v1"),
        help="Intent URN this node serves. env: IICP_INTENT",
    )
    serve.add_argument(
        "--max-concurrent",
        type=int,
        default=int(_env("IICP_MAX_CONCURRENT", "4") or "4"),
        help="Concurrent task cap (excess gets 429 IICP-E021). env: IICP_MAX_CONCURRENT",
    )
    serve.add_argument(
        "--node-id",
        default=_env("IICP_NODE_ID"),
        help="Stable node ID. env: IICP_NODE_ID. Auto-generated if absent.",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=int(_env("IICP_PORT", "8020") or "8020"),
        help="HTTP listen port. env: IICP_PORT",
    )
    serve.add_argument(
        "--host",
        default=_env("IICP_HOST", "0.0.0.0"),
        help="HTTP bind host. env: IICP_HOST",
    )
    serve.add_argument(
        "--skip-registration",
        action="store_true",
        default=(_env("IICP_SKIP_REGISTRATION", "false") or "false").lower() == "true",
        help="Skip directory registration (development / offline mode). "
        "env: IICP_SKIP_REGISTRATION",
    )
    serve.add_argument(
        "--auto-detect-nat",
        action="store_true",
        default=(_env("IICP_AUTO_DETECT_NAT", "false") or "false").lower() == "true",
        help="Run detect_nat() at startup to claim a public endpoint via "
        "UPnP / external-IP probe. Overrides --public-endpoint when a higher-"
        "tier endpoint is discovered. env: IICP_AUTO_DETECT_NAT",
    )
    serve.add_argument(
        "--external-ip-probe-url",
        default=_env("IICP_EXTERNAL_IP_PROBE_URL"),
        help="Optional HTTPS URL returning the operator's public IPv4 in plain "
        "text (e.g. https://api.ipify.org). Used as fallback when UPnP "
        "discovery succeeds but GetExternalIPAddress is auth-gated. "
        "env: IICP_EXTERNAL_IP_PROBE_URL",
    )

    return p


async def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.backend_url or not args.model:
        sys.stderr.write(
            "ERROR: --backend-url and --model are required "
            "(or set IICP_BACKEND_URL and IICP_BACKEND_MODEL).\n"
        )
        return 2

    node_id = args.node_id or f"sdk-{args.model.replace(':', '-')}-{uuid.uuid4().hex[:8]}"
    public_endpoint = args.public_endpoint or f"http://localhost:{args.port}"

    cfg = NodeConfig(
        node_id=node_id,
        endpoint=public_endpoint,
        intent=args.intent,
        model=args.model,
        region=args.region,
        directory_url=args.directory_url,
        max_concurrent=args.max_concurrent,
    )
    node = IicpNode(cfg)

    # Optional ADR-041 NAT detection. Runs detect_nat() to discover a public
    # endpoint via UPnP / external-IP probe and applies the result to the node
    # config — this populates transport_method + nat_type + transport_metadata
    # so the directory's binary NATted validator (#334) accepts the
    # registration without manual port-forwarding.
    if args.auto_detect_nat:
        from iicp_client.nat_detection import detect_nat
        try:
            profile = await detect_nat(
                args.host,
                args.port,
                operator_public_endpoint=args.public_endpoint or None,
                external_ip_probe_url=args.external_ip_probe_url,
            )
            logger.info(
                "NAT detection: tier=%d method=%s public_endpoint=%s",
                profile.tier,
                profile.transport_method,
                getattr(profile, "public_endpoint", None) or "(none)",
            )
            node.apply_nat_profile(profile)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NAT detection failed: %s — continuing without it", exc)

    # The handler expects base_url; the CLI's --backend-url is the OpenAI-
    # compatible root. Append /v1 if not already present (Ollama serves at
    # both /v1/chat/completions and the bare /api/chat, but the SDK helper
    # talks the OpenAI dialect on /v1).
    base_url = args.backend_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    handler = openai_compat_handler(
        base_url=base_url,
        model=args.model,
    )

    token: str | None = None
    if not args.skip_registration:
        try:
            token = await node.register()
            logger.info("Registered as %s (token=%s…)", node_id, (token or "")[:8])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registration failed: %s — continuing without heartbeat", exc)

    logger.info(
        "Serving %s on %s:%d — backend %s (model=%s, max_concurrent=%d)",
        args.intent,
        args.host,
        args.port,
        args.backend_url,
        args.model,
        args.max_concurrent,
    )
    await node.serve(handler, host=args.host, port=args.port, node_token=token)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return asyncio.run(_serve(args))
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
