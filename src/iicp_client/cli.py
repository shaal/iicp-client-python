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
import json
import logging
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass as _dc

from iicp_client import IicpNode, NodeConfig
from iicp_client.backends import BACKEND_TYPES, get_backend_handler
from iicp_client.identity import (
    NodeIdentity,
    OperatorIdentity,
    config_dir,
    list_nodes,
    load_node,
    load_operator,
    save_node,
    save_operator,
)

logger = logging.getLogger("iicp-node")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iicp-node",
        description="Run an IICP provider node backed by an OpenAI-compatible server.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "init",
        help="Interactive wizard — set up operator identity + first node config.",
    )
    sub.add_parser(
        "list",
        help="List node configs saved under ~/.iicp/nodes/.",
    )

    serve = sub.add_parser("serve", help="Register and serve a node.")
    serve.add_argument(
        "--node",
        default=_env("IICP_NODE_NAME"),
        help="Load config from ~/.iicp/nodes/<NAME>.json (created by `iicp-node init`). "
        "Other flags override file values when both are set.",
    )
    serve.add_argument(
        "--backend-url",
        default=_env("IICP_BACKEND_URL"),
        help="OpenAI-compatible backend URL (Ollama / vLLM / LM Studio). "
        "env: IICP_BACKEND_URL",
    )
    serve.add_argument(
        "--backend-type",
        default=_env("IICP_BACKEND_TYPE", "openai_compat"),
        choices=list(BACKEND_TYPES),
        help="Inference backend engine. env: IICP_BACKEND_TYPE",
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
        # Default ON: auto-detection runs unless operator explicitly sets
        # --public-endpoint or disables via IICP_AUTO_DETECT_NAT=false.
        default=(_env("IICP_AUTO_DETECT_NAT", "true") or "true").lower() != "false",
        help="Run detect_nat() at startup to claim a public endpoint via "
        "UPnP / external-IP probe. Overrides --public-endpoint when a higher-"
        "tier endpoint is discovered. Default: ON. env: IICP_AUTO_DETECT_NAT",
    )
    serve.add_argument(
        "--external-ip-probe-url",
        # Default to a well-known stable external IP probe URL so CGNAT detection
        # and UPnP external-IP fallback work without operator configuration.
        default=_env("IICP_EXTERNAL_IP_PROBE_URL") or "https://api.ipify.org",
        help="HTTPS URL returning the operator's public IPv4 in plain text. "
        "Used as fallback when UPnP succeeds but GetExternalIPAddress is "
        "auth-gated (common on FRITZ!Box). Default: https://api.ipify.org. "
        "env: IICP_EXTERNAL_IP_PROBE_URL",
    )
    serve.add_argument(
        "--relay-worker-endpoint",
        default=_env("IICP_RELAY_WORKER_ENDPOINT"),
        help="R2 relay-as-last-resort: <host>:<port> of a relay node to connect "
        "outbound to (e.g. relay.example.com:9485). When set, this node acts "
        "as a relay-worker — inbound tasks are forwarded through the relay for "
        "operators behind CGNAT. env: IICP_RELAY_WORKER_ENDPOINT",
    )

    return p


async def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # CIP toggle via env var — same hook the TS + Rust SDKs use. Safe-off
    # default; operator opts in by exporting IICP_CIP_ALLOW_WORKER=true.
    if (os.environ.get("IICP_CIP_ALLOW_WORKER", "") or "").lower() in ("1", "true", "yes"):
        from iicp_client.cip_policy import configure_policy
        configure_policy(enabled=True, allow_worker=True, allow_coordinator=True)

    # If --node <name> points at a saved config, fill any unset CLI flags
    # from the file. Explicit flags still win — operators iterate by passing
    # `--model phi3:mini` while keeping the rest from disk.
    saved: NodeIdentity | None = None
    if getattr(args, "node", None):
        saved = load_node(args.node)
        if saved is None:
            sys.stderr.write(
                f"ERROR: no saved config at ~/.iicp/nodes/{args.node}.json. "
                "Run `iicp-node init` first.\n"
            )
            return 2
        args.backend_url = args.backend_url or saved.backend_url
        args.model = args.model or saved.model
        args.public_endpoint = args.public_endpoint or saved.public_endpoint
        args.directory_url = args.directory_url or saved.directory_url
        args.region = args.region or saved.region
        args.intent = args.intent or saved.intent
        args.node_id = args.node_id or saved.node_id
        if args.max_concurrent == 4:
            args.max_concurrent = saved.max_concurrent
        if args.port == 8020:
            args.port = saved.port
        if args.host == "0.0.0.0":
            args.host = saved.host
        if not args.auto_detect_nat and saved.auto_detect_nat:
            args.auto_detect_nat = True
        if not args.external_ip_probe_url and saved.external_ip_probe_url:
            args.external_ip_probe_url = saved.external_ip_probe_url

    if not args.backend_url or not args.model:
        sys.stderr.write(
            "ERROR: --backend-url and --model are required "
            "(or set IICP_BACKEND_URL and IICP_BACKEND_MODEL, "
            "or load via `--node <name>` after `iicp-node init`).\n"
        )
        return 2

    node_id = (args.node_id or str(uuid.uuid4()))[:36]
    public_endpoint = args.public_endpoint or f"http://localhost:{args.port}"

    relay_worker_ep: str | None = getattr(args, "relay_worker_endpoint", None)
    cfg = NodeConfig(
        node_id=node_id,
        endpoint=public_endpoint,
        intent=args.intent,
        model=args.model,
        region=args.region,
        directory_url=args.directory_url,
        max_concurrent=args.max_concurrent,
        relay_worker_endpoint=relay_worker_ep or None,
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
            # Tier ≥ 3 (unreachable/CGNAT with no IPv6 fallback) and no relay
            # configured → auto-elect a relay from the directory and configure it.
            # This is the fully automatic relay-as-last-resort path.
            if getattr(profile, "tier", 0) >= 3 and not relay_worker_ep:
                logger.info(
                    "NAT tier=%d: no direct or IPv6 endpoint available — "
                    "querying directory for relay-capable peers.",
                    profile.tier,
                )
                elected_relay = await _auto_elect_relay(
                    args.directory_url, cfg.intent, node_id
                )
                if elected_relay:
                    relay_host, relay_port = elected_relay
                    relay_worker_ep = f"{relay_host}:{relay_port}"
                    cfg.relay_worker_endpoint = relay_worker_ep
                    logger.info(
                        "NAT: auto-elected relay %s:%d — node will register "
                        "via relay when connection is established.",
                        relay_host, relay_port,
                    )
                else:
                    logger.warning(
                        "NAT tier=%d: no relay-capable peers found in directory. "
                        "Enabling mesh to discover relays via gossip. "
                        "Set IICP_RELAY_WORKER_ENDPOINT=<host>:<port> to specify "
                        "a relay manually.",
                        profile.tier,
                    )
                    cfg.enable_mesh = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("NAT detection failed: %s — continuing without it", exc)
    else:
        # #343 — Even without full NAT detection, if the public_endpoint is
        # a bracketed IPv6 URL, attempt to open the UPnP IGDv2 firewall
        # pinhole proactively. This is the path the maintainer's setup hits:
        # operator gives `http://[2a0a:...]:8020`, expects the router pinhole
        # to open automatically. Previously the SDK skipped detect_nat in
        # this case and never tried AddPinhole.
        if "[" in (args.public_endpoint or ""):
            try:
                from iicp_client.nat_detection import (
                    NatProfile,
                    _maybe_open_v6_pinhole_for_endpoint,
                )
                synth = NatProfile(
                    tier=0,
                    transport_method="direct",
                    public_endpoint=args.public_endpoint,
                    detection_log=[],
                )
                _maybe_open_v6_pinhole_for_endpoint(synth, args.port)
                for line in synth.detection_log:
                    logger.info("v6: %s", line)
                node.apply_nat_profile(synth)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IPv6 pinhole attempt failed: %s", exc)

    # The handler expects base_url; the CLI's --backend-url is the OpenAI-
    # compatible root. Append /v1 if not already present (Ollama serves at
    # both /v1/chat/completions and the bare /api/chat, but the SDK helper
    # talks the OpenAI dialect on /v1).
    base_url = args.backend_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    handler = get_backend_handler(
        args.backend_type,
        base_url=base_url,
        model=args.model,
    )

    # GAP-6: probe the backend for all available models and advertise them.
    # _ollama_models is best-effort; on any error it returns [] and we fall
    # back to the single configured model.
    discovered = _ollama_models(args.backend_url)
    if discovered:
        extra = [m for m in discovered if m != args.model]
        if extra:
            cfg.capabilities = extra
            logger.info("GAP-6: advertising %d additional models: %s", len(extra), extra[:6])

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


async def _auto_elect_relay(
    directory_url: str, intent: str, node_id: str
) -> tuple[str, int] | None:
    """Query the directory for relay-capable nodes and elect one deterministically.

    Used when NAT detection returns tier≥3 (CGNAT with no usable IPv6 path).
    Returns (relay_host, relay_port) or None if no relay-capable peer is found.
    """
    import hashlib
    try:
        url = f"{directory_url.rstrip('/')}/v1/discover"
        with urllib.request.urlopen(  # noqa: S310
            f"{url}?intent={urllib.parse.quote(intent)}&relay_capable=true",
            timeout=5,
        ) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.debug("relay discovery failed: %s", exc)
        return None

    candidates = [
        n for n in data.get("nodes", [])
        if n.get("relay_capable") and n.get("endpoint")
    ]
    if not candidates:
        return None

    def _score(node: dict) -> tuple:
        load = float(node.get("load", 0.0))
        h = hashlib.sha256(f"{node_id}:{node['node_id']}".encode()).hexdigest()
        return (load, h)

    elected = min(candidates, key=_score)
    endpoint = elected["endpoint"].rstrip("/")
    # Derive relay host from HTTP endpoint URL
    try:
        parsed = urllib.parse.urlparse(endpoint)
        relay_host = parsed.hostname or ""
    except Exception:  # noqa: BLE001
        relay_host = ""
    relay_port = elected.get("relay_accept_port") or 9485
    if not relay_host:
        return None
    return relay_host, int(relay_port)


def _prompt(question: str, default: str = "") -> str:
    """Plain stdin prompt with default. Returns the user's answer (or default)."""
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"{question}{suffix}: ")
    sys.stdout.flush()
    line = sys.stdin.readline().strip()
    return line or default


def _ollama_models(backend_url: str) -> list[str]:
    """Best-effort: list models from a local Ollama. Empty list on any error."""
    try:
        with urllib.request.urlopen(
            backend_url.rstrip("/") + "/api/tags", timeout=2
        ) as resp:
            data = json.loads(resp.read().decode())
            return sorted({m["name"] for m in data.get("models", [])})
    except Exception:  # noqa: BLE001
        return []


def _cmd_init(args: argparse.Namespace) -> int:
    """Interactive setup wizard. Creates operator identity if absent,
    creates a new node config under ~/.iicp/nodes/<name>.json."""
    print(f"IICP node setup — config dir: {config_dir()}\n")

    # 1) Operator identity ──────────────────────────────────────────────────
    op = load_operator()
    if op is None:
        print("No operator identity yet. Creating one — credits earned by every")
        print("node you run will accumulate to this operator_id.\n")
        display = _prompt("Display name", os.environ.get("USER", "operator"))
        contact = _prompt("Contact (email, leave blank for none)", "")
        op = OperatorIdentity.generate(display_name=display, contact=contact)
        save_operator(op)
        print(f"  ✓ created {op.operator_id}\n")
    else:
        print(f"Existing operator: {op.operator_id} ({op.display_name})\n")

    # 2) Backend (Ollama / vLLM / LM Studio) ───────────────────────────────
    backend_url = _prompt(
        "Backend URL (OpenAI-compatible)", "http://localhost:11434"
    )
    models = _ollama_models(backend_url)
    if models:
        print(f"  Detected models at {backend_url}: {', '.join(models[:6])}")
        model_default = models[0]
    else:
        model_default = "qwen2.5:0.5b"
    model = _prompt("Model to advertise", model_default)

    # 3) Node-specific ──────────────────────────────────────────────────────
    name_default = model.replace(":", "-").replace(".", "-").lower()
    name = _prompt("Local node name (used as ~/.iicp/nodes/<NAME>.json)", name_default)
    intent = _prompt("Intent URN", "urn:iicp:intent:llm:chat:v1")
    region = _prompt("Region tag", "eu-central")
    directory_url = _prompt("Directory URL", "https://iicp.network/api")
    port_str = _prompt("Local HTTP port", "8020")
    port = int(port_str)
    public_endpoint = _prompt(
        "Public endpoint URL (leave blank if you'll use --auto-detect-nat)",
        "",
    )
    auto_nat = _prompt("Auto-detect NAT via UPnP / external-IP probe? (y/N)", "n")
    auto_detect_nat = auto_nat.lower().startswith("y")
    external_probe = ""
    if auto_detect_nat:
        external_probe = _prompt(
            "External-IP probe URL (fallback when UPnP fails)",
            "https://api.ipify.org",
        )

    # 4) Persist ───────────────────────────────────────────────────────────
    try:
        node = NodeIdentity.generate(
            operator_id=op.operator_id,
            name=name,
            backend_url=backend_url,
            model=model,
            intent=intent,
            region=region,
            directory_url=directory_url,
            port=port,
            public_endpoint=public_endpoint,
            auto_detect_nat=auto_detect_nat,
            external_ip_probe_url=external_probe,
        )
    except ValueError as exc:
        sys.stderr.write(f"\nERROR: {exc}\n")
        return 2
    saved_to = save_node(node)
    print()
    print(f"  ✓ saved {saved_to}")
    print(f"  ✓ node_id = {node.node_id}")
    print()

    # ── Dependency check + auto-install + docs link (#346) ────────────────
    print("Checking dependencies …")
    issues = _check_dependencies(backend_url)
    _print_dep_status(issues)
    if any(i.severity == "missing" and i.installable for i in issues):
        ans = _prompt("Install missing optional deps now? (Y/n)", "y").lower()
        if ans.startswith("y"):
            _install_missing(issues)
    print()
    print("Documentation:")
    print("  Operator quickstart: https://iicp.network/docs/sdk-quickstart-docker")
    print("  CLI reference:       iicp-node --help / iicp-node serve --help")
    print("  Spec:                https://iicp.network/spec")
    print()
    print(f"Run: iicp-node serve --node {name}")
    return 0


# ── #346 — dependency checker + auto-install ────────────────────────────────


@_dc
class _DepIssue:
    name: str
    severity: str  # "missing" | "warn" | "ok"
    message: str
    installable: bool = False
    pip_extra: str = ""


def _check_dependencies(backend_url: str) -> list[_DepIssue]:
    """Probe runtime + optional deps + backend reachability."""
    out: list[_DepIssue] = []

    # 1) Backend reachability
    try:
        with urllib.request.urlopen(
            backend_url.rstrip("/") + "/api/tags", timeout=2
        ) as resp:
            ok = resp.status == 200
        if ok:
            out.append(_DepIssue("backend", "ok", f"reachable at {backend_url}"))
        else:
            out.append(_DepIssue("backend", "warn", f"backend HTTP {resp.status}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_DepIssue("backend", "warn", f"{backend_url} unreachable: {exc}"))

    # 2) Optional Python deps mapped to pip extras
    optional = [
        ("cbor2", "iicp-tcp", "native IICP TCP transport (port 9484)"),
        ("upnpclient", "nat", "UPnP NAT detection + IPv6 firewall pinhole"),
        ("ifaddr", "nat", "interface enumeration for NAT detection"),
        ("prometheus_client", "metrics", "/metrics endpoint"),
    ]
    for mod, extra, purpose in optional:
        try:
            __import__(mod)
            out.append(_DepIssue(mod, "ok", purpose))
        except ImportError:
            out.append(_DepIssue(
                mod,
                "missing",
                f"{purpose} (not installed)",
                installable=True,
                pip_extra=extra,
            ))

    # 3) IPv6 routing surface (advisory — doesn't gate anything)
    try:
        import asyncio

        from iicp_client.nat_detection import detect_ipv6
        v6 = asyncio.get_event_loop().run_until_complete(detect_ipv6(0, timeout_s=1.5)) \
            if not asyncio.get_event_loop().is_running() else None
    except Exception:  # noqa: BLE001
        v6 = None
    if v6 and v6.global_v6_available:
        msg = f"{len(v6.addresses)} global IPv6 address(es)"
        if v6.external_v6_reachable:
            msg += "; outbound v6 reachable"
        out.append(_DepIssue("ipv6", "ok", msg))
    elif v6:
        out.append(_DepIssue(
            "ipv6",
            "warn",
            "no global IPv6 — direct hosting will require IPv4 + tunnel",
        ))

    return out


def _print_dep_status(issues: list[_DepIssue]) -> None:
    glyph = {"ok": "  ✓", "warn": "  !", "missing": "  ✗"}
    for i in issues:
        print(f"{glyph.get(i.severity, '  ?')} {i.name:18}  {i.message}")


def _install_missing(issues: list[_DepIssue]) -> None:
    """Run `pip install iicp-client[<extras>]` for the missing-optional set."""
    extras = sorted({
        i.pip_extra
        for i in issues
        if i.severity == "missing" and i.installable and i.pip_extra
    })
    if not extras:
        return
    spec = f"iicp-client[{','.join(extras)}]"
    print(f"\n  → pip install --upgrade {spec}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", spec],
            check=True,
        )
        print("  ✓ done")
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"  ✗ pip install failed (exit={exc.returncode})\n")


def _cmd_list(_args: argparse.Namespace) -> int:
    op = load_operator()
    if op:
        print(f"Operator: {op.operator_id} ({op.display_name})\n")
    nodes = list_nodes()
    if not nodes:
        print("No node configs yet. Run `iicp-node init`.")
        return 0
    width = max(len(n.name) for n in nodes)
    print(f"{'NAME'.ljust(width)}  MODEL                BACKEND")
    print(f"{'-' * width}  -------------------- --------------------------------")
    for n in nodes:
        print(
            f"{n.name.ljust(width)}  {n.model[:20].ljust(20)} {n.backend_url[:48]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return asyncio.run(_serve(args))
    if args.cmd == "init":
        return _cmd_init(args)
    if args.cmd == "list":
        return _cmd_list(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
