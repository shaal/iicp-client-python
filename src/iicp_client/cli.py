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
import signal
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass as _dc
from importlib.metadata import version as _pkg_version

from iicp_client import IicpNode, NodeConfig
from iicp_client.backends import BACKEND_TYPES, get_backend_handler
from iicp_client.identity import (
    NodeIdentity,
    OperatorIdentity,
    config_dir,
    list_nodes,
    load_node,
    load_operator,
    no_identity_notice,
    save_node,
    save_operator,
)
from iicp_client.node_log import setup_node_log
from iicp_client.node_log import write_event as _log_event

logger = logging.getLogger("iicp-node")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _find_available_port(host: str, start: int, max_tries: int = 64) -> int:
    """Return the first bindable TCP port >= ``start`` on ``host``.

    The official IICP port 9484 is the starting point; when running multiple
    nodes on one host (each model on its own port → its own pinhole) the second
    node auto-increments to 9485, the third to 9486, and so on. Probes by
    attempting a real bind so the chosen port is genuinely free before NAT
    detection opens a pinhole and the directory registration advertises it.
    """
    bind_host = host if host not in ("", "0.0.0.0") else "0.0.0.0"
    for candidate in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((bind_host, candidate))
                return candidate
            except OSError:
                continue
    return start  # exhausted — let serve() surface the real bind error


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iicp-node",
        description="Run an IICP provider node backed by an OpenAI-compatible server.",
    )
    try:
        _ver = _pkg_version("iicp-client")
    except Exception:
        _ver = "unknown"
    p.add_argument("--version", "-V", action="version", version=f"iicp-node {_ver}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "help",
        help="Print this top-level usage and exit.",
    )
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
        # #410 — default EMPTY so a saved-node config (--node) can supply backend_url.
        # The localhost:11434 fallback is applied AFTER saved-config restore, giving
        # the correct precedence: flag > env > saved-config > built-in default.
        default=_env("IICP_BACKEND_URL") or "",
        help="OpenAI-compatible backend URL (Ollama / vLLM / LM Studio). "
        "env: IICP_BACKEND_URL (default http://localhost:11434)",
    )
    serve.add_argument(
        "--backend-type",
        default=_env("IICP_BACKEND_TYPE", "openai_compat"),
        choices=list(BACKEND_TYPES),
        help="Inference backend engine. env: IICP_BACKEND_TYPE",
    )
    serve.add_argument(
        "--backend-api-key",
        # #5 — Bearer key for an auth-requiring OpenAI-compat backend (LM Studio,
        # hosted providers). Empty = no Authorization header (local Ollama).
        default=_env("IICP_BACKEND_API_KEY", ""),
        help="Bearer API key for an auth'd backend (LM Studio, hosted). "
        "env: IICP_BACKEND_API_KEY (default empty = none)",
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
        default=_env("IICP_REGION"),
        help="Region tag (e.g. us-east, eu-central). env: IICP_REGION. "
        "If unset and none is saved, registers as 'unknown' (never assumes a region).",
    )
    serve.add_argument(
        "--intent",
        default=_env("IICP_INTENT", "urn:iicp:intent:llm:chat:v1"),
        help="Intent URN this node serves. env: IICP_INTENT",
    )
    serve.add_argument(
        "--max-concurrent",
        type=int,
        # default=None as a "not supplied" sentinel so a saved-node value can be
        # restored; the env/built-in default (4) is applied AFTER saved-config
        # restore in _serve(). Passing the default value on the CLI no longer
        # silently loses to the saved config.
        default=None,
        help="Concurrent task cap (excess gets 429 IICP-E021). "
        "env: IICP_MAX_CONCURRENT (default 4)",
    )
    serve.add_argument(
        "--node-id",
        default=_env("IICP_NODE_ID"),
        help="Stable node ID. env: IICP_NODE_ID. Auto-generated if absent.",
    )
    serve.add_argument(
        "--port",
        type=int,
        # default=None sentinel — see --max-concurrent. env/built-in (9484) applied
        # after saved-config restore in _serve().
        default=None,
        help="HTTP listen port. env: IICP_PORT (default 9484)",
    )
    serve.add_argument(
        "--host",
        # default=None sentinel — see --max-concurrent. env/built-in (::) applied
        # after saved-config restore in _serve(). Fixes the prior bug where the
        # restore guard compared against "0.0.0.0" but the default was "::", so a
        # saved host was never restored.
        default=None,
        help="HTTP bind host. env: IICP_HOST (default ::)",
    )
    serve.add_argument(
        "--skip-registration",
        action="store_true",
        default=(_env("IICP_SKIP_REGISTRATION", "false") or "false").lower() == "true",
        help="Skip directory registration (development / offline mode). env: IICP_SKIP_REGISTRATION",
    )
    serve.add_argument(
        "--force",
        action="store_true",
        default=(_env("IICP_FORCE", "false") or "false").lower() == "true",
        help="Take over the single-instance lock if another process serves this node_id. env: IICP_FORCE",
    )
    serve.add_argument(
        "--auto-detect-nat",
        action=argparse.BooleanOptionalAction,
        # Default ON: auto-detection runs unless operator passes
        # --no-auto-detect-nat, sets --public-endpoint, or disables via
        # IICP_AUTO_DETECT_NAT=false. BooleanOptionalAction registers both
        # --auto-detect-nat and --no-auto-detect-nat so the CLI off-switch works.
        default=(_env("IICP_AUTO_DETECT_NAT", "true") or "true").lower() != "false",
        help="Run detect_nat() at startup to claim a public endpoint via "
        "UPnP / external-IP probe. Overrides --public-endpoint when a higher-"
        "tier endpoint is discovered. Disable with --no-auto-detect-nat. "
        "Default: ON. env: IICP_AUTO_DETECT_NAT",
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
    serve.add_argument(
        "--tunnel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="#520 NAT-ladder rung 5: expose this node via a zero-account "
        "Cloudflare Quick Tunnel (requires cloudflared on PATH; never "
        "auto-installed). Default: automatic — used only when every other "
        "NAT path fails (tier ≥ 3, no relay found). --tunnel forces it on; "
        "--no-tunnel disables the automatic escalation. env: IICP_TUNNEL=1/0",
    )
    serve.add_argument(
        "--relay-capable",
        action=argparse.BooleanOptionalAction,
        default=bool(_env("IICP_RELAY_CAPABLE", "").lower() in ("1", "true", "yes")),
        help="Advertise this node as a relay server for CGNAT/tier-4 operators. "
        "Opens relay_accept_port (default 9485) and registers relay_capable=true "
        "in the directory. Use with a publicly routable endpoint (direct or "
        "Cloudflare Tunnel). env: IICP_RELAY_CAPABLE",
    )
    serve.add_argument(
        "--relay-accept-port",
        type=int,
        default=int(_env("IICP_RELAY_ACCEPT_PORT", "9485")),
        help="TCP port for the relay accept server (only used with --relay-capable). "
        "Default: 9485. Note: relay bind authentication is pending (#510) — only "
        "run a relay accept port on networks you trust until the signed-bind "
        "mechanism ships. env: IICP_RELAY_ACCEPT_PORT",
    )
    serve.add_argument(
        "--log-dir",
        default=_env("IICP_LOG_DIR"),
        help="Directory for persistent log files (<node_id>.log + events.jsonl). "
        "Default: ~/.iicp/logs/. env: IICP_LOG_DIR",
    )
    serve.add_argument(
        "--with-proxy",
        action="store_true",
        help="Also run the local compat proxy gateway (loopback 127.0.0.1:9483) in this "
        "process, supervised + crash-isolated from the node. Needs the [proxy] extra. "
        "Port override: IICP_PROXY_PORT.",
    )

    query = sub.add_parser(
        "query",
        help="Discover mesh nodes and submit a chat task.",
    )
    query.add_argument("prompt", nargs="+", help="Prompt text to send.")
    query.add_argument(
        "--directory-url",
        default=_env("IICP_DIRECTORY_URL", "https://iicp.network/api"),
        help="IICP directory base URL. env: IICP_DIRECTORY_URL",
    )
    query.add_argument(
        "--intent",
        default=_env("IICP_INTENT", "urn:iicp:intent:llm:chat:v1"),
        help="Intent URN to query. env: IICP_INTENT",
    )
    query.add_argument("--model", default=None, help="Pin to a specific model on the remote node.")
    query.add_argument("--max-tokens", type=int, default=None, help="Limit response length.")
    query.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="Request timeout in milliseconds.",
    )
    query.add_argument(
        "--node",
        default=_env("IICP_NODE", ""),
        help="Node config name (for self-query neutrality). env: IICP_NODE",
    )

    credits = sub.add_parser(
        "credits",
        help="Show this node's earned / spent / balance credits.",
    )
    credits.add_argument("--node", default=None, help="Load token + node_id from ~/.iicp/nodes/<NAME>.json")
    credits.add_argument("--node-id", dest="node_id", default=None, help="Node id (if not using --node).")
    credits.add_argument(
        "--token",
        default=_env("IICP_NODE_TOKEN", "") or None,
        help="Node token. env: IICP_NODE_TOKEN",
    )
    credits.add_argument(
        "--directory-url",
        default=None,
        help="IICP directory base URL (defaults to the saved node's / env / iicp.network).",
    )
    credits.add_argument("--json", action="store_true", help="Print the raw summary JSON.")
    credits.add_argument(
        "--verify",
        action="store_true",
        help="Cryptographically audit each award against the directory's signed CREDIT_AWARD log.",
    )

    # #460 — operator-identity management (mutable nickname over the immutable operator_id).
    operator = sub.add_parser("operator", help="Manage your operator identity.")
    op_sub = operator.add_subparsers(dest="op_cmd", required=True)
    op_rename = op_sub.add_parser(
        "rename",
        help="Change your public display_name (signed by your operator key; reflected on "
        "every node + the leaderboard).",
    )
    op_rename.add_argument("name", help="New display name (1-64 chars, no control characters).")
    op_rename.add_argument(
        "--directory-url",
        default=None,
        help="IICP directory base URL (defaults to env / iicp.network).",
    )
    op_sub.add_parser(
        "encrypt",
        help="Password-encrypt the operator secret at rest (#460). Set $IICP_OPERATOR_PASSPHRASE "
        "to unlock it headlessly during `serve`.",
    )
    op_sub.add_parser("decrypt", help="Remove at-rest encryption — restore the plaintext secret.")

    # ── proxy (ADR-050) — local OpenAI/Ollama/Anthropic-compat gateway ────────────
    proxy = sub.add_parser(
        "proxy",
        help="Run the local OpenAI/Ollama/Anthropic-compat gateway (consumer; loopback; "
        "does NOT register with the directory). Needs the [proxy] extra.",
    )
    proxy.add_argument(
        "--port", type=int, default=int(_env("IICP_PROXY_PORT", "9483") or "9483"),
        help="Listen port (env IICP_PROXY_PORT, default 9483 — reserved IICP proxy band).",
    )
    proxy.add_argument(
        "--host", default=_env("IICP_PROXY_HOST", "127.0.0.1") or "127.0.0.1",
        help="Bind host (env IICP_PROXY_HOST, default 127.0.0.1 — loopback only).",
    )
    proxy.add_argument(
        "--config", default="proxy.toml",
        help="Optional proxy.toml path (env IICP_PROXY_* override individual fields).",
    )

    # ── mcp-gateway — bridge a local MCP server as an IICP provider node ──────────
    gw = sub.add_parser(
        "mcp-gateway",
        help="Bridge a local MCP server into the IICP mesh as a registered provider node.",
    )
    gw.add_argument(
        "--mcp-url",
        default=_env("IICP_MCP_URL", "http://localhost:8001") or "http://localhost:8001",
        help="Local MCP server base URL (env IICP_MCP_URL, default http://localhost:8001).",
    )
    gw.add_argument(
        "--tools",
        default=_env("IICP_MCP_TOOLS", "") or "",
        help="Comma-separated MCP tool names to advertise (env IICP_MCP_TOOLS). Required.",
    )
    gw.add_argument(
        "--node-id",
        default=_env("IICP_NODE_ID", "") or "",
        help="Node ID (env IICP_NODE_ID, auto-generated if absent).",
    )
    gw.add_argument(
        "--public-endpoint",
        default=_env("IICP_PUBLIC_ENDPOINT", "") or "",
        help="Externally reachable URL for this gateway (env IICP_PUBLIC_ENDPOINT).",
    )
    gw.add_argument(
        "--directory-url",
        default=_env("IICP_DIRECTORY_URL", "https://iicp.network/api/v1") or "https://iicp.network/api/v1",
        help="IICP directory URL (env IICP_DIRECTORY_URL).",
    )
    gw.add_argument(
        "--region",
        default=_env("IICP_REGION", "local") or "local",
        help="Region tag (env IICP_REGION, default local).",
    )
    gw.add_argument(
        "--port",
        type=int,
        default=int(_env("IICP_PORT", "9484") or "9484"),
        help="Listen port (env IICP_PORT, default 9484).",
    )
    gw.add_argument(
        "--host",
        default=_env("IICP_HOST", "::") or "::",
        help="Bind host (env IICP_HOST, default :: — dual-stack).",
    )

    return p


async def _cmd_credits_async(args: argparse.Namespace) -> int:
    """`iicp-node credits` — earned / spent / balance from the directory's
    reconcile-checked GET /v1/credits/summary (#456). Figures come authenticated
    from the directory (not the local config), so editing the saved file cannot
    inflate them; `reconciles` flags a ledger that does not add up."""

    saved = load_node(args.node) if args.node else None
    if args.node and saved is None:
        sys.stderr.write(
            f"ERROR: no saved config at ~/.iicp/nodes/{args.node}.json — run `iicp-node init` / `serve` first.\n"
        )
        return 1
    # UX: with no --node / --node-id, fall back to a saved config so a bare
    # `iicp-node credits` just works for the common single-node setup. Prefer an
    # explicit `default.json`; otherwise use the sole saved node if exactly one
    # exists. If 'default' exists but has no cached token, auto-fall-through to the
    # single node that does have a token rather than failing with a confusing error.
    if saved is None and not args.node_id:
        all_nodes = list_nodes()
        default_node = next((n for n in all_nodes if n.name == "default"), None)
        if len(all_nodes) == 1:
            saved = all_nodes[0]
        elif default_node is not None:
            if default_node.node_token:
                saved = default_node
            else:
                with_token = [n for n in all_nodes if n.node_token]
                if len(with_token) == 1:
                    sys.stderr.write(
                        f"[iicp-node] '{default_node.name}' has no cached token"
                        f" — using '{with_token[0].name}' instead\n"
                    )
                    saved = with_token[0]
                elif len(with_token) > 1:
                    # Multiple nodes have tokens — show all of them.
                    directory_url = (
                        args.directory_url
                        or _env("IICP_DIRECTORY_URL", "https://iicp.network/api")
                    )
                    sys.stderr.write(
                        f"[iicp-node] no --node given — showing credits for all"
                        f" {len(with_token)} nodes:\n"
                    )
                    # One node failing must not hide the others — show every
                    # node, then exit non-zero if any failed (2026-06-11).
                    failed = 0
                    for idx, n in enumerate(with_token):
                        if idx > 0:
                            print()
                        rc = await _fetch_and_display_credits(
                            n.directory_url or directory_url,
                            n.node_id, n.node_token, n.name,
                            args.json, args.verify,
                        )
                        if rc != 0:
                            sys.stderr.write(
                                f"ERROR: credits fetch failed for node '{n.name}'"
                                " — continuing with remaining nodes\n"
                            )
                            failed += 1
                    if failed:
                        sys.stderr.write(
                            f"ERROR: {failed}/{len(with_token)} node(s) failed\n"
                        )
                        return 1
                    return 0
                else:
                    saved = default_node  # no tokens anywhere; "run serve" fires below
    node_id = args.node_id or (saved.node_id if saved else None)
    token = args.token or (saved.node_token if saved else None)
    directory_url = (
        args.directory_url
        or (saved.directory_url if saved else None)
        or _env("IICP_DIRECTORY_URL", "https://iicp.network/api")
    )
    if not node_id:
        names = [n.name for n in list_nodes()]
        if names:
            sys.stderr.write(
                "ERROR: node_id required — multiple saved node configs exist, so the "
                "node is ambiguous.\n"
                f"  Saved nodes: {', '.join(names)}\n"
                "  Re-run with `--node <NAME>` (or `--node-id <ID>`).\n"
            )
        else:
            sys.stderr.write(
                "ERROR: node_id required (use --node NAME or --node-id ID).\n"
                "  No saved node configs found under ~/.iicp/nodes/ — run "
                "`iicp-node init` first.\n"
            )
        return 1
    if not token:
        sys.stderr.write(
            "ERROR: no node_token — run `iicp-node serve` once (it caches the token), "
            "or pass --token / $IICP_NODE_TOKEN\n"
        )
        return 1

    label = args.node or node_id
    return await _fetch_and_display_credits(
        directory_url, node_id, token, label, args.json, args.verify
    )


async def _fetch_and_display_credits(
    directory_url: str,
    node_id: str,
    token: str,
    label: str,
    as_json: bool,
    verify: bool,
) -> int:
    """Shared fetch+display logic for one node's credits summary."""
    import httpx

    url = directory_url.rstrip("/") + f"/v1/credits/summary?node_id={node_id}"
    # Transient failures (network error, 5xx, undecodable body) get ONE retry
    # after a short pause — shared-hosting blips and deploy windows otherwise
    # surface as one-shot CLI errors (observed 2026-06-11).
    resp = None
    body = None
    last_err = ""
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url, headers={"Authorization": f"Bearer {token}"}
                )
        except Exception as exc:  # noqa: BLE001
            last_err = f"request failed: {exc}"
            resp = None
        if resp is not None:
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                last_err = f"bad response (HTTP {resp.status_code})"
                body = None
            if body is not None and resp.status_code < 500:
                break  # success or definitive 4xx — no retry
            if body is not None:
                msg = (
                    (body.get("error") or {}).get("message", "request rejected")
                    if isinstance(body, dict)
                    else "request rejected"
                )
                last_err = f"HTTP {resp.status_code}: {msg}"
        if attempt == 1:
            await asyncio.sleep(2.0)

    if resp is None or body is None or resp.status_code >= 500:
        sys.stderr.write(f"ERROR: {last_err}\n")
        return 1
    if resp.status_code >= 400:
        msg = (
            (body.get("error") or {}).get("message", "request rejected")
            if isinstance(body, dict)
            else "request rejected"
        )
        sys.stderr.write(f"ERROR: HTTP {resp.status_code}: {msg}\n")
        return 1

    if as_json:
        print(json.dumps(body, indent=2))
        return 0

    earned = float(body.get("total_earned", 0.0))
    spent = float(body.get("total_spent", 0.0))
    balance = float(body.get("balance", 0.0))
    tx = int(body.get("tx_count", 0))
    reconciles = bool(body.get("reconciles", False))
    tpc = int(body.get("tokens_per_credit", 1000))
    print(f"IICP credits — {label}")
    print(f"  Earned (income)   {earned:>12.3f}")
    print(f"  Spent             {spent:>12.3f}")
    print("  ─────────────────────────────")
    check = "✓ reconciles" if reconciles else "✗ DOES NOT RECONCILE"
    print(f"  Balance           {balance:>12.3f}   {check}   (≈ {int(balance * tpc)} tokens)")
    print(f"  {tx} transactions · `iicp-node credits --json` for raw")
    if not reconciles:
        sys.stderr.write(
            "[iicp-node] WARNING: balance != earned − spent — the ledger does not "
            "reconcile; do not trust these figures.\n"
        )
    if verify:
        try:
            vsum, vcount, vfailed = await _verify_credit_awards(directory_url, node_id)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[iicp-node] --verify failed: {exc}\n")
            return 1
        print("  ── cryptographic verification (signed CREDIT_AWARD log) ──")
        if vfailed > 0:
            sys.stderr.write(
                f"[iicp-node] ✗ {vfailed} award event(s) FAILED Ed25519 verification — "
                "tampered or inconsistent event log. Do NOT trust these figures.\n"
            )
            return 1
        print(
            f"  ✓ {vcount} award(s) cryptographically verified · {vsum:.3f} credits "
            "(Ed25519, signed by the directory)"
        )
        free_tier = earned - vsum
        if free_tier > 0.0001:
            print(
                f"  · {free_tier:.3f} credits are free-tier allocation "
                "(directory-granted, not signed task awards)"
            )
    return 0


async def _cmd_operator_rename_async(args: argparse.Namespace) -> int:
    """`iicp-node operator rename <name>` (#460) — change the public, mutable display_name
    over the immutable operator_id. The operator signs the canonical rename bytes with their
    own key, so the directory authenticates the change by signature alone (no node token);
    one signed call updates the single operator record, reflected on every node + the
    leaderboard. Updates the local operator.json on success. Never sends the secret/contact."""
    import time

    import httpx

    from iicp_client.delegation import sign_rename

    op = load_operator()
    if op is None:
        sys.stderr.write("ERROR: no operator identity — run `iicp-node init` first.\n")
        return 1
    if not op.is_key_backed():
        sys.stderr.write(
            "ERROR: legacy keyless operator identity (operator_id is a UUID, not a key) — "
            "cannot sign a rename. Regenerate with a key-backed identity (#464).\n"
        )
        return 1
    new_name = args.name
    if not new_name or len(new_name) > 64 or any(ord(c) < 0x20 or ord(c) == 0x7F for c in new_name):
        sys.stderr.write("ERROR: display name must be 1-64 chars with no control characters.\n")
        return 1

    directory_url = args.directory_url or _env("IICP_DIRECTORY_URL", "https://iicp.network/api")
    ts = int(time.time())
    sig = sign_rename(op.signing_key(), new_name, op.operator_id, ts)
    payload = {"operator_pub": op.operator_id, "display_name": new_name, "ts": ts, "sig": sig}
    url = directory_url.rstrip("/") + "/v1/operator/rename"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: request failed: {exc}\n")
        return 1

    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = {}
    if resp.status_code >= 400:
        msg = (
            (body.get("error") or {}).get("message", "request rejected")
            if isinstance(body, dict)
            else "request rejected"
        )
        sys.stderr.write(f"ERROR: HTTP {resp.status_code}: {msg}\n")
        return 1

    # Persist the new name locally so the next `serve` re-asserts it at register.
    op.display_name = body.get("display_name", new_name) if isinstance(body, dict) else new_name
    save_operator(op)
    print(f"Renamed operator display_name to {op.display_name!r}.")
    return 0


def _operator_passphrase(prompt: str, *, confirm: bool) -> str | None:
    """Resolve a passphrase: $IICP_OPERATOR_PASSPHRASE if set (headless/CI), else an
    interactive getpass prompt (this command is operator-run, so a prompt is fine here —
    only `serve` must stay non-interactive)."""
    import getpass
    import os

    env = os.environ.get("IICP_OPERATOR_PASSPHRASE")
    if env:
        return env
    pw = getpass.getpass(prompt)
    if confirm and pw != getpass.getpass("Confirm passphrase: "):
        sys.stderr.write("ERROR: passphrases do not match.\n")
        return None
    return pw or None


def _cmd_operator_encrypt(args: argparse.Namespace) -> int:
    """`iicp-node operator encrypt` (#460) — seal the operator secret at rest under a passphrase."""
    op = load_operator()
    if op is None:
        sys.stderr.write("ERROR: no operator identity — run `iicp-node init` first.\n")
        return 1
    if op.is_encrypted():
        sys.stdout.write("Operator secret is already encrypted at rest.\n")
        return 0
    if not op.is_key_backed():
        sys.stderr.write("ERROR: legacy keyless operator identity has nothing to encrypt (#464).\n")
        return 1
    pw = _operator_passphrase("New operator passphrase: ", confirm=True)
    if not pw:
        sys.stderr.write("ERROR: a non-empty passphrase is required.\n")
        return 1
    save_operator(op.encrypt_at_rest(pw))
    sys.stdout.write(
        "Operator secret encrypted at rest (AES-256-GCM / PBKDF2). Set $IICP_OPERATOR_PASSPHRASE "
        "to unlock it headlessly during `serve`.\n"
    )
    return 0


def _cmd_operator_decrypt(args: argparse.Namespace) -> int:
    """`iicp-node operator decrypt` (#460) — restore the plaintext secret at rest."""
    op = load_operator()
    if op is None:
        sys.stderr.write("ERROR: no operator identity — run `iicp-node init` first.\n")
        return 1
    if not op.is_encrypted():
        sys.stdout.write("Operator secret is already stored in plaintext.\n")
        return 0
    pw = _operator_passphrase("Operator passphrase: ", confirm=False)
    if not pw:
        sys.stderr.write("ERROR: a passphrase is required to decrypt.\n")
        return 1
    try:
        plain = op.decrypt_at_rest(pw)
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    save_operator(plain)
    sys.stdout.write("Operator secret decrypted (now stored in plaintext at rest).\n")
    return 0


async def _verify_credit_awards(directory_url: str, node_id: str) -> tuple[float, int, int]:
    """#456 --verify: cryptographically confirm this node's CREDIT_AWARD income against the
    directory's signed event log (defends against a lying directory). Resolves the directory's
    Ed25519 key from /.well-known/did.json and re-derives + verifies each award signature.
    Returns (verified_sum, verified_count, failed_count)."""
    import base64
    import hashlib
    from urllib.parse import urlsplit

    import httpx
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    # #458 hash-chain genesis root: SHA256_hex("iicp:dir:event-log:genesis:v1"). Bound into the
    # signing input as prev_hash for a genesis/legacy event; the directory serves the real link otherwise.
    genesis_root = "c44802bedf3e63b5a3f1634c5d19263634f92f26dd15401b09b06dd53a80cf9d"
    sp = urlsplit(directory_url)
    origin = f"{sp.scheme}://{sp.netloc}"  # did.json lives at the host root, not under /api
    verified_sum, verified, failed = 0.0, 0, 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        did = (await client.get(f"{origin}/.well-known/did.json")).json()
        x = did["verificationMethod"][0]["publicKeyJwk"]["x"]
        pub = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
        vk = Ed25519PublicKey.from_public_bytes(pub)

        since = 0
        while True:
            resp = await client.get(
                f"{directory_url.rstrip('/')}/v1/events",
                params={"event_types": "CREDIT_AWARD", "since_seq": since, "limit": 500},
            )
            events = resp.json().get("events", [])
            if not events:
                break
            max_seq = since
            for e in events:
                seq = int(e.get("seq", 0))
                max_seq = max(max_seq, seq)
                if e.get("event_type") != "CREDIT_AWARD" or e.get("node_id") != node_id:
                    continue
                sig = e.get("sig")
                if not sig:
                    continue
                payload = e.get("payload", {})
                # Canonical form == the directory's (recursive key-sort, no-space, unescaped).
                canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                payload_hash = hashlib.sha256(canon.encode()).hexdigest()
                # #458: prev_hash (tamper-evident chain) is bound into the signing input.
                prev_hash = e.get("prev_hash") or genesis_root
                signing_input = (
                    f'{e.get("event_id", "")}:CREDIT_AWARD:{seq}:{int(e.get("ts_ms", 0))}'
                    f":{payload_hash}:{prev_hash}"
                )
                msg = hashlib.sha256(signing_input.encode()).digest()
                try:
                    vk.verify(bytes.fromhex(sig), msg)
                    verified += 1
                    verified_sum += float(payload.get("amount", 0.0))
                except (InvalidSignature, ValueError):
                    failed += 1
            if len(events) < 500 or max_seq <= since:
                break
            since = max_seq
    return verified_sum, verified, failed


async def _cmd_query_async(args: argparse.Namespace) -> int:
    from iicp_client.client import IicpClient
    from iicp_client.types import ClientConfig, TaskConstraints, TaskRequest

    prompt_text = " ".join(args.prompt)
    payload: dict = {"messages": [{"role": "user", "content": prompt_text}]}
    if args.model:
        payload["model"] = args.model
    if args.max_tokens:
        payload["max_tokens"] = args.max_tokens

    cfg = ClientConfig(
        directory_url=args.directory_url,
        timeout_ms=args.timeout_ms,
    )
    client = IicpClient(cfg)
    # #488: resolve source_node_id from saved node config for self-query neutrality.
    _query_source_node_id: str | None = None
    if getattr(args, "node", None):
        _saved_query = load_node(args.node)
        if _saved_query:
            _query_source_node_id = _saved_query.node_id
    req = TaskRequest(
        intent=args.intent,
        payload=payload,
        constraints=TaskConstraints(timeout_ms=args.timeout_ms),
        source_node_id=_query_source_node_id,
    )
    print(f"[iicp-node] Discovering nodes for {args.intent}...", file=sys.stderr)
    try:
        resp = await client.submit_async(req)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Spec terminal success status is "success" (was "completed"); accept both.
    if resp.status in ("success", "completed") and resp.result:
        content = resp.result.get("content") or json.dumps(resp.result, indent=2)
        print(content)
        if resp.metrics and resp.metrics.node_id:
            print(f"[iicp-node] routed to node {resp.metrics.node_id[:8]}", file=sys.stderr)
            if resp.metrics.latency_ms is not None:
                print(f"[iicp-node] latency {resp.metrics.latency_ms:.0f}ms", file=sys.stderr)
        return 0

    print(f"[iicp-node] task status: {resp.status}", file=sys.stderr)
    return 1


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
            sys.stderr.write(f"ERROR: no saved config at ~/.iicp/nodes/{args.node}.json. Run `iicp-node init` first.\n")
            return 2
        args.backend_url = args.backend_url or saved.backend_url
        args.model = args.model or saved.model
        args.public_endpoint = args.public_endpoint or saved.public_endpoint
        args.directory_url = args.directory_url or saved.directory_url
        args.region = args.region or saved.region
        args.intent = args.intent or saved.intent
        args.node_id = args.node_id or saved.node_id
        # Sentinel restore: --host/--port/--max-concurrent default to None when not
        # supplied on the CLI. Precedence is flag > env > saved-config > built-in.
        # The env value (if set) wins over the saved config; otherwise restore the
        # saved value. Built-in defaults are applied below for the all-unset case.
        if args.max_concurrent is None and _env("IICP_MAX_CONCURRENT") is None:
            args.max_concurrent = saved.max_concurrent
        if args.port is None and _env("IICP_PORT") is None:
            args.port = saved.port
        if args.host is None and _env("IICP_HOST") is None:
            args.host = saved.host
        # auto_detect_nat is a real bool (BooleanOptionalAction). Only restore the
        # saved opt-in when the operator did not pass --auto-detect-nat /
        # --no-auto-detect-nat and no env override is set — otherwise the explicit
        # CLI/env choice (including the --no- off-switch) is honoured.
        if (
            getattr(args, "auto_detect_nat_explicit", None) is None
            and _env("IICP_AUTO_DETECT_NAT") is None
            and not args.auto_detect_nat
            and saved.auto_detect_nat
        ):
            args.auto_detect_nat = True
        if not args.external_ip_probe_url and saved.external_ip_probe_url:
            args.external_ip_probe_url = saved.external_ip_probe_url

    # Apply env / built-in defaults for any sentinel still unset (no saved config,
    # or saved config did not provide the value). flag > env > built-in.
    if args.max_concurrent is None:
        args.max_concurrent = int(_env("IICP_MAX_CONCURRENT", "4") or "4")
    if args.port is None:
        args.port = int(_env("IICP_PORT", "9484") or "9484")
    if args.host is None:
        args.host = _env("IICP_HOST", "::")

    # #410 — built-in fallback applied LAST (after flag/env/saved-config), so the
    # Ollama default never shadows a saved-node backend_url. #414/C1 — an `anthropic`
    # backend defaults to the Anthropic API, not localhost Ollama.
    if not args.backend_url:
        args.backend_url = (
            "https://api.anthropic.com"
            if getattr(args, "backend_type", "") == "anthropic"
            else "http://localhost:11434"
        )

    # Onboarding: if no --model given, auto-select the first model the backend advertises
    # so a bare `iicp-node serve` just works (parity with Rust/TS).
    if not args.model and args.backend_url:
        _models = _ollama_models(args.backend_url, getattr(args, "backend_api_key", "") or "")
        if _models:
            args.model = _models[0]
            sys.stderr.write(f"no --model given — auto-selected '{args.model}' from {args.backend_url}\n")

    if not args.backend_url or not args.model:
        sys.stderr.write(
            "ERROR: --model is required (--backend-url defaults to http://localhost:11434). "
            "Set IICP_BACKEND_MODEL, or load via `--node <name>` after `iicp-node init`.\n"
        )
        return 2

    # Resolve the actual listen port before NAT detection: start at the
    # requested port (default 9484, the official IICP port) and auto-increment
    # to the next free port. This keeps one port per node (multiple models on
    # one node share it) while N nodes on one host each get a distinct port →
    # distinct pinhole. Skipped when the operator supplies an explicit
    # --public-endpoint (they own the port mapping in that case).
    if not args.public_endpoint:
        resolved_port = _find_available_port(args.host, args.port)
        if resolved_port != args.port:
            logger.info(
                "Port %d in use — auto-incremented to first free port %d.",
                args.port,
                resolved_port,
            )
        args.port = resolved_port

    node_id = (args.node_id or str(uuid.uuid4()))[:36]
    public_endpoint = args.public_endpoint or f"http://localhost:{args.port}"
    _log_dir_override: str | None = getattr(args, "log_dir", None)
    setup_node_log(node_id, _log_dir_override)

    relay_worker_ep: str | None = getattr(args, "relay_worker_endpoint", None)
    # #520 rung 5 — tri-state: True (forced), False (disabled), None (auto:
    # only when every other NAT path fails). CLI flag wins over IICP_TUNNEL.
    _tunnel_pref: bool | None = getattr(args, "tunnel", None)
    if _tunnel_pref is None:
        _env_tunnel = (_env("IICP_TUNNEL") or "").lower()
        if _env_tunnel in ("1", "true", "yes"):
            _tunnel_pref = True
        elif _env_tunnel in ("0", "false", "no"):
            _tunnel_pref = False
    _tunnel = None  # QuickTunnel handle — closed in the serve finally block
    _backend_flavor = _detect_backend_flavor(
        args.backend_url, getattr(args, "backend_api_key", "") or "", args.backend_type
    )
    sys.stderr.write(f"backend detected: {_backend_flavor}\n")
    # #463/#464 — bind the operator identity: issue a delegation FROM the (key-backed) operator
    # identity for this node and advertise the public display_name. The directory verifies the
    # delegation (operator_pub == operator_id) and records the operator record. Never sends the
    # operator's secret key or contact/email.
    _op = load_operator()
    _op_delegation = None
    _op_display_name = None
    _op_created_at = None
    _op_integrity_hash = None
    _identity_notice = no_identity_notice(_op)
    if _identity_notice is not None:
        # #503 — anonymous registration accrues no founder/recognition standing;
        # say so loudly instead of silently excluding the operator. Non-fatal.
        sys.stderr.write(_identity_notice + "\n")
    else:
        from iicp_client.delegation import issue_delegation

        _op_delegation = issue_delegation(_op.signing_key(), node_id)
        _op_display_name = _op.display_name or None
        _op_created_at = _op.created_at
        _op_integrity_hash = _op.operator_integrity_hash or None

    cfg = NodeConfig(
        node_id=node_id,
        endpoint=public_endpoint,
        intent=args.intent,
        model=args.model,
        backend=_backend_flavor,
        region=args.region,
        directory_url=args.directory_url,
        max_concurrent=args.max_concurrent,
        relay_worker_endpoint=relay_worker_ep or None,
        relay_capable=getattr(args, "relay_capable", False),
        relay_accept_port=getattr(args, "relay_accept_port", 9485),
        operator_delegation=_op_delegation,
        operator_display_name=_op_display_name,
        operator_created_at=_op_created_at,
        operator_integrity_hash=_op_integrity_hash,
        # TC-9c — pre-load saved HMAC key so receipts work immediately on restart.
        node_hmac_key=saved.node_hmac_key or "" if saved else "",
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
                    "NAT tier=%d: no direct or IPv6 endpoint available — querying directory for relay-capable peers.",
                    profile.tier,
                )
                elected_relay = await _auto_elect_relay(args.directory_url, cfg.intent, node_id)
                if elected_relay:
                    relay_host, relay_port = elected_relay
                    relay_worker_ep = f"{relay_host}:{relay_port}"
                    cfg.relay_worker_endpoint = relay_worker_ep
                    logger.info(
                        "NAT: auto-elected relay %s:%d — node will register via relay when connection is established.",
                        relay_host,
                        relay_port,
                    )
                else:
                    # #520 rung 5: no relay anywhere → Quick Tunnel (unless
                    # the operator disabled it with --no-tunnel/IICP_TUNNEL=0).
                    if _tunnel_pref is not False:
                        _tunnel = _open_tunnel_rung(node, args.port, forced=False)
                        if _tunnel is not None:
                            public_endpoint = _tunnel.url
                    if _tunnel is None:
                        logger.warning(
                            "NAT tier=%d: no relay-capable peers found in directory "
                            "and no tunnel available. Enabling mesh to discover "
                            "relays via gossip. Set IICP_RELAY_WORKER_ENDPOINT="
                            "<host>:<port> to specify a relay manually.",
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

    # #520 — `--tunnel` forces rung 5 regardless of NAT tier (e.g. an operator
    # who KNOWS they're unreachable, or wants an https endpoint for browser
    # consumers without touching the router).
    if _tunnel_pref is True and _tunnel is None:
        _tunnel = _open_tunnel_rung(node, args.port, forced=True)
        if _tunnel is not None:
            public_endpoint = _tunnel.url

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
        # #5 — Bearer key for auth'd backends (LM Studio, hosted). Empty = no header.
        api_key=getattr(args, "backend_api_key", "") or "",
    )

    # GAP-6: probe the backend for all available models and advertise them.
    # _ollama_models is best-effort; on any error it returns [] and we fall
    # back to the single configured model.
    discovered = _ollama_models(args.backend_url, getattr(args, "backend_api_key", "") or "")
    if discovered:
        extra = [m for m in discovered if m != args.model]
        if extra:
            cfg.capabilities = extra
            logger.info("GAP-6: advertising %d additional models: %s", len(extra), extra[:6])

    # #494 — wire backend_url into NodeConfig so heartbeat can probe live model list.
    cfg.backend_url = args.backend_url or ""
    cfg.backend_api_key = getattr(args, "backend_api_key", "") or ""

    # NAT-4 guard: if endpoint is non-routable and no relay configured, skip
    # registration to avoid a confusing 422 from the directory's RoutableEndpoint check.
    _ep = public_endpoint.lower()
    _ep_is_local = any(
        _ep.startswith(p)
        for p in ("http://localhost", "http://127.", "http://0.0.0.0", "http://192.168.", "http://10.")
    )
    if _ep_is_local and not relay_worker_ep and not args.skip_registration:
        logger.warning(
            "No routable endpoint detected and no relay configured — "
            "skipping directory registration. Node will accept direct connections "
            "but will not appear in discover results. "
            "Set IICP_PUBLIC_ENDPOINT=<url> or IICP_RELAY_WORKER_ENDPOINT=<host>:<port> to register."
        )
        args.skip_registration = True

    # #405 — single-instance lock: refuse a second LIVE process for this node_id
    # (the token-rotation war). Distinct node_ids are unaffected. Fails open.
    from iicp_client.instance_lock import InstanceLock, NodeAlreadyServingError

    try:
        _instance_lock = InstanceLock.acquire(node_id, force=getattr(args, "force", False))
    except NodeAlreadyServingError as exc:
        logger.error(str(exc))
        return 2

    # #457 / ADR-040 — advertise the native IICP binary transport. serve() multiplexes it
    # onto the SAME socket as HTTP (first-byte detection), so transport_endpoint shares the
    # endpoint's host:port with the iicp:// scheme. Derived from the FINAL endpoint (after NAT
    # profile application); register() only sends it when registering (skip_registration gates
    # the non-routable case) → advertise-when-reachable. Opt out: IICP_DISABLE_NATIVE_TRANSPORT=1.
    if not args.skip_registration and os.environ.get("IICP_DISABLE_NATIVE_TRANSPORT") != "1":
        from iicp_client.node import derive_native_endpoint

        _native_ep = derive_native_endpoint(node._cfg.endpoint)
        if _native_ep:
            node._cfg.transport_endpoint = _native_ep

    # #404 — register with bounded backoff retry. On persistent failure, pass an
    # empty token (NOT None) so the heartbeat loop still starts and re-registers on
    # the first 401 (#399 path) once the directory is reachable — the self-healing
    # watchdog, instead of the old "continuing without heartbeat" dead end.
    # None is reserved for --skip-registration (no heartbeat by design).
    token: str | None = None
    if not args.skip_registration:
        for attempt in range(1, 4):
            try:
                token = await node.register()
                logger.info("Registered as %s (token=%s…)", node_id, (token or "")[:8])
                _log_event(node_id, "register_ok", f"endpoint={public_endpoint}", _log_dir_override)
                # #456 / TC-9c — cache token + HMAC key in the saved config so
                # `iicp-node credits` can authenticate and CIPWorkerReceipts work
                # immediately on restart (best-effort).
                if getattr(args, "node", None) and token:
                    saved = load_node(args.node)
                    if saved is not None:
                        saved.node_token = token
                        hmac_key = node.node_hmac_key()
                        if hmac_key:
                            saved.node_hmac_key = hmac_key
                        try:
                            save_node(saved)
                        except OSError:
                            pass
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= 3:
                    logger.warning(
                        "Registration failed after %d attempts: %s — starting heartbeat loop "
                        "anyway; it will re-register on the first 401",
                        attempt,
                        exc,
                    )
                    _log_event(node_id, "register_fail", f"error={exc} attempts={attempt}", _log_dir_override)
                    token = ""  # empty (not None) → heartbeat loop starts and self-heals
                    break
                backoff = 2**attempt
                logger.warning("Registration attempt %d failed: %s — retrying in %ds", attempt, exc, backoff)
                await asyncio.sleep(backoff)

    logger.info(
        "Serving %s on %s:%d — backend %s (model=%s, max_concurrent=%d)",
        args.intent,
        args.host,
        args.port,
        args.backend_url,
        args.model,
        args.max_concurrent,
    )
    _log_event(
        node_id,
        "serve_start",
        f"port={args.port} model={args.model} intent={args.intent}",
        _log_dir_override,
    )
    proxy_task: asyncio.Task | None = None
    if getattr(args, "with_proxy", False):
        proxy_task = asyncio.create_task(_run_cohosted_proxy())
    try:
        await node.serve(handler, host=args.host, port=args.port, node_token=token)
    finally:
        if proxy_task is not None:
            proxy_task.cancel()  # node exited → stop the co-hosted proxy
        if _tunnel is not None:
            _tunnel.close()  # #520 — tear the Quick Tunnel down with the node
        _instance_lock.release()  # #405 — free the pidfile on shutdown
    return 0


def _open_tunnel_rung(node: IicpNode, local_port: int, *, forced: bool):
    """#520 rung 5: open a Quick Tunnel and wire it into the node lifecycle.

    Setup (binary detection), initiation (spawn + URL parse), supervision
    (watchdog respawns + re-registers on URL rotation) and the install hint
    all live here; teardown happens in the serve `finally` (+ atexit).
    Returns the QuickTunnel handle, or None when unavailable/failed —
    callers fall through to the next rung.

    Must be called from async context (captures the running loop so the
    watchdog thread can marshal `node.register()` back onto it).
    """
    from iicp_client.tunnel import INSTALL_HINT, cloudflared_path, open_quick_tunnel

    if not cloudflared_path():
        logger.warning(INSTALL_HINT)
        return None
    try:
        tunnel = open_quick_tunnel(local_port)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quick Tunnel failed to start: %s — continuing without it", exc)
        return None

    node._cfg.endpoint = tunnel.url  # noqa: SLF001 — same pattern as _on_relay_bind
    node._cfg.transport_method = "external_tunnel"  # noqa: SLF001
    loop = asyncio.get_running_loop()

    def _on_new_url(url: str) -> None:
        # Quick Tunnel URLs rotate per process — re-register with the new one.
        node._cfg.endpoint = url  # noqa: SLF001
        asyncio.run_coroutine_threadsafe(node.register(), loop)

    def _on_dead() -> None:
        logger.error(
            "Quick Tunnel permanently down — this node is no longer publicly "
            "reachable. Restart `iicp-node serve` to recover."
        )

    tunnel.watch(_on_new_url, _on_dead)

    # Teardown must survive SIGTERM too: a plain `kill` bypasses finally/atexit
    # in CPython, which would orphan the cloudflared child. Close the tunnel,
    # then re-raise with the default handler so exit semantics stay unchanged.
    def _terminate(signum: int, _frame: object) -> None:
        tunnel.close()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except ValueError:
        pass  # not on the main thread — finally/atexit still cover normal exits

    logger.info(
        "NAT rung 5%s: public https endpoint via Quick Tunnel — %s "
        "(zero-account; URL rotates on restart)",
        " (forced)" if forced else "",
        tunnel.url,
    )
    return tunnel


async def _auto_elect_relay(directory_url: str, intent: str, node_id: str) -> tuple[str, int] | None:
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

    candidates = [n for n in data.get("nodes", []) if n.get("relay_capable") and n.get("endpoint")]
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


def _detect_backend_flavor(backend_url: str, api_key: str = "", backend_type: str = "openai_compat") -> str:
    """Detect the backend server flavor for the `backend` node-detail field:
    ollama / lmstudio / vllm / llamacpp / anthropic / custom. Mirrors
    iicp-client-rust. For non-OpenAI dialects the configured backend_type is
    authoritative; for openai_compat it fingerprints /v1/models response headers
    — X-Powered-By:Express → lmstudio (LM Studio also serves Ollama-compatible
    /api/version + /api/tags, so the Express header is the discriminator, not those
    endpoints), uvicorn/vllm → vllm, llama → llamacpp, else probe /api/version →
    ollama, else custom (generic OpenAI-compatible)."""
    if backend_type in ("anthropic", "vllm", "llamacpp"):
        return backend_type
    base = backend_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _ok(path: str):
        try:
            req = urllib.request.Request(f"{root}{path}", headers=headers)
            return urllib.request.urlopen(req, timeout=2)
        except Exception:  # noqa: BLE001
            return None

    resp = _ok("/v1/models")
    if resp is not None:
        with resp:
            powered = (resp.headers.get("X-Powered-By") or "").lower()
            server = (resp.headers.get("Server") or "").lower()
        if "express" in powered:
            return "lmstudio"
        if "vllm" in server or "uvicorn" in server:
            return "vllm"
        if "llama.cpp" in server or "llama-server" in server:
            return "llamacpp"
        v = _ok("/api/version")
        if v is not None:
            v.close()
            return "ollama"
        return "custom"
    # No /v1/models (older Ollama) — try the proprietary endpoint.
    v = _ok("/api/version")
    if v is not None:
        v.close()
        return "ollama"
    return "custom"


def _ollama_models(backend_url: str, api_key: str = "") -> list[str]:
    """Best-effort: list backend models. Empty list on any error.

    #409 — strip a trailing /v1 to a root so the probe URLs are well-formed
    whether the operator passed `http://host:11434` (Ollama) or
    `http://host:1234/v1` (LM Studio / OpenAI-compat). Tries Ollama /api/tags
    then OpenAI /v1/models, attaching the Bearer key (LM Studio /v1/models 401s
    without it) so multi-intent discovery works against auth'd backends.
    """
    base = backend_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # Ollama /api/tags ({"models":[{"name":...}]})
    try:
        req = urllib.request.Request(f"{root}/api/tags", headers=headers)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            names = sorted({m["name"] for m in data.get("models", [])})
            if names:
                return names
    except Exception:  # noqa: BLE001
        pass
    # OpenAI-compat /v1/models ({"data":[{"id":...}]})
    try:
        req = urllib.request.Request(f"{root}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", []) if m.get("id")]
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
    backend_url = _prompt("Backend URL (OpenAI-compatible)", "http://localhost:11434")
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
    region = _prompt("Region tag (e.g. us-east, eu-central; blank = unknown)", "unknown")
    directory_url = _prompt("Directory URL", "https://iicp.network/api")
    port_str = _prompt("Local HTTP port", "9484")
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
    if any(i.severity in ("optional", "missing") and i.installable for i in issues):
        ans = _prompt("Enable optional deps now? (your node runs without them) (Y/n)", "y").lower()
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
    # "ok"       — present
    # "optional" — opt-in capability not installed; node runs fine without it
    # "warn"     — degraded runtime state (backend unreachable, no IPv6)
    # "missing"  — required dependency absent
    severity: str
    message: str
    installable: bool = False
    pip_extra: str = ""


def _check_dependencies(backend_url: str) -> list[_DepIssue]:
    """Probe runtime + optional deps + backend reachability."""
    out: list[_DepIssue] = []

    # 1) Backend reachability
    try:
        with urllib.request.urlopen(backend_url.rstrip("/") + "/api/tags", timeout=2) as resp:
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
            out.append(
                _DepIssue(
                    mod,
                    "optional",
                    f"{purpose} (optional — not installed)",
                    installable=True,
                    pip_extra=extra,
                )
            )

    # 3) IPv6 routing surface (advisory — doesn't gate anything)
    try:
        import asyncio

        from iicp_client.nat_detection import detect_ipv6

        v6 = (
            asyncio.get_event_loop().run_until_complete(detect_ipv6(0, timeout_s=1.5))
            if not asyncio.get_event_loop().is_running()
            else None
        )
    except Exception:  # noqa: BLE001
        v6 = None
    if v6 and v6.global_v6_available:
        msg = f"{len(v6.addresses)} global IPv6 address(es)"
        if v6.external_v6_reachable:
            msg += "; outbound v6 reachable"
        out.append(_DepIssue("ipv6", "ok", msg))
    elif v6:
        out.append(
            _DepIssue(
                "ipv6",
                "warn",
                "no global IPv6 — direct hosting will require IPv4 + tunnel",
            )
        )

    return out


def _print_dep_status(issues: list[_DepIssue]) -> None:
    glyph = {"ok": "  ✓", "optional": "  ○", "warn": "  !", "missing": "  ✗"}
    for i in issues:
        print(f"{glyph.get(i.severity, '  ?')} {i.name:18}  {i.message}")


def _install_missing(issues: list[_DepIssue]) -> None:
    """Run `pip install iicp-client[<extras>]` for the missing-optional set."""
    extras = sorted(
        {i.pip_extra for i in issues if i.severity in ("optional", "missing") and i.installable and i.pip_extra}
    )
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
        print(f"{n.name.ljust(width)}  {n.model[:20].ljust(20)} {n.backend_url[:48]}")
    return 0


async def _run_cohosted_proxy() -> None:
    """`serve --with-proxy` (2-C) — run the compat gateway on loopback alongside the
    node, supervised so a proxy failure logs but never drops the network-facing node.
    The proxy is forced to 127.0.0.1 when co-hosted (consumer trust boundary)."""
    try:
        import uvicorn

        from iicp_client.proxy.config import ProxyConfig
        from iicp_client.proxy.main import create_app
    except ModuleNotFoundError as exc:
        logger.error(
            "--with-proxy needs the [proxy] extra (missing %s); the node continues "
            "WITHOUT the proxy. Install: pip install 'iicp-client[proxy]'",
            exc.name,
        )
        return
    pcfg = ProxyConfig.from_toml("proxy.toml")
    pcfg.host = "127.0.0.1"  # force loopback when co-hosted
    server = uvicorn.Server(
        uvicorn.Config(create_app(pcfg), host="127.0.0.1", port=pcfg.port, log_level="warning", server_header=False)
    )
    try:
        logger.info("co-hosted proxy → http://127.0.0.1:%d (OpenAI/Ollama/Anthropic compat)", pcfg.port)
        await server.serve()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — supervise: proxy crash must not drop the node
        logger.error("co-hosted proxy crashed (node keeps running): %s", exc)


def _cmd_proxy(args: argparse.Namespace) -> int:
    """`iicp-node proxy` (ADR-050) — run the compat gateway. Lazy-imports the
    [proxy] extra so plain library / `serve` / `query` use never needs FastAPI."""
    try:
        import uvicorn

        from iicp_client.proxy.config import ProxyConfig
        from iicp_client.proxy.main import create_app
    except ModuleNotFoundError as exc:
        print(
            f"The proxy gateway needs the optional [proxy] extra (missing: {exc.name}).\n"
            "  Install:  pip install 'iicp-client[proxy]'  (or pipx install 'iicp-client[proxy]')",
            file=sys.stderr,
        )
        return 2
    cfg = ProxyConfig.from_toml(args.config)
    # Flag/env precedence: explicit --host/--port override the TOML/env-loaded config.
    cfg.host = args.host
    cfg.port = args.port
    print(f"iicp-node proxy → http://{cfg.host}:{cfg.port} (OpenAI/Ollama/Anthropic compat; no directory registration)")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, server_header=False)
    return 0


def _cmd_mcp_gateway(args: argparse.Namespace) -> int:
    """``iicp-node mcp-gateway`` — bridge a local MCP server as an IICP provider node.

    Wraps each tool as ``urn:iicp:intent:mcp:<tool>:v1``, registers with the
    directory, runs a heartbeat loop, and serves ``POST /v1/task`` by forwarding
    to the MCP server's ``tools/call`` JSON-RPC endpoint.
    """
    import re
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    _DANGEROUS = {"bash", "shell", "exec", "run_command", "eval"}

    def _tool_to_intent(name: str) -> str:
        safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
        return f"urn:iicp:intent:mcp:{safe}:v1"

    raw_tools = [t.strip() for t in (args.tools or "").split(",") if t.strip()]
    active_tools = [t for t in raw_tools if t.lower() not in _DANGEROUS]
    if not active_tools:
        sys.stderr.write(
            "ERROR: --tools is required. Provide a comma-separated list of MCP tool names.\n"
            "  Example: iicp-node mcp-gateway --tools read_file,list_dir --mcp-url http://localhost:8001\n"
        )
        return 2

    node_id = args.node_id or f"gateway-mcp-{uuid.uuid4().hex[:8]}"
    directory_url = (args.directory_url or "https://iicp.network/api/v1").rstrip("/")
    mcp_url = (args.mcp_url or "http://localhost:8001").rstrip("/")
    region = args.region or "local"
    port = args.port or 9484
    host = args.host or "::"
    public_endpoint = args.public_endpoint or f"http://localhost:{port}"
    node_token_env = _env("IICP_NODE_TOKEN", "") or ""

    intents = [_tool_to_intent(t) for t in active_tools]
    _live: dict = {"token": node_token_env, "mcp_rpc_id": 0}

    def _register() -> str:
        payload = {
            "node_id": node_id,
            "region": region,
            "endpoint": public_endpoint,
            "intents": intents,
            "mcp_tools": active_tools,
            "protocol_version": "1.0",
        }
        headers = {"Authorization": f"Bearer {_live['token']}"} if _live["token"] else {}
        r = httpx.post(f"{directory_url}/register", json=payload, headers=headers, timeout=10.0)
        r.raise_for_status()
        return r.json().get("node_token", _live["token"])

    def _heartbeat() -> None:
        payload = {"node_id": node_id, "intents": intents, "load": 0.0, "status": "active"}
        try:
            httpx.post(
                f"{directory_url}/heartbeat",
                json=payload,
                headers={"Authorization": f"Bearer {_live['token']}"},
                timeout=10.0,
            ).raise_for_status()
        except httpx.HTTPError:
            pass

    def _call_mcp(tool_name: str, arguments: dict) -> object:
        _live["mcp_rpc_id"] += 1
        rpc = {
            "jsonrpc": "2.0",
            "id": _live["mcp_rpc_id"],
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        r = httpx.post(f"{mcp_url}/mcp", json=rpc, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise ValueError(data["error"].get("message", "MCP error"))
        return data.get("result")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # suppress access log
            pass

        def _send_json(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/iicp/health":
                self._send_json(200, {
                    "status": "ok",
                    "node_id": node_id,
                    "active_tools": active_tools,
                    "mcp_server": mcp_url,
                    "timestamp": int(time.time()),
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/task":
                self._send_json(404, {"error": "not found"})
                return
            auth = self.headers.get("Authorization", "")
            if not auth or auth != f"Bearer {_live['token']}":
                self._send_json(401, {"error": "Unauthorized"})
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            payload = body.get("payload", {})
            tool_name: str = payload.get("tool_name", "")
            if not tool_name:
                m = re.search(r"urn:iicp:intent:mcp:([^:]+):v1", body.get("intent", ""))
                if m:
                    tool_name = m.group(1)
            if not tool_name:
                self._send_json(400, {"error": "Cannot determine tool name from payload or intent"})
                return
            if tool_name.lower() in _DANGEROUS:
                self._send_json(403, {"error": "Tool not permitted"})
                return
            if active_tools and tool_name not in active_tools:
                self._send_json(404, {"error": "Tool not available on this gateway"})
                return
            task_id = body.get("task_id", str(uuid.uuid4()))
            try:
                result = _call_mcp(tool_name, payload.get("arguments", {}))
            except httpx.HTTPError:
                self._send_json(502, {"error": "MCP server unreachable"})
                return
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            self._send_json(200, {"task_id": task_id, "status": "completed", "result": result})

    # Register + start
    try:
        _live["token"] = _register()
        sys.stdout.write(
            f"iicp-node mcp-gateway registered as {node_id!r} with {len(active_tools)} tool(s): "
            f"{', '.join(active_tools)}\n"
            f"  IICP endpoint: {public_endpoint}\n"
            f"  MCP server:    {mcp_url}\n"
        )
    except httpx.HTTPError as exc:
        sys.stderr.write(f"WARNING: directory registration failed ({exc}) — running without listing\n")

    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(30):
            _heartbeat()

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

    bind_host = "::" if host in ("", "::") else host
    server = ThreadingHTTPServer((bind_host, port), _Handler)
    sys.stdout.write(f"  Listening on {bind_host}:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.cmd == "help":
        parser.print_help()
        return 0
    if args.cmd == "serve":
        # Record whether the operator explicitly toggled the NAT flag on the CLI
        # so saved-config restore can honour an explicit --no-auto-detect-nat
        # rather than silently restoring a saved opt-in. (BooleanOptionalAction
        # collapses the on/off forms to a plain bool, losing explicitness.)
        args.auto_detect_nat_explicit = (
            True
            if ("--auto-detect-nat" in raw_argv or "--no-auto-detect-nat" in raw_argv)
            else None
        )
        return asyncio.run(_serve(args))
    if args.cmd == "proxy":
        return _cmd_proxy(args)
    if args.cmd == "mcp-gateway":
        return _cmd_mcp_gateway(args)
    if args.cmd == "init":
        return _cmd_init(args)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "query":
        return asyncio.run(_cmd_query_async(args))
    if args.cmd == "credits":
        return asyncio.run(_cmd_credits_async(args))
    if args.cmd == "operator":
        if args.op_cmd == "rename":
            return asyncio.run(_cmd_operator_rename_async(args))
        if args.op_cmd == "encrypt":
            return _cmd_operator_encrypt(args)
        if args.op_cmd == "decrypt":
            return _cmd_operator_decrypt(args)
        parser.error(f"unknown operator subcommand: {args.op_cmd}")
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
