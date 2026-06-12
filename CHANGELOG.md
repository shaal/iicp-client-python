# Changelog

All notable changes to the IICP Python SDK (`iicp-client`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
within the scope of the IICP Software axis (see [`VERSIONING.md`](https://github.com/RobLe3/iicp.network/blob/main/project/VERSIONING.md)
in the main repo).

## [0.7.58] — 2026-06-12

### Security — relay session cap (red-team F5)

- The relay caps concurrent worker sessions (default 256); new binds past the
  cap are rejected (HTTP 503 `IICP-E039` / TCP `RELAY_ACK` error), closing a
  bind-flood memory-exhaustion DoS. A rebind of an existing worker_id is exempt.

### Added — `iicp-node update --check`

- Read-only check for a newer published release (numeric version compare) with
  the exact upgrade command. Exit 10 when a newer release exists, 0 otherwise.

## [0.7.57] — 2026-06-12

### Added — automatic Quick-Tunnel escalation (NAT ladder rung 5, #520)

- When every NAT path fails (no direct endpoint, no UPnP pinhole, no IPv6
  GUA, no relay-capable peer in the directory), the node now exposes itself
  via a zero-account Cloudflare Quick Tunnel automatically: detect
  `cloudflared` on PATH (never auto-installed — one actionable install hint
  when missing), spawn it, register the issued `https://*.trycloudflare.com`
  URL as the endpoint (`transport_method=external_tunnel`), supervise the
  child (bounded respawn ×3), and tear it down with the node on every exit
  path.
- `--tunnel` forces the rung regardless of NAT tier (e.g. to get an https
  endpoint for browser consumers without touching the router);
  `--no-tunnel` / `IICP_TUNNEL=0` disables the automatic escalation.

## [0.7.56] — 2026-06-12

(Also includes the never-published 0.7.55 changes: MCP gateway as a built-in
`iicp-node mcp-gateway` feature.)

### Added — HTTP long-poll relay worker transport (#450)

- Relay-capable nodes accept browser-compatible workers over plain HTTP:
  `POST /v1/relay/bind` (bearer session token; 409 on alive-rebind, #510
  interim-C), `GET /v1/relay/pull` (long-poll ≤25 s), `POST /v1/relay/result`,
  `POST /v1/relay/unbind` — same session registry as TCP RELAY_BIND workers.
- Path-scoped worker endpoints `{relay}/v1/relay-for/<worker_id>/v1/task` +
  `/iicp/health`: published consumers route through the relay with no client
  changes. RELAY_ACK gains additive field 4 (the relay's HTTP task port).

### Fixed — relay-bound workers were silently misattributed

- Relay workers previously advertised the bare relay endpoint, so consumer
  dispatches executed **on the relay itself** instead of forwarding (and used
  the non-HTTP accept port). Workers now register the path-scoped endpoint.

### Changed — CORS on every node HTTP endpoint

- All node responses carry `Access-Control-Allow-Origin: *` and every path
  answers `OPTIONS` preflights. Web pages (e.g. iicp.network/browser-node)
  are first-class consumers: an https-exposed node now serves browser
  dispatches directly. No new capability — CORS only ever gated browsers;
  curl was never restricted.

## [0.7.54] — 2026-06-11

### Fixed — `iicp-node credits` resilience

- Transient failures (network error, 5xx, undecodable body) are retried once after
  a 2s pause — deploy windows / shared-hosting blips no longer surface as one-shot
  CLI errors (`HTTP 500` / `bad response: error decoding response body`).
- All-nodes listing (bare `iicp-node credits` with multiple saved nodes): one
  node's failure no longer aborts the whole listing — every node is shown and the
  command exits non-zero with an `N/M node(s) failed` summary.

## [0.7.53] — 2026-06-11

### Added — model-drift re-registration (#494)

- Each heartbeat tick compares the backend's live model list against the registered
  set and automatically re-registers when they diverge — directory registration no
  longer goes stale when Ollama loads/unloads models.

## [0.7.52] — 2026-06-10

### Added

- #496 Phase-2 consumer token support.
- `models[]` array on the `/iicp/health` endpoint (#494).
- #503 loud CLI notice when serving without an operator identity.

## [0.7.51] — 2026-06-10

### Added — health_models heartbeat reporting (#494)

- **`backend_url` / `backend_api_key`** in `NodeConfig` — when set, each heartbeat probes
  the backend's live model list (`/api/tags` for Ollama, `/v1/models` for OpenAI-compatible
  backends) and sends `health_models=[...]` in the heartbeat payload.
- The directory (≥ v1.10.28) uses `health_models` to filter `?model=` discover queries
  to nodes whose backend actually has that model loaded, eliminating stale-model routing.
- Probe failures are soft — heartbeat still fires without `health_models` (backward compat).
- 3 behavior tests added (`test_serve.py`).

## [0.7.40] — 2026-06-07

### Fixed — CLI usability hardening (no friction for new operators)

- **`proxy` now listed in `iicp-node --help`** + all serve flags documented.
- **Every subcommand `--help`/`-h` prints usage** instead of crashing.
- **Friendly parse errors** — unknown flags print `ERROR: …` (exit 2) instead of tracebacks.
- **`iicp-node serve --model X` works without `--backend-url`** — `localhost:11434` default applied unconditionally.
- **`--no-auto-detect-nat`** off-switch; `iicp-node help` prints usage; `credits` auto-resolves single node. Cross-flavour CLI parity (3-C).

## [0.7.39] — 2026-06-07

### Added — unified client: local OpenAI/Ollama/Anthropic-compat proxy (ADR-050, #476)

- **`iicp-node proxy`** — a local compat gateway on `127.0.0.1:9483`. Speaks OpenAI
  (`/v1/chat/completions`, `/v1/models`), Ollama (`/api/chat`, `/api/generate`, `/api/tags`),
  and Anthropic (`/v1/messages`) and routes each request across the IICP mesh.
- **`iicp-node serve --with-proxy`** — co-host the proxy next to a provider node in one process.
- **CIP consumer gating** in the proxy path — `IICP-E036` → 402, `IICP-E022` → 503.
- One client now does **node + query + proxy**; the standalone `iicp-proxy` package is retired.

## [0.7.36–0.7.38] — 2026-06-03..06

- Maintenance + lockstep version alignment across Python/TS/Rust SDKs (3-C). No API changes.

## [0.7.35] — 2026-06-03

### Added — native Anthropic backend + audio chat modality (#414, capability roadmap)

- **`backend_type="anthropic"`** — speaks the Anthropic Messages API directly; defaults
  `backend_url` to `https://api.anthropic.com`.
- **Audio modality detection** — model names containing `audio`, `voxtral`, or `omni`
  advertise `input_modalities: ["audio"]`.

### Added — heartbeat liveness challenge (ADR-047 Part A, #411)

- The heartbeat loop answers the directory's liveness challenge.

## [0.7.34] — 2026-06-03

### Added — operator delegation at registration (ADR-045 Phase A, #407)

- The node signs an ed25519 operator delegation on `register`.

## [0.7.33] — 2026-06-03

### Added — multimodal capability advertising (ADR-046, #408)

- `build_capabilities` advertises `input_modalities` (text + image for vision models).

## [0.7.32] — 2026-06-03

### Added — multi-intent advertising (#409)

- A node advertises every intent its backend serves (chat + embedding).

## [0.7.31] — 2026-06-02

### Fixed — backend_url precedence regression-lock (#410)

## [0.7.30] — 2026-06-02

### Added — Bearer auth for OpenAI-compat backends (#5)

- **`--backend-api-key` / `IICP_BACKEND_API_KEY`**.

## [0.7.29] — 2026-06-02

### Fixed — single-instance lock prevents duplicate-node thrash (#405)

- Per-node pidfile; `--force` / `IICP_FORCE` to take over.

## [0.7.28] — 2026-06-02

### Fixed — node no longer needs restart to reconnect (#404, reliability)

- Registration retries with backoff; heartbeat loop re-registers on 401.

## [0.7.27] — 2026-06-02

### Fixed — CIP policy now enforced on incoming tasks (#403, security)

- `cip_gate` rejects tool-execution-domain intents unless the operator opted in.

## [0.7.26] — 2026-06-02

### Added — transport on parsed discover nodes (#397)

- `Node.transport: list[str]` — protocols each node speaks.

## [0.7.25] — 2026-06-02

### Fixed — node recovers after the directory drops it (#399)

- Heartbeat loop re-registers on node-unknown rejection (404/401/410).

## [0.7.24] — 2026-06-02

### Changed — onboarding clarity

- `iicp-node init` distinguishes optional capabilities from real problems.

## [0.5.x] — 2026-05-27

- 0.5.3: CBOR wire-compat fix (integer keys, RFC 8949); 3×3 cross-SDK matrix verified.
- 0.5.2: ConcurrencyGate parity port (Tier 2 Item 5).
- 0.5.1: CONF self-conformance probes (Tier 2 Item 4).
- 0.5.0: ADR-019 declarative pricing + HMAC signing (Tier 2 Item 3).

## Earlier 0.x releases

See git log — the Tier 1 ports (transport_endpoint, IICP TCP, UPnP, openai_compat,
NAT observability) and Tier 2 items (CIP policy, pricing, conformance, ConcurrencyGate)
shipped across iter-1409..1440 of the main repo's FORGE loop.
