"""Behavior tests for the 0.7.40 CLI usability fixes (commit 35034cd).

Each test asserts a real behavior of `iicp_client.cli` that would FAIL if the
0.7.40 fix were reverted — not a smoke test:

1. `help` top-level alias prints usage + exits 0 (was `invalid choice: 'help'`).
2. `--no-auto-detect-nat` off-switch (BooleanOptionalAction) actually flips the
   resolved bool; bare `serve` and `--auto-detect-nat` stay enabled.
3. `credits` no-arg node resolution: sole node auto-selected, `default.json`
   preferred, ambiguous >=2 errors with the node names, zero nodes points at
   `iicp-node init`. Resolution is asserted via the post-resolution token branch
   so no network call is made.
4. `serve --model X` backend-url fallback: localhost:11434 (openai_compat) /
   api.anthropic.com (anthropic) when no --backend-url/env is supplied.

All node-config tests redirect HOME via the IICP_HOME env var to a tmp_path so
the real ~/.iicp is never touched, and the network layer is never reached.
"""

from __future__ import annotations

import asyncio

import pytest

from iicp_client import cli
from iicp_client.identity import NodeIdentity, save_node


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def iicp_home(tmp_path, monkeypatch):
    """Point ~/.iicp at a throwaway tmp dir via IICP_HOME (config_dir() honours it)."""
    home = tmp_path / "iicp_home"
    monkeypatch.setenv("IICP_HOME", str(home))
    # Make sure no stray env leaks into the resolution branches under test.
    for var in (
        "IICP_NODE_TOKEN",
        "IICP_DIRECTORY_URL",
        "IICP_AUTO_DETECT_NAT",
        "IICP_BACKEND_URL",
        "IICP_BACKEND_TYPE",
        "IICP_PORT",
        "IICP_HOST",
        "IICP_MAX_CONCURRENT",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def _make_node(name: str, *, token: str | None = None) -> NodeIdentity:
    node = NodeIdentity.generate(
        operator_id="op-test",
        name=name,
        backend_url="http://localhost:11434",
        model="qwen2.5:0.5b",
    )
    node.node_token = token
    save_node(node)
    return node


# --------------------------------------------------------------------------- #
# Fix 1 — `help` top-level command alias
# --------------------------------------------------------------------------- #
def test_help_alias_returns_zero_and_prints_usage(capsys):
    rc = cli.main(["help"])
    assert rc == 0
    out = capsys.readouterr().out
    # Top-level usage banner, and the command list must enumerate `proxy`.
    assert "usage" in out.lower()
    assert "proxy" in out


def test_help_alias_is_a_real_command_not_an_error():
    # Reverting the fix made `help` an `invalid choice`, which argparse turns
    # into SystemExit(2). The fix must let `help` resolve to rc 0 instead.
    rc = cli.main(["help"])
    assert rc == 0


# --------------------------------------------------------------------------- #
# Fix 2 — `--no-auto-detect-nat` off-switch (BooleanOptionalAction)
# --------------------------------------------------------------------------- #
def _parse_serve(*extra: str, monkeypatch):
    # Isolate from a developer's exported IICP_AUTO_DETECT_NAT.
    monkeypatch.delenv("IICP_AUTO_DETECT_NAT", raising=False)
    parser = cli._build_parser()
    return parser.parse_args(["serve", *extra])


def test_no_auto_detect_nat_disables(monkeypatch):
    args = _parse_serve("--no-auto-detect-nat", monkeypatch=monkeypatch)
    assert args.auto_detect_nat is False


def test_serve_default_auto_detect_nat_enabled(monkeypatch):
    args = _parse_serve(monkeypatch=monkeypatch)
    assert args.auto_detect_nat is True


def test_auto_detect_nat_explicit_on(monkeypatch):
    args = _parse_serve("--auto-detect-nat", monkeypatch=monkeypatch)
    assert args.auto_detect_nat is True


def test_no_auto_detect_nat_flag_is_registered(monkeypatch):
    # Pre-fix, --auto-detect-nat was a store_true with no off-switch, so
    # `--no-auto-detect-nat` raised "unrecognized arguments" → SystemExit.
    # The fix registers it via BooleanOptionalAction; parsing must not raise.
    args = _parse_serve("--no-auto-detect-nat", monkeypatch=monkeypatch)
    assert args.auto_detect_nat is False


# --------------------------------------------------------------------------- #
# Fix 3 — `credits` no-arg node resolution
# --------------------------------------------------------------------------- #
def _run_credits(monkeypatch, *extra: str) -> tuple[int, str]:
    """Run `credits` past arg parsing and capture (rc, stderr).

    Nodes saved without a token short-circuit at the post-resolution token
    check, so a successful node resolution is proven WITHOUT any network call.
    Guard against accidental network use by forbidding httpx.AsyncClient.
    """
    import iicp_client.cli as cli_mod

    def _no_network(*a, **k):  # pragma: no cover - defensive
        raise AssertionError("network must not be reached in resolution test")

    monkeypatch.setattr("httpx.AsyncClient", _no_network, raising=False)

    parser = cli_mod._build_parser()
    args = parser.parse_args(["credits", *extra])
    captured: dict[str, str] = {"err": ""}

    real_write = cli_mod.sys.stderr.write

    def _capture(s):
        captured["err"] += s
        return len(s)

    monkeypatch.setattr(cli_mod.sys.stderr, "write", _capture)
    try:
        rc = asyncio.run(cli_mod._cmd_credits_async(args))
    finally:
        monkeypatch.setattr(cli_mod.sys.stderr, "write", real_write, raising=False)
    return rc, captured["err"]


def test_credits_sole_node_auto_selected(iicp_home, monkeypatch):
    _make_node("solo", token=None)
    rc, err = _run_credits(monkeypatch)
    # Resolution succeeded (single node found) → it got PAST node_id-required
    # and failed instead at the token check.
    assert "node_id required" not in err
    assert "node_token" in err
    assert rc == 1


def test_credits_default_node_preferred_among_many(iicp_home, monkeypatch):
    _make_node("alpha", token=None)
    _make_node("default", token=None)
    _make_node("beta", token=None)
    rc, err = _run_credits(monkeypatch)
    # `default.json` is preferred over the ambiguity error even with >=2 nodes.
    assert "node_id required" not in err
    assert "node_token" in err
    assert rc == 1


def test_credits_default_no_token_falls_back_to_sole_token_node(iicp_home, monkeypatch):
    # Regression: `default.json` exists but has no cached token; exactly one other
    # node has a token. `credits` must auto-select that node rather than failing with
    # "no node_token — run serve" on a node that was never served.
    _make_node("default", token=None)
    _make_node("ollama", token="tok-ollama")
    rc, err = _run_credits(monkeypatch)
    # Should NOT fail at token check — it found "ollama" and used it.
    assert "no cached token" in err  # hint about the fallback printed to stderr
    assert "ollama" in err
    # The token check passes → fails at the network call (which is forbidden), but
    # the resolution error is gone — this proves the right branch was taken.
    assert "node_id required" not in err
    assert rc == 1  # fails at token (since tok-ollama is accepted), or mock-network


def test_credits_default_no_token_multiple_token_nodes_shows_all(iicp_home, monkeypatch):
    # 0.7.44 fix: when multiple nodes have tokens, the command shows credits for ALL
    # of them rather than emitting a listing error. Behavior test: if the fix is
    # reverted, "showing credits for all" disappears and "no cached token" reappears.
    _make_node("default", token=None)
    _make_node("ollama", token="tok-a")
    _make_node("lmstudio", token="tok-b")

    import iicp_client.cli as cli_mod

    fetched_for: list[str] = []

    async def _mock_fetch(directory_url, node_id, token, label, as_json, verify):
        fetched_for.append(label)
        return 0

    monkeypatch.setattr(cli_mod, "_fetch_and_display_credits", _mock_fetch)

    captured_err: list[str] = []
    monkeypatch.setattr(cli_mod.sys.stderr, "write", lambda s: captured_err.append(s) or len(s))

    parser = cli_mod._build_parser()
    args = parser.parse_args(["credits"])
    rc = asyncio.run(cli_mod._cmd_credits_async(args))

    err = "".join(captured_err)
    assert rc == 0
    assert "showing credits for all" in err  # new routing message
    assert "ollama" in fetched_for
    assert "lmstudio" in fetched_for
    assert "no cached token" not in err  # old listing behavior must be gone


def test_credits_ambiguous_lists_node_names(iicp_home, monkeypatch):
    _make_node("alpha", token=None)
    _make_node("beta", token=None)
    rc, err = _run_credits(monkeypatch)
    assert rc == 1
    assert "node_id required" in err
    # The error must enumerate the saved node names (the 0.7.40 improvement).
    assert "alpha" in err
    assert "beta" in err


def test_credits_zero_nodes_points_to_init(iicp_home, monkeypatch):
    rc, err = _run_credits(monkeypatch)
    assert rc == 1
    assert "node_id required" in err
    assert "iicp-node init" in err


# --------------------------------------------------------------------------- #
# Fix 4 — `serve --model X` backend-url fallback
# --------------------------------------------------------------------------- #
class _StopAfterResolution(Exception):
    """Sentinel raised right after backend_url resolution to halt _serve()."""


def _resolve_backend_url(monkeypatch, *extra: str) -> str:
    """Drive _serve() up to (and not past) backend_url resolution.

    A --model is supplied so the model auto-select (network) is skipped, and
    _find_available_port — the first call after resolution — is patched to raise
    a sentinel so no port is ever bound / no server starts.
    """
    parser = cli._build_parser()
    args = parser.parse_args(["serve", "--model", "test-model", "--skip-registration", *extra])
    args.auto_detect_nat_explicit = None

    def _stop(*a, **k):
        raise _StopAfterResolution

    monkeypatch.setattr(cli, "_find_available_port", _stop)

    with pytest.raises(_StopAfterResolution):
        asyncio.run(cli._serve(args))
    return args.backend_url


def test_serve_backend_url_defaults_to_ollama(iicp_home, monkeypatch):
    url = _resolve_backend_url(monkeypatch)
    assert url == "http://localhost:11434"


def test_serve_backend_url_anthropic_default(iicp_home, monkeypatch):
    url = _resolve_backend_url(monkeypatch, "--backend-type", "anthropic")
    assert url == "https://api.anthropic.com"


def test_serve_explicit_backend_url_wins(iicp_home, monkeypatch):
    url = _resolve_backend_url(monkeypatch, "--backend-url", "http://example.test:1234")
    assert url == "http://example.test:1234"


# --------------------------------------------------------------------------- #
# Fix WQ-066 — `serve --relay-capable` flag (0.7.45)
# --------------------------------------------------------------------------- #

def test_relay_capable_flag_parses(monkeypatch):
    """--relay-capable sets relay_capable=True on the parsed namespace (0.7.45).
    Without the fix, argparse raises 'unrecognized arguments' → test fails."""
    monkeypatch.delenv("IICP_RELAY_CAPABLE", raising=False)
    args = _parse_serve("--relay-capable", monkeypatch=monkeypatch)
    assert args.relay_capable is True


def test_relay_capable_default_off(monkeypatch):
    """relay_capable defaults to False when flag and env are absent."""
    monkeypatch.delenv("IICP_RELAY_CAPABLE", raising=False)
    args = _parse_serve(monkeypatch=monkeypatch)
    assert args.relay_capable is False


def test_relay_capable_env_var(monkeypatch):
    """IICP_RELAY_CAPABLE=1 enables relay_capable without the CLI flag."""
    monkeypatch.setenv("IICP_RELAY_CAPABLE", "1")
    args = _parse_serve(monkeypatch=monkeypatch)
    assert args.relay_capable is True


def test_relay_accept_port_flag_parses(monkeypatch):
    """--relay-accept-port sets relay_accept_port to the given integer."""
    monkeypatch.delenv("IICP_RELAY_ACCEPT_PORT", raising=False)
    args = _parse_serve("--relay-accept-port", "9490", monkeypatch=monkeypatch)
    assert args.relay_accept_port == 9490


def test_relay_accept_port_default(monkeypatch):
    """relay_accept_port defaults to 9485 when flag and env are absent."""
    monkeypatch.delenv("IICP_RELAY_ACCEPT_PORT", raising=False)
    args = _parse_serve(monkeypatch=monkeypatch)
    assert args.relay_accept_port == 9485
