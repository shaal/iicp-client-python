# ADR-016: IICP client SDK conformance
"""Unit tests for nat_detection — ADR-041 tier-0 + tier-1 detection logic.

UPnP-IGD discovery isn't reachable in CI so tier-1 success is exercised by
mocking `_try_upnp_mapping` to return a synthetic `_UpnpResult`. The other
helpers (`_looks_routable`, `_probe_external_ip`, `_detect_cgnat`, dual-port
mapping, tier-0 fall-through) are testable without external services.
"""

from __future__ import annotations

from unittest.mock import patch

from iicp_client.nat_detection import (
    Ipv6Profile,
    _detect_cgnat,
    _looks_routable,
    _probe_external_ip,
    _UpnpResult,
    detect_nat,
)

# ── _looks_routable ────────────────────────────────────────────────────────


class TestLooksRoutable:
    def test_accepts_public_dns(self):
        assert _looks_routable("http://node.example.com:8080")

    def test_accepts_public_ipv4(self):
        # Note: 203.0.113.0/24 is RFC 5737 TEST-NET-3; Python's ipaddress
        # considers it is_private. Use a non-documentation IP for the test.
        assert _looks_routable("http://8.8.8.8:8080")
        assert _looks_routable("http://1.1.1.1:443")

    def test_rejects_localhost(self):
        assert not _looks_routable("http://localhost:8080")

    def test_rejects_127_loopback(self):
        assert not _looks_routable("http://127.0.0.1:8080")

    def test_rejects_rfc1918(self):
        assert not _looks_routable("http://192.168.1.1:8080")
        assert not _looks_routable("http://10.0.0.5:8080")
        assert not _looks_routable("http://172.20.0.5:8080")

    def test_rejects_link_local(self):
        assert not _looks_routable("http://169.254.5.5:8080")

    def test_rejects_reserved_suffixes(self):
        assert not _looks_routable("http://node.local:8080")
        assert not _looks_routable("http://node.test:8080")
        assert not _looks_routable("http://service.internal:8080")
        assert not _looks_routable("http://node.example:8080")

    def test_rejects_bare_hostname(self):
        # Docker-compose service names: single-word host with no dots
        assert not _looks_routable("http://adapter-llama:8080")

    def test_rejects_garbage(self):
        assert not _looks_routable("not-a-url")


# ── _detect_cgnat ──────────────────────────────────────────────────────────


class TestDetectCgnat:
    def test_returns_warning_for_cgn_hostname(self):
        with patch("socket.gethostbyaddr", return_value=("cgn-89-1-216-20.nc.de", [], [])):
            warning = _detect_cgnat("89.1.216.20")
        assert warning is not None
        assert "CGNAT" in warning

    def test_returns_warning_for_cgnat_keyword(self):
        with patch("socket.gethostbyaddr", return_value=("cgnat-pool.example.com", [], [])):
            warning = _detect_cgnat("100.65.0.1")
        assert warning is not None

    def test_returns_none_for_normal_hostname(self):
        with patch("socket.gethostbyaddr", return_value=("node1.example.com", [], [])):
            warning = _detect_cgnat("8.8.8.5")
        assert warning is None

    def test_returns_none_when_dns_fails(self):
        with patch("socket.gethostbyaddr", side_effect=OSError("no reverse DNS")):
            warning = _detect_cgnat("8.8.8.5")
        assert warning is None


# ── _probe_external_ip ─────────────────────────────────────────────────────


class TestProbeExternalIp:
    async def test_returns_public_ipv4(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(200, text="8.8.8.5\n")
        ip = await _probe_external_ip("https://api.ipify.test")
        assert ip == "8.8.8.5"

    async def test_rejects_rfc1918(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(200, text="192.168.1.1")
        assert await _probe_external_ip("https://api.ipify.test") is None

    async def test_rejects_loopback(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(200, text="127.0.0.1")
        assert await _probe_external_ip("https://api.ipify.test") is None

    async def test_rejects_cgnat_100_64(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(200, text="100.64.5.5")
        assert await _probe_external_ip("https://api.ipify.test") is None

    async def test_returns_none_on_http_error(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(500)
        assert await _probe_external_ip("https://api.ipify.test") is None

    async def test_handles_json_response(self, respx_mock):
        respx_mock.get("https://api.ipify.test").respond(200, text='{"ip": "8.8.8.5"}')
        # regex picks up the first IPv4-shaped token from the body
        assert await _probe_external_ip("https://api.ipify.test") == "8.8.8.5"


# ── detect_nat tier-0 ───────────────────────────────────────────────────────


class TestDetectNatTier0:
    async def test_operator_endpoint_routable_returns_tier_0(self):
        profile = await detect_nat(
            "0.0.0.0",
            8080,
            operator_public_endpoint="http://node.example.com:8080",
        )
        assert profile.tier == 0
        assert profile.transport_method == "direct"
        assert profile.public_endpoint == "http://node.example.com:8080"
        assert profile.is_reachable()

    async def test_operator_endpoint_non_routable_falls_through(self):
        # localhost is non-routable; tier-0 should fall through, hit tier-1
        # UPnP which fails because we're in a test environment, then return
        # tier 4 unreachable. detect_v6=False isolates the v4 path — the
        # ADR-043 §10 IPv6 fallback (iter-1467) would otherwise upgrade
        # this to tier-1 on any host with a working IPv6 GUA.
        with patch(
            "iicp_client.nat_detection._try_upnp_mapping",
            side_effect=ImportError("upnpclient not installed (test)"),
        ):
            profile = await detect_nat(
                "0.0.0.0",
                8080,
                operator_public_endpoint="http://localhost:8080",
                detect_v6=False,
            )
        assert profile.tier == 4
        assert profile.transport_method == "unreachable"
        # Audit log should show the fall-through reason
        assert any("non-routable" in line for line in profile.detection_log)

    async def test_no_operator_endpoint_runs_tier_1(self):
        with patch(
            "iicp_client.nat_detection._try_upnp_mapping",
            side_effect=ImportError("upnpclient not installed (test)"),
        ):
            profile = await detect_nat("0.0.0.0", 8080, detect_v6=False)
        assert profile.tier == 4
        # Operator guidance should be present so the user knows what to do
        assert profile.operator_guidance is not None

    async def test_ipv6_gua_auto_elected_as_tier_0(self):
        """#416 — a dual-stack host with a global IPv6 GUA + working v6 listener
        and NO public IPv4 must auto-elect http://[GUA]:port as a tier-0 direct
        endpoint with zero manual IICP_PUBLIC_ENDPOINT. Parity with the Rust SDK;
        fails before the IPv6 tier-0 election was added (fell through to tier-4)."""
        gua = "2a0a:a543:df54:0:888:9777:7743:8ae"
        v6 = Ipv6Profile(global_v6_available=True, stable_v6_available=True, addresses=[gua])
        v6.listener_v6_ok = True

        async def _fake_detect_ipv6(*_a, **_k):
            return v6

        with (
            patch("iicp_client.nat_detection.detect_ipv6", _fake_detect_ipv6),
            patch("iicp_client.nat_detection._detect_public_v4_on_interfaces", return_value=None),
        ):
            profile = await detect_nat("0.0.0.0", 9484)
        assert profile.tier == 0
        assert profile.transport_method == "direct"
        assert profile.public_endpoint == f"http://[{gua}]:9484"
        assert profile.is_reachable()

    async def test_ipv6_election_prefers_stable_over_privacy(self):
        """When both a rotating RFC 4941 privacy address and a stable EUI-64 GUA
        are present, the stable one is advertised so the endpoint doesn't churn."""
        privacy = "2a0a:a543:df54:0:1234:5678:9abc:def0"  # no ff:fe → privacy
        stable = "2a0a:a543:df54:0:020c:29ff:fe33:4455"  # ff:fe → EUI-64 stable
        v6 = Ipv6Profile(global_v6_available=True, stable_v6_available=True, addresses=[privacy, stable])
        v6.listener_v6_ok = True

        async def _fake_detect_ipv6(*_a, **_k):
            return v6

        with (
            patch("iicp_client.nat_detection.detect_ipv6", _fake_detect_ipv6),
            patch("iicp_client.nat_detection._detect_public_v4_on_interfaces", return_value=None),
        ):
            profile = await detect_nat("0.0.0.0", 9484)
        assert profile.tier == 0
        assert profile.public_endpoint == f"http://[{stable}]:9484"


# ── detect_nat tier-1 (mocked UPnP) ─────────────────────────────────────────


class TestDetectNatTier1Mocked:
    async def test_upnp_success_returns_tier_1_with_public_endpoint(self):
        fake = _UpnpResult(
            success=True,
            external_ip="8.8.8.5",
            external_port=8080,
            mapped_ports=[8080],
            igd_device="FakeRouter",
        )

        async def fake_try(_ports, *, lease_seconds):
            return fake

        with patch("iicp_client.nat_detection._try_upnp_mapping", fake_try):
            with patch("iicp_client.nat_detection._detect_cgnat", return_value=None):
                # detect_v6=False isolates the UPnP/v4 path — on a host with a real
                # IPv6 GUA the #416 tier-0 v6 election would otherwise pre-empt UPnP.
                profile = await detect_nat("0.0.0.0", 8080, detect_v6=False)
        assert profile.tier == 1
        assert profile.transport_method == "upnp_mapped"
        assert profile.public_endpoint == "http://8.8.8.5:8080"
        assert profile.transport_endpoint is None  # transport_port not requested
        assert profile.is_reachable()

    async def test_upnp_success_dual_port_returns_transport_endpoint(self):
        fake = _UpnpResult(
            success=True,
            external_ip="8.8.8.5",
            external_port=8080,
            mapped_ports=[8080, 9484],
            igd_device="FakeRouter",
        )

        async def fake_try(_ports, *, lease_seconds):
            return fake

        with patch("iicp_client.nat_detection._try_upnp_mapping", fake_try):
            with patch("iicp_client.nat_detection._detect_cgnat", return_value=None):
                profile = await detect_nat("0.0.0.0", 8080, transport_port=9484, detect_v6=False)
        assert profile.public_endpoint == "http://8.8.8.5:8080"
        assert profile.transport_endpoint == "iicp://8.8.8.5:9484"

    async def test_upnp_succeeds_but_cgnat_detected_returns_tier_4(self):
        """#339: WAN IP looks public but reverse-DNS indicates carrier-grade NAT.
        Detector must NOT advertise this as reachable — UPnP mapping is useless
        when the carrier CGNs above the router."""
        fake = _UpnpResult(
            success=True,
            external_ip="89.1.216.20",
            external_port=8080,
            mapped_ports=[8080],
            igd_device="FakeRouter",
        )

        async def fake_try(_ports, *, lease_seconds):
            return fake

        with patch("iicp_client.nat_detection._try_upnp_mapping", fake_try):
            with patch(
                "iicp_client.nat_detection._detect_cgnat",
                return_value="reverse-DNS suggests CGNAT",
            ):
                # detect_v6=False — exercise the v4-only CGNAT path. With
                # IPv6 enabled, ADR-043 §10 fallback would upgrade to tier-1
                # on any host with a working IPv6 GUA (iter-1467 behaviour).
                profile = await detect_nat("0.0.0.0", 8080, detect_v6=False)
        assert profile.tier == 4
        assert not profile.is_reachable()
        assert "CGNAT" in (profile.operator_guidance or "")

    async def test_upnp_no_external_ip_with_probe_fallback(self, respx_mock):
        """Issue #331 Phase A: FRITZ!Box accepts AddPortMapping but refuses
        GetExternalIPAddress. The probe URL fallback recovers the WAN IP."""
        respx_mock.get("https://api.ipify.test").respond(200, text="8.8.8.99")
        fake = _UpnpResult(
            success=True,
            external_ip="",  # IGD refused
            external_port=8080,
            mapped_ports=[8080],
            igd_device="FakeFRITZ",
        )

        async def fake_try(_ports, *, lease_seconds):
            return fake

        with patch("iicp_client.nat_detection._try_upnp_mapping", fake_try):
            with patch("iicp_client.nat_detection._detect_cgnat", return_value=None):
                profile = await detect_nat(
                    "0.0.0.0",
                    8080,
                    external_ip_probe_url="https://api.ipify.test",
                    detect_v6=False,
                )
        assert profile.tier == 1
        assert profile.public_endpoint == "http://8.8.8.99:8080"


# ── ADR-043 §4/§10 — IPv6 fallback (iter-1467) ──────────────────────────────


class TestIpv6Fallback:
    """#416 + ADR-047 supersede the pre-#416 design (v6 GUA = tier-1 fallback gated
    on verified *outbound* v6). A host with a global IPv6 GUA + a working v6 listener
    is now elected as a **tier-0 direct** endpoint — early, before UPnP, and without
    requiring outbound reachability. Rationale: inbound reachability (what matters for
    a serving node) can't be proven locally; the directory un-hide model tolerates a
    node it can't dial-back-probe; clients fall back if a dial fails. This mirrors the
    Rust SDK's tier-0 v6 election exactly (full-parity rule)."""

    async def test_cgnat_v4_gua_elected_tier_0_direct(self):
        # CGNAT v4 (UPnP would map) but a working v6 GUA exists → the v6 GUA is
        # elected tier-0 DIRECTLY, ahead of the v4 UPnP path (direct > mapped).
        # Pre-#416 this returned tier-1 (v6 fallback after the v4 attempt).
        from iicp_client.nat_detection import Ipv6Profile

        fake_v4 = _UpnpResult(
            success=True,
            external_ip="89.1.216.20",
            external_port=8080,
            mapped_ports=[8080],
            igd_device="FakeRouter",
        )

        async def fake_try(_ports, *, lease_seconds):
            return fake_v4

        async def fake_v6(_port, *, timeout_s=3.0):
            return Ipv6Profile(
                global_v6_available=True,
                stable_v6_available=False,
                addresses=["2a0a:a543:df54::1"],
                listener_v6_ok=True,
                external_v6_reachable=True,
            )

        with patch("iicp_client.nat_detection._try_upnp_mapping", fake_try):
            with patch(
                "iicp_client.nat_detection._detect_cgnat",
                return_value="reverse-DNS suggests CGNAT",
            ):
                with patch("iicp_client.nat_detection.detect_ipv6", fake_v6):
                    profile = await detect_nat("0.0.0.0", 8080)

        assert profile.tier == 0
        assert profile.transport_method == "direct"
        assert profile.public_endpoint == "http://[2a0a:a543:df54::1]:8080"
        assert profile.ipv6 is not None
        assert profile.ipv6.global_v6_available

    async def test_v6_gua_elected_tier_0_even_without_outbound(self):
        """#416 parity: a bound GUA is elected tier-0 even when the outbound v6
        probe fails — outbound is not a proxy for inbound serving reachability, and
        the directory un-hide model tolerates a node it can't verify. (Pre-#416 this
        stayed tier-4; the outbound gate is superseded.)"""
        from iicp_client.nat_detection import Ipv6Profile

        async def fake_v6(_port, *, timeout_s=3.0):
            return Ipv6Profile(
                global_v6_available=True,
                addresses=["2a0a:a543:df54::1"],
                listener_v6_ok=True,
                external_v6_reachable=False,  # outbound v6 probe fails — no longer a gate
            )

        with patch(
            "iicp_client.nat_detection._try_upnp_mapping",
            side_effect=ImportError("upnpclient missing"),
        ):
            with patch("iicp_client.nat_detection.detect_ipv6", fake_v6):
                profile = await detect_nat("0.0.0.0", 8080)
        assert profile.tier == 0
        assert profile.transport_method == "direct"
        assert profile.public_endpoint == "http://[2a0a:a543:df54::1]:8080"

    async def test_v6_detection_can_be_disabled(self):
        with patch(
            "iicp_client.nat_detection._try_upnp_mapping",
            side_effect=ImportError("upnpclient missing"),
        ):
            profile = await detect_nat("0.0.0.0", 8080, detect_v6=False)
        assert profile.ipv6 is None


async def test_detect_ipv6_uses_interface_candidates_not_hostname():
    """#416 — detect_ipv6 must enumerate GUAs via the interface-aware candidate
    scan (ifconfig/ifaddr), NOT hostname resolution (which is empty on macOS where
    the host's name doesn't resolve to its GUA). Regression: the node falsely
    reported global_v6_available=False and refused to register over working IPv6."""
    import iicp_client.nat_detection as nat

    with patch.object(nat, "_local_global_ipv6_candidates", return_value=["2a0a:dead:beef::1"]):
        # hostname method returns nothing (simulating the macOS bug) — must not matter
        with patch.object(nat, "_list_global_ipv6_addresses", return_value=[]):
            p = await nat.detect_ipv6(9484)
    assert p.global_v6_available is True
    assert "2a0a:dead:beef::1" in p.addresses
