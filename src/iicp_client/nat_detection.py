# SPDX-License-Identifier: Apache-2.0
"""UPnP NAT detection + dual-port mapping (ADR-041 tier-0 + tier-1).

Port of iicp-adapter's `adapter/network/nat_detector.py` (iter-1410) into
the iicp-client-python SDK as part of the adapter→hybrid-client migration
(tracker iicp.network#340 Tier 1 Item 4).

This module gives an SDK-hosted node automatic public-endpoint discovery:

  - Tier 0 — operator-configured public endpoint (passed in by caller),
             validated against the same `_looks_routable` heuristics the
             directory's `RoutableEndpoint` validator applies. Falls through
             to tier 1 when the URL is non-routable or contains an `example.com`
             placeholder.
  - Tier 1 — UPnP / NAT-PMP via the `upnpclient` library: discover the local
             IGD, query the WAN IP, request port mappings for HTTP control AND
             native IICP transport (`endpoint` port + `transport_port` per
             spec/iicp-dir.md v0.7.0), and return a `NatProfile` carrying
             both URLs.
  - Tier 4 — unreachable: operator gets actionable guidance (manual port-
             forward, tunnel service, switch to IPv6, etc.).

Two operator-facing diagnostic improvements over the original adapter port:

  1. **CGNAT reverse-DNS heuristic** (#339): when the WAN IP's hostname
     contains `cgn`, `cgnat`, `shared`, `ds-lite`, etc., the detector treats
     the UPnP mapping as ineffective and surfaces a clear "your ISP runs
     carrier-grade NAT" message — the original adapter logged tier=1 success
     for CGN-allocated IPs that were not actually reachable from the internet.
  2. **External IP probe fallback** (#331 Phase A): when UPnP `AddPortMapping`
     succeeds but `GetExternalIPAddress` returns nothing usable (FRITZ!Box
     auth-restricted case), the detector fetches the WAN IP from an operator-
     configured HTTPS probe URL (e.g. `https://api.ipify.org`).

`upnpclient` and `ifaddr` are optional deps installed via the `[nat]` extra.
The module gracefully degrades to tier 4 when they're absent.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Public NatProfile ────────────────────────────────────────────────────────


@dataclass
class Ipv6Profile:
    """IPv6-side qualification result (#342, ADR-043 §4).

    Distinguishes the three orthogonal states from the spec doc:
      1. global IPv6 address exists on a local interface
      2. local IPv6 listener is actually bound (service code reachable)
      3. external IPv6 reachability (return path from the internet)

    'stable' tracks RFC 4941 privacy addresses vs. EUI-64 / manual.
    A privacy address is NOT suitable for a long-lived server identity.

    Pinhole fields (#343, ADR-043 §5) populated when UPnP v6 firewall
    pinhole automation succeeds:
      - pinhole_active: True iff router accepted AddPinhole
      - pinhole_unique_id: opaque handle returned by the IGD; use to
        delete or renew the pinhole
      - pinhole_lease_seconds: lease the IGD granted (may differ from
        the request)
    """
    global_v6_available: bool = False
    stable_v6_available: bool = False
    addresses: list[str] = field(default_factory=list)
    listener_v6_ok: bool = False  # the SDK's bind(0.0.0.0) is v4-only by default
    external_v6_reachable: bool = False  # outbound v6 connectivity to probe target
    pinhole_active: bool = False
    pinhole_unique_id: int | None = None
    pinhole_lease_seconds: int | None = None
    pinhole_inbound_allowed: bool | None = None  # router reports firewall lets in pinholes
    error: str | None = None


@dataclass
class NatProfile:
    """Result of `detect_nat()` — describes what the SDK can advertise.

    Mirrors the adapter's NatProfile shape so existing operator-facing docs
    (docs/nat-aware-adapter-setup.md, ADR-041) apply unchanged.
    """

    tier: int  # 0..4 per ADR-041
    # ADR-041 transport_method: 'direct' / 'upnp_mapped' / 'stun_hole_punch' /
    # 'turn_relay' / 'external_tunnel' / 'unreachable'
    transport_method: str
    public_endpoint: str | None = None  # HTTP control plane URL (`http://host:port`)
    transport_endpoint: str | None = None  # spec v0.7.0 native IICP URL (`iicp://host:9484`)
    internal_endpoint: str | None = None
    operator_guidance: str | None = None
    detection_log: list[str] = field(default_factory=list)
    # ADR-043 §4 — IPv6 side of the qualification result (#342). Populated by
    # detect_ipv6() when detect_nat is called with v6 detection enabled.
    ipv6: Ipv6Profile | None = None

    def is_reachable(self) -> bool:
        return self.tier <= 3 and self.public_endpoint is not None


# ── Public entry point ───────────────────────────────────────────────────────


async def detect_nat(
    bind_host: str,
    bind_port: int,
    operator_public_endpoint: str | None = None,
    *,
    upnp_lease_seconds: int = 3600,
    timeout_s: float = 5.0,
    external_ip_probe_url: str | None = None,
    transport_port: int | None = None,
    detect_v6: bool = True,
) -> NatProfile:
    """Run ADR-041 tier-0 + tier-1 detection.

    Arguments:
        bind_host: local bind address (typically `0.0.0.0` or the LAN IP).
        bind_port: HTTP control port (the SDK's IicpNode.serve() port).
        operator_public_endpoint: optional pre-configured `http://...` URL.
            When set AND `_looks_routable()` accepts it, returned as tier 0
            without UPnP. When non-routable (or contains "example.com"),
            falls through to tier 1.
        upnp_lease_seconds: lease duration to request from the IGD.
        timeout_s: per-step timeout (UPnP discovery + probe URL fetch).
        external_ip_probe_url: opt-in WAN-IP probe URL (e.g. api.ipify.org).
            Used as a fallback when UPnP AddPortMapping succeeds but the IGD
            refuses GetExternalIPAddress (FRITZ!Box auth-restricted case).
        transport_port: optional native IICP TCP port (default 9484 per
            spec/iicp-dir.md v0.7.0). When set AND distinct from `bind_port`,
            the detector asks UPnP to map BOTH ports and returns a
            transport_endpoint URL alongside the HTTP public_endpoint.
    """
    profile = NatProfile(tier=4, transport_method="unreachable")
    profile.internal_endpoint = f"http://{bind_host}:{bind_port}"

    # ADR-043 §4 — IPv6 qualification runs in parallel to the v4 NAT path.
    # Side-effect free until the operator chooses to bind on `[::]`.
    if detect_v6:
        try:
            profile.ipv6 = await detect_ipv6(bind_port, timeout_s=min(timeout_s, 3.0))
            profile.detection_log.append(
                f"ipv6: global={profile.ipv6.global_v6_available} "
                f"stable={profile.ipv6.stable_v6_available} "
                f"listener={profile.ipv6.listener_v6_ok} "
                f"reachable_out={profile.ipv6.external_v6_reachable}"
            )
        except Exception as exc:  # noqa: BLE001
            profile.detection_log.append(f"ipv6: probe error — {exc}")

    # Tier 0 — operator-configured public endpoint
    if operator_public_endpoint:
        if _looks_routable(operator_public_endpoint):
            profile.detection_log.append(
                f"tier-0: operator-configured public_endpoint={operator_public_endpoint!r}"
            )
            t0 = NatProfile(
                tier=0,
                transport_method="direct",
                public_endpoint=operator_public_endpoint,
                internal_endpoint=profile.internal_endpoint,
                detection_log=profile.detection_log,
                ipv6=profile.ipv6,
            )
            # #343 / ADR-043 §5 — even when the operator gives us an explicit
            # endpoint, if it's IPv6 we should still try to open an inbound
            # firewall pinhole via UPnP. Previously the tier-0 path returned
            # immediately, leaving the router firewall closed and the directory
            # unable to dial back → public_reachable=false.
            _maybe_open_v6_pinhole_for_endpoint(t0, bind_port)
            return t0
        profile.detection_log.append(
            f"tier-0: operator-configured public_endpoint={operator_public_endpoint!r} "
            f"non-routable — falling through to tier-1 UPnP detection."
        )

    # Tier 0 auto-detect — cloud VM with a public IPv4 directly on a local interface.
    # Runs before UPnP so bare-metal VPS nodes (Hetzner, DigitalOcean) get a direct
    # endpoint without waiting for IGD discovery.
    auto_v4 = _detect_public_v4_on_interfaces()
    if auto_v4:
        auto_url = f"http://{auto_v4}:{bind_port}"
        profile.detection_log.append(
            f"tier-0: auto-detected public IPv4 on local interface → {auto_url!r}"
        )
        t0 = NatProfile(
            tier=0,
            transport_method="direct",
            public_endpoint=auto_url,
            internal_endpoint=profile.internal_endpoint,
            detection_log=profile.detection_log,
            ipv6=profile.ipv6,
        )
        return t0

    # Tier 1 — UPnP
    ports_to_map: list[int] = [bind_port]
    if transport_port and transport_port != bind_port:
        ports_to_map.append(transport_port)

    try:
        upnp = await asyncio.wait_for(
            _try_upnp_mapping(ports_to_map, lease_seconds=upnp_lease_seconds),
            timeout=timeout_s,
        )
    except TimeoutError:
        profile.detection_log.append(f"tier-1: UPnP discovery timed out after {timeout_s}s")
        upnp = None
    except ImportError as exc:
        profile.detection_log.append(f"tier-1: upnp library not installed: {exc}")
        upnp = None
    except Exception as exc:  # noqa: BLE001
        profile.detection_log.append(f"tier-1: UPnP error: {exc}")
        upnp = None

    if upnp and upnp.success:
        # External-IP probe fallback for routers that AddPortMapping but refuse
        # GetExternalIPAddress (FRITZ!Box with default UPnP auth config).
        if not upnp.external_ip or upnp.external_ip == "0.0.0.0":
            if external_ip_probe_url:
                probed = await _probe_external_ip(
                    external_ip_probe_url, timeout_s=min(timeout_s, 5.0)
                )
                if probed:
                    profile.detection_log.append(
                        f"tier-1: external IP probe {external_ip_probe_url!r} returned {probed}"
                    )
                    upnp.external_ip = probed
                else:
                    profile.detection_log.append(
                        f"tier-1: external IP probe {external_ip_probe_url!r} "
                        "returned no valid IPv4"
                    )
            if not upnp.external_ip or upnp.external_ip == "0.0.0.0":
                profile.operator_guidance = (
                    f"UPnP mapped port {bind_port} but the router did not return its WAN IP. "
                    f"Set nat_external_ip_probe_url to an HTTPS probe service (e.g. "
                    f"https://api.ipify.org) OR set public_endpoint manually."
                )
                return profile

        # #339 — CGNAT reverse-DNS heuristic. Even when the WAN IP is in a
        # normal-looking IPv4 range (89.x, 95.x, ...), German cable carriers
        # like NetCologne DS-Lite the residential IPs through CGN gateways.
        # The hostname `cgn-89-1-216-20.nc.de` is the smoking gun. The
        # original adapter port reported tier=1 success for such IPs even
        # though inbound TCP was filtered at the carrier layer.
        cgnat_warning = _detect_cgnat(upnp.external_ip)
        if cgnat_warning:
            profile.detection_log.append(f"tier-1: {cgnat_warning}")
            # CGNAT IPv4 unreachable — but if the host has a working IPv6 GUA,
            # advertise that instead (ADR-043 §10 'ELSE IF IPv6 reachable').
            v6_profile = _try_ipv6_fallback(profile, bind_port, transport_port)
            if v6_profile is not None:
                return v6_profile
            profile.operator_guidance = (
                f"WARNING: your WAN IP {upnp.external_ip} appears to be inside a "
                f"carrier-grade NAT pool (reverse-DNS suggests CGNAT). UPnP-mapped "
                f"ports are typically not reachable from the internet in this case. "
                f"Options: (a) ask your ISP for a native IPv4 lease, "
                f"(b) use an external tunnel (Cloudflare Tunnel, tailscale funnel), "
                f"(c) switch to IPv6 if your network supports it."
            )
            return profile  # tier 4 — unreachable in practice

        # ADR-041 §3 — the URLs we advertise must use the EXTERNAL ports the
        # IGD actually assigned. With AddAnyPortMapping fallback, the assigned
        # external can differ from our internal bind_port / transport_port
        # (e.g. another LAN host already owns external 9484).
        ext_bind = upnp.port_mapping.get(bind_port, bind_port)
        public_url = f"http://{upnp.external_ip}:{ext_bind}"
        transport_url: str | None = None
        if transport_port and transport_port in upnp.mapped_ports and transport_port != bind_port:
            ext_transport = upnp.port_mapping.get(transport_port, transport_port)
            transport_url = f"iicp://{upnp.external_ip}:{ext_transport}"
            profile.detection_log.append(
                f"tier-1: UPnP mapped {bind_port}→{ext_bind} ({public_url}) AND "
                f"{transport_port}→{ext_transport} ({transport_url}) "
                f"(spec v0.7.0 dual-endpoint; AddAnyPortMapping used if ext≠internal)"
            )
        else:
            profile.detection_log.append(
                f"tier-1: UPnP mapped {bind_port}→{ext_bind} ({public_url})"
            )

        return NatProfile(
            tier=1,
            transport_method="upnp_mapped",
            public_endpoint=public_url,
            transport_endpoint=transport_url,
            internal_endpoint=profile.internal_endpoint,
            detection_log=profile.detection_log,
        )

    # UPnP failed — explain why + give actionable guidance
    if upnp is None:
        profile.detection_log.append(
            "tier-1: UPnP discovery returned nothing (SSDP broadcast filtered? library missing?)"
        )
    elif not upnp.igd_device:
        profile.detection_log.append(f"tier-1: no IGD device responded — {upnp.error}")
    else:
        profile.detection_log.append(
            f"tier-1: IGD found ({upnp.igd_device}) but mapping refused — {upnp.error}"
        )

    # External-IP-only fallback: when UPnP failed entirely but the operator
    # has port-forwarding wired manually (or there's a router-managed mapping
    # we couldn't detect), the external_ip_probe_url still gives us the WAN
    # IP. Construct a tier-1-ish public_endpoint optimistically and let the
    # directory's Layer-2 assertLive probe verify reachability. Failure on
    # that side will surface clearly via IICP-E036 (#331 Phase B).
    if external_ip_probe_url:
        probed = await _probe_external_ip(external_ip_probe_url, timeout_s=min(timeout_s, 5.0))
        if probed:
            public_url = f"http://{probed}:{bind_port}"
            transport_url: str | None = None
            if transport_port and transport_port != bind_port:
                transport_url = f"iicp://{probed}:{transport_port}"
            profile.detection_log.append(
                f"tier-1-fallback: UPnP failed but external IP probe returned {probed} — "
                f"advertising {public_url} (verification deferred to directory Layer-2 probe)"
            )
            return NatProfile(
                tier=1,
                transport_method="external_tunnel",
                public_endpoint=public_url,
                transport_endpoint=transport_url,
                internal_endpoint=profile.internal_endpoint,
                detection_log=profile.detection_log,
                operator_guidance=(
                    "UPnP discovery failed. Advertising the external IP from the probe URL "
                    f"({probed}). For this to actually be reachable, your router needs a "
                    f"port-forward rule for port {bind_port} → this host. If "
                    "registration fails with IICP-E036, add the port-forward or "
                    "use a tunnel (Cloudflare Tunnel, ngrok)."
                ),
            )

    # IPv6 GUA fallback (ADR-043 §10): when v4 paths fail entirely but the
    # host has a routable IPv6 + verified outbound v6 connectivity, the
    # operator can host over v6 — bypasses CGNAT entirely.
    v6_profile = _try_ipv6_fallback(profile, bind_port, transport_port)
    if v6_profile is not None:
        return v6_profile

    profile.operator_guidance = (
        "No automatic port mapping available. Options:\n"
        "  1. Configure your router to forward an external port to this host\n"
        "  2. Set public_endpoint to your real external URL\n"
        "  3. Use an external tunnel (Cloudflare Tunnel, ngrok, tailscale funnel)\n"
        "See iicp.network/docs/nat-aware-adapter-setup.md for the details."
    )
    return profile


# ── UPnP helpers ─────────────────────────────────────────────────────────────


@dataclass
class _UpnpResult:
    success: bool
    external_ip: str | None = None
    # Primary external port — what the IGD actually assigned for the first
    # internal port. May differ from the internal when AddAnyPortMapping was
    # used as fallback to AddPortMapping.
    external_port: int | None = None
    # All internal ports that we successfully mapped to SOME external port.
    mapped_ports: list[int] = field(default_factory=list)
    # internal_port → assigned_external_port (often identity, but not when
    # AddAnyPortMapping picks a non-canonical port due to conflict).
    port_mapping: dict[int, int] = field(default_factory=dict)
    igd_device: str | None = None
    error: str | None = None


async def _try_upnp_mapping(
    internal_ports: list[int] | int, *, lease_seconds: int = 3600
) -> _UpnpResult:
    """Discover the local IGD and request port mappings for one or more ports.

    Returns success when at least the PRIMARY port (first in the list) is
    mapped; additional ports are best-effort and recorded in `mapped_ports`.
    """
    if isinstance(internal_ports, int):
        internal_ports = [internal_ports]
    return await asyncio.get_running_loop().run_in_executor(
        None, _upnp_mapping_blocking, list(internal_ports), lease_seconds
    )


def _upnp_mapping_blocking(internal_ports: list[int], lease_seconds: int) -> _UpnpResult:
    """Blocking implementation — imports upnpclient lazily so the missing
    dep is recoverable (returns tier-4 with a clear error rather than crashing).
    """
    try:
        import upnpclient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "upnpclient not installed — install with: pip install 'iicp-client[nat]'"
        ) from exc

    if not internal_ports:
        return _UpnpResult(success=False, error="no ports specified")
    primary_port = internal_ports[0]

    # Discover IGD devices (typically the local router responds via SSDP)
    devices = upnpclient.discover(timeout=3)
    igd = None
    for d in devices:
        for service in getattr(d, "service_map", {}).values():
            if "WANIPConn" in service.service_type or "WANPPPConn" in service.service_type:
                igd = d
                break
        if igd:
            break

    if not igd:
        return _UpnpResult(success=False, error="no IGD device responded to SSDP discovery")

    try:
        wan_svc = next(
            s
            for s in igd.service_map.values()
            if "WANIPConn" in s.service_type or "WANPPPConn" in s.service_type
        )
        ext_ip = wan_svc.GetExternalIPAddress()["NewExternalIPAddress"]
    except Exception as exc:  # noqa: BLE001
        return _UpnpResult(success=False, error=f"GetExternalIPAddress failed: {exc}")

    # Pick the local interface IP on the IGD's LAN subnet. When a VPN is
    # active, the socket-to-8.8.8.8 trick returns the VPN-tunnel IP, not the
    # LAN IP — FRITZ!Box rejects AddPortMapping for NewInternalClient that
    # isn't on its LAN (UPnP error 606 / 718).
    local_ip = _detect_local_ip_matching_igd(igd) or _detect_local_ip_for_default_gateway()

    # Strategy: 1:1 AddPortMapping first (canonical port preserved); on conflict
    # fall back to AddAnyPortMapping (IGDv2) which lets the IGD pick a free
    # external port. We track the assigned port so the SDK can advertise the
    # actual iicp:// URL the directory should dial.
    assigned_primary = _map_one_v4_port(wan_svc, primary_port, local_ip, lease_seconds)
    if assigned_primary is None:
        return _UpnpResult(
            success=False,
            external_ip=ext_ip,
            error=(
                f"AddPortMapping + AddAnyPortMapping both failed for "
                f"primary port {primary_port} (NewInternalClient={local_ip})"
            ),
            igd_device=str(igd),
        )

    # `mapped` tracks INTERNAL ports that succeeded (callers check
    # `internal in mapped` to decide whether to advertise that endpoint).
    # `port_map_external` carries the internal→external mapping when the
    # IGD assigned a non-canonical external (AddAnyPortMapping fallback).
    mapped: list[int] = [primary_port]
    port_map_external: dict[int, int] = {primary_port: assigned_primary}
    for extra in internal_ports[1:]:
        assigned = _map_one_v4_port(wan_svc, extra, local_ip, lease_seconds)
        if assigned is not None:
            mapped.append(extra)
            port_map_external[extra] = assigned
        else:
            logger.warning(
                "UPnP: failed to map additional port %d (primary %d → %d already mapped)",
                extra,
                primary_port,
                assigned_primary,
            )

    return _UpnpResult(
        success=True,
        external_ip=ext_ip,
        external_port=assigned_primary,
        mapped_ports=mapped,
        port_mapping=port_map_external,
        igd_device=str(igd),
    )


def _map_one_v4_port(wan_svc, internal_port: int, local_ip: str, lease_seconds: int) -> int | None:
    """Map a single port via UPnP v4, returning the assigned EXTERNAL port.

    Tries 1:1 AddPortMapping first (preserves canonical ports like 9484).
    On `ConflictInMappingEntry` (the IGD already has a mapping for that
    external port from another host), falls back to AddAnyPortMapping
    (IGDv2 §2.5.13 — added precisely for this race). Returns None when
    both fail.
    """
    desc = f"iicp-client (ADR-041 tier-1) {internal_port}"
    try:
        wan_svc.AddPortMapping(
            NewRemoteHost="",
            NewExternalPort=internal_port,
            NewProtocol="TCP",
            NewInternalPort=internal_port,
            NewInternalClient=local_ip,
            NewEnabled="1",
            NewPortMappingDescription=desc,
            NewLeaseDuration=lease_seconds,
        )
        return internal_port
    except Exception as exc:  # noqa: BLE001
        # 1:1 conflict (most common cause: another host already has this
        # external port mapped). Try AddAnyPortMapping; the IGD picks a
        # free external port in its dynamic pool.
        logger.info(
            "UPnP: AddPortMapping %d failed (%s); retrying via AddAnyPortMapping",
            internal_port,
            exc,
        )
    try:
        result = wan_svc.AddAnyPortMapping(
            NewRemoteHost="",
            NewExternalPort=internal_port,
            NewProtocol="TCP",
            NewInternalPort=internal_port,
            NewInternalClient=local_ip,
            NewEnabled="1",
            NewPortMappingDescription=desc,
            NewLeaseDuration=lease_seconds,
        )
        assigned = int(result.get("NewReservedPort", 0))
        if assigned > 0:
            logger.info(
                "UPnP: AddAnyPortMapping assigned external port %d for internal %d",
                assigned,
                internal_port,
            )
            return assigned
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "UPnP: AddAnyPortMapping failed for internal %d: %s",
            internal_port,
            exc,
        )
    return None


def _detect_local_ip_matching_igd(igd) -> str | None:
    """Find a local interface IP on the same /24 subnet as the IGD device.

    Mirrors the adapter's helper to avoid the VPN-tunnel-IP gotcha.
    """
    from urllib.parse import urlparse

    location = getattr(igd, "location", "") or ""
    parsed = urlparse(location)
    igd_host = parsed.hostname
    if not igd_host:
        return None
    m = re.match(r"^(\d+\.\d+\.\d+\.)(\d+)$", igd_host)
    if not m:
        return None
    igd_prefix = m.group(1)

    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            addr = info[4][0]
            if addr.startswith(igd_prefix):
                return addr
    except OSError:
        pass

    try:
        import ifaddr  # type: ignore[import-untyped]

        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and ip.ip.startswith(igd_prefix):
                    return ip.ip
    except ImportError:
        pass
    return None


def _detect_local_ip_for_default_gateway() -> str:
    """Fallback when IGD-based interface detection misses (or when no IGD).

    Uses the UDP-socket trick: open a UDP socket toward 8.8.8.8 (no traffic
    actually sent), then ask the OS which local IP it chose. Warning: this
    returns the VPN-tunnel IP when a VPN is active — prefer the IGD-matching
    helper above whenever possible.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _detect_public_v4_on_interfaces() -> str | None:
    """Cloud-VM auto-detect: returns the first public-routable IPv4 found on
    a local network interface.

    Covers bare-metal VPS scenarios (Hetzner, DigitalOcean, Vultr, Linode)
    where the public IP is assigned directly to an interface (eth0/ens3/etc.)
    rather than via NAT. On AWS/GCP the instance typically sees a private IP
    only — this returns None and the caller falls through to UPnP and beyond.

    Uses `ifaddr` when available (iicp-client[nat] dep); falls back to the
    UDP-socket default-route trick which reveals the primary IP only.
    """
    candidates: list[str] = []
    try:
        import ifaddr  # type: ignore[import-untyped]

        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                # ifaddr returns IPv4 as plain str, IPv6 as (str, scope_id) tuple
                if isinstance(ip.ip, str):
                    candidates.append(ip.ip)
    except ImportError:
        pass

    # Fallback: OS-chosen source IP for the default-gateway route.
    candidates.append(_detect_local_ip_for_default_gateway())

    for ip_str in candidates:
        try:
            addr = ipaddress.IPv4Address(ip_str)
        except ValueError:
            continue
        if (
            not addr.is_private
            and not addr.is_loopback
            and not addr.is_link_local
            and not addr.is_multicast
            and not addr.is_reserved
            and not addr.is_unspecified
            and not (
                ipaddress.IPv4Address("100.64.0.0")
                <= addr
                <= ipaddress.IPv4Address("100.127.255.255")
            )
        ):
            return ip_str
    return None


# ── External-IP probe + routability helpers ──────────────────────────────────


async def _probe_external_ip(url: str, *, timeout_s: float = 5.0) -> str | None:
    """Issue #331 Phase A: fetch the WAN IPv4 from an HTTPS probe URL.

    Validates the response is a public IPv4 outside RFC1918, loopback, link-
    local, multicast, reserved, and the 100.64/10 CGNAT range. Returns None
    on any failure so callers can treat it as "probe did not help."
    """
    try:
        import httpx
    except ImportError:
        logger.warning("nat_external_ip_probe_url set but httpx not available")
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return None
        body = resp.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("external IP probe failed: %s", exc)
        return None

    m = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", body)
    if not m:
        return None
    candidate = m.group(1)
    try:
        addr = ipaddress.IPv4Address(candidate)
    except ValueError:
        return None
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return None
    if ipaddress.IPv4Address("100.64.0.0") <= addr <= ipaddress.IPv4Address("100.127.255.255"):
        return None  # RFC 6598 CGNAT — UPnP can't help
    return candidate


def _looks_routable(url: str) -> bool:
    """Mirror of the directory's `RoutableEndpoint` validator (iicp.network
    iter-1365). Returns True if `url`'s host could plausibly be reached from
    a public client. The directory will reject anything else at registration.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    never_routable = {"localhost", "0.0.0.0", "::1", "::"}
    if host in never_routable:
        return False
    suffixes = (
        ".localhost",
        ".local",
        ".test",
        ".example",
        ".invalid",
        ".lan",
        ".internal",
    )
    if any(host.endswith(s) for s in suffixes):
        return False
    try:
        addr = ipaddress.ip_address(host)
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            return False
        return True
    except ValueError:
        pass
    # Bare hostname without TLD = likely Docker service name
    if "." not in host:
        return False
    return True


# ── CGNAT detection (iicp.network #339) ──────────────────────────────────────

# Hostname substrings that strongly suggest carrier-grade NAT (DS-Lite, etc.)
_CGNAT_HINTS = ("cgn", "cgnat", "ds-lite", "dslite", "nat64")

# Hostname substrings that suggest shared/non-routable infrastructure — softer
# signal than CGNAT_HINTS but still worth flagging.
_SHARED_HINTS = ("shared",)


def _detect_cgnat(external_ip: str) -> str | None:
    """Reverse-DNS heuristic for ISP carrier-grade NAT.

    Returns a warning string when the hostname for `external_ip` matches a
    CGNAT pattern, or None when no signal is detected. Doesn't network-call
    on every detection — only does a single PTR lookup with a short timeout.
    """
    try:
        # gethostbyaddr is blocking; cap with a socket-level timeout would
        # require setting socket.setdefaulttimeout globally, which we don't
        # want. The DNS resolver is typically <100ms; if it stalls it's the
        # operator's resolver problem, not ours.
        hostname = socket.gethostbyaddr(external_ip)[0].lower()
    except (socket.herror, socket.gaierror, OSError):
        return None
    if any(h in hostname for h in _CGNAT_HINTS):
        return (
            f"reverse-DNS for {external_ip} = {hostname!r} suggests CGNAT — "
            f"UPnP mapping likely not externally reachable"
        )
    if any(h in hostname for h in _SHARED_HINTS):
        return (
            f"reverse-DNS for {external_ip} = {hostname!r} suggests shared/CGNAT "
            f"infrastructure — verify external reachability"
        )
    return None


# ── IPv6 qualification (#342, ADR-043 §4) ────────────────────────────────────


async def detect_ipv6(
    bind_port: int,
    *,
    probe_url: str = "https://api6.ipify.org",
    timeout_s: float = 3.0,
) -> Ipv6Profile:
    """Probe the IPv6 surface of the local host.

    Three orthogonal checks per the maintainer-supplied service qualification
    reference (§4 of the doc; ADR-043 §4):

      1. Does any interface have a global IPv6 address (2000::/3 GUA)?
      2. Is the SDK able to bind an IPv6 socket on ``bind_port``? The default
         ``bind=0.0.0.0`` only listens on IPv4 — operators have to bind ``::``
         (dual-stack) or ``::1`` (v6-only) to be reachable over v6.
      3. Can we reach a known IPv6-only probe target on the internet?
         (outbound connectivity test — does NOT prove inbound).

    The result is purely advisory — the directory's Layer-2 dial-back
    (#326 / iter-1458) remains the source of truth for inbound reachability.
    """
    profile = Ipv6Profile()
    profile.addresses = _list_global_ipv6_addresses()
    profile.global_v6_available = bool(profile.addresses)
    profile.stable_v6_available = any(
        not _is_privacy_v6(a) for a in profile.addresses
    )

    # Bind test — can we open a v6 socket on the requested port?
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("::", bind_port))
        profile.listener_v6_ok = True
    except OSError as exc:
        profile.listener_v6_ok = False
        profile.error = f"v6 bind failed: {exc}"
    finally:
        s.close()

    # Outbound reachability test — does v6 routing actually work?
    if profile.global_v6_available:
        profile.external_v6_reachable = await _probe_outbound_ipv6(probe_url, timeout_s)

    return profile


def _list_global_ipv6_addresses() -> list[str]:
    """Return all IPv6 GUA addresses bound to local interfaces.

    Filters out link-local (fe80::/10), unique-local (fc00::/7), loopback,
    multicast, and unspecified — only globally-routable 2000::/3 returned.
    """
    found: set[str] = []
    try:
        addrs = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET6)
    except OSError:
        return []
    for fam, _stype, _proto, _canon, sockaddr in addrs:
        if fam != socket.AF_INET6:
            continue
        ip = sockaddr[0].split("%")[0]  # strip zone-id (fe80::1%en0)
        try:
            addr = ipaddress.IPv6Address(ip)
        except ValueError:
            continue
        if addr.is_global and not addr.is_unspecified:
            found.append(ip)
    return sorted(set(found))


def _is_privacy_v6(addr: str) -> bool:
    """Heuristic: RFC 4941 privacy addresses use a random 64-bit interface
    identifier (no `ff:fe` middle marker that EUI-64 has). This isn't strict
    — operators can also have manual stable addresses without `ff:fe`. The
    qualification result lists raw addresses; this helper is a hint, not a
    proof. Per ADR-043 §7: 'IF IPv6 prefix changes frequently THEN use DDNS.'
    """
    try:
        a = ipaddress.IPv6Address(addr)
    except ValueError:
        return False
    # EUI-64 has the form xxxx:xxxx:xxxx:xxxx where bits 64..71 == 0xff and
    # bits 72..79 == 0xfe in the interface identifier. Heuristic.
    iface_id = int(a) & ((1 << 64) - 1)
    eui64_marker = (iface_id >> 24) & 0xFFFF
    return eui64_marker != 0xFFFE


def _local_global_ipv6_candidates() -> list[str]:
    """Enumerate this host's global IPv6 addresses (2000::/3 GUA).

    Returned ranked from MOST-likely-to-be-pinhole-accepted to least, based on
    AVM/FRITZ!Box behaviour observed 2026-05-27: AddPinhole only authorises
    addresses currently in the router's neighbor cache, which is the
    **current temporary** (RFC 4941 privacy) address — NOT the "secured"
    RFC 7217 stable-private one, NOT deprecated temporaries.

    Heuristic on macOS (the only platform we have a confirmed FRITZ test
    against): prefer addresses listed first whose flags do NOT include
    "deprecated" or "secured". On Linux we don't have that flag visibility
    via ifaddr, so fall back to "first GUA per interface".
    """
    out: list[str] = []
    try:
        import sys

        if sys.platform == "darwin":
            import subprocess

            r = subprocess.run(
                ["ifconfig"], capture_output=True, text=True, check=False
            )
            iface_name: str | None = None
            current_temp: list[str] = []
            secured: list[str] = []
            deprecated_or_other: list[str] = []
            for line in r.stdout.splitlines():
                if line and line[0].isalpha():
                    iface_name = line.split(":")[0]
                    continue
                line = line.strip()
                if not line.startswith("inet6 "):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                addr = parts[1].split("%")[0]
                if not addr.startswith(("2", "3")):
                    continue  # not GUA
                flags = " ".join(parts[2:]).lower()
                if "deprecated" in flags:
                    deprecated_or_other.append(addr)
                elif "secured" in flags:
                    secured.append(addr)
                elif "temporary" in flags or "autoconf" in flags:
                    current_temp.append(addr)
                else:
                    deprecated_or_other.append(addr)
            out = current_temp + secured + deprecated_or_other
        else:
            # Linux / other: best-effort via ifaddr (an iicp-client[nat] dep).
            try:
                import ifaddr  # type: ignore[import-untyped]

                for ad in ifaddr.get_adapters():
                    for ip in ad.ips:
                        s = ip.ip[0] if isinstance(ip.ip, tuple) else ip.ip
                        if isinstance(s, str) and s.startswith(("2", "3")):
                            out.append(s)
            except ImportError:
                pass
    except Exception:  # noqa: BLE001
        pass
    # Dedupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for a in out:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def _maybe_open_v6_pinhole_for_endpoint(profile: NatProfile, bind_port: int) -> None:
    """If the chosen public_endpoint is IPv6, try to open an inbound pinhole.

    Mutates `profile.ipv6` + `profile.detection_log` in place. Idempotent and
    safe to call multiple times (the SDK falls through to it from tier-0 AND
    tier-1 paths). The operator-facing payoff: a node advertising
    `http://[2a0a:...]:8020` actually becomes reachable from the directory's
    dial-back probe instead of sitting silently in `internal_nodes`.

    Behaviour 2026-05-27 (after FRITZ!Box AddPinhole investigation): if the
    initial v6 in the endpoint URL returns UPnP error 606, iterate through
    the host's other GUAs (ranked by [_local_global_ipv6_candidates]) and
    retry — FRITZ only authorises the address currently in its neighbor
    cache, which on macOS is the "current temporary" RFC 4941 address, NOT
    the "secured" RFC 7217 one the OS often enumerates first. If a
    different GUA succeeds, the endpoint URL is REWRITTEN on the profile
    so the directory sees the actually-pinhole'd address.
    """
    endpoint = profile.public_endpoint or ""
    # Parse [ipv6]:port out of http://[hex:hex::]:8020 — bracketed-form only.
    import re

    m = re.match(r"https?://\[([0-9a-fA-F:]+)\](?::(\d+))?", endpoint)
    if not m:
        return
    v6_host = m.group(1)
    port_in_url = int(m.group(2)) if m.group(2) else bind_port
    scheme = endpoint.split("://", 1)[0]
    # GUA 2000::/3 only — RFC 4193 (fc00::/7) and link-local (fe80::/10) skip.
    try:
        import ipaddress

        addr = ipaddress.IPv6Address(v6_host)
        if not addr.is_global:
            profile.detection_log.append(
                f"v6 pinhole: skip — {v6_host} is not a global IPv6 (GUA 2000::/3 required)"
            )
            return
    except ValueError:
        return

    # Ranked candidate list: requested address first, then the other GUAs
    # this host has (most-likely-to-be-accepted first).
    candidates = [v6_host] + [
        c for c in _local_global_ipv6_candidates() if c != v6_host
    ]
    chosen: str | None = None
    chosen_result: tuple[int, int, bool] | None = None
    for cand in candidates:
        profile.detection_log.append(
            f"v6 pinhole: attempting AddPinhole for [{cand}]:{port_in_url}"
        )
        result = _try_upnp_ipv6_pinhole(cand, port_in_url)
        if result is not None:
            chosen = cand
            chosen_result = result
            break

    if chosen is None or chosen_result is None:
        profile.detection_log.append(
            "v6 pinhole: not opened on any local GUA — if your router is a "
            "FRITZ!Box, enable 'Internet → Filters → IPv6 → Selbständige "
            "Portfreigaben durch das Gerät erlauben' (or equivalent on "
            "other vendors). Error 606 from the IGD = router-side ACL "
            "block, NOT a SOAP problem."
        )
        if profile.ipv6 is None:
            profile.ipv6 = Ipv6Profile()
        profile.ipv6.pinhole_active = False
        return

    uid, lease, allowed = chosen_result
    profile.detection_log.append(
        f"v6 pinhole: AddPinhole OK — uid={uid} lease={lease}s on [{chosen}]"
    )
    if chosen != v6_host:
        # Rewrite the public_endpoint so the directory advertises the v6
        # that's ACTUALLY pinhole'd. Operator's original URL pointed at a
        # router-rejected address (typical macOS RFC 7217 path).
        new_endpoint = f"{scheme}://[{chosen}]:{port_in_url}"
        profile.detection_log.append(
            f"v6 pinhole: rewriting public_endpoint {endpoint} → {new_endpoint} "
            f"(original v6 rejected by IGD, pinhole opened on different local GUA)"
        )
        profile.public_endpoint = new_endpoint
    if profile.ipv6 is None:
        profile.ipv6 = Ipv6Profile()
    profile.ipv6.pinhole_active = True
    profile.ipv6.pinhole_unique_id = uid
    profile.ipv6.pinhole_lease_seconds = lease
    profile.ipv6.pinhole_inbound_allowed = allowed


def _try_upnp_ipv6_pinhole(
    internal_v6: str,
    internal_port: int,
    *,
    lease_seconds: int = 3600,
    protocol: int = 6,  # TCP
) -> tuple[int, int, bool] | None:
    """Open an inbound IPv6 firewall pinhole via UPnP IGDv2
    ``WANIPv6FirewallControl::AddPinhole`` (#343, ADR-043 §5).

    Returns ``(unique_id, granted_lease_seconds, inbound_pinhole_allowed)`` on
    success. Returns None when:
      - upnpclient not installed
      - no IGD found
      - the IGD doesn't expose WANIPv6FirewallControl
      - the IGD reports InboundPinholeAllowed=False
      - AddPinhole errored

    The pinhole authorises inbound TCP traffic to ``internal_v6:internal_port``
    from any external host. Operators close the pinhole on shutdown via
    ``delete_ipv6_pinhole(unique_id)`` to avoid leaving a stale open hole when
    the node restarts with a different identity.
    """
    try:
        import upnpclient  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        devices = upnpclient.discover(timeout=3)
    except Exception as exc:  # noqa: BLE001
        logger.debug("v6 pinhole: UPnP discovery failed: %s", exc)
        return None
    for d in devices:
        for svc in getattr(d, "services", []):
            stype = getattr(svc, "service_type", "")
            if "WANIPv6FirewallControl" not in stype:
                continue
            # Check the firewall policy first — some IGDs disable inbound
            # pinholes by default even when the service exists.
            try:
                status = svc.GetFirewallStatus()
                inbound_allowed = bool(status.get("InboundPinholeAllowed", False))
            except Exception as exc:  # noqa: BLE001
                logger.debug("v6 pinhole: GetFirewallStatus failed: %s", exc)
                continue
            if not inbound_allowed:
                logger.info(
                    "v6 pinhole: IGD %s reports InboundPinholeAllowed=False; "
                    "router-side admin must enable inbound pinholes",
                    getattr(d, "friendly_name", "?"),
                )
                return None
            # Request the pinhole. RemoteHost="" means "any"; RemotePort=0
            # means "any port". TCP protocol number = 6.
            try:
                result = svc.AddPinhole(
                    RemoteHost="",
                    RemotePort=0,
                    InternalClient=internal_v6,
                    InternalPort=internal_port,
                    Protocol=protocol,
                    LeaseTime=lease_seconds,
                )
                uid = int(result.get("UniqueID", 0))
                return (uid, lease_seconds, True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "v6 pinhole: AddPinhole failed on %s: %s",
                    getattr(d, "friendly_name", "?"),
                    exc,
                )
                return None
    return None


def delete_ipv6_pinhole(unique_id: int) -> bool:
    """Close a previously-opened IPv6 firewall pinhole via UPnP
    ``WANIPv6FirewallControl::DeletePinhole``.

    Returns True on success, False on any failure (including the IGD not
    being reachable any more — best-effort cleanup, leases auto-expire).
    """
    try:
        import upnpclient  # type: ignore[import-untyped]
    except ImportError:
        return False
    try:
        devices = upnpclient.discover(timeout=3)
    except Exception:  # noqa: BLE001
        return False
    for d in devices:
        for svc in getattr(d, "services", []):
            if "WANIPv6FirewallControl" not in getattr(svc, "service_type", ""):
                continue
            try:
                svc.DeletePinhole(UniqueID=unique_id)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("v6 pinhole: DeletePinhole(%s) failed: %s", unique_id, exc)
                continue
    return False


def _try_ipv6_fallback(
    profile: NatProfile,
    bind_port: int,
    transport_port: int | None,
) -> NatProfile | None:
    """ADR-043 §10 — when the IPv4 path can't expose this node (CGNAT or
    UPnP failure), advertise the host's IPv6 GUA instead. Returns a tier-1
    NatProfile when v6 is usable, None otherwise so the caller falls back to
    its existing tier-4 message.

    Inbound v6 reachability isn't proven here — outbound connectivity to a v6
    probe is the most we can verify locally. Router firewall pinholes for v6
    inbound are tracked under #343. The directory's Layer-2 dial-back is the
    truth test (assertLive over v6 hits IICP-E036 if the firewall is closed).
    """
    if not profile.ipv6:
        return None
    if not (profile.ipv6.global_v6_available and profile.ipv6.external_v6_reachable):
        return None
    v6_addr = profile.ipv6.addresses[0]
    public_url = f"http://[{v6_addr}]:{bind_port}"
    transport_url: str | None = None
    if transport_port and transport_port != bind_port:
        transport_url = f"iicp://[{v6_addr}]:{transport_port}"

    # #343 — attempt the UPnP IPv6 firewall pinhole. Best-effort: when it
    # succeeds the operator doesn't need to manually configure their router;
    # when it fails the existing guidance still points them at it.
    pinhole = _try_upnp_ipv6_pinhole(v6_addr, bind_port)
    pinhole_log = ""
    if pinhole is not None:
        uid, lease, inbound_ok = pinhole
        profile.ipv6.pinhole_active = True
        profile.ipv6.pinhole_unique_id = uid
        profile.ipv6.pinhole_lease_seconds = lease
        profile.ipv6.pinhole_inbound_allowed = inbound_ok
        pinhole_log = (
            f" + UPnP pinhole opened (uid={uid}, lease={lease}s — "
            "renew before expiry via UpdatePinhole, close on shutdown)"
        )
    else:
        profile.ipv6.pinhole_active = False
        pinhole_log = (
            " (UPnP pinhole not opened — router didn't accept AddPinhole; "
            "manual firewall rule still required)"
        )

    profile.detection_log.append(
        f"tier-1-ipv6: advertising {public_url}{pinhole_log}"
    )
    guidance = (
        f"Advertising IPv6 GUA {v6_addr}. Inbound IPv4 isn't available "
        "(no UPnP success / CGNAT), but your IPv6 surface is routable. "
    )
    if profile.ipv6.pinhole_active:
        guidance += (
            f"The SDK opened a UPnP firewall pinhole (port {bind_port} → "
            f"{v6_addr}, lease {profile.ipv6.pinhole_lease_seconds}s); the "
            "directory will Layer-2 dial-back to confirm reachability."
        )
    else:
        guidance += (
            f"For external clients to reach this node over IPv6, ensure your "
            f"router's firewall allows inbound TCP on port {bind_port} → "
            f"{v6_addr}. The directory will Layer-2 dial-back to verify."
        )
    return NatProfile(
        tier=1,
        transport_method="direct",  # IPv6 GUA is directly addressable
        public_endpoint=public_url,
        transport_endpoint=transport_url,
        internal_endpoint=profile.internal_endpoint,
        detection_log=profile.detection_log,
        ipv6=profile.ipv6,
        operator_guidance=guidance,
    )


async def _probe_outbound_ipv6(url: str, timeout_s: float) -> bool:
    """Best-effort: try to reach an IPv6-only probe URL. Returns True iff the
    request completed via IPv6 (httpx auto-selects family based on DNS, so a
    v6-only hostname like api6.ipify.org is the right way to force v6).
    """
    try:
        import httpx
    except ImportError:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False
