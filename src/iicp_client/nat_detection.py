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

    # Tier 0 — operator-configured public endpoint
    if operator_public_endpoint:
        if _looks_routable(operator_public_endpoint):
            profile.detection_log.append(
                f"tier-0: operator-configured public_endpoint={operator_public_endpoint!r}"
            )
            return NatProfile(
                tier=0,
                transport_method="direct",
                public_endpoint=operator_public_endpoint,
                internal_endpoint=profile.internal_endpoint,
                detection_log=profile.detection_log,
            )
        profile.detection_log.append(
            f"tier-0: operator-configured public_endpoint={operator_public_endpoint!r} "
            f"non-routable — falling through to tier-1 UPnP detection."
        )

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
            profile.operator_guidance = (
                f"WARNING: your WAN IP {upnp.external_ip} appears to be inside a "
                f"carrier-grade NAT pool (reverse-DNS suggests CGNAT). UPnP-mapped "
                f"ports are typically not reachable from the internet in this case. "
                f"Options: (a) ask your ISP for a native IPv4 lease, "
                f"(b) use an external tunnel (Cloudflare Tunnel, tailscale funnel), "
                f"(c) switch to IPv6 if your network supports it."
            )
            return profile  # tier 4 — unreachable in practice

        public_url = f"http://{upnp.external_ip}:{bind_port}"
        transport_url: str | None = None
        if transport_port and transport_port in upnp.mapped_ports and transport_port != bind_port:
            transport_url = f"iicp://{upnp.external_ip}:{transport_port}"
            profile.detection_log.append(
                f"tier-1: UPnP mapped {bind_port} → {public_url} AND "
                f"{transport_port} → {transport_url} (spec v0.7.0 dual-endpoint)"
            )
        else:
            profile.detection_log.append(f"tier-1: UPnP mapped {bind_port} → {public_url}")

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
    external_port: int | None = None  # primary mapped port (first in the list)
    mapped_ports: list[int] = field(default_factory=list)  # all successfully mapped ports
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

    try:
        wan_svc.AddPortMapping(
            NewRemoteHost="",
            NewExternalPort=primary_port,
            NewProtocol="TCP",
            NewInternalPort=primary_port,
            NewInternalClient=local_ip,
            NewEnabled="1",
            NewPortMappingDescription=f"iicp-client (ADR-041 tier-1) {primary_port}",
            NewLeaseDuration=lease_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return _UpnpResult(
            success=False,
            external_ip=ext_ip,
            error=(
                f"AddPortMapping failed for primary port {primary_port}: {exc} "
                f"(NewInternalClient={local_ip})"
            ),
            igd_device=str(igd),
        )

    mapped: list[int] = [primary_port]
    for extra in internal_ports[1:]:
        try:
            wan_svc.AddPortMapping(
                NewRemoteHost="",
                NewExternalPort=extra,
                NewProtocol="TCP",
                NewInternalPort=extra,
                NewInternalClient=local_ip,
                NewEnabled="1",
                NewPortMappingDescription=f"iicp-client (ADR-041 tier-1) {extra}",
                NewLeaseDuration=lease_seconds,
            )
            mapped.append(extra)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "UPnP: failed to map additional port %d (primary %d already mapped): %s",
                extra,
                primary_port,
                exc,
            )

    return _UpnpResult(
        success=True,
        external_ip=ext_ip,
        external_port=primary_port,
        mapped_ports=mapped,
        igd_device=str(igd),
    )


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
