"""ADR-043 §9/§11 — ServiceQualification: 8-category exposure enum + structured result.

Maps a NatProfile to the canonical exposure_mode string and a ServiceQualification
object that the directory can store as `nodes.exposure_mode` and consumers can use
for routing decisions (prefer direct vs relay vs tunnel).

Usage::

    from iicp_client import IicpNode, qualify_service
    profile = await detect_nat(...)
    sq = qualify_service(profile)
    print(sq.exposure_mode)  # e.g. "ipv4_public_direct"

Or standalone (runs detection internally)::

    sq = await qualify_service_async(bind_host="0.0.0.0", bind_port=8020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iicp_client.nat_detection import NatProfile

# ── 8-category exposure enum (ADR-043 §9) ────────────────────────────────────

EXPOSURE_MODES = frozenset({
    "outbound_only",
    "ipv4_public_direct",
    "ipv4_cgnat_blocked",
    "ipv6_direct_firewall_required",
    "ipv6_direct_pinhole_available",
    "relay_required",
    "tunnel_required",
    "dual_stack_available",
})


@dataclass
class Ipv4Qualification:
    public_ip: str | None = None
    cgnat: bool = False
    upnp_mapped: bool = False


@dataclass
class Ipv6Qualification:
    routable: bool = False
    pinhole_ok: bool = False
    address: str | None = None


@dataclass
class ExposureQualification:
    public_endpoint: str | None = None
    transport_endpoint: str | None = None


@dataclass
class ServiceQualification:
    """ADR-043 §11 — structured result of service qualification.

    ``exposure_mode`` is the canonical 8-category enum used for directory
    routing decisions and the ``nodes.exposure_mode`` column.
    """

    exposure_mode: str
    ipv4: Ipv4Qualification = field(default_factory=Ipv4Qualification)
    ipv6: Ipv6Qualification = field(default_factory=Ipv6Qualification)
    exposure: ExposureQualification = field(default_factory=ExposureQualification)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "exposure_mode": self.exposure_mode,
            "ipv4": {
                "public_ip": self.ipv4.public_ip,
                "cgnat": self.ipv4.cgnat,
                "upnp_mapped": self.ipv4.upnp_mapped,
            },
            "ipv6": {
                "routable": self.ipv6.routable,
                "pinhole_ok": self.ipv6.pinhole_ok,
                "address": self.ipv6.address,
            },
            "exposure": {
                "public_endpoint": self.exposure.public_endpoint,
                "transport_endpoint": self.exposure.transport_endpoint,
            },
            "recommendation": self.recommendation,
        }


# ── Core mapping (NatProfile → ServiceQualification) ─────────────────────────

def qualify_service(profile: NatProfile) -> ServiceQualification:
    """Map a NatProfile to an ADR-043 ServiceQualification.

    Derives the 8-category ``exposure_mode`` from tier/transport_method/detection_log
    and populates the structured sub-objects for directory storage.

    Synchronous — pass an already-awaited NatProfile.  For a fully autonomous
    detection + qualification flow use ``qualify_service_async()``.
    """
    ipv4_q = Ipv4Qualification()
    ipv6_q = Ipv6Qualification()
    exposure_q = ExposureQualification(
        public_endpoint=profile.public_endpoint,
        transport_endpoint=profile.transport_endpoint,
    )

    # Populate IPv4 sub-object
    if profile.public_endpoint:
        import re
        m = re.search(r"https?://([^:/]+)", profile.public_endpoint)
        if m:
            ipv4_q.public_ip = m.group(1)
    ipv4_q.cgnat = any(
        kw in log.lower() for log in profile.detection_log
        for kw in ("cgnat", "ds-lite", "carrier-grade")
    )
    ipv4_q.upnp_mapped = profile.transport_method in ("upnp_mapped",)

    # Populate IPv6 sub-object
    if profile.ipv6:
        ipv6_q.routable = bool(profile.ipv6.global_v6_available)
        ipv6_q.pinhole_ok = profile.ipv6.pinhole_active
        ipv6_q.address = profile.ipv6.addresses[0] if profile.ipv6.addresses else None

    exposure_mode = _derive_exposure_mode(profile, ipv4_q, ipv6_q)
    recommendation = _build_recommendation(exposure_mode, profile)

    return ServiceQualification(
        exposure_mode=exposure_mode,
        ipv4=ipv4_q,
        ipv6=ipv6_q,
        exposure=exposure_q,
        recommendation=recommendation,
    )


def _derive_exposure_mode(
    profile: NatProfile,
    ipv4_q: Ipv4Qualification,
    ipv6_q: Ipv6Qualification,
) -> str:
    ipv6_available = ipv6_q.routable

    if profile.tier == 3:
        return "relay_required"

    if profile.tier == 2 or profile.transport_method == "external_tunnel":
        return "tunnel_required"

    if profile.tier == 4 or profile.public_endpoint is None:
        return "ipv4_cgnat_blocked" if ipv4_q.cgnat else "outbound_only"

    # tier 0 or 1 — some form of direct/mapped reachability
    ipv4_reachable = profile.public_endpoint is not None

    if ipv4_reachable and ipv6_available and ipv6_q.pinhole_ok:
        return "dual_stack_available"

    if not ipv4_reachable and ipv6_available:
        return "ipv6_direct_pinhole_available" if ipv6_q.pinhole_ok else "ipv6_direct_firewall_required"

    if ipv4_reachable:
        return "ipv4_public_direct"

    return "outbound_only"


def _build_recommendation(mode: str, profile: NatProfile) -> str:
    messages = {
        "ipv4_public_direct": "Direct IPv4 connection available. No additional setup needed.",
        "dual_stack_available": "Dual-stack (IPv4 + IPv6) available. Consumers can reach you on either path.",
        "ipv6_direct_pinhole_available": "IPv6 direct connection available with firewall pinhole. IPv4 unreachable.",
        "ipv6_direct_firewall_required": "IPv6 address routable but firewall is blocking. Open the relevant port.",
        "relay_required": "Behind CGNAT or strict firewall — use relay mode (iicp-node --relay-worker-endpoint).",
        "tunnel_required": "External tunnel detected (ngrok/Tailscale). Advertise the tunnel URL as public endpoint.",
        "ipv4_cgnat_blocked": "Carrier-grade NAT detected. Relay mode is the recommended path.",
        "outbound_only": "No inbound connectivity detected. Set --public-endpoint manually or use relay mode.",
    }
    base = messages.get(mode, "Unknown exposure mode.")
    guidance = profile.operator_guidance
    return f"{base} {guidance}".strip() if guidance else base


# ── Async convenience wrapper ─────────────────────────────────────────────────

async def qualify_service_async(
    bind_host: str = "0.0.0.0",
    bind_port: int = 8020,
    transport_port: int = 9484,
    *,
    detect_v6: bool = True,
    timeout_s: float = 15.0,
) -> ServiceQualification:
    """Run NAT detection and return a ServiceQualification in one call.

    Equivalent to ``detect_nat(...)`` followed by ``qualify_service(profile)``.
    """
    from iicp_client.nat_detection import detect_nat
    profile = await detect_nat(
        bind_host=bind_host,
        bind_port=bind_port,
        transport_port=transport_port,
        detect_v6=detect_v6,
        timeout_s=timeout_s,
    )
    return qualify_service(profile)
