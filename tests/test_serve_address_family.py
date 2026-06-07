# SPDX-License-Identifier: Apache-2.0
"""serve() must bind the multiplexer listener to the address family of --host.

The CLI defaults --host/IICP_HOST to "::" (IPv6 any). serve() used to hardcode
``socket.socket(socket.AF_INET, ...)`` then ``bind((host, port))``, so binding an
IPv6 host to an AF_INET socket raised ``gaierror: Address family for hostname not
supported`` — i.e. ``iicp-node serve`` crashed on its own default host. The test
suite never caught it because every serve test passes host="127.0.0.1" (AF_INET).

These tests pin the family-resolution helper and prove a listener can actually be
created+bound on the default IPv6 host.
"""

from __future__ import annotations

import socket

import pytest

from iicp_client.node import _listen_family


def _ipv6_available() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        s.bind(("::", 0))
    except OSError:
        return False
    finally:
        s.close()
    return True


def test_listen_family_ipv4_hosts() -> None:
    assert _listen_family("0.0.0.0", 0) == socket.AF_INET
    assert _listen_family("127.0.0.1", 0) == socket.AF_INET


def test_listen_family_ipv6_any() -> None:
    if not _ipv6_available():
        pytest.skip("IPv6 not available on this host")
    # The CLI default host. Pre-fix this resolved to AF_INET6 but the socket was
    # hardcoded AF_INET, so the bind blew up.
    assert _listen_family("::", 0) == socket.AF_INET6


def test_default_ipv6_host_binds_without_crashing() -> None:
    """Regression: the default "::" host must produce a bindable listener.

    Reproduces the serve() listener setup (resolve family → create socket → bind)
    on the default IPv6 host and asserts it does not raise gaierror.
    """
    if not _ipv6_available():
        pytest.skip("IPv6 not available on this host")
    host = "::"
    family = _listen_family(host, 0)
    assert family == socket.AF_INET6
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        # Pre-fix this line raised socket.gaierror on an AF_INET socket.
        listener.bind((host, 0))
        listener.listen(128)
    finally:
        listener.close()


def test_ipv4_host_still_binds() -> None:
    """The previously-working AF_INET path keeps working."""
    family = _listen_family("127.0.0.1", 0)
    assert family == socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
    finally:
        listener.close()
