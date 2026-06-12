# ADR-016: IICP client SDK conformance — #520 Quick-Tunnel escalation (rung 5)
"""Behavior tests for the automatic tunnel lifecycle (fail if #520 reverts):
setup (binary detection), initiation (spawn + URL parse + timeout), teardown
(close terminates the child, idempotent), supervision (watchdog respawn with
NEW url → re-register callback; bounded → on_dead).

A fake `cloudflared` (tiny python script) stands in for the real binary, so
the suite needs no network and no Cloudflare."""

from __future__ import annotations

import stat
import sys
import threading
import time

import pytest

from iicp_client.tunnel import (
    INSTALL_HINT,
    cloudflared_path,
    open_quick_tunnel,
)

FAKE_OK = """#!{python}
import sys, time
sys.stdout.write("INF | starting tunnel\\n")
sys.stdout.write("INF | https://{name}.trycloudflare.com\\n")
sys.stdout.flush()
time.sleep({lifetime})
"""

FAKE_SILENT = """#!{python}
import time
time.sleep(60)
"""


def _fake_bin(tmp_path, template: str, name: str = "fake-fox-1234", lifetime: float = 60.0) -> str:
    p = tmp_path / "cloudflared"
    p.write_text(template.format(python=sys.executable, name=name, lifetime=lifetime))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


class TestSetup:
    def test_cloudflared_path_none_when_absent(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert cloudflared_path() is None

    def test_open_raises_install_hint_when_absent(self, monkeypatch):
        monkeypatch.setattr("iicp_client.tunnel.cloudflared_path", lambda: None)
        with pytest.raises(FileNotFoundError) as ei:
            open_quick_tunnel(9484)
        assert "brew install cloudflared" in str(ei.value)
        assert INSTALL_HINT.startswith("cloudflared not found")


class TestInitiation:
    def test_parses_url_from_output(self, tmp_path):
        t = open_quick_tunnel(9484, binary=_fake_bin(tmp_path, FAKE_OK))
        try:
            assert t.url == "https://fake-fox-1234.trycloudflare.com"
            assert t.local_port == 9484
            assert t.process.poll() is None  # still running
        finally:
            t.close()

    def test_timeout_when_no_url(self, tmp_path):
        with pytest.raises(RuntimeError, match="no tunnel URL"):
            open_quick_tunnel(9484, timeout=0.5, binary=_fake_bin(tmp_path, FAKE_SILENT))


class TestTeardown:
    def test_close_terminates_child_and_is_idempotent(self, tmp_path):
        t = open_quick_tunnel(9484, binary=_fake_bin(tmp_path, FAKE_OK))
        assert t.process.poll() is None
        t.close()
        assert t.process.poll() is not None  # child gone
        t.close()  # second close must not raise


class TestSupervision:
    def test_watchdog_respawns_with_new_url(self, tmp_path):
        t = open_quick_tunnel(9484, binary=_fake_bin(tmp_path, FAKE_OK, lifetime=60))
        new_urls: list[str] = []
        got_url = threading.Event()
        t.watch(lambda u: (new_urls.append(u), got_url.set()), lambda: None)
        t.process.kill()  # simulate unexpected death
        assert got_url.wait(timeout=10), "watchdog did not respawn in time"
        assert new_urls and new_urls[0].startswith("https://")
        assert t.process.poll() is None  # fresh child running
        t.close()

    def test_watchdog_gives_up_after_max_respawns(self, tmp_path):
        # A binary that dies instantly after printing → every respawn dies too.
        t = open_quick_tunnel(9484, binary=_fake_bin(tmp_path, FAKE_OK, lifetime=0.01))
        dead = threading.Event()
        t.watch(lambda _u: None, dead.set)
        assert dead.wait(timeout=30), "on_dead never fired"
        assert t._respawns >= 1
        t.close()

    def test_close_suppresses_watchdog(self, tmp_path):
        t = open_quick_tunnel(9484, binary=_fake_bin(tmp_path, FAKE_OK))
        fired = threading.Event()
        t.watch(lambda _u: fired.set(), lambda: fired.set())
        t.close()  # intentional teardown — watchdog must NOT respawn
        time.sleep(0.5)
        assert not fired.is_set()


class TestCliWiring:
    def test_serve_has_tunnel_flags(self):
        from iicp_client.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["serve", "--model", "m", "--tunnel"])
        assert ns.tunnel is True
        ns = parser.parse_args(["serve", "--model", "m", "--no-tunnel"])
        assert ns.tunnel is False
        ns = parser.parse_args(["serve", "--model", "m"])
        assert ns.tunnel is None  # auto
