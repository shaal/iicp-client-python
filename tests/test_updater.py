# ADR-016: IICP client SDK conformance — #521 self-updater P1 (read-only check)
"""Behavior tests for the version-check logic + the `update` CLI command.
Network is monkeypatched — no real PyPI call."""

from __future__ import annotations

import pytest

from iicp_client import updater
from iicp_client.cli import _build_parser, main
from iicp_client.updater import auto_update_tick


class TestVersionCompare:
    @pytest.mark.parametrize(
        ("cur", "latest", "expected"),
        [
            ("0.7.56", "0.7.57", True),
            ("0.7.57", "0.7.57", False),
            ("0.7.57", "0.7.56", False),
            ("0.7.9", "0.7.10", True),  # numeric, not lexicographic
            ("1.0.0", "0.9.9", False),
            ("v0.7.56", "0.7.57", True),  # leading v tolerated
        ],
    )
    def test_is_outdated(self, cur, latest, expected):
        assert updater.is_outdated(cur, latest) is expected

    def test_parse_version_truncates_prerelease(self):
        assert updater.parse_version("1.2.3rc1") == (1, 2, 3)
        assert updater.parse_version("0.7.57") == (0, 7, 57)


class TestCheckUpdate:
    def test_outdated_verdict(self):
        v = updater.check_update("0.7.56", "0.7.57")
        assert v["outdated"] is True
        assert v["command"] == "pip install -U iicp-client"

    def test_unknown_latest_is_not_outdated(self):
        v = updater.check_update("0.7.57", None)
        assert v["outdated"] is False


class TestUpdateCli:
    def test_subcommand_parses(self):
        ns = _build_parser().parse_args(["update", "--check"])
        assert ns.cmd == "update"
        assert ns.check is True

    # _cmd_update does `from iicp_client.updater import latest_pypi_version`
    # at call time, so patch the source module (the binding it imports from).
    def test_exit_10_when_outdated(self, monkeypatch, capsys):
        monkeypatch.setattr(updater, "latest_pypi_version", lambda *a, **k: "99.0.0")
        code = main(["update", "--check"])
        assert code == 10
        assert "newer release is available" in capsys.readouterr().out

    def test_exit_0_when_current(self, monkeypatch, capsys):
        from iicp_client import __version__
        monkeypatch.setattr(updater, "latest_pypi_version", lambda *a, **k: __version__)
        code = main(["update", "--check"])
        assert code == 0
        assert "up to date" in capsys.readouterr().out

    def test_exit_0_when_registry_unreachable(self, monkeypatch, capsys):
        monkeypatch.setattr(updater, "latest_pypi_version", lambda *a, **k: None)
        code = main(["update", "--check"])
        assert code == 0
        assert "could not reach PyPI" in capsys.readouterr().out


# ── P2 auto-updater (#521) ──────────────────────────────────────────────────────


def _spy():
    calls = []
    return calls, (lambda *a: calls.append(a))


def test_auto_update_tick_upgrades_and_reexecs_when_newer():
    logs, log_fn = _spy()
    reexec_calls = []
    result = auto_update_tick(
        "0.7.59", "0.7.60", True,
        upgrade_fn=lambda: True,
        reexec_fn=lambda: reexec_calls.append(1),
        log_fn=log_fn,
    )
    assert result == "upgraded"
    assert reexec_calls == [1]  # re-exec attempted exactly once


def test_auto_update_tick_noop_when_current():
    result = auto_update_tick(
        "0.7.60", "0.7.60", True,
        upgrade_fn=lambda: (_ for _ in ()).throw(AssertionError("must not upgrade")),
        reexec_fn=lambda: (_ for _ in ()).throw(AssertionError("must not reexec")),
        log_fn=lambda *a: None,
    )
    assert result == "current"


def test_auto_update_tick_disabled_is_noop():
    assert auto_update_tick("0.7.59", "0.7.60", False, lambda: True, lambda: None, lambda *a: None) == "disabled"


def test_auto_update_tick_unknown_latest_is_noop():
    assert auto_update_tick("0.7.59", None, True, lambda: True, lambda: None, lambda *a: None) == "unknown"


def test_auto_update_tick_failed_upgrade_does_not_reexec():
    reexec_calls = []
    result = auto_update_tick(
        "0.7.59", "0.7.60", True,
        upgrade_fn=lambda: False,
        reexec_fn=lambda: reexec_calls.append(1),
        log_fn=lambda *a: None,
    )
    assert result == "upgrade-failed"
    assert reexec_calls == []  # no restart on a failed upgrade
