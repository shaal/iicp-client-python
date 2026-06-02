"""#405 — single-instance lock per node_id."""

from __future__ import annotations

import subprocess

import pytest

from iicp_client.instance_lock import InstanceLock, NodeAlreadyServingError


def test_live_foreign_pid_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("IICP_HOME", str(tmp_path))
    # a real, same-user, signalable live process holding the lock
    child = subprocess.Popen(["sleep", "30"])
    try:
        run = tmp_path / "run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "dup.pid").write_text(str(child.pid))
        with pytest.raises(NodeAlreadyServingError):
            InstanceLock.acquire("dup", force=False)
        # --force takes over
        InstanceLock.acquire("dup", force=True)
    finally:
        child.terminate()
        child.wait()


def test_distinct_nodes_and_release(tmp_path, monkeypatch):
    monkeypatch.setenv("IICP_HOME", str(tmp_path))
    a = InstanceLock.acquire("node-a", force=False)
    b = InstanceLock.acquire("node-b", force=False)  # distinct → no conflict
    assert a and b
    a.release()
    # re-acquirable after release
    InstanceLock.acquire("node-a", force=False)
