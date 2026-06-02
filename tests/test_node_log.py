"""Tests for iicp_client.node_log — persistent node log writer."""
import json
import logging
from pathlib import Path

import pytest

from iicp_client.node_log import setup_node_log, write_event


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def test_write_event_creates_files(tmp_log_dir: Path) -> None:
    write_event("n1", "register_ok", "endpoint=http://localhost:9484", str(tmp_log_dir))
    assert (tmp_log_dir / "events.jsonl").exists()


def test_write_event_jsonl_is_valid(tmp_log_dir: Path) -> None:
    write_event("n1", "heartbeat_ok", "seq=1", str(tmp_log_dir))
    line = (tmp_log_dir / "events.jsonl").read_text().strip()
    record = json.loads(line)
    assert record["event"] == "heartbeat_ok"
    assert record["node_id"] == "n1"
    assert record["details"] == "seq=1"
    assert "ts" in record


def test_write_event_multiple_appends(tmp_log_dir: Path) -> None:
    write_event("n1", "register_ok", "", str(tmp_log_dir))
    write_event("n1", "heartbeat_ok", "seq=1", str(tmp_log_dir))
    write_event("n1", "heartbeat_ok", "seq=2", str(tmp_log_dir))
    lines = (tmp_log_dir / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(ln)["event"] for ln in lines]
    assert events == ["register_ok", "heartbeat_ok", "heartbeat_ok"]


def test_setup_node_log_adds_file_handler(tmp_log_dir: Path) -> None:
    setup_node_log("testnode", str(tmp_log_dir))
    logger = logging.getLogger("iicp-node")
    file_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
        and "testnode" in h.baseFilename
    ]
    assert len(file_handlers) >= 1


def test_setup_node_log_no_duplicate_handlers(tmp_log_dir: Path) -> None:
    setup_node_log("dedup", str(tmp_log_dir))
    before = len(logging.getLogger("iicp-node").handlers)
    setup_node_log("dedup", str(tmp_log_dir))
    after = len(logging.getLogger("iicp-node").handlers)
    assert after == before


def test_setup_node_log_writes_via_logger(tmp_log_dir: Path) -> None:
    lgr = logging.getLogger("iicp-node")
    lgr.setLevel(logging.INFO)
    setup_node_log("writetest", str(tmp_log_dir))
    lgr.info("hello from test")
    # Flush all file handlers to ensure content is written.
    for h in lgr.handlers:
        h.flush()
    log_file = tmp_log_dir / "writetest.log"
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()


def test_log_dir_from_env(tmp_log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IICP_LOG_DIR", str(tmp_log_dir))
    write_event("env-node", "serve_start", "port=9484")
    assert (tmp_log_dir / "events.jsonl").exists()
