# SPDX-License-Identifier: Apache-2.0
"""#456 — `iicp-node credits` CLI.

Renders a 200 summary, and ERRORS on 401: a forged/wrong token cannot fabricate
credits (the figures come authenticated from the directory, not the local config).
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from iicp_client.cli import main


def _serve_once(status: int, body: str) -> int:
    """Single-shot mock of GET /v1/credits/summary — std-only, no test deps."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *_args):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.handle_request, daemon=True).start()
    return port


def test_credits_renders_on_200(capsys):
    body = (
        '{"node_id":"n1","total_earned":142.5,"total_spent":38.25,"balance":104.25,'
        '"tx_count":2,"reconciles":true,"unit":"credit","tokens_per_credit":1000}'
    )
    port = _serve_once(200, body)
    rc = main(
        ["credits", "--node-id", "n1", "--token", "t",
         "--directory-url", f"http://127.0.0.1:{port}", "--json"]
    )
    assert rc == 0
    assert '"total_earned": 142.5' in capsys.readouterr().out


def test_credits_errs_on_forged_token_401():
    body = '{"error":{"code":"unauthorized","message":"invalid node_token"}}'
    port = _serve_once(401, body)
    rc = main(
        ["credits", "--node-id", "n1", "--token", "forged",
         "--directory-url", f"http://127.0.0.1:{port}"]
    )
    # A forged/wrong token must be rejected — the local config cannot fabricate credits.
    assert rc == 1


# --- #456 --verify: cryptographic audit + tamper rejection ---
import asyncio  # noqa: E402
import base64  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json as _json  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from iicp_client.cli import _verify_credit_awards  # noqa: E402


def _make_signed_events(sk: Ed25519PrivateKey, payload: dict):
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    pub_b64 = base64.urlsafe_b64encode(pub).decode().rstrip("=")
    event_id, seq, ts_ms = "11111111-1111-1111-1111-111111111111", 2, 1_700_000_000_000
    canon = _json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    ph = hashlib.sha256(canon.encode()).hexdigest()
    msg = hashlib.sha256(f"{event_id}:CREDIT_AWARD:{seq}:{ts_ms}:{ph}".encode()).digest()
    sig = sk.sign(msg).hex()
    ev = {"event_id": event_id, "event_type": "CREDIT_AWARD", "seq": seq, "ts_ms": ts_ms,
          "node_id": "n1", "payload": payload, "sig": sig}
    return pub_b64, {"events": [ev]}


def _serve(pub_b64: str, events: dict):
    did = {"verificationMethod": [{"publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": pub_b64}}]}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = _json.dumps(did if "did.json" in self.path else events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_verify_accepts_valid_award():
    sk = Ed25519PrivateKey.generate()
    pub, events = _make_signed_events(sk, {"amount": 5.0, "new_balance": 5.0, "task_id": "t1"})
    srv, port = _serve(pub, events)
    try:
        vsum, ok, failed = asyncio.run(_verify_credit_awards(f"http://127.0.0.1:{port}", "n1"))
        assert (vsum, ok, failed) == (5.0, 1, 0)
    finally:
        srv.shutdown()


def test_verify_rejects_tampered_amount():
    sk = Ed25519PrivateKey.generate()
    pub, events = _make_signed_events(sk, {"amount": 5.0, "new_balance": 5.0, "task_id": "t1"})
    tampered = copy.deepcopy(events)
    tampered["events"][0]["payload"]["amount"] = 9999.0  # mutate, keep original sig
    srv, port = _serve(pub, tampered)
    try:
        _vsum, ok, failed = asyncio.run(_verify_credit_awards(f"http://127.0.0.1:{port}", "n1"))
        assert ok == 0 and failed >= 1, "tampered amount must not verify"
    finally:
        srv.shutdown()
