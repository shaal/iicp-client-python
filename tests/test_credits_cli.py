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
