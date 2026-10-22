"""Local server: same collectors, re-run per poll.

Bound to 127.0.0.1.  This serves a read-only view of a working directory;
it must never be exposed.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Tuple

from .payload import collect
from .render import render


class Router:
    """Path -> (status, content_type, body).  Pure enough to unit-test."""

    def __init__(
        self,
        project_root: Path,
        registry_path: Optional[Path],
        *,
        poll_seconds: float = 10.0,
    ) -> None:
        self.project_root = Path(project_root)
        self.registry_path = registry_path
        self.poll_seconds = float(poll_seconds)

    def _payload(self) -> dict:
        return collect(
            self.project_root,
            self.registry_path,
            mode="serve",
            poll_window_seconds=self.poll_seconds,
        )

    def handle(self, path: str) -> Tuple[int, str, str]:
        if path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", render(self._payload())
        if path.startswith("/api/"):
            key = path[len("/api/"):].strip("/")
            doc = self._payload()
            if key in ("gates", "results", "fleet", "live", "chain"):
                return 200, "application/json", json.dumps(doc.get(key, {}), default=str)
            if key == "all":
                return 200, "application/json", json.dumps(doc, default=str)
        return 404, "text/plain; charset=utf-8", "not found"


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, ctype, body = router.handle(self.path)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return  # quiet

    return Handler


def serve(
    project_root: Path,
    registry_path: Optional[Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_seconds: float = 10.0,
) -> None:
    router = Router(project_root, registry_path, poll_seconds=poll_seconds)
    server = HTTPServer((host, port), make_handler(router))
    print(f"[dashboard] http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()
