"""Serve a session_analysis package as a local HTML visualizer."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent / "viz"

_STATIC_CONTENT_TYPES = {
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


def resolve_static_file(name_or_path: str) -> Path | None:
    """Resolve a /static/<name_or_path> request to a file under VIZ_DIR --
    basename only, so "../serve.py" or any other traversal attempt can
    never escape VIZ_DIR. Factored out (not just inlined in Handler.do_GET)
    so interstellar's combined server can reuse this exact rule instead of
    writing its own copy -- see interstellar/combined_serve.py."""
    fp = VIZ_DIR / Path(name_or_path).name
    return fp if fp.is_file() else None


def static_content_type(fp: Path) -> str:
    return _STATIC_CONTENT_TYPES.get(fp.suffix, "application/octet-stream")


def resolve_data_file(package_dir: Path, rel: str) -> Path | None:
    """Resolve a /data/<rel> request to a file under package_dir, guarding
    path traversal by resolving and checking containment. Same reuse
    rationale as resolve_static_file above."""
    package_dir = Path(package_dir).resolve()
    fp = (package_dir / rel).resolve()
    if not str(fp).startswith(str(package_dir)) or not fp.is_file():
        return None
    return fp


def data_content_type(fp: Path) -> str:
    if fp.suffix == ".jsonl":
        return "application/x-ndjson; charset=utf-8"
    return "application/json; charset=utf-8"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, package_dir: Path, report_url: str | None = None,
                 report_session_id: str | None = None, **kwargs):
        self.package_dir = package_dir.resolve()
        # View-switcher link to the matching interstellar review report,
        # if one was configured (see serve()/main()'s --report-url /
        # --report-session-id) -- exposed to app.js via /meta.json below.
        # Neither is trusted/validated here; it's a string handed to the
        # client, which only ever renders it as an href/label.
        self.report_url = report_url or ""
        self.report_session_id = report_session_id or ""
        super().__init__(*args, directory=str(VIZ_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send_file(VIZ_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/meta.json":
            return self._send_json({
                "report_url": self.report_url or None,
                "report_session_id": self.report_session_id or None,
            })
        if path.startswith("/static/"):
            fp = resolve_static_file(path[len("/static/") :])
            if fp is None:
                self.send_error(404)
                return
            return self._send_file(fp, static_content_type(fp))
        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            fp = resolve_data_file(self.package_dir, rel)
            if fp is None:
                self.send_error(404, f"missing {rel}")
                return
            return self._send_file(fp, data_content_type(fp))
        self.send_error(404)

    def _send_file(self, fp: Path, content_type: str) -> None:
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def serve(package_dir: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          report_url: str | None = None, report_session_id: str | None = None) -> None:
    package_dir = package_dir.resolve()
    if not (package_dir / "manifest.json").is_file():
        raise SystemExit(f"error: {package_dir} is not a package (missing manifest.json)")
    if not VIZ_DIR.is_dir():
        raise SystemExit(f"error: viz assets missing at {VIZ_DIR}")

    handler = functools.partial(
        Handler, package_dir=package_dir,
        report_url=report_url, report_session_id=report_session_id,
    )
    # allow reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"serving package {package_dir}")
        print(f"open {url}")
        if open_browser:
            threading.Thread(
                target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                daemon=True,
            ).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local HTML visualizer for a session_analysis package")
    p.add_argument("package_dir", type=Path, help="Normalized package directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="Do not open a browser")
    p.add_argument("--report-url", default=None,
                    help="URL of the matching interstellar review report, if any "
                         "(rendered as a view-switcher link in the topbar)")
    p.add_argument("--report-session-id", default=None,
                    help="session_id the --report-url report is actually for, "
                         "used to verify/label the switcher link -- a report_url "
                         "given without this renders the link but marks it "
                         "unverified rather than claiming a match")
    args = p.parse_args(argv)
    serve(
        args.package_dir, host=args.host, port=args.port, open_browser=not args.no_open,
        report_url=args.report_url, report_session_id=args.report_session_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
