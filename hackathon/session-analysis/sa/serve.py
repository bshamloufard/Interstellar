"""Serve a session_analysis package as a local HTML visualizer."""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent / "viz"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, package_dir: Path, **kwargs):
        self.package_dir = package_dir.resolve()
        super().__init__(*args, directory=str(VIZ_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send_file(VIZ_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            # only allow basenames under viz/
            name = Path(rel).name
            fp = VIZ_DIR / name
            if not fp.is_file():
                self.send_error(404)
                return
            ctype = {
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".html": "text/html; charset=utf-8",
            }.get(fp.suffix, "application/octet-stream")
            return self._send_file(fp, ctype)
        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            # prevent path traversal
            fp = (self.package_dir / rel).resolve()
            if not str(fp).startswith(str(self.package_dir)) or not fp.is_file():
                self.send_error(404, f"missing {rel}")
                return
            ctype = "application/json; charset=utf-8"
            if fp.suffix == ".jsonl":
                ctype = "application/x-ndjson; charset=utf-8"
            return self._send_file(fp, ctype)
        self.send_error(404)

    def _send_file(self, fp: Path, content_type: str) -> None:
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def serve(package_dir: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    package_dir = package_dir.resolve()
    if not (package_dir / "manifest.json").is_file():
        raise SystemExit(f"error: {package_dir} is not a package (missing manifest.json)")
    if not VIZ_DIR.is_dir():
        raise SystemExit(f"error: viz assets missing at {VIZ_DIR}")

    handler = functools.partial(Handler, package_dir=package_dir)
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
    args = p.parse_args(argv)
    serve(args.package_dir, host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
