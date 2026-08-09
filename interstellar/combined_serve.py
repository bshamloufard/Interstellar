"""interstellar/combined_serve.py — one loopback server, two real tabs.

Serves Alex's trace visualizer (hackathon/session-analysis/sa) and our own
review report (interstellar/report.py) as top-level pages on ONE origin:

    /            -> redirect to whichever tab is available (prefers /trace)
    /trace       -> Alex's viz index.html, with a tab bar injected
    /review      -> our report index.html, with a tab bar injected
    /static/*    -> Alex's viz assets       (his existing routes, reused)
    /data/*      -> Alex's package JSON     (his existing routes, reused)
    /report.json -> ours                    (unchanged)
    /run, /status-> ours                    (unchanged)

Real page navigations, not iframes and not a client-side router -- each
side keeps its own top-level page and asset paths; this module only reuses
their existing serving logic and injects a small shared tab bar into each
page's HTML at request time (a string insert right after "<body>", not a
template system).

Either side may be missing (no --trace-package given, or a report out_dir
with no index.html yet): the tab bar always shows both tabs, but an
unavailable one renders as a disabled label with a reason, never a link to
a broken destination -- and navigating to that route directly renders an
explanatory page (200), never a 500.

Entry point:
    python3 -m interstellar.combined_serve <out_dir> [--trace-package DIR] [--port 4242]

`interstellar/cli.py` does not call this module (out of this module's
ownership) -- run it directly, or see report.serve()/sa.serve.serve() for
the single-page-per-server alternative this supersedes for anyone who
wants one browser tab for both views.
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SA_ROOT = _REPO_ROOT / "hackathon" / "session-analysis"
# hackathon/session-analysis's `sa` package is not installed / not on
# sys.path by default (it has no pyproject.toml) -- this is the one cross-
# tree import in the project, so it gets the one sys.path insert, done
# once, guarded, rather than a copy of sa.serve's file-resolution logic.
if str(_SA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SA_ROOT))

from sa.serve import (  # noqa: E402  (see sys.path note above)
    VIZ_DIR,
    data_content_type,
    resolve_data_file,
    resolve_static_file,
    static_content_type,
)

from interstellar import report as report_mod  # noqa: E402

DEFAULT_PORT = 4242


def _read_json(fp: Path):
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class _Sides:
    """What's actually servable on each tab. Availability and session ids
    are (re)computed per call, not cached at server start -- report.json
    changes after a Re-run, and recomputing a couple of small JSON reads
    per request is cheap compared to serving a stale mismatch verdict."""

    def __init__(self, out_dir, trace_package_dir):
        self.out_dir = Path(out_dir).resolve() if out_dir else None
        self.trace_package_dir = Path(trace_package_dir).resolve() if trace_package_dir else None

    def review_available(self):
        return bool(self.out_dir and (self.out_dir / "index.html").is_file())

    def review_report(self):
        return _read_json(self.out_dir / "report.json") if self.out_dir else None

    def review_session_id(self):
        rpt = self.review_report()
        return ((rpt or {}).get("session") or {}).get("session_id") or ""

    def trace_available(self):
        return bool(self.trace_package_dir and (self.trace_package_dir / "manifest.json").is_file())

    def trace_session_id(self):
        if not self.trace_package_dir:
            return ""
        manifest = _read_json(self.trace_package_dir / "manifest.json") or {}
        return manifest.get("session_id") or ""

    def unavailable_reason(self, side):
        if side == "review":
            if not self.out_dir:
                return "no review report directory was given to this server"
            if not (self.out_dir / "index.html").is_file():
                return f"{self.out_dir} has no index.html yet (run `interstellar review` first)"
            return ""
        if not self.trace_package_dir:
            return "no --trace-package was given to this server"
        if not (self.trace_package_dir / "manifest.json").is_file():
            return f"{self.trace_package_dir} has no manifest.json (run `python -m sa normalize` first)"
        return ""


# --- shared tab bar -----------------------------------------------------
#
# Reuses the .view-switcher/.view-seg/.view-link/.pill classes already
# defined in BOTH report.py's _STYLE and sa/viz/styles.css (added for the
# earlier cross-port view-switcher) -- injecting this markup needs no new
# CSS in either page's own stylesheet, only in the one small self-
# contained shell page this module serves when a route is unavailable
# (see _SHELL_STYLE) where there is no host stylesheet to borrow from.

def _tab_seg_html(label, path, available, is_active, reason):
    if is_active:
        return f'<span class="view-seg view-seg-active">{html.escape(label)}</span>'
    if not available:
        return (
            f'<span class="view-seg view-link view-seg-unavailable" aria-disabled="true" '
            f'title="unavailable: {html.escape(reason)}">{html.escape(label)}</span>'
        )
    return f'<a class="view-seg view-link" href="{path}">{html.escape(label)}</a>'


def _tab_shell_html(sides, active):
    """The injected shell: a two-tab switcher (Trace/Review), a session id
    pill, and -- only when both sides are available AND disagree -- a loud
    mismatch banner. Never claims one session when the two sides disagree;
    see the module docstring."""
    trace_ok = sides.trace_available()
    review_ok = sides.review_available()
    trace_id = sides.trace_session_id() if trace_ok else ""
    review_id = sides.review_session_id() if review_ok else ""
    mismatch = bool(trace_ok and review_ok and trace_id and review_id and trace_id != review_id)

    bar = (
        '<div class="view-switcher" role="tablist" aria-label="View">'
        + _tab_seg_html("Trace", "/trace", trace_ok, active == "trace", sides.unavailable_reason("trace"))
        + _tab_seg_html("Review", "/review", review_ok, active == "review", sides.unavailable_reason("review"))
        + "</div>"
    )
    shown_id = trace_id or review_id
    pill = (
        f'<span class="pill">session <strong>{html.escape(shown_id)}</strong></span>'
        if shown_id else ""
    )
    banner = ""
    if mismatch:
        banner = (
            '<div class="tab-mismatch-banner" role="alert">&#9888; session mismatch — '
            f"trace is <strong>{html.escape(trace_id)}</strong>, "
            f"review is <strong>{html.escape(review_id)}</strong> — "
            "these are NOT the same session, do not read them as one.</div>"
        )
    return f'<div class="tab-shell">{bar}{pill}</div>{banner}'


def _inject_tab_shell(page_html, sides, active):
    """Insert the tab shell right after the page's opening <body> tag --
    the one, simple, dependency-free way to add shared chrome to a page
    this module doesn't otherwise touch (Alex's static index.html, or our
    own render_html() output) without understanding or restyling
    everything else on it."""
    shell = _tab_shell_html(sides, active)
    marker = "<body>"
    idx = page_html.find(marker)
    if idx == -1:
        return shell + page_html
    insert_at = idx + len(marker)
    return page_html[:insert_at] + "\n" + shell + page_html[insert_at:]


# --- the small self-contained shell page for an unavailable route -------
#
# Duplicates a handful of CSS rules also defined in report.py's _STYLE and
# sa/viz/styles.css (same token values, so it still "looks native" -- see
# the brief) rather than linking to either: this page can be reached when
# NEITHER underlying page exists to serve (e.g. --trace-package was never
# given), so it cannot depend on either stylesheet being reachable.
_SHELL_STYLE = """
:root {
  --bg: #0b0e14; --bg-elev: #11151c; --border: #1e2633; --text: #e6edf7;
  --muted: #8b97a8; --faint: #5c6b7e; --accent: #3b82f6;
  --accent-dim: rgba(59, 130, 246, 0.15); --bad: #f87171;
  --bad-dim: rgba(248, 113, 113, 0.14);
  --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--font); }
"""

# Shared by both report.py's _STYLE and sa/viz/styles.css (each defines
# .view-switcher/.view-seg/.pill already for the cross-port switcher) --
# this module adds only the two pieces neither page already has.
_TAB_EXTRA_CSS = """
.tab-shell { display: flex; align-items: center; gap: 0.6rem; padding: 0.6rem 0.9rem;
  border-bottom: 1px solid var(--border); background: var(--bg-elev); font-family: var(--font); }
.view-seg-unavailable { color: var(--faint) !important; cursor: not-allowed; }
.tab-mismatch-banner { background: var(--bad-dim); border-bottom: 1px solid var(--bad);
  color: var(--bad); font-weight: 600; font-size: 12.5px; padding: 0.45rem 0.9rem; font-family: var(--font); }
.view-switcher { display: inline-flex; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; font-size: 11px; font-family: var(--mono); }
.view-seg { padding: 0.2rem 0.55rem; white-space: nowrap; }
.view-seg-active { color: var(--text); background: var(--accent-dim); font-weight: 600; }
.view-link { color: var(--muted); background: var(--bg); border-left: 1px solid var(--border);
  text-decoration: none; }
.view-link:hover { color: var(--accent); }
.pill { border: 1px solid var(--border); background: var(--bg); color: var(--muted);
  border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 11px; font-family: var(--mono); }
.pill strong { color: var(--text); font-weight: 600; }
.tab-empty { max-width: 640px; margin: 3rem auto; padding: 0 1rem; text-align: center; }
.tab-empty h1 { font-size: 18px; }
.tab-empty p { color: var(--muted); }
"""


def _unavailable_page_html(sides, side):
    label = "Trace" if side == "trace" else "Review"
    reason = sides.unavailable_reason(side)
    shell = _tab_shell_html(sides, side)
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{label} unavailable</title>
<style>{_SHELL_STYLE}{_TAB_EXTRA_CSS}</style></head>
<body>
{shell}
<main class="tab-empty">
  <h1>{html.escape(label)} unavailable</h1>
  <p>{html.escape(reason)}</p>
</main>
</body>
</html>"""


def _nothing_configured_html():
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"/><title>Nothing configured</title>
<style>{_SHELL_STYLE}</style></head>
<body>
<main class="tab-empty">
  <h1>Nothing to show</h1>
  <p>This server was started without a review report directory or a
  --trace-package. Pass at least one.</p>
</main>
</body>
</html>"""


# --- HTTP handler ---------------------------------------------------------

class _CombinedHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, sides, run_manager, **kwargs):
        self._sides = sides
        self._run_manager = run_manager
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._redirect_root()
        if path in ("/trace", "/trace/"):
            return self._serve_trace()
        if path in ("/review", "/review/"):
            return self._serve_review()
        if path == "/status":
            if self._run_manager is None:
                return self._send_json(503, {"error": "run manager unavailable"})
            return self._send_json(200, self._run_manager.snapshot())
        if path == "/report.json":
            return self._serve_report_json()
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path.startswith("/data/"):
            return self._serve_data(path[len("/data/"):])
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path.split("?", 1)[0] != "/run":
            self.send_error(404)
            return
        if self._run_manager is None:
            self._send_json(503, {"error": "run manager unavailable"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > 4096:
            self._send_json(400, {"error": "request body too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) if length else b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        k, err = report_mod.validate_run_k(body)
        if err:
            self._send_json(400, {"error": err})
            return
        ok, err = self._run_manager.start(k)
        if not ok:
            status = 409 if err == "a review cycle is already running" else 400
            self._send_json(status, {"error": err})
            return
        self._send_json(202, self._run_manager.snapshot())

    # --- route handlers ---------------------------------------------------

    def _redirect_root(self):
        if self._sides.trace_available():
            target = "/trace"
        elif self._sides.review_available():
            target = "/review"
        else:
            return self._send_html(200, _nothing_configured_html())
        self.send_response(302)
        self.send_header("Location", target)
        self.end_headers()

    def _serve_trace(self):
        if not self._sides.trace_available():
            return self._send_html(200, _unavailable_page_html(self._sides, "trace"))
        page = (VIZ_DIR / "index.html").read_text(encoding="utf-8")
        return self._send_html(200, _inject_tab_shell(page, self._sides, "trace"))

    def _serve_review(self):
        if not self._sides.review_available():
            return self._send_html(200, _unavailable_page_html(self._sides, "review"))
        report_dict = self._sides.review_report()
        if report_dict is None:
            return self._send_html(200, _unavailable_page_html(self._sides, "review"))
        page = report_mod.render_html(report_dict)
        return self._send_html(200, _inject_tab_shell(page, self._sides, "review"))

    def _serve_report_json(self):
        if not self._sides.out_dir:
            self.send_error(404)
            return
        fp = self._sides.out_dir / "report.json"
        if not fp.is_file():
            self.send_error(404)
            return
        self._send_bytes(200, fp.read_bytes(), "application/json; charset=utf-8")

    def _serve_static(self, rel):
        fp = resolve_static_file(rel)
        if fp is None:
            self.send_error(404)
            return
        self._send_bytes(200, fp.read_bytes(), static_content_type(fp))

    def _serve_data(self, rel):
        if not self._sides.trace_package_dir:
            self.send_error(404)
            return
        fp = resolve_data_file(self._sides.trace_package_dir, rel)
        if fp is None:
            self.send_error(404, f"missing {rel}")
            return
        self._send_bytes(200, fp.read_bytes(), data_content_type(fp))

    # --- response helpers ---------------------------------------------------

    def _send_html(self, status, page_html):
        self._send_bytes(status, page_html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, status, payload):
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_bytes(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class _LoopbackTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def make_server(out_dir, trace_package_dir, *, host="127.0.0.1", port=DEFAULT_PORT, argv_builder=None):
    """Build (but do not start) the combined loopback server. Split out
    from serve() so tests can bind an ephemeral port (port=0). Either
    out_dir or trace_package_dir may be None/missing -- see _Sides."""
    sides = _Sides(out_dir, trace_package_dir)
    run_manager = report_mod.make_run_manager(sides.out_dir, argv_builder=argv_builder) if sides.out_dir else None
    handler = functools.partial(_CombinedHandler, sides=sides, run_manager=run_manager)
    return _LoopbackTCPServer((host, port), handler)


def serve(out_dir, trace_package_dir, *, host="127.0.0.1", port=DEFAULT_PORT, open_browser=True, argv_builder=None):
    """Serve both tabs on one loopback server, bound to 127.0.0.1 only."""
    with make_server(out_dir, trace_package_dir, host=host, port=port, argv_builder=argv_builder) as httpd:
        bound_host, bound_port = httpd.server_address
        url = f"http://{bound_host}:{bound_port}/"
        print(f"serving combined dashboard at {url}")
        if out_dir:
            print(f"  review: {Path(out_dir).resolve()}")
        if trace_package_dir:
            print(f"  trace:  {Path(trace_package_dir).resolve()}")
        if open_browser:
            threading.Thread(
                target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True,
            ).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m interstellar.combined_serve",
        description="Serve Alex's trace visualizer and our review report as "
                     "tabs on one loopback server.",
    )
    p.add_argument("out_dir", type=Path, nargs="?", default=None,
                    help="review report output dir (report.json + index.html "
                         "from `interstellar review`); omit to serve only --trace-package")
    p.add_argument("--trace-package", type=Path, default=None,
                    help="Alex's normalized session_analysis package dir "
                         "(manifest.json + data/*.json, from `python -m sa normalize`); "
                         "omit to serve only the review report")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = p.parse_args(argv)
    if not args.out_dir and not args.trace_package:
        raise SystemExit("error: need at least one of <out_dir> or --trace-package")
    serve(
        args.out_dir, args.trace_package, host=args.host, port=args.port,
        open_browser=not args.no_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
