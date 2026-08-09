"""Tests for interstellar/combined_serve.py -- the one-server, two-tab
dashboard (supersedes the earlier cross-port view-switcher for anyone
using this launcher; see combined_serve.py's module docstring).

Checks: the route table (/, /trace, /review, /static, /data, /report.json,
/run, /status) against a real loopback server; graceful degradation when
one side is missing (available tab renders, missing tab shows a reason,
never a 500); the session-mismatch banner renders when both sides are
configured but disagree, and is absent when they agree or when only one
side is configured (nothing to compare); Alex's path-traversal guards are
still enforced through the reused resolve_static_file/resolve_data_file
functions (not a looser copy); and /run only ever reads `k` from the
request, exactly like report.py's own endpoint (see RunEndpointTests in
test_report.py -- this is the same contract on new routes).
"""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from interstellar import combined_serve
from interstellar import report as report_mod


def _make_review_out(tmp_path, session_id, k=5, title="Test session"):
    out_dir = tmp_path / "out"
    rpt = report_mod.build(
        session={
            "session_id": session_id, "title": title, "prompt": "do the thing",
            "trace_file": "/tmp/fake-trace.json",
        },
        baseline_version={"version_id": "base-1", "skills": 1, "mcp": 0},
        patch_results=[], k=k, cost_usd=0.01,
    )
    report_mod.write(rpt, out_dir)
    return out_dir


def _make_trace_package(tmp_path, session_id, name="pkg"):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "manifest.json").write_text(
        json.dumps({"session_id": session_id, "kind": "manifest"}), encoding="utf-8",
    )
    (pkg / "session.json").write_text(
        json.dumps({"session_id": session_id, "id": session_id}), encoding="utf-8",
    )
    return pkg


class SidesTests(unittest.TestCase):
    """Pure logic: availability, session ids, unavailable reasons -- no
    server involved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_both_sides_available_and_matching(self):
        out = _make_review_out(self.tmp_path, "sess-A")
        pkg = _make_trace_package(self.tmp_path, "sess-A")
        sides = combined_serve._Sides(out, pkg)
        self.assertTrue(sides.review_available())
        self.assertTrue(sides.trace_available())
        self.assertEqual(sides.review_session_id(), "sess-A")
        self.assertEqual(sides.trace_session_id(), "sess-A")

    def test_review_missing_reports_reason(self):
        sides = combined_serve._Sides(None, None)
        self.assertFalse(sides.review_available())
        self.assertIn("no review report directory", sides.unavailable_reason("review"))

    def test_review_out_dir_without_index_html_reports_reason(self):
        empty_out = self.tmp_path / "empty_out"
        empty_out.mkdir()
        sides = combined_serve._Sides(empty_out, None)
        self.assertFalse(sides.review_available())
        self.assertIn("no index.html", sides.unavailable_reason("review"))

    def test_trace_missing_reports_reason(self):
        sides = combined_serve._Sides(None, None)
        self.assertFalse(sides.trace_available())
        self.assertIn("no --trace-package", sides.unavailable_reason("trace"))

    def test_trace_package_without_manifest_reports_reason(self):
        empty_pkg = self.tmp_path / "empty_pkg"
        empty_pkg.mkdir()
        sides = combined_serve._Sides(None, empty_pkg)
        self.assertFalse(sides.trace_available())
        self.assertIn("no manifest.json", sides.unavailable_reason("trace"))


class TabShellRenderingTests(unittest.TestCase):
    """Pure rendering: _tab_shell_html / _inject_tab_shell."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_matching_sessions_render_no_mismatch_banner(self):
        out = _make_review_out(self.tmp_path, "same-session")
        pkg = _make_trace_package(self.tmp_path, "same-session")
        sides = combined_serve._Sides(out, pkg)
        shell = combined_serve._tab_shell_html(sides, "review")
        self.assertNotIn("tab-mismatch-banner", shell)
        self.assertIn("same-session", shell)

    def test_mismatched_sessions_render_loud_banner_naming_both_ids(self):
        out = _make_review_out(self.tmp_path, "review-session")
        pkg = _make_trace_package(self.tmp_path, "trace-session")
        sides = combined_serve._Sides(out, pkg)
        shell = combined_serve._tab_shell_html(sides, "review")
        self.assertIn("tab-mismatch-banner", shell)
        self.assertIn("review-session", shell)
        self.assertIn("trace-session", shell)
        self.assertIn("NOT the same session", shell)

    def test_only_one_side_configured_never_shows_mismatch(self):
        # Nothing to compare against -- must not fabricate a mismatch.
        out = _make_review_out(self.tmp_path, "solo-session")
        sides = combined_serve._Sides(out, None)
        shell = combined_serve._tab_shell_html(sides, "review")
        self.assertNotIn("tab-mismatch-banner", shell)

    def test_unavailable_tab_is_a_disabled_span_not_a_link(self):
        out = _make_review_out(self.tmp_path, "sess")
        sides = combined_serve._Sides(out, None)
        shell = combined_serve._tab_shell_html(sides, "review")
        self.assertIn("view-seg-unavailable", shell)
        self.assertIn('aria-disabled="true"', shell)
        # never a clickable link to a route that doesn't exist
        self.assertNotIn('href="/trace"', shell)

    def test_inject_tab_shell_inserts_right_after_body_tag(self):
        page = "<html><body><h1>hi</h1></body></html>"
        sides = combined_serve._Sides(None, None)
        out = combined_serve._inject_tab_shell(page, sides, "review")
        self.assertTrue(out.startswith("<html><body>\n<div class=\"tab-shell\">"))
        self.assertIn("<h1>hi</h1>", out)


def _shutdown(httpd, thread):
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()


class RouteTests(unittest.TestCase):
    """Real loopback server, real requests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def _serve(self, out_dir, trace_package_dir, argv_builder=None,
               first_run_trace_file=None, first_run_args=None):
        httpd = combined_serve.make_server(
            out_dir, trace_package_dir, host="127.0.0.1", port=0, argv_builder=argv_builder,
            first_run_trace_file=first_run_trace_file, first_run_args=first_run_args,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown, httpd, thread)
        return httpd

    def _conn(self, httpd):
        return http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)

    def _get(self, httpd, path):
        conn = self._conn(httpd)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        headers = dict(resp.getheaders())
        conn.close()
        return resp.status, body, headers

    # --- both sides available, matching session --------------------------

    def test_both_sides_full_route_table(self):
        out = _make_review_out(self.tmp_path, "sess-A")
        pkg = _make_trace_package(self.tmp_path, "sess-A")
        httpd = self._serve(out, pkg)

        status, _, headers = self._get(httpd, "/")
        self.assertEqual(status, 302)
        self.assertEqual(headers.get("Location"), "/trace")

        status, body, _ = self._get(httpd, "/trace")
        self.assertEqual(status, 200)
        self.assertIn(b"tab-shell", body)
        self.assertIn(b"sess-A", body)

        status, body, _ = self._get(httpd, "/review")
        self.assertEqual(status, 200)
        self.assertIn(b"tab-shell", body)
        self.assertIn(b"sess-A", body)
        self.assertNotIn(b"tab-mismatch-banner", body)

        status, body, headers = self._get(httpd, "/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers.get("Content-Type", ""))

        status, body, _ = self._get(httpd, "/data/session.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session_id"], "sess-A")

        status, body, _ = self._get(httpd, "/report.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session"]["session_id"], "sess-A")

        status, body, _ = self._get(httpd, "/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "idle")

    # --- degradation: one side missing ------------------------------------

    def test_missing_trace_package_degrades_cleanly(self):
        out = _make_review_out(self.tmp_path, "sess-B")
        httpd = self._serve(out, None)

        status, _, headers = self._get(httpd, "/")
        self.assertEqual(status, 302)
        self.assertEqual(headers.get("Location"), "/review")

        status, body, _ = self._get(httpd, "/trace")
        self.assertEqual(status, 200)  # rendered, not a 500 or bare 404
        self.assertIn(b"Trace unavailable", body)
        self.assertIn(b"no --trace-package", body)

        status, body, _ = self._get(httpd, "/review")
        self.assertEqual(status, 200)
        self.assertIn(b"sess-B", body)

        status, _, _ = self._get(httpd, "/data/session.json")
        self.assertEqual(status, 404)

    def test_missing_review_out_dir_degrades_cleanly(self):
        pkg = _make_trace_package(self.tmp_path, "sess-C")
        httpd = self._serve(None, pkg)

        status, _, headers = self._get(httpd, "/")
        self.assertEqual(status, 302)
        self.assertEqual(headers.get("Location"), "/trace")

        status, body, _ = self._get(httpd, "/review")
        self.assertEqual(status, 200)
        self.assertIn(b"Review unavailable", body)
        self.assertIn(b"no review report directory", body)

        status, _, _ = self._get(httpd, "/report.json")
        self.assertEqual(status, 404)

    def test_neither_side_configured_renders_placeholder_never_500(self):
        httpd = self._serve(None, None)
        status, body, _ = self._get(httpd, "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Nothing to show", body)

    # --- mismatch surfaced on a real page ---------------------------------

    def test_mismatched_sessions_surface_banner_on_both_tabs(self):
        out = _make_review_out(self.tmp_path, "review-sess")
        pkg = _make_trace_package(self.tmp_path, "trace-sess")
        httpd = self._serve(out, pkg)

        for path in ("/trace", "/review"):
            status, body, _ = self._get(httpd, path)
            self.assertEqual(status, 200)
            self.assertIn(b"tab-mismatch-banner", body, f"{path} should show the mismatch")
            self.assertIn(b"review-sess", body)
            self.assertIn(b"trace-sess", body)

    # --- path traversal guards (reused from sa.serve, not a looser copy) --

    def test_static_traversal_is_blocked(self):
        out = _make_review_out(self.tmp_path, "sess-D")
        pkg = _make_trace_package(self.tmp_path, "sess-D")
        httpd = self._serve(out, pkg)
        status, _, _ = self._get(httpd, "/static/../combined_serve.py")
        self.assertEqual(status, 404)
        status, _, _ = self._get(httpd, "/static/../../report.py")
        self.assertEqual(status, 404)

    def test_data_traversal_is_blocked(self):
        out = _make_review_out(self.tmp_path, "sess-E")
        pkg = _make_trace_package(self.tmp_path, "sess-E")
        httpd = self._serve(out, pkg)
        status, _, _ = self._get(httpd, "/data/../../../../etc/passwd")
        self.assertEqual(status, 404)

    def test_scratch_secrets_next_to_report_json_stay_unreachable(self):
        # Same C-5 guard as report.py's own server: out_dir may hold
        # scratch/<patch>/...*/auth.json alongside report.json.
        out = _make_review_out(self.tmp_path, "sess-F")
        (out / "scratch").mkdir()
        (out / "scratch" / "auth.json").write_text('{"token": "secret"}', encoding="utf-8")
        httpd = self._serve(out, None)
        status, body, _ = self._get(httpd, "/scratch/auth.json")
        self.assertEqual(status, 404)
        self.assertNotIn(b"secret", body)

    # --- /run + /status on the combined server ----------------------------

    def test_run_and_status_work_on_combined_server(self):
        out = _make_review_out(self.tmp_path, "sess-G")

        def fake_argv_builder(*, trace_file, out_dir, k, args):
            return [sys.executable, "-c", "import time; time.sleep(0.1)"]

        httpd = self._serve(out, None, argv_builder=fake_argv_builder)
        conn = self._conn(httpd)
        conn.request("POST", "/run", body=json.dumps({"k": 10}).encode("utf-8"),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 202)
        self.assertEqual(body["status"], "running")

        deadline = time.time() + 10
        status = self._get(httpd, "/status")[1]
        status = json.loads(status)
        while status["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
            status = json.loads(self._get(httpd, "/status")[1])
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["returncode"], 0)

    def test_run_rejects_k_outside_allow_list(self):
        out = _make_review_out(self.tmp_path, "sess-H")
        httpd = self._serve(out, None)
        conn = self._conn(httpd)
        conn.request("POST", "/run", body=json.dumps({"k": 7}).encode("utf-8"),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertIn("k must be one of", body["error"])

    def test_run_returns_503_when_no_review_out_dir(self):
        pkg = _make_trace_package(self.tmp_path, "sess-I")
        httpd = self._serve(None, pkg)
        conn = self._conn(httpd)
        conn.request("POST", "/run", body=json.dumps({"k": 3}).encode("utf-8"),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 503)

    # --- server binding ------------------------------------------------------

    def test_make_server_binds_loopback_only(self):
        out = _make_review_out(self.tmp_path, "sess-J")
        httpd = combined_serve.make_server(out, None, host="127.0.0.1", port=0)
        try:
            host, port = httpd.server_address
            self.assertEqual(host, "127.0.0.1")
            self.assertNotEqual(port, 0)
        finally:
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
