"""Tests for interstellar/report.py.

Builds a realistic Report fixture directly from the types.py constructors
(mirroring what stats.py + the cli.py orchestrator will hand report.build())
and checks: build() produces every documented Report key; write() emits an
index.html containing the diff text and the caveats verbatim; the HTML has
no external http(s) resource references; serve() binds loopback only; and
an insufficient-samples statistic never renders as a fabricated number.
"""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path

from interstellar import report
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFECT_SKILL_TOKENS,
    PATCH_SKILL_TRUNCATE,
    make_grade,
    make_harness_version,
    make_judge_verdict,
    make_patch,
    make_patch_application,
    make_patch_result,
    make_report,
    make_run_result,
)

SAMPLE_DIFF = """--- a/skills/verbose-skill/SKILL.md
+++ b/skills/verbose-skill/SKILL.md
@@ -10,3 +10,1 @@
-This is a wordy paragraph that never gets used by the agent in practice.
-Another wasted line kept here for historical reasons only.
+Trimmed.
"""


def _run(arm, repeat, ok=True):
    return make_run_result(
        arm=arm, repeat=repeat, ok=ok, prompt="do the thing", home="/tmp/home",
        workdir="/tmp/wd", cost_usd=0.01, wall_s=1.0, turns=2,
    )


def _grade(repeat, verdict="treatment", consistent=True):
    return make_grade(
        repeat=repeat,
        control_efficiency={
            "total_tokens": 10000, "cost_usd": 0.05, "wall_ms": 4000,
            "tool_calls": 12, "skill_tokens_est": 2000,
            "skill_wasted_tokens_est": 900, "tool_result_tokens_est": 1200,
            "mcp_startup_ms": 300, "duplicate_calls": 1, "turns": 4,
        },
        treatment_efficiency={
            "total_tokens": 7000, "cost_usd": 0.035, "wall_ms": 3200,
            "tool_calls": 10, "skill_tokens_est": 900,
            "skill_wasted_tokens_est": 100, "tool_result_tokens_est": 1100,
            "mcp_startup_ms": 300, "duplicate_calls": 1, "turns": 4,
        },
        judge=make_judge_verdict(verdict=verdict, consistent=consistent, reason="treatment is more concise"),
    )


def _well_measured_patch_result():
    patch = make_patch(
        patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="verbose-skill",
        rationale="This skill's SKILL.md repeats itself; cited lines are never read by the agent.",
        expected_effect=EFFECT_SKILL_TOKENS, source_recommendation="rec-3",
        line_ranges=[[10, 11]], diff=SAMPLE_DIFF,
    )
    application = make_patch_application(patch_id="p1", applied=True, diff=SAMPLE_DIFF, lines_changed=3)
    matrix_summary = {
        "prompt": "do the thing", "k": 3, "patch_id": "p1",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p1",
        "arms": {
            ARM_CONTROL: [_run(ARM_CONTROL, i) for i in range(3)],
            ARM_TREATMENT: [_run(ARM_TREATMENT, i) for i in range(3)],
        },
    }
    grades = [_grade(i) for i in range(3)]
    statistics = {
        "efficiency": {
            "skill_tokens_est": {
                "control_median": 2000, "treatment_median": 900,
                "delta_abs": -1100, "delta_pct": -0.55,
                "ci": {"mean": -1080.0, "lo": -1400.0, "hi": -800.0, "level": 0.95},
            },
            "total_tokens": {
                "control_median": 10000, "treatment_median": 7000,
                "delta_abs": -3000, "delta_pct": -0.30,
                "ci": {"mean": -2950.0, "lo": -3600.0, "hi": -2200.0, "level": 0.95},
            },
        },
        "win_rate": {"wins": 3, "ties": 0, "losses": 0, "n": 3, "rate": 1.0, "lo": 0.4, "hi": 1.0},
        "sign_test": {"p": 0.25, "n_nonzero": 3, "direction": "treatment"},
        "pass_k": {"control": 1.0, "treatment": 1.0},
    }
    gate = {"accepted": True, "reasons": ["win-rate lower CI >= 0.5", "no high-severity regressions", "skill_tokens_est improved 55%"]}
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="Trims the verbose-skill SKILL.md and cuts skill tokens ~55% with no regressions across 3/3 runs.",
    )


def _insufficient_patch_result():
    patch = make_patch(
        patch_id="p2", kind=PATCH_SKILL_TRUNCATE, target="another-skill",
        rationale="Trims a redundant example block.",
        expected_effect=EFFECT_SKILL_TOKENS, diff=SAMPLE_DIFF,
    )
    application = make_patch_application(patch_id="p2", applied=True, diff=SAMPLE_DIFF, lines_changed=2)
    matrix_summary = {
        "prompt": "do the thing", "k": 1, "patch_id": "p2",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p2",
        "arms": {ARM_CONTROL: [_run(ARM_CONTROL, 0)], ARM_TREATMENT: [_run(ARM_TREATMENT, 0)]},
    }
    grades = [_grade(0, verdict="tie", consistent=False)]
    statistics = {
        "efficiency": {
            "skill_tokens_est": {
                "control_median": 2000, "treatment_median": 1500,
                "delta_abs": -500, "delta_pct": -0.25,
                "ci": {"mean": -500.0, "lo": None, "hi": None, "level": 0.95, "note": "insufficient_samples"},
            },
        },
        "win_rate": {"wins": 0, "ties": 1, "losses": 0, "n": 1, "rate": None, "lo": None, "hi": None, "note": "insufficient_samples"},
        "sign_test": {"p": None, "n_nonzero": 0, "direction": None},
        "pass_k": {"control": 1.0, "treatment": 1.0},
    }
    gate = {"accepted": False, "reasons": ["only 1 repeat: insufficient samples to clear the win-rate gate"]}
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="Too few repeats (n=1) to accept or reject this patch with confidence.",
    )


def _session():
    return {
        "session_id": "019fe438-a9e4-7f42-ab5a-3c715f066f87",
        "title": "Fix the pager welcome screen",
        "prompt": "do the thing",
        "trace_file": "traces/corpus/skill_bloat_001.json",
    }


def _baseline_version():
    hv = make_harness_version(
        version_id="base-abc", skills=[], mcp_servers=[], config_text="", source="live",
    )
    return {"version_id": hv["version_id"], "skills": 4, "mcp": 2}


class BuildTests(unittest.TestCase):
    def test_build_produces_every_documented_key(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_well_measured_patch_result()], k=3, cost_usd=1.2345,
        )
        expected_keys = set(make_report(
            session={}, baseline_version={}, patch_results=[], caveats=[],
        ).keys())
        self.assertEqual(set(rpt.keys()), expected_keys)
        self.assertEqual(rpt["k"], 3)
        self.assertAlmostEqual(rpt["cost_usd"], 1.2345)
        self.assertEqual(len(rpt["patch_results"]), 1)
        self.assertTrue(rpt["generated_at"])

    def test_caveats_always_include_k_and_isolation_note(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[], k=5,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("k=5", joined)
        self.assertIn("isolated sandbox", joined)

    def test_caveats_flag_insufficient_samples(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_insufficient_patch_result()], k=1,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("Insufficient samples", joined)
        self.assertIn("p2", joined)

    def test_caveats_report_judge_consistency_rate(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_well_measured_patch_result(), _insufficient_patch_result()], k=3,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("Judge consistency", joined)
        self.assertIn("3/4", joined)  # 3 consistent judged pairs out of 4 total

    def test_missing_verdict_line_is_rejected(self):
        pr = _well_measured_patch_result()
        pr["verdict_line"] = ""
        with self.assertRaises(ValueError):
            report.build(
                session=_session(), baseline_version=_baseline_version(),
                patch_results=[pr], k=3,
            )


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "out"

    def _build_and_write(self, patch_results):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=patch_results, k=3, cost_usd=2.5,
        )
        report.write(rpt, self.out_dir)
        return rpt

    def test_write_emits_report_json_matching_build_output(self):
        rpt = self._build_and_write([_well_measured_patch_result()])
        on_disk = json.loads((self.out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, rpt)

    def test_html_contains_diff_text(self):
        self._build_and_write([_well_measured_patch_result()])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn("Trimmed.", html_text)
        self.assertIn("verbose-skill/SKILL.md", html_text)

    def test_html_contains_caveats_verbatim(self):
        rpt = self._build_and_write([_well_measured_patch_result()])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        for caveat in rpt["caveats"]:
            self.assertIn(report._esc(caveat), html_text)

    def test_html_contains_gate_reasons_for_accept_and_reject(self):
        self._build_and_write([_well_measured_patch_result(), _insufficient_patch_result()])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn("win-rate lower CI", html_text)
        self.assertIn("insufficient samples to clear the win-rate gate", html_text)
        self.assertIn("ACCEPT", html_text)
        self.assertIn("REJECT", html_text)

    def test_html_never_fabricates_a_number_for_insufficient_samples(self):
        self._build_and_write([_insufficient_patch_result()])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        # The insufficient-sample metric's CI must render as the caveat text,
        # not as a numeric interval.
        self.assertIn("insufficient samples", html_text)
        self.assertNotIn("[None, None]", html_text)

    def test_html_has_no_external_http_resource_references(self):
        self._build_and_write([_well_measured_patch_result(), _insufficient_patch_result()])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("http://", html_text)
        self.assertNotIn("https://", html_text)

    def test_html_renders_with_zero_patches(self):
        self._build_and_write([])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn("No candidate patches", html_text)

    def test_html_escapes_untrusted_text(self):
        pr = _well_measured_patch_result()
        pr["patch"]["rationale"] = "<script>alert(1)</script> & more"
        self._build_and_write([pr])
        html_text = (self.out_dir / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", html_text)
        self.assertIn("&lt;script&gt;", html_text)


class ServeTests(unittest.TestCase):
    def test_make_server_binds_loopback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            rpt = report.build(
                session=_session(), baseline_version=_baseline_version(),
                patch_results=[_well_measured_patch_result()], k=3,
            )
            report.write(rpt, out_dir)
            httpd = report.make_server(out_dir, host="127.0.0.1", port=0)
            try:
                host, port = httpd.server_address
                self.assertEqual(host, "127.0.0.1")
                self.assertNotEqual(port, 0)
                # confirm it really is loopback-reachable and refuses to be
                # anything but 127.0.0.1 (no accidental 0.0.0.0 bind)
                self.assertEqual(httpd.socket.getsockname()[0], "127.0.0.1")
            finally:
                httpd.server_close()

    def test_make_server_requires_written_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                report.make_server(Path(tmp) / "nope", port=0)


if __name__ == "__main__":
    unittest.main()
