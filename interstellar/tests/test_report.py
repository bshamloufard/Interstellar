"""Tests for interstellar/report.py.

Builds realistic Report fixtures directly from the types.py constructors
(mirroring what stats.py + the cli.py orchestrator hand report.build()) and
checks: build() produces every documented Report key; the caveats block is
first-class (small-n floor, partial isolation, control-is-a-rerun, judge
consistency, insufficient-sample flags) and rendered prominently, never as
a number-free dash; the three-state gate (ACCEPTED / PROVISIONAL /
DIRECTIONAL) renders honestly instead of collapsing to accept/reject;
sign-flip permutation p-values at the sample-size floor get the "as extreme
as this permits" phrasing instead of a bare p; degenerate pass^k renders as
a flag, not a percentage; low judge position-consistency is visibly
flagged next to the win rate; the HTML has no external http(s) references;
and serve() binds loopback only.
"""

from __future__ import annotations

import json
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


def _efficiency_entry(control_median, treatment_median, abs_delta, pct_delta, *, lo=None, hi=None, note=None, distinct_resamples=None):
    """Matches the pinned statistics["efficiency"][metric] shape exactly:
    control_median/treatment_median/abs_delta/pct_delta/lo/hi/note, flat —
    no nested "bootstrap" sub-dict, no separate "mean" (abs_delta already
    is the point estimate)."""
    entry = {
        "lower_is_better": True,
        "n": 3,
        "control_median": control_median,
        "treatment_median": treatment_median,
        "abs_delta": abs_delta,
        "pct_delta": pct_delta,
        "lo": lo,
        "hi": hi,
        "note": note,
    }
    if distinct_resamples is not None:
        entry["distinct_resamples"] = distinct_resamples
    return entry


def _accepted_patch_result():
    """k=5, clean win: gate reaches full ACCEPTED (research-methods.md: n<5
    can never accept, so this fixture uses k=5 specifically to exercise the
    ACCEPTED path honestly)."""
    patch = make_patch(
        patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="verbose-skill",
        rationale="This skill's SKILL.md repeats itself; cited lines are never read by the agent.",
        expected_effect=EFFECT_SKILL_TOKENS, source_recommendation="rec-3",
        line_ranges=[[10, 11]], diff=SAMPLE_DIFF,
    )
    application = make_patch_application(patch_id="p1", applied=True, diff=SAMPLE_DIFF, lines_changed=3)
    matrix_summary = {
        "prompt": "do the thing", "k": 5, "patch_id": "p1",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p1",
        "arms": {
            ARM_CONTROL: [_run(ARM_CONTROL, i) for i in range(5)],
            ARM_TREATMENT: [_run(ARM_TREATMENT, i) for i in range(5)],
        },
    }
    grades = [_grade(i) for i in range(5)]
    statistics = {
        "n": 5,
        "primary_effect": "skill_tokens_est",
        "efficiency": {
            "skill_tokens_est": _efficiency_entry(2000, 900, -1100, -0.55, lo=-1400.0, hi=-800.0),
            "total_tokens": _efficiency_entry(10000, 7000, -3000, -0.30, lo=-3600.0, hi=-2200.0),
        },
        "win_rate": {"wins": 5, "ties": 0, "losses": 0, "n": 5, "rate": 1.0, "lo": 0.65, "hi": 1.0, "note": None},
        "judge_consistency": {"pairs": 10, "agreed": 9, "rate": 0.9, "note": None},
        "permutation": {"p": 0.0625, "n": 5, "direction": "positive", "min_achievable_p": 0.0625, "note": None},
        "pass_k": {"value": 1.0, "n": 5, "c": 5, "k": 5, "degenerate": True, "note": None},
        "regressions": [],
    }
    gate = {
        "accepted": True, "provisional": False, "directional": "favorable",
        "reasons": ["win-rate lower CI 0.65 >= 0.5", "no high-severity regressions", "skill_tokens_est improved: paired-diff CI [-1400, -800] entirely > 0"],
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="Trims the verbose-skill SKILL.md and cuts skill tokens ~55% with no regressions across 5/5 runs.",
    )


def _provisional_patch_result():
    """k=3: honest small-n case. Evidence leans favorable but the gate
    cannot ACCEPT at this k — PROVISIONAL, not a red reject. Also carries
    one medium-severity regression, to exercise that rendering path."""
    patch = make_patch(
        patch_id="p2", kind=PATCH_SKILL_TRUNCATE, target="another-skill",
        rationale="Trims a redundant example block.",
        expected_effect=EFFECT_SKILL_TOKENS, diff=SAMPLE_DIFF,
    )
    application = make_patch_application(patch_id="p2", applied=True, diff=SAMPLE_DIFF, lines_changed=2)
    matrix_summary = {
        "prompt": "do the thing", "k": 3, "patch_id": "p2",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p2",
        "arms": {
            ARM_CONTROL: [_run(ARM_CONTROL, i) for i in range(3)],
            ARM_TREATMENT: [_run(ARM_TREATMENT, i) for i in range(3)],
        },
    }
    grades = [_grade(i, consistent=(i != 2)) for i in range(3)]
    statistics = {
        "n": 3,
        "primary_effect": "skill_tokens_est",
        "efficiency": {
            "skill_tokens_est": _efficiency_entry(2000, 1500, -500, -0.25, lo=None, hi=None, note="insufficient_samples", distinct_resamples=10),
        },
        "win_rate": {
            "wins": 3, "ties": 0, "losses": 0, "n": 3, "rate": 1.0, "lo": None, "hi": None,
            "note": "insufficient_samples", "distinct_resamples": 10,
        },
        "judge_consistency": {"pairs": 3, "agreed": 2, "rate": 2 / 3, "note": None},
        "permutation": {"p": 0.25, "n": 3, "direction": "positive", "min_achievable_p": 0.25, "note": None},
        "pass_k": {"value": 1.0, "n": 3, "c": 3, "k": 3, "degenerate": True, "note": None},
        "regressions": [
            {"kind": "duplicate_calls", "severity": "medium", "detail": "1 new duplicate tool-call pair", "evidence": "repeat 1: read_file(x2)"},
        ],
    }
    gate = {
        "accepted": False, "provisional": True, "directional": "favorable",
        "reasons": ["0 losses out of 3 judged pairs", "insufficient samples for a credible CI on skill_tokens_est: not accepted on a point estimate alone"],
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="Too few repeats (k=3) to accept outright; the signal so far favors the patch.",
    )


def _directional_patch_result():
    """k=1: no evidence either way; DIRECTIONAL: NEUTRAL, and a per-patch
    caveat naming this exact patch's target (the skill.remove-is-a-no-op
    style warning)."""
    patch = make_patch(
        patch_id="p3", kind=PATCH_SKILL_TRUNCATE, target="third-skill",
        rationale="Removes an unused example section.",
        expected_effect=EFFECT_SKILL_TOKENS, diff="",
    )
    application = make_patch_application(patch_id="p3", applied=True, diff="", lines_changed=0)
    matrix_summary = {
        "prompt": "do the thing", "k": 1, "patch_id": "p3",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p3",
        "arms": {ARM_CONTROL: [_run(ARM_CONTROL, 0)], ARM_TREATMENT: [_run(ARM_TREATMENT, 0)]},
    }
    grades = [_grade(0, verdict="tie", consistent=False)]
    statistics = {
        "n": 1,
        "primary_effect": "skill_tokens_est",
        "efficiency": {
            "skill_tokens_est": _efficiency_entry(2000, 2000, 0, 0.0, lo=None, hi=None, note="insufficient_samples"),
        },
        "win_rate": {"wins": 0, "ties": 1, "losses": 0, "n": 1, "rate": None, "lo": None, "hi": None, "note": "insufficient_samples"},
        "judge_consistency": {"pairs": 1, "agreed": 0, "rate": 0.0, "note": None},
        "permutation": {"p": None, "n": 0, "direction": "none", "min_achievable_p": None, "note": None},
        "pass_k": {"value": 1.0, "n": 1, "c": 1, "k": 1, "degenerate": True, "note": None},
        "regressions": [],
    }
    gate = {
        "accepted": False, "provisional": False, "directional": "neutral",
        "reasons": ["only 1 repeat: no decided pairs, no reading possible"],
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="One repeat, a tie: no evidence either way yet.",
    )


def _all_ties_patch_result():
    """The exact integration smoke-test win_rate reported by team-lead:
    all judged pairs tie (rate=None, note="no_decided_pairs"). A common,
    deliberate result — the judge found the two harnesses equivalent —
    and must render as such, not as a broken/empty widget."""
    patch = make_patch(
        patch_id="p4", kind=PATCH_SKILL_TRUNCATE, target="tied-skill",
        rationale="Removes a comment block the agent never reads.",
        expected_effect=EFFECT_SKILL_TOKENS, diff=SAMPLE_DIFF,
    )
    application = make_patch_application(patch_id="p4", applied=True, diff=SAMPLE_DIFF, lines_changed=1)
    matrix_summary = {
        "prompt": "do the thing", "k": 3, "patch_id": "p4",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p4",
        "arms": {
            ARM_CONTROL: [_run(ARM_CONTROL, i) for i in range(3)],
            ARM_TREATMENT: [_run(ARM_TREATMENT, i) for i in range(3)],
        },
    }
    grades = [_grade(i, verdict="tie", consistent=True) for i in range(3)]
    statistics = {
        "n": 3,
        "primary_effect": "skill_tokens_est",
        "efficiency": {
            "skill_tokens_est": _efficiency_entry(2000, 1900, -100, -0.05, lo=None, hi=None, note="insufficient_samples"),
        },
        "win_rate": {"wins": 0, "ties": 3, "losses": 0, "n": 3, "rate": None, "lo": None, "hi": None, "note": "no_decided_pairs"},
        "judge_consistency": {"pairs": 3, "agreed": 3, "rate": 1.0, "note": None},
        "permutation": {"p": None, "n": 0, "direction": "none", "min_achievable_p": None, "note": None},
        "pass_k": {"value": 1.0, "n": 3, "c": 3, "k": 3, "degenerate": True, "note": None},
        "regressions": [],
    }
    gate = {
        "accepted": False, "provisional": False, "directional": "neutral",
        "reasons": ["quality: 0 losses but 0 decided pairs (all ties) — no win-rate evidence either way"],
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=statistics, gate=gate,
        verdict_line="All three judged pairs came back tied: the two harnesses look equivalent so far.",
    )


def _session():
    return {
        "session_id": "019fe438-a9e4-7f42-ab5a-3c715f066f87",
        "title": "Fix the pager welcome screen",
        "prompt": "do the thing",
        "trace_file": "traces/corpus/skill_bloat_001.json",
    }


def _baseline_version():
    return {"version_id": "base-abc", "skills": 4, "mcp": 2}


class BuildTests(unittest.TestCase):
    def test_build_produces_every_documented_key(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_accepted_patch_result()], k=5, cost_usd=1.2345,
        )
        expected_keys = set(make_report(
            session={}, baseline_version={}, patch_results=[], caveats=[],
        ).keys())
        self.assertEqual(set(rpt.keys()), expected_keys)
        self.assertEqual(rpt["k"], 5)
        self.assertAlmostEqual(rpt["cost_usd"], 1.2345)
        self.assertEqual(len(rpt["patch_results"]), 1)
        self.assertTrue(rpt["generated_at"])

    def test_caveats_state_the_exact_small_n_floor(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[], k=3,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("k=3", joined)
        self.assertIn("0.25", joined)  # 2*(0.5**3), the exact permutation floor at n=3

    def test_caveats_include_partial_isolation_and_control_is_rerun(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[], k=5,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("GROK_HOME", joined)
        self.assertIn("~/.agents/skills/", joined)
        self.assertIn("not the user's original recorded session", joined)

    def test_caveats_flag_insufficient_samples_by_patch_id(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_provisional_patch_result()], k=3,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("p2", joined)
        self.assertIn("Point estimates only", joined)

    def test_caveats_report_judge_consistency_rate(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_accepted_patch_result(), _provisional_patch_result()], k=5,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("Judge consistency", joined)
        # summed from each patch's own statistics["judge_consistency"]:
        # 9/10 (accepted fixture) + 2/3 (provisional fixture) = 11/13
        self.assertIn("11/13", joined)

    def test_extra_caveats_are_appended_not_dropped(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[], k=3, extra_caveats=["skill.remove for another-skill is a no-op: also present under ~/.agents/skills/"],
        )
        self.assertIn("skill.remove for another-skill is a no-op: also present under ~/.agents/skills/", rpt["caveats"])

    def test_missing_verdict_line_is_rejected(self):
        pr = _accepted_patch_result()
        pr["verdict_line"] = ""
        with self.assertRaises(ValueError):
            report.build(
                session=_session(), baseline_version=_baseline_version(),
                patch_results=[pr], k=5,
            )


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "out"

    def _build_and_write(self, patch_results, k=3, extra_caveats=None):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=patch_results, k=k, cost_usd=2.5, extra_caveats=extra_caveats,
        )
        report.write(rpt, self.out_dir)
        return rpt

    def _html(self):
        return (self.out_dir / "index.html").read_text(encoding="utf-8")

    def test_write_emits_report_json_matching_build_output(self):
        rpt = self._build_and_write([_accepted_patch_result()], k=5)
        on_disk = json.loads((self.out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, rpt)

    def test_html_contains_diff_text(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("Trimmed.", html_text)
        self.assertIn("verbose-skill/SKILL.md", html_text)

    def test_html_contains_caveats_verbatim_and_before_the_patches(self):
        rpt = self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        for caveat in rpt["caveats"]:
            self.assertIn(report._esc(caveat), html_text)
        # the caveats panel must come before the first patch card in the
        # document, i.e. prominent / not below the fold
        self.assertLess(html_text.index('class="caveats"'), html_text.index('class="patch-card"'))

    def test_gate_badge_accepted(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn('class="badge accept"', html_text)
        self.assertIn("ACCEPTED", html_text)

    def test_gate_badge_provisional_is_not_a_reject(self):
        self._build_and_write([_provisional_patch_result()], k=3)
        html_text = self._html()
        self.assertIn('class="badge provisional"', html_text)
        self.assertIn("PROVISIONAL", html_text)
        self.assertNotIn("REJECT<", html_text)
        self.assertNotIn('class="badge reject"', html_text)

    def test_gate_badge_directional_neutral_is_not_red(self):
        self._build_and_write([_directional_patch_result()], k=1)
        html_text = self._html()
        self.assertIn('class="badge directional-neutral"', html_text)
        self.assertIn("DIRECTIONAL", html_text)
        self.assertNotIn('class="badge reject"', html_text)

    def test_gate_reasons_render_for_every_state(self):
        self._build_and_write(
            [_accepted_patch_result(), _provisional_patch_result(), _directional_patch_result()], k=5,
        )
        html_text = self._html()
        self.assertIn("no high-severity regressions", html_text)
        self.assertIn("insufficient samples for a credible CI on skill_tokens_est", html_text)
        self.assertIn("no decided pairs, no reading possible", html_text)

    def test_legacy_two_state_gate_falls_back_to_reject_without_crashing(self):
        pr = _accepted_patch_result()
        pr["gate"] = {"accepted": False, "reasons": ["legacy shape, no provisional/directional keys"]}
        self._build_and_write([pr], k=5)
        html_text = self._html()
        self.assertIn('class="badge reject"', html_text)

    def test_permutation_test_floor_phrasing(self):
        self._build_and_write([_provisional_patch_result()], k=3)
        html_text = self._html()
        self.assertIn("as extreme as n=3 permits", html_text)
        self.assertIn("not evidence of no effect", html_text)

    def test_pass_k_degenerate_renders_as_flag_not_percentage(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("flag, n=k", html_text)
        self.assertIn("all k succeeded", html_text)
        # must not be misrepresented as a 100% rate
        self.assertNotIn("100% &rarr;", html_text)

    def test_low_position_consistency_is_flagged_next_to_win_rate(self):
        self._build_and_write([_provisional_patch_result()], k=3)
        html_text = self._html()
        self.assertIn("consistency-low", html_text)
        self.assertIn("position consistency: 67%", html_text)
        self.assertIn("below the ~70-77% published baseline", html_text)

    def test_high_position_consistency_is_not_flagged(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("position consistency: 90%", html_text)

    def test_insufficient_ci_shows_point_estimate_never_a_dash(self):
        self._build_and_write([_provisional_patch_result()], k=3)
        html_text = self._html()
        # abs_delta is the point estimate (its own table column) and must
        # render regardless of the CI being unavailable
        self.assertIn("-500.00", html_text)
        self.assertIn("insufficient samples for a CI", html_text)
        self.assertIn("10 distinct resample", html_text)
        self.assertNotIn("<td>—</td><td>—</td></tr>", html_text)

    def test_all_ties_renders_as_deliberate_result_not_broken(self):
        self._build_and_write([_all_ties_patch_result()], k=3)
        html_text = self._html()
        self.assertIn("3 ties, no decided pairs", html_text)
        self.assertIn("the judge found the two harnesses equivalent", html_text)
        self.assertNotIn("win rate: n/a", html_text)
        self.assertNotIn('<p class="muted">No judged pairs.</p>', html_text)

    def test_regressions_render_with_severity(self):
        self._build_and_write([_provisional_patch_result()], k=3)
        html_text = self._html()
        self.assertIn('class="regressions"', html_text)
        self.assertIn("regression-medium", html_text)
        self.assertIn("1 new duplicate tool-call pair", html_text)

    def test_no_regressions_renders_no_empty_section(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertNotIn('class="regressions"', html_text)

    def test_primary_effect_highlights_matching_metric_row(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn('<span class="tag">target</span>', html_text)
        self.assertIn("Skill tokens (est.)", html_text)

    def test_pass_k_shows_c_over_n_detail(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("c=5/n=5", html_text)

    def test_permutation_n_field_used_when_n_nonzero_absent(self):
        self._build_and_write([_accepted_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("as extreme as n=5 permits", html_text)

    def test_per_patch_caveat_surfaces_inline_on_matching_card(self):
        caveat = "skill.remove for another-skill is a no-op: it also exists under an uncontrolled skill root."
        self._build_and_write([_provisional_patch_result()], k=3, extra_caveats=[caveat])
        html_text = self._html()
        # appears once in the persistent panel...
        self.assertIn(report._esc(caveat), html_text)
        # ...and again inline on the matching patch card
        self.assertIn('class="patch-caveats"', html_text)
        self.assertEqual(html_text.count(report._esc(caveat)), 2)

    def test_html_has_no_external_http_resource_references(self):
        self._build_and_write([_accepted_patch_result(), _provisional_patch_result(), _directional_patch_result()], k=5)
        html_text = self._html()
        self.assertNotIn("http://", html_text)
        self.assertNotIn("https://", html_text)

    def test_html_renders_with_zero_patches(self):
        self._build_and_write([], k=3)
        html_text = self._html()
        self.assertIn("No candidate patches", html_text)
        # caveats still render even with no patches
        self.assertIn('class="caveats"', html_text)

    def test_html_escapes_untrusted_text(self):
        pr = _accepted_patch_result()
        pr["patch"]["rationale"] = "<script>alert(1)</script> & more"
        self._build_and_write([pr], k=5)
        html_text = self._html()
        self.assertNotIn("<script>alert(1)</script>", html_text)
        self.assertIn("&lt;script&gt;", html_text)


class ServeTests(unittest.TestCase):
    def test_make_server_binds_loopback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            rpt = report.build(
                session=_session(), baseline_version=_baseline_version(),
                patch_results=[_accepted_patch_result()], k=5,
            )
            report.write(rpt, out_dir)
            httpd = report.make_server(out_dir, host="127.0.0.1", port=0)
            try:
                host, port = httpd.server_address
                self.assertEqual(host, "127.0.0.1")
                self.assertNotEqual(port, 0)
                self.assertEqual(httpd.socket.getsockname()[0], "127.0.0.1")
            finally:
                httpd.server_close()

    def test_make_server_requires_written_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                report.make_server(Path(tmp) / "nope", port=0)


if __name__ == "__main__":
    unittest.main()
