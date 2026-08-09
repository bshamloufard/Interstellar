"""Tests for interstellar/report.py.

Fixtures build a `grades` list from the types.py constructors and feed it
through the REAL `interstellar.stats.summarize()` / `stats.gate()` (pure,
no grok-dev, no I/O) rather than hand-assembling a `statistics`/`gate`
dict — the same call cli.py will make. This is deliberate: an earlier
round of this module hand-typed a `statistics` shape that looked plausible
but didn't match stats.py's actual output (nested "bootstrap", primary_effect
as an EFFECT_* name requiring a mapping, not an EFFICIENCY_METRICS name
directly), and that kind of mismatch is exactly what calling the real
function catches that a hand-typed fixture cannot.

Checks: build() produces every documented Report key; the caveats block is
first-class (small-n floor, partial isolation, control-is-a-rerun, judge
consistency, insufficient-sample flags) and rendered prominently, never a
number-free dash; the gate's five real states (ACCEPTED / ACCEPTED ·
PROVISIONAL / PROVISIONAL · reading / DIRECTIONAL · reading / REJECTED)
render distinctly instead of collapsing to accept/reject; sign-flip
permutation p-values at the sample-size floor get the "as extreme as this
permits" phrasing instead of a bare p; degenerate pass^k renders as a flag,
not a percentage; low judge position-consistency is visibly flagged next
to the win rate; all-ties win_rate renders as a deliberate result, not a
broken widget; regressions render with severity; the primary_effect ->
EFFICIENCY_METRICS mapping correctly highlights the target row; the HTML
has no external http(s) references; and serve() binds loopback only.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interstellar import report, stats
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
    make_regression,
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

BASE_EFFICIENCY = {
    "total_tokens": 10000, "cost_usd": 0.05, "wall_ms": 4000,
    "tool_calls": 12, "skill_tokens_est": 2000,
    "skill_wasted_tokens_est": 900, "tool_result_tokens_est": 1200,
    "mcp_startup_ms": 300, "duplicate_calls": 1, "turns": 4,
}


def _run(arm, repeat, ok=True):
    return make_run_result(
        arm=arm, repeat=repeat, ok=ok, prompt="do the thing", home="/tmp/home",
        workdir="/tmp/wd", cost_usd=0.01, wall_s=1.0, turns=2,
    )


def _synthetic_grades(n, *, skill_tokens_improvement=0, verdict="treatment",
                       consistent=True, high_regression_on_first=False):
    """n paired grades. `skill_tokens_improvement` > 0 means the treatment
    used that many fewer skill_tokens_est than control (an improvement,
    since lower_is_better); < 0 means the treatment used more (a
    regression). `verdict` is the judge's call for every grade (mix wins
    and losses by concatenating two calls with different verdicts)."""
    grades = []
    for i in range(n):
        control_eff = dict(BASE_EFFICIENCY)
        treatment_eff = dict(BASE_EFFICIENCY)
        treatment_eff["skill_tokens_est"] = control_eff["skill_tokens_est"] - skill_tokens_improvement
        regs = []
        if high_regression_on_first and i == 0:
            regs = [make_regression(
                kind="mcp_failure", severity="high",
                detail="mcp server flaky-mcp failed in treatment but not control",
                evidence="exit code 1",
            )]
        grades.append(make_grade(
            repeat=i, control_efficiency=control_eff, treatment_efficiency=treatment_eff,
            judge=make_judge_verdict(verdict=verdict, consistent=consistent, reason="test verdict"),
            regressions=regs,
        ))
    return grades


def _patch_result(patch_id, target, grades, *, rationale, verdict_line, diff=SAMPLE_DIFF, applied=True):
    """Builds a real make_patch_result() by calling stats.summarize() and
    stats.gate() on `grades` — the same call cli.py makes — instead of
    hand-assembling `statistics`/`gate`. `applied` and `diff` are
    independent: a patch that ran and was measured but happens to have no
    diff text to show (diff="") is still `applied=True` -- conflating
    "empty diff" with "never applied" is exactly the C-1 bug this module
    now guards against, so the fixture must not make that mistake either."""
    summary = stats.summarize(grades, primary_effect=EFFECT_SKILL_TOKENS)
    gate_result = stats.gate(summary)
    patch = make_patch(
        patch_id=patch_id, kind=PATCH_SKILL_TRUNCATE, target=target,
        rationale=rationale, expected_effect=EFFECT_SKILL_TOKENS,
        source_recommendation=f"rec-{patch_id}", diff=diff,
    )
    application = make_patch_application(patch_id=patch_id, applied=applied, diff=diff, lines_changed=2 if diff else 0)
    n = len(grades)
    matrix_summary = {
        "prompt": "do the thing", "k": n, "patch_id": patch_id,
        "control_version_id": "base-abc", "treatment_version_id": f"treat-{patch_id}",
        "arms": {
            ARM_CONTROL: [_run(ARM_CONTROL, i) for i in range(n)],
            ARM_TREATMENT: [_run(ARM_TREATMENT, i) for i in range(n)],
        },
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=grades, statistics=summary, gate=gate_result, verdict_line=verdict_line,
    )


def _accepted_patch_result():
    """n=12 (>= stats.MIN_SAMPLES_FOR_CI=10): clean, unanimous win with a
    real primary-metric improvement -- the full CI-backed ACCEPTED path."""
    grades = _synthetic_grades(12, skill_tokens_improvement=1100, verdict="treatment", consistent=True)
    return _patch_result(
        "p1", "verbose-skill", grades,
        rationale="This skill's SKILL.md repeats itself; cited lines are never read by the agent.",
        verdict_line="Trims the verbose-skill SKILL.md and cuts skill tokens with no regressions across 12/12 runs.",
    )


def _accepted_provisional_patch_result():
    """n=7 (in stats.MIN_SAMPLES_FOR_ACCEPT..MIN_SAMPLES_FOR_CI, i.e. 5-9):
    zero losses and a positive point estimate pass the only check this
    zone has -- accepted=True, but provisional=True, since there is no CI
    to back it. Must render as a qualified accept, not a clean one."""
    grades = _synthetic_grades(7, skill_tokens_improvement=1100, verdict="treatment", consistent=True)
    return _patch_result(
        "p2", "another-skill", grades,
        rationale="Trims a redundant example block.",
        verdict_line="Passed the only check available at this sample size; not enough repeats yet for a full accept.",
    )


def _provisional_reject_patch_result():
    """n=5 (the small-n zone's floor), one loss among the five: the zone's
    only pass criterion (zero losses) fails, but there still isn't a CI to
    back a confident reject either -- provisional=True, accepted=False."""
    grades = (
        _synthetic_grades(4, skill_tokens_improvement=1100, verdict="treatment", consistent=True)
        + _synthetic_grades(1, skill_tokens_improvement=1100, verdict="control", consistent=True)
    )
    return _patch_result(
        "p5", "loss-skill", grades,
        rationale="Trims an example block that one run's judge preferred kept.",
        verdict_line="One judged loss in the no-CI zone: can't accept, but not enough data for a confident reject either.",
    )


def _directional_patch_result():
    """n=1 (< stats.MIN_SAMPLES_FOR_ACCEPT=5): acceptance is impossible by
    construction -- DIRECTIONAL, not a red REJECT."""
    grades = _synthetic_grades(1, skill_tokens_improvement=0, verdict="tie", consistent=False)
    return _patch_result(
        "p3", "third-skill", grades, diff="",
        rationale="Removes an unused example section.",
        verdict_line="One repeat, a tie: no evidence either way yet.",
    )


def _all_ties_patch_result():
    """n=3 (< 5), every judged pair a tie -- a common, deliberate result
    (the judge found the two harnesses equivalent), not missing data, and
    must render as such rather than a broken/empty widget."""
    grades = _synthetic_grades(3, skill_tokens_improvement=100, verdict="tie", consistent=True)
    return _patch_result(
        "p4", "tied-skill", grades,
        rationale="Removes a comment block the agent never reads.",
        verdict_line="All three judged pairs came back tied: the two harnesses look equivalent so far.",
    )


def _all_ties_large_n_patch_result():
    """n=10 (>= stats.MIN_SAMPLES_FOR_CI): all-ties here hits win_rate's
    OTHER note, "no_decided_pairs" (verified against the real function --
    at n<10 it's "insufficient_samples" instead; both must render the same
    deliberate "N ties" text, not just one of them)."""
    grades = _synthetic_grades(10, skill_tokens_improvement=0, verdict="tie", consistent=True)
    return _patch_result(
        "p7", "wash-skill", grades,
        rationale="Reorders two independent paragraphs.",
        verdict_line="Ten judged pairs, all ties: the harnesses are behaviorally equivalent on this prompt.",
    )


def _rejected_patch_result():
    """n=12 (>=10): treatment loses convincingly on both quality and the
    primary metric, plus a high-severity regression -- a real, well-
    powered REJECT, distinct from the small-n DIRECTIONAL/PROVISIONAL
    states above."""
    grades = _synthetic_grades(
        12, skill_tokens_improvement=-1100, verdict="control", consistent=True,
        high_regression_on_first=True,
    )
    return _patch_result(
        "p6", "risky-skill", grades,
        rationale="Removes error-handling code the agent actually relies on.",
        verdict_line="Regresses skill tokens and quality, with a high-severity MCP failure: reject.",
    )


def _not_run_patch_result():
    """C-1: the patch was never applied at all -- no grades, no
    measurement. cli.py's not-applied gate result still carries
    directional/reasons fields (stats.gate() knows nothing about
    `application["applied"]`), so without the applied-first check this
    would fall through to a red REJECTED and read as "measured and lost"
    instead of "never built.\""""
    patch = make_patch(
        patch_id="p9", kind=PATCH_SKILL_TRUNCATE, target="stale-skill",
        rationale="Trims a section whose line numbers may have drifted.",
        expected_effect=EFFECT_SKILL_TOKENS, source_recommendation="rec-p9", diff="",
    )
    application = make_patch_application(
        patch_id="p9", applied=False, diff="",
        reason="lint: cited line range [10, 999] exceeds file length (42 lines)",
    )
    matrix_summary = {
        "prompt": "do the thing", "k": 0, "patch_id": "p9",
        "control_version_id": "base-abc", "treatment_version_id": "treat-p9",
        "arms": {ARM_CONTROL: [], ARM_TREATMENT: []},
    }
    gate = {
        "accepted": False, "provisional": False, "directional": "neutral",
        "reasons": ["patch not applied: lint: cited line range [10, 999] exceeds file length (42 lines)"],
    }
    return make_patch_result(
        patch=patch, application=application, matrix_summary=matrix_summary,
        grades=[], statistics={}, gate=gate,
        verdict_line="Never applied: the cited line range no longer matches the file.",
    )


def _high_regression_but_favorable_gate_patch_result():
    """C-2: treatment wins every judged pair right up until it crashes the
    harness on one repeat with a high-severity regression. win/loss counts
    and the efficiency point estimate alone (what stats.gate()'s
    `directional` is computed from) still read favorable -- the badge must
    override that itself rather than trust the gate to have noticed."""
    grades = _synthetic_grades(5, skill_tokens_improvement=1100, verdict="treatment", consistent=True, high_regression_on_first=True)
    return _patch_result(
        "p10", "crashy-skill", grades,
        rationale="Removes a fallback path the harness turns out to depend on.",
        verdict_line="Won every judged pair, but crashed the harness once with a high-severity failure.",
    )


def _accepted_zero_decided_pairs_patch_result():
    """I-1: n=12 (full CI zone), every judged pair a TIE (0 wins, 0
    losses). stats.gate()'s quality check passes on losses==0 -- true
    regardless of ties -- and the primary metric's CI genuinely clears
    zero, so accepted=True with ZERO quality evidence behind it. Must
    render as a qualified badge adjacent to the "0 decided pairs" fact,
    not a plain green ACCEPTED indistinguishable from a real win."""
    grades = _synthetic_grades(12, skill_tokens_improvement=1100, verdict="tie", consistent=True)
    return _patch_result(
        "p8", "quiet-win-skill", grades,
        rationale="Removes a paragraph the judge never seemed to notice either way.",
        verdict_line="Judge called every pair a tie, but skill tokens measurably dropped.",
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
            patch_results=[_accepted_patch_result()], k=12, cost_usd=1.2345,
        )
        expected_keys = set(make_report(
            session={}, baseline_version={}, patch_results=[], caveats=[],
        ).keys())
        self.assertEqual(set(rpt.keys()), expected_keys)
        self.assertEqual(rpt["k"], 12)
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
            patch_results=[_accepted_provisional_patch_result()], k=7,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("p2", joined)
        self.assertIn("Point estimates only", joined)

    def test_caveats_report_judge_consistency_rate(self):
        # _accepted_patch_result: 12/12 consistent; _directional_patch_result: 0/1 -> 12/13 overall
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_accepted_patch_result(), _directional_patch_result()], k=12,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("Judge consistency", joined)
        self.assertIn("12/13", joined)

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
                patch_results=[pr], k=12,
            )


class IntervalTextTests(unittest.TestCase):
    """Unit coverage for the flat-shape fallback: real stats.py nests the
    CI under "bootstrap", but _interval_text must also handle a flat
    lo/hi/note dict (an earlier pinned shape, and a plausible future
    producer) without crashing or mis-rendering."""

    def test_nested_bootstrap_shape(self):
        m = {"abs_delta": 1100, "bootstrap": {"lo": 800.0, "hi": 1400.0, "level": 0.95, "note": None}}
        self.assertIn("[800.00, 1,400.00]", report._interval_text(m))

    def test_flat_shape_fallback(self):
        m = {"abs_delta": 1100, "lo": 800.0, "hi": 1400.0, "level": 0.95, "note": None}
        self.assertIn("[800.00, 1,400.00]", report._interval_text(m))

    def test_insufficient_samples_note_no_interval(self):
        m = {"bootstrap": {"lo": None, "hi": None, "note": "insufficient_samples", "distinct_resamples": 10}}
        text = report._interval_text(m)
        self.assertIn("insufficient samples for a CI", text)
        self.assertIn("10 distinct resample", text)

    def test_low_confidence_note_keeps_interval_but_flags_it(self):
        m = {"bootstrap": {"lo": 0.0, "hi": 0.0, "note": "low_confidence", "distinct_resamples": 92378}}
        text = report._interval_text(m)
        self.assertIn("[0.00, 0.00]", text)
        self.assertIn("low confidence", text)


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
        rpt = self._build_and_write([_accepted_patch_result()], k=12)
        on_disk = json.loads((self.out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, rpt)

    def test_html_contains_diff_text(self):
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn("Trimmed.", html_text)
        self.assertIn("verbose-skill/SKILL.md", html_text)

    def test_html_contains_caveats_verbatim_and_before_the_patches(self):
        rpt = self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        for caveat in rpt["caveats"]:
            self.assertIn(report._esc(caveat), html_text)
        self.assertLess(html_text.index('class="caveats"'), html_text.index('class="patch-card"'))

    # --- the five real gate states ------------------------------------

    def test_gate_badge_accepted_full_ci(self):
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn('class="badge accept">ACCEPTED<', html_text)
        self.assertNotIn('class="badge accept-provisional"', html_text)

    def test_gate_badge_accepted_provisional(self):
        self._build_and_write([_accepted_provisional_patch_result()], k=7)
        html_text = self._html()
        self.assertIn('class="badge accept-provisional">ACCEPTED &middot; PROVISIONAL</span>', html_text)

    def test_gate_badge_provisional_reject_is_not_red(self):
        self._build_and_write([_provisional_reject_patch_result()], k=5)
        html_text = self._html()
        self.assertIn('class="badge directional-favorable">PROVISIONAL &middot; FAVORABLE</span>', html_text)
        self.assertNotIn('class="badge reject"', html_text)

    def test_gate_badge_directional_neutral_is_not_red(self):
        self._build_and_write([_directional_patch_result()], k=1)
        html_text = self._html()
        self.assertIn('class="badge directional-neutral">DIRECTIONAL &middot; NEUTRAL</span>', html_text)
        self.assertNotIn('class="badge reject"', html_text)

    def test_gate_badge_directional_favorable_at_small_n(self):
        self._build_and_write([_all_ties_patch_result()], k=3)
        html_text = self._html()
        self.assertIn('class="badge directional-favorable">DIRECTIONAL &middot; FAVORABLE</span>', html_text)

    def test_gate_badge_regressed_overrides_rejected_when_high_severity_present(self):
        # _rejected_patch_result carries a high-severity regression, so C-2's
        # override fires before the normal 5-state table is even consulted --
        # REGRESSED, not a bare REJECTED, and the count is in the badge text
        # itself (inside .patch-head, not just in the collapsible body).
        self._build_and_write([_rejected_patch_result()], k=12)
        html_text = self._html()
        self.assertIn('class="badge regressed">REGRESSED &middot; 1 HIGH-SEVERITY</span>', html_text)
        self.assertNotIn('class="badge reject"', html_text)

    def test_gate_badge_rejected_confident_no_regression(self):
        grades = _synthetic_grades(12, skill_tokens_improvement=-1100, verdict="control", consistent=True)
        pr = _patch_result(
            "p11", "losing-skill", grades,
            rationale="Removes a paragraph that turns out to matter.",
            verdict_line="Regresses skill tokens and quality: reject, no crash involved.",
        )
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertIn('class="badge reject">REJECTED</span>', html_text)

    def test_gate_badge_rejected_all_ties_at_full_n(self):
        self._build_and_write([_all_ties_large_n_patch_result()], k=10)
        html_text = self._html()
        self.assertIn('class="badge reject">REJECTED</span>', html_text)
        # still a deliberate win-rate reading, not a broken widget
        self.assertIn("10 ties, no decided pairs", html_text)

    def test_legacy_two_state_gate_falls_back_to_reject_without_crashing(self):
        pr = _accepted_patch_result()
        pr["gate"] = {"accepted": False, "reasons": ["legacy shape, no provisional/directional keys"]}
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertIn('class="badge reject">REJECT</span>', html_text)

    def test_gate_reasons_render_for_every_state(self):
        self._build_and_write(
            [_accepted_patch_result(), _provisional_reject_patch_result(), _directional_patch_result()], k=12,
        )
        html_text = self._html()
        self.assertIn("no high-severity regressions", html_text)
        self.assertIn("cannot accept without zero losses at this sample size", html_text)
        self.assertIn("insufficient_samples_for_acceptance", html_text)

    # --- win rate / all-ties / consistency ------------------------------

    def test_all_ties_small_n_renders_as_deliberate_result_not_broken(self):
        self._build_and_write([_all_ties_patch_result()], k=3)
        html_text = self._html()
        self.assertIn("3 ties, no decided pairs", html_text)
        self.assertIn("the judge found the two harnesses equivalent", html_text)
        self.assertNotIn('<p class="muted">No judged pairs.</p>', html_text)

    def test_all_ties_large_n_renders_as_deliberate_result_not_broken(self):
        self._build_and_write([_all_ties_large_n_patch_result()], k=10)
        html_text = self._html()
        self.assertIn("10 ties, no decided pairs", html_text)

    def test_low_position_consistency_is_flagged(self):
        self._build_and_write([_directional_patch_result()], k=1)
        html_text = self._html()
        self.assertIn("consistency-low", html_text)
        self.assertIn("position consistency: 0%", html_text)
        self.assertIn("below the ~70-77% published baseline", html_text)

    def test_high_position_consistency_is_not_flagged(self):
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn("position consistency: 100%", html_text)
        self.assertNotIn('class="stat-line consistency-low"', html_text)

    # --- efficiency table / primary_effect mapping ----------------------

    def test_insufficient_ci_shows_point_estimate_never_a_dash(self):
        self._build_and_write([_accepted_provisional_patch_result()], k=7)
        html_text = self._html()
        self.assertIn("+1,100.00", html_text)  # abs_delta, its own column, always shown
        self.assertIn("insufficient samples for a CI", html_text)
        self.assertIn("1716 distinct resample", html_text)
        self.assertNotIn("<td>—</td><td>—</td></tr>", html_text)

    def test_primary_effect_name_maps_to_efficiency_metric_and_highlights_row(self):
        # regression test for the bug where statistics["primary_effect"]
        # (an EFFECT_* name, e.g. "skill_tokens") was used directly as an
        # EFFICIENCY_METRICS key (e.g. "skill_tokens_est") without going
        # through EFFECT_TO_EFFICIENCY_METRIC, silently failing to highlight
        # anything.
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn('<span class="tag">target</span>', html_text)
        self.assertIn("Skill tokens (est.)", html_text)

    # --- permutation test / pass^k ---------------------------------------

    def test_permutation_floor_phrasing_suppressed_when_test_is_well_powered(self):
        # M-3: at n=12 the floor (min_achievable_p ~= 0.00049) is far below
        # 0.05 -- this test IS well-powered, so the "as extreme as this
        # permits" hedge (written for the k=3/5 case) must not appear and
        # hedge away a genuinely strong result.
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertNotIn("as extreme as n=12 permits", html_text)
        self.assertIn("p=0.000488", html_text)

    def test_permutation_floor_phrasing_small_n(self):
        self._build_and_write([_provisional_reject_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("as extreme as n=5 permits", html_text)

    def test_pass_k_degenerate_shows_flag_and_c_over_n_without_redundant_note(self):
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn("all k succeeded", html_text)
        self.assertIn("flag, n=k", html_text)
        self.assertIn("c=12/n=12", html_text)
        # the raw stats.py note string would restate what "(flag, n=k...)"
        # already says -- must be suppressed, not printed twice
        self.assertNotIn("n_equals_k", html_text)

    # --- regressions -------------------------------------------------------

    def test_regressions_render_with_high_severity(self):
        self._build_and_write([_rejected_patch_result()], k=12)
        html_text = self._html()
        self.assertIn('class="regressions"', html_text)
        self.assertIn("regression-high", html_text)
        self.assertIn("mcp server flaky-mcp failed in treatment", html_text)

    def test_no_regressions_renders_no_empty_section(self):
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertNotIn('class="regressions"', html_text)

    # --- caveats / escaping / safety ---------------------------------------

    def test_per_patch_caveat_surfaces_inline_on_matching_card(self):
        caveat = "skill.remove for loss-skill is a no-op: it also exists under an uncontrolled skill root."
        self._build_and_write([_provisional_reject_patch_result()], k=5, extra_caveats=[caveat])
        html_text = self._html()
        self.assertIn(report._esc(caveat), html_text)
        self.assertIn('class="patch-caveats"', html_text)
        self.assertEqual(html_text.count(report._esc(caveat)), 2)

    def test_html_has_no_external_http_resource_references(self):
        self._build_and_write(
            [_accepted_patch_result(), _provisional_reject_patch_result(), _directional_patch_result(), _rejected_patch_result()],
            k=12,
        )
        html_text = self._html()
        self.assertNotIn("http://", html_text)
        self.assertNotIn("https://", html_text)

    def test_html_renders_with_zero_patches(self):
        self._build_and_write([], k=3)
        html_text = self._html()
        self.assertIn("No candidate patches", html_text)
        self.assertIn('class="caveats"', html_text)

    def test_html_escapes_untrusted_text(self):
        pr = _accepted_patch_result()
        pr["patch"]["rationale"] = "<script>alert(1)</script> & more"
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertNotIn("<script>alert(1)</script>", html_text)
        self.assertIn("&lt;script&gt;", html_text)


class ServeTests(unittest.TestCase):
    def test_make_server_binds_loopback_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            rpt = report.build(
                session=_session(), baseline_version=_baseline_version(),
                patch_results=[_accepted_patch_result()], k=12,
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
