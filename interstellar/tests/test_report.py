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

import http.client
import json
import sys
import tempfile
import threading
import time
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

# A del/add pair that's clearly "the same line, modified" (SequenceMatcher
# ratio well above _CLOSE_MATCH_RATIO) -- the word-level highlighting test's
# fixture. Only "typo" -> "typoo" differs.
WORD_LEVEL_DIFF = """--- a/skills/typo-skill/SKILL.md
+++ b/skills/typo-skill/SKILL.md
@@ -3,3 +3,3 @@
 unchanged context line
-This line has a small typo in it
+This line has a small typoo in it
 trailing context line
"""

# Header claims a 60-line old-file span starting at line 66 (old_start=66,
# old_count=60 -> old_end=125) -- the "removes N of M lines" fraction test
# only needs the hunk header's arithmetic to be internally consistent, not
# a literal 60-line diff body (see _truncate_fraction_text: numerator comes
# from application["lines_changed"]/patch["line_ranges"], denominator from
# the hunk header alone).
FRACTION_DIFF = """--- a/skills/big-skill/SKILL.md
+++ b/skills/big-skill/SKILL.md
@@ -66,60 +66,1 @@
-first removed line of a long cut block
-second removed line of a long cut block
+kept summary line
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


def _patch_result(patch_id, target, grades, *, rationale, verdict_line, diff=SAMPLE_DIFF,
                   applied=True, line_ranges=None, lines_changed=None):
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
        source_recommendation=f"rec-{patch_id}", diff=diff, line_ranges=line_ranges,
    )
    if lines_changed is None:
        lines_changed = 2 if diff else 0
    application = make_patch_application(patch_id=patch_id, applied=applied, diff=diff, lines_changed=lines_changed)
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


def _no_data_patch_result():
    """stats.py fix-round-3: every run failed on at least one arm (a real
    k=3 cycle hit this on a transient auth error). stats.gate() short-
    circuits to directional="unknown" before any other check runs. Must
    render as NO DATA, not a bare REJECTED that reads as "measured and
    lost" -- distinct from both NOT RUN (never applied) and a real
    directional="neutral" (measured and came out even)."""
    grades = [
        make_grade(
            repeat=i, control_efficiency={}, treatment_efficiency={}, judge=None,
            control_ok=False, treatment_ok=False,
        )
        for i in range(3)
    ]
    return _patch_result(
        "p13", "flaky-skill", grades,
        rationale="Trims a rarely-hit error path.",
        verdict_line="All three repeats failed on a transient auth error before either arm produced usable data.",
    )


def _partial_failure_patch_result():
    """1 of 3 repeats usable (control failed once, treatment failed once,
    on different repeats). statistics["n"] now means usable pairs (1), not
    total repeats attempted (3) -- must stay visible as "1 of 3", not
    silently reported as n=1 with no context for where the other 2 went."""
    grades = [
        make_grade(
            repeat=0, control_efficiency=dict(BASE_EFFICIENCY),
            treatment_efficiency={**BASE_EFFICIENCY, "skill_tokens_est": 900},
            judge=make_judge_verdict(verdict="treatment", consistent=True),
            control_ok=True, treatment_ok=True,
        ),
        make_grade(repeat=1, control_efficiency={}, treatment_efficiency={}, judge=None, control_ok=False, treatment_ok=True),
        make_grade(repeat=2, control_efficiency={}, treatment_efficiency={}, judge=None, control_ok=True, treatment_ok=False),
    ]
    return _patch_result(
        "p14", "half-measured-skill", grades,
        rationale="Trims an example block.",
        verdict_line="Only one of three repeats produced usable data; the other two failed on one arm each.",
    )


def _word_level_diff_patch_result():
    """A close-matching del/add pair (see WORD_LEVEL_DIFF) -- exercises the
    word-level highlighting path of the diff view."""
    grades = _synthetic_grades(3, skill_tokens_improvement=50, verdict="tie", consistent=True)
    return _patch_result(
        "p16", "typo-skill", grades, diff=WORD_LEVEL_DIFF,
        rationale="Fixes a typo the agent occasionally copies verbatim.",
        verdict_line="Trivial one-word fix, too few repeats to say more.",
    )


def _truncate_fraction_patch_result():
    """A skill.truncate patch whose line_ranges/lines_changed/diff hunk
    header are consistent enough to exercise the "removes N of M lines"
    fraction line end to end (see FRACTION_DIFF)."""
    grades = _synthetic_grades(3, skill_tokens_improvement=200, verdict="tie", consistent=True)
    return _patch_result(
        "p15", "big-skill", grades, diff=FRACTION_DIFF,
        line_ranges=[[66, 125]], lines_changed=60,
        rationale="Trims a large dead block of duplicated guidance.",
        verdict_line="Large truncation, small sample so far.",
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
        # build() adds "rerun" on top of make_report()'s frozen shape (Task
        # 2's stored re-run config) -- report.json is this module's own
        # artifact, not the types.py contract, so this is an intentional
        # addition, not a drift bug. See build()'s docstring.
        expected_keys = set(make_report(
            session={}, baseline_version={}, patch_results=[], caveats=[],
        ).keys()) | {"rerun"}
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
        pr = _all_ties_patch_result()
        self.assertEqual(pr["gate"].get("quality_evidence"), "ties_only")  # sanity
        self._build_and_write([pr], k=3)
        html_text = self._html()
        # quality_evidence != "decided" -> the efficiency-only suffix is
        # part of the badge itself, not only in the reasons list
        self.assertIn(
            'class="badge directional-favorable">DIRECTIONAL &middot; FAVORABLE '
            '(efficiency only, no decided quality pairs)</span>',
            html_text,
        )

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


class FixRound3ReviewTests(unittest.TestCase):
    """Every Critical/Important from review-report.md, verified against the
    real HTML the fix produces (not just that the function returns
    something) -- the review's own method."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "out"

    def _build_and_write(self, patch_results, k=3, **kwargs):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=patch_results, k=k, cost_usd=2.5, **kwargs,
        )
        report.write(rpt, self.out_dir)
        return rpt

    def _html(self):
        return (self.out_dir / "index.html").read_text(encoding="utf-8")

    # --- C-1: never-applied patch must not read as "measured and lost" ---

    def test_never_applied_patch_renders_not_run_never_reject(self):
        self._build_and_write([_not_run_patch_result()], k=0)
        html_text = self._html()
        self.assertIn('class="badge not-run">NOT RUN</span>', html_text)
        self.assertNotIn('class="badge reject"', html_text)
        self.assertIn("Not applied: lint: cited line range", html_text)

    # --- C-2: a high-severity regression must dominate the badge, visibly,
    # regardless of what accepted/provisional/directional say ---

    def test_high_severity_regression_overrides_a_favorable_looking_gate(self):
        pr = _high_regression_but_favorable_gate_patch_result()
        # sanity: the win/loss counts alone (what a reader would infer
        # without the override) really do look favorable -- 4 wins, 0
        # losses -- so this is exercising report.py's own override, not
        # merely re-checking that stats.gate() already forces
        # directional="unfavorable" on a high-severity regression (it
        # does, independently -- see stats.py's own final override; this
        # test is the "do not rely on that alone" half of the fix).
        wr = pr["statistics"]["win_rate"]
        self.assertEqual(wr["losses"], 0)
        self.assertGreater(wr["wins"], 0)
        self._build_and_write([pr], k=5)
        html_text = self._html()
        self.assertIn('class="badge regressed">REGRESSED &middot; 1 HIGH-SEVERITY</span>', html_text)
        self.assertNotIn("FAVORABLE", html_text.split('<div class="patch-body">')[0])

    # --- C-3: no paired data must say so, never a dash plus a guessed why ---

    def test_metric_with_no_paired_data_says_so_not_a_dash(self):
        pr = _accepted_patch_result()
        # zero out one metric's pairing entirely, as efficiency_deltas()
        # does for a metric no grade reported on either side
        pr["statistics"]["efficiency"]["turns"] = {
            "lower_is_better": True, "n": 0,
            "control_median": None, "treatment_median": None,
            "abs_delta": None, "pct_delta": None, "bootstrap": None,
        }
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertIn("no paired data", html_text)
        self.assertNotIn("control ≈ 0", html_text)
        self.assertNotIn("<td>—</td><td>—</td><td>n/a</td>", html_text)

    # --- stats.py fix-round-3: zero-usable-pairs must not read as a
    # measured loss, and a partial failure must stay visible ---

    def test_zero_usable_pairs_renders_no_data_never_reject(self):
        pr = _no_data_patch_result()
        self.assertEqual(pr["gate"]["directional"], "unknown")  # sanity
        self._build_and_write([pr], k=3)
        html_text = self._html()
        self.assertIn('class="badge no-data">NO DATA</span>', html_text)
        self.assertNotIn('class="badge reject"', html_text)
        self.assertIn("no successful paired runs: 0 of 3 runs produced usable data", html_text)

    def test_no_data_is_distinct_from_not_run_and_from_real_neutral(self):
        # three genuinely different truths, three genuinely different badges
        self._build_and_write(
            [_no_data_patch_result(), _not_run_patch_result(), _directional_patch_result()], k=3,
        )
        html_text = self._html()
        self.assertIn('class="badge no-data">NO DATA</span>', html_text)
        self.assertIn('class="badge not-run">NOT RUN</span>', html_text)
        self.assertIn('class="badge directional-neutral">DIRECTIONAL &middot; NEUTRAL</span>', html_text)

    def test_partial_failure_shows_usable_of_total_not_bare_n(self):
        pr = _partial_failure_patch_result()
        self.assertEqual(pr["statistics"]["data"]["pairs_total"], 3)
        self.assertEqual(pr["statistics"]["data"]["pairs_usable"], 1)
        self.assertEqual(pr["statistics"]["n"], 1)  # sanity: n means usable, not total
        self._build_and_write([pr], k=3)
        html_text = self._html()
        self.assertIn("1 of 3 paired repeats usable", html_text)
        self.assertIn("control failed 1", html_text)
        self.assertIn("treatment failed 1", html_text)
        # must not silently render the bare, misleading "1 paired repeats run"
        self.assertNotIn("1 paired repeats run", html_text)

    # --- C-5: serve() must never expose anything besides the two report
    # files, regardless of what else lives in out_dir (credentials, other
    # scratch state) ---

    def test_serve_never_exposes_files_outside_the_allow_list(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_accepted_patch_result()], k=12,
        )
        report.write(rpt, self.out_dir)
        # simulate cli.py's real layout: out_dir also holds a materialized
        # harness home with a credentials file, plus a directory
        (self.out_dir / "scratch").mkdir()
        secret = self.out_dir / "scratch" / "auth.json"
        secret.write_text('{"token": "super-secret"}', encoding="utf-8")

        httpd = report.make_server(self.out_dir, host="127.0.0.1", port=0)
        try:
            import http.client
            import threading
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                port = httpd.server_address[1]
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

                conn.request("GET", "/scratch/auth.json")
                resp = conn.getresponse()
                body = resp.read()
                self.assertEqual(resp.status, 404)
                self.assertNotIn(b"super-secret", body)

                conn.request("GET", "/scratch/")
                resp = conn.getresponse()
                resp.read()
                self.assertEqual(resp.status, 404)

                conn.request("GET", "/index.html")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                resp.read()

                conn.request("GET", "/report.json")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                resp.read()
                conn.close()
            finally:
                httpd.shutdown()
                thread.join(timeout=5)
        finally:
            httpd.server_close()

    def test_make_server_does_not_mutate_global_tcpserver_class(self):
        import socketserver as _ss
        before = _ss.TCPServer.allow_reuse_address
        httpd = report.make_server(self._write_minimal(), host="127.0.0.1", port=0)
        try:
            self.assertEqual(_ss.TCPServer.allow_reuse_address, before)
        finally:
            httpd.server_close()

    def _write_minimal(self):
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[_accepted_patch_result()], k=12,
        )
        report.write(rpt, self.out_dir)
        return self.out_dir

    # --- I-1: accepted with zero decided pairs must be visually qualified ---

    def test_accepted_with_zero_decided_pairs_is_qualified_not_plain_green(self):
        pr = _accepted_zero_decided_pairs_patch_result()
        self.assertTrue(pr["gate"]["accepted"])  # sanity: gate really did accept
        self.assertEqual(pr["statistics"]["win_rate"]["wins"], 0)
        self.assertEqual(pr["statistics"]["win_rate"]["losses"], 0)
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertIn('class="badge accept-provisional">ACCEPTED &middot; NO DECIDED PAIRS</span>', html_text)
        self.assertNotIn('class="badge accept">ACCEPTED</span>', html_text)

    # --- I-2: the three different Ns must be distinguishable ---

    def test_repeats_and_judged_pairs_are_labeled_distinctly(self):
        self._build_and_write([_provisional_reject_patch_result()], k=5)
        html_text = self._html()
        self.assertIn("5 paired repeats run", html_text)
        self.assertIn("(5 judged pair", html_text)

    # --- I-3: per-patch caveat matching must not staple the generic
    # aggregate onto every card, and must not false-positive on short names

    def test_aggregate_insufficient_samples_caveat_not_duplicated_per_card(self):
        rpt = self._build_and_write([_provisional_reject_patch_result()], k=5)
        aggregate = next(c for c in rpt["caveats"] if c.startswith("Point estimates only"))
        html_text = self._html()
        # appears once, in the persistent panel -- not also inline on the card
        self.assertEqual(html_text.count(report._esc(aggregate)), 1)

    def test_per_patch_caveat_uses_word_boundary_not_raw_substring(self):
        pr = _provisional_reject_patch_result()
        pr["patch"]["target"] = "rules"
        # "rules" is a substring of "overrules" but not the same word --
        # a raw `in` check would false-positive on this; word-boundary
        # matching must not.
        unrelated = "The isolation note overrules per-call token limits in this cycle"
        specific = "rules.append for rules duplicates an existing line"
        self._build_and_write([pr], k=5, extra_caveats=[unrelated, specific])
        html_text = self._html()
        self.assertIn(report._esc(specific), html_text)
        card = html_text.split('id="p5"')[1].split("</article>")[0]
        self.assertNotIn(report._esc(unrelated), card)

    # --- I-5: dropped-before-ranking patches get real information, not one
    # joined sentence ---

    def test_dropped_patches_render_with_rationale_and_lineage(self):
        dropped = [make_patch(
            patch_id="d1", kind=PATCH_SKILL_TRUNCATE, target="stale-target",
            rationale="Trims a section the analyzer flagged as bloated.",
            expected_effect=EFFECT_SKILL_TOKENS, source_recommendation="rec-7",
            skipped_reason="lint: line range out of bounds",
        )]
        rpt = report.build(
            session=_session(), baseline_version=_baseline_version(),
            patch_results=[], dropped_patches=dropped, k=3,
        )
        joined = " ".join(rpt["caveats"])
        self.assertIn("d1", joined)
        self.assertIn("stale-target", joined)
        self.assertIn("rec-7", joined)
        self.assertIn("lint: line range out of bounds", joined)
        self.assertIn("Trims a section the analyzer flagged as bloated.", joined)

    # --- I-6: .badge.kind and regression severity chips must be theme-aware ---

    def test_badge_kind_and_regression_chip_have_light_mode_tokens(self):
        html_text = render_module_style()
        self.assertIn("--kind-fg:", html_text)
        self.assertIn("--sev-chip-bg:", html_text)
        # both tokens must be redefined inside the light-mode block, not
        # only declared once in the dark (default) :root
        light_block = html_text.split("prefers-color-scheme: light")[1].split("}\n}")[0]
        self.assertIn("--kind-fg:", light_block)
        self.assertIn("--sev-chip-bg:", light_block)

    # --- additional finding: sign convention + explicit direction wording ---

    def test_a_real_saving_renders_good_and_says_saved_not_bad(self):
        # skill_tokens_est: control=2000, treatment=900 -- a 1100-token
        # SAVING on a lower_is_better metric. abs_delta = control-treatment
        # = +1100 (stats.py convention); must render "good"/"saved", never
        # "bad" with an unqualified "+55.0%" that reads as an increase.
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertIn('tr class="good hl"', html_text)
        self.assertNotIn('tr class="bad hl"', html_text)
        self.assertIn("+1,100.00 saved", html_text)

    def test_a_real_regression_renders_bad_and_says_cost_more(self):
        grades = _synthetic_grades(12, skill_tokens_improvement=-1100, verdict="control", consistent=True)
        pr = _patch_result(
            "p12", "worse-skill", grades, rationale="test",
            verdict_line="Regresses.",
        )
        self._build_and_write([pr], k=12)
        html_text = self._html()
        self.assertIn('tr class="bad hl"', html_text)
        self.assertIn("cost more", html_text)

    # --- cache_sensitive flag: mark cache-warmth-affected metrics without
    # letting them discount the deterministic target metric next to them ---

    def test_cache_sensitive_metrics_are_marked_target_metric_is_not(self):
        pr = _accepted_patch_result()
        eff = pr["statistics"]["efficiency"]
        eff["total_tokens"]["cache_sensitive"] = True
        eff["cost_usd"]["cache_sensitive"] = True
        # skill_tokens_est (the target metric here) is left unmarked --
        # deterministic, not cache-affected, per the team-lead's report
        self._build_and_write([pr], k=12)
        html_text = self._html()
        # Target the metrics table specifically -- the diff view (Task 1)
        # renders its own <tbody> earlier in the page, so a bare first
        # "<tbody>" match would grab diff rows instead of metric rows.
        metrics_section = html_text.split('table class="metrics"')[1]
        rows = metrics_section.split("<tbody>")[1].split("</tbody>")[0].split("</tr>")
        total_row = next(r for r in rows if "Total tokens" in r)
        cost_row = next(r for r in rows if "Cost (USD)" in r)
        target_row = next(r for r in rows if "Skill tokens (est.)" in r)
        self.assertIn('class="tag cache-tag"', total_row)
        self.assertIn('class="tag cache-tag"', cost_row)
        self.assertNotIn('class="tag cache-tag"', target_row)
        self.assertIn("prompt-cache warmth", html_text)
        self.assertIn("indicative, not as a precise measurement", html_text)
        # legend appears exactly once per table, not once per flagged row
        self.assertEqual(html_text.count('class="cache-legend'), 1)

    def test_cache_sensitive_flag_absent_renders_no_marker_no_legend(self):
        # older reports (statistics predating the flag) must render clean --
        # every real-stats fixture already omits the key entirely, so this
        # is the default path, not a special case
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertNotIn('class="tag cache-tag"', html_text)
        self.assertNotIn('class="cache-legend', html_text)

    # --- quality_evidence: a favorable/unfavorable reading with zero
    # decided quality pairs behind it must say so on the badge itself ---

    def test_quality_evidence_none_gets_efficiency_only_suffix(self):
        # zero judged pairs at all -- gate()'s "none" variant. Exercised as
        # a direct _gate_badge unit test since a real stats.summarize()/
        # gate() call that reaches directional="favorable" with genuinely
        # zero judged pairs (not merely all-ties) isn't reachable through
        # ordinary synthetic grades -- efficiency-only-with-no-judge-calls
        # is exactly the edge case the field exists to describe honestly.
        gate = {
            "accepted": False, "provisional": False, "directional": "favorable",
            "quality_evidence": "none",
            "reasons": [
                "insufficient_samples_for_acceptance: only 0 judged pair(s) -- k<5 cannot support an accept decision at any confidence",
                'directional reading ("favorable") rests on efficiency alone: no judged quality pairs at all',
            ],
        }
        badge = report._gate_badge(gate, applied=True, high_severity_regressions=0, win_rate={"wins": 0, "ties": 0, "losses": 0, "n": 0})
        self.assertIn("DIRECTIONAL &middot; FAVORABLE (efficiency only, no decided quality pairs)", badge)

    def test_quality_evidence_absent_renders_no_suffix(self):
        # older gate dicts (predating this field) must render exactly as
        # before -- no suffix, no crash.
        gate = {
            "accepted": False, "provisional": True, "directional": "unfavorable",
            "reasons": ["provisional quality pass: ..."],
        }
        badge = report._gate_badge(gate)
        self.assertIn('PROVISIONAL &middot; UNFAVORABLE</span>', badge)
        self.assertNotIn("efficiency only", badge)

    def test_quality_evidence_neutral_reading_never_gets_the_suffix(self):
        # the suffix only qualifies an optimistic/pessimistic claim --
        # "neutral" has no claim to qualify, even at quality_evidence="none"
        gate = {
            "accepted": False, "provisional": False, "directional": "neutral",
            "quality_evidence": "none",
            "reasons": ["insufficient_samples_for_acceptance: only 0 judged pair(s)"],
        }
        badge = report._gate_badge(gate)
        self.assertIn("DIRECTIONAL &middot; NEUTRAL</span>", badge)
        self.assertNotIn("efficiency only", badge)

    # --- Minors worth a regression test ---

    def test_esc_none_is_empty_not_a_dash(self):
        self.assertEqual(report._esc(None), "")

    def test_pass_k_zero_runs_has_sane_message(self):
        pk_line = report._pass_k_line({"pass_k": {"value": None, "n": 0, "c": 0, "k": 0, "degenerate": False, "note": "insufficient_samples"}})
        self.assertIn("no runs completed", pk_line)
        self.assertNotIn("pass&#94;0", pk_line)

    def test_zero_width_ci_flagged_as_degenerate(self):
        text = report._interval_text({"bootstrap": {"lo": 50.0, "hi": 50.0, "note": None, "level": 0.95}})
        self.assertIn("[50.00, 50.00]", text)
        self.assertIn("degenerate", text)

    def test_no_external_script_reference_in_output(self):
        # Task 2 gives the page real inline JS (the run-panel poller) --
        # the page is no longer script-free, but it must still never
        # reference an external script (a <script src="..."> would be a
        # non-self-contained resource, exactly what the http(s)-reference
        # test below also guards against from a different angle).
        self._build_and_write([_accepted_patch_result()], k=12)
        html_text = self._html()
        self.assertNotIn("<script src=", html_text)
        self.assertIn("<script>", html_text)


def render_module_style():
    """The raw _STYLE string, for CSS-structure assertions that don't need
    a full report build."""
    return report._STYLE


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
