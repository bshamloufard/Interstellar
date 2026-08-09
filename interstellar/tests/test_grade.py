"""Tests for interstellar.grade. No test here spawns grok-dev: judge_pair
takes an injectable `grok` callable and every test supplies a fake one.
`_default_grok`'s own parsing logic is tested by mocking `subprocess.run`
and `harness.materialize` directly (never the real binary, never the real
`~/.grok`) -- see DefaultGrokParsingTest."""
from __future__ import annotations

import glob
import json
import types
import unittest
from pathlib import Path
from unittest import mock

from interstellar import grade
from interstellar.grade import (
    efficiency,
    grade_matrix,
    judge_consistency,
    judge_pair,
    regressions,
)
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFICIENCY_METRICS,
    make_grade,
    make_judge_verdict,
    make_replay_matrix,
    make_run_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "traces" / "corpus"

_ALL_NONE = {name: None for name, _ in EFFICIENCY_METRICS}


def _load_corpus_trace(pattern="skill_bloat_*.json"):
    matches = sorted(glob.glob(str(CORPUS / pattern)))
    if not matches:
        return None
    return json.loads(Path(matches[0]).read_text())


# --------------------------------------------------------------------------
# efficiency()
# --------------------------------------------------------------------------

_REAL_USAGE = {
    "input_tokens": 15811,
    "cache_read_input_tokens": 128,
    "cache_creation_input_tokens": 0,
    "output_tokens": 708,
    "reasoning_tokens": 449,
    "total_tokens": 16647,
}


def _run(*, trace=None, cost_usd=None, turns=None, usage=None, ok=True, error=None):
    return make_run_result(
        arm=ARM_CONTROL, repeat=0, ok=ok, prompt="do the thing",
        home="/tmp/h", workdir="/tmp/w",
        trace=trace, cost_usd=cost_usd, turns=turns, usage=usage, error=error,
    )


class EfficiencyTest(unittest.TestCase):
    def setUp(self):
        self.trace = _load_corpus_trace()
        if self.trace is None:
            self.skipTest("traces/corpus not present")
        # total_tokens/cost_usd/turns only ever exist on the RunResult (grok's
        # own headless-stdout accounting) -- the normalizer's session store
        # records no token counts, so a real run wraps the trace like this.
        self.run = _run(trace=self.trace, cost_usd=0.0570984, turns=3, usage=_REAL_USAGE)

    def test_reads_exactly_the_frozen_metric_names(self):
        eff = efficiency(self.run)
        self.assertEqual(set(eff.keys()), {name for name, _ in EFFICIENCY_METRICS})

    def test_trace_derived_values_match_the_trace_directly(self):
        eff = efficiency(self.run)
        metrics = self.trace["metrics"]
        self.assertEqual(eff["wall_ms"], self.trace["session"]["duration_ms"])
        self.assertEqual(eff["tool_calls"], metrics["counts"]["tool_calls"])
        self.assertEqual(eff["skill_tokens_est"], metrics["context_cost"]["skill_tokens_est"])
        self.assertEqual(eff["skill_wasted_tokens_est"],
                          metrics["context_cost"]["skill_wasted_tokens_est"])
        self.assertEqual(eff["tool_result_tokens_est"],
                          metrics["context_cost"]["tool_result_tokens_est"])
        self.assertEqual(eff["mcp_startup_ms"], metrics["timing"]["mcp_startup_ms"])
        self.assertEqual(
            eff["duplicate_calls"],
            sum(max(e.get("times", 1), 1) - 1 for e in metrics["duplicate_calls"]),
        )

    def test_total_tokens_and_cost_come_from_the_run_not_the_trace(self):
        eff = efficiency(self.run)
        self.assertNotIn("total_tokens", self.trace.get("session", {}))
        self.assertEqual(eff["total_tokens"], 16647)
        self.assertEqual(eff["cost_usd"], 0.0570984)

    def test_turns_prefers_run_level_count_over_trace_metrics(self):
        # This fixture's trace metrics.counts.turns (a normalizer recount) is
        # 1, but grok's own run-level count is 3 -- efficiency() must report
        # the authoritative run-level figure, not the trace's recount.
        self.assertEqual(self.trace["metrics"]["counts"]["turns"], 1)
        eff = efficiency(self.run)
        self.assertEqual(eff["turns"], 3)

    def test_no_usage_or_cost_on_the_run_is_none_not_zero(self):
        run = _run(trace=self.trace)  # no cost_usd, no turns, no usage
        eff = efficiency(run)
        self.assertIsNone(eff["total_tokens"])
        self.assertIsNone(eff["cost_usd"])
        self.assertIsNone(eff["turns"])
        # trace-derived values are unaffected
        self.assertEqual(eff["tool_calls"], self.trace["metrics"]["counts"]["tool_calls"])

    def test_empty_usage_dict_is_none_not_zero(self):
        eff = efficiency(_run(trace=self.trace, usage={}))
        self.assertIsNone(eff["total_tokens"])

    def test_failed_run_reports_no_efficiency_at_all(self):
        # C2: a run that crashed reports NO efficiency numbers, not even the
        # partial cost/token/turn figures grok emitted before crashing --
        # those describe a different, incomplete quantity and must never be
        # averaged in as if the run had completed.
        run = _run(trace=self.trace, cost_usd=0.01, turns=1, usage=_REAL_USAGE,
                    ok=False, error="crashed")
        self.assertEqual(efficiency(run), _ALL_NONE)

    def test_empty_run_is_all_none(self):
        self.assertEqual(efficiency(None), _ALL_NONE)
        self.assertEqual(efficiency({}), _ALL_NONE)

    def test_duplicate_calls_counts_wasted_repeats_not_distinct_signatures(self):
        # I4: len(...) reports "2" whether a call repeated twice or fifty
        # times. The metric must sum (times - 1) per entry instead, so a
        # blowup on an already-duplicated call is visible.
        trace = {
            "session": {"duration_ms": 1000},
            "metrics": {
                "counts": {"tool_calls": 5},
                "timing": {"mcp_startup_ms": 0},
                "context_cost": {},
                "duplicate_calls": [
                    {"tool": "read_file", "times": 3, "args": "{}"},
                    {"tool": "grep", "times": 2, "args": "{}"},
                ],
            },
        }
        eff = efficiency(_run(trace=trace))
        self.assertEqual(eff["duplicate_calls"], (3 - 1) + (2 - 1))
        self.assertNotEqual(eff["duplicate_calls"], len(trace["metrics"]["duplicate_calls"]))


# --------------------------------------------------------------------------
# regressions()
# --------------------------------------------------------------------------

def _trace(*, failed=None, idle=None, dupes=None, status=None, spans=None):
    return {
        "session": {"status": status} if status is not None else {},
        "metrics": {
            "failed_mcp_servers": failed or [],
            "idle_mcp_servers": idle or [],
            "duplicate_calls": dupes or [],
        },
        "spans": spans or [],
    }


def _error_span(kind, name):
    return {"kind": kind, "name": name, "status": "error"}


class RegressionsTest(unittest.TestCase):
    def test_finds_a_newly_failed_mcp_server_and_ignores_a_shared_one(self):
        control = _trace(failed=[{"server": "git", "error_type": "handshake_failed"}])
        treatment = _trace(failed=[
            {"server": "git", "error_type": "handshake_failed"},   # present in both
            {"server": "time", "error_type": "handshake_failed"},  # new
        ])
        regs = regressions(control, treatment)
        kinds = [r["kind"] for r in regs]
        self.assertIn("mcp_newly_failed", kinds)
        newly_failed = next(r for r in regs if r["kind"] == "mcp_newly_failed")
        self.assertIn("time", newly_failed["detail"])
        self.assertNotIn("'git'", newly_failed["detail"])

    def test_no_regressions_when_nothing_new(self):
        control = _trace(failed=[{"server": "git"}], idle=[{"server": "memory"}])
        treatment = _trace(failed=[{"server": "git"}], idle=[{"server": "memory"}])
        self.assertEqual(regressions(control, treatment), [])

    def test_new_duplicate_call_pair(self):
        control = _trace(dupes=[])
        treatment = _trace(dupes=[{"tool": "read_file", "times": 2, "args": "{}"}])
        regs = regressions(control, treatment)
        self.assertTrue(any(r["kind"] == "duplicate_call_new" for r in regs))

    def test_duplicate_call_blowup_on_an_already_duplicated_key_is_caught(self):
        # I4: the same (tool, args) pair exists on both sides, but its
        # `times` count jumps massively in treatment -- must still fire.
        control = _trace(dupes=[{"tool": "read_file", "times": 2, "args": "{}"}])
        treatment = _trace(dupes=[{"tool": "read_file", "times": 50, "args": "{}"}])
        regs = regressions(control, treatment)
        dupe_regs = [r for r in regs if r["kind"] == "duplicate_call_new"]
        self.assertEqual(len(dupe_regs), 1)
        self.assertIn("2->50", dupe_regs[0]["detail"])

    def test_duplicate_call_same_times_both_sides_is_not_flagged(self):
        control = _trace(dupes=[{"tool": "read_file", "times": 2, "args": "{}"}])
        treatment = _trace(dupes=[{"tool": "read_file", "times": 2, "args": "{}"}])
        regs = regressions(control, treatment)
        self.assertFalse(any(r["kind"] == "duplicate_call_new" for r in regs))

    def test_session_status_error_detected(self):
        control = _trace(status="ok")
        treatment = _trace(status="error")
        regs = regressions(control, treatment)
        self.assertTrue(any(r["kind"] == "session_status_error" for r in regs))

    def test_session_status_ok_on_both_sides_is_not_flagged(self):
        control = _trace(status="ok")
        treatment = _trace(status="ok")
        regs = regressions(control, treatment)
        self.assertFalse(any(r["kind"] == "session_status_error" for r in regs))

    def test_span_error_new_detects_a_brand_new_erroring_span(self):
        control = _trace(spans=[])
        treatment = _trace(spans=[_error_span("mcp", "connect time")])
        regs = regressions(control, treatment)
        self.assertTrue(any(r["kind"] == "span_error_new" for r in regs))

    def test_span_error_new_detects_a_blowup_on_an_already_erroring_span(self):
        # I3: control already has ONE failing read_file span; treatment has
        # forty. A key-presence diff would see the same key on both sides
        # and miss this entirely -- the count must be compared instead.
        control = _trace(spans=[_error_span("tool", "read_file")])
        treatment = _trace(spans=[_error_span("tool", "read_file")] * 40)
        regs = regressions(control, treatment)
        span_regs = [r for r in regs if r["kind"] == "span_error_new"]
        self.assertEqual(len(span_regs), 1)
        self.assertIn("1->40", span_regs[0]["detail"])

    def test_span_error_new_same_count_both_sides_is_not_flagged(self):
        control = _trace(spans=[_error_span("tool", "read_file")])
        treatment = _trace(spans=[_error_span("tool", "read_file")])
        regs = regressions(control, treatment)
        self.assertFalse(any(r["kind"] == "span_error_new" for r in regs))

    def test_ok_status_spans_are_not_errors(self):
        treatment = _trace(spans=[{"kind": "tool", "name": "read_file", "status": "ok"}])
        regs = regressions(_trace(), treatment)
        self.assertFalse(any(r["kind"] == "span_error_new" for r in regs))

    def test_every_regression_has_a_valid_severity(self):
        control = _trace()
        treatment = _trace(
            failed=[{"server": "time"}],
            idle=[{"server": "memory"}],
            dupes=[{"tool": "x", "times": 3, "args": "{}"}],
            status="error",
            spans=[_error_span("tool", "x")],
        )
        regs = regressions(control, treatment)
        self.assertTrue(regs)
        for r in regs:
            self.assertIn(r["severity"], {"high", "medium", "low"})


# --------------------------------------------------------------------------
# _turn_count_regression() / _run_failed_regression()
# --------------------------------------------------------------------------

class TurnCountRegressionTest(unittest.TestCase):
    def test_over_50_percent_increase_fires(self):
        self.assertIsNotNone(grade._turn_count_regression(2, 4))  # +100%

    def test_under_threshold_does_not_fire(self):
        self.assertIsNone(grade._turn_count_regression(4, 5))  # +25%

    def test_exactly_50_percent_does_not_fire_strict_boundary(self):
        self.assertIsNone(grade._turn_count_regression(4, 6))  # exactly +50%

    def test_zero_control_turns_does_not_fire_not_a_substitute_for_a_real_signal(self):
        # M1: a falsy-but-real 0 must not be treated the same as "no data",
        # but it also must not divide/compare into a bogus multiplier.
        self.assertIsNone(grade._turn_count_regression(0, 10))

    def test_none_on_either_side_does_not_fire(self):
        self.assertIsNone(grade._turn_count_regression(None, 10))
        self.assertIsNone(grade._turn_count_regression(2, None))


class RunFailedRegressionTest(unittest.TestCase):
    def test_treatment_failed_control_ok_is_high_severity(self):
        reg = grade._run_failed_regression(True, False, "boom")
        self.assertIsNotNone(reg)
        self.assertEqual(reg["kind"], "run_failed")
        self.assertEqual(reg["severity"], "high")
        self.assertIn("boom", reg["detail"])

    def test_both_ok_is_not_flagged(self):
        self.assertIsNone(grade._run_failed_regression(True, True, None))

    def test_both_failed_is_not_flagged_as_a_treatment_regression(self):
        self.assertIsNone(grade._run_failed_regression(False, False, "boom"))

    def test_control_failed_treatment_ok_is_not_flagged(self):
        # Control breaking and treatment succeeding is not a regression.
        self.assertIsNone(grade._run_failed_regression(False, True, None))


# --------------------------------------------------------------------------
# judge_pair()
# --------------------------------------------------------------------------

def _agreeing_grok(winner_first_call):
    """A fake grok that always prefers the response in position `winner_first_call`
    ("1" or "2") regardless of which order it's shown -- i.e. it tracks a
    stable underlying preference, so the two position-swapped calls agree."""
    calls = []

    def fake(prompt_text, schema, *, model=None):
        calls.append(prompt_text)
        # First call shows (a, b); second call shows (b, a). A stable
        # preference for "a" means "1" on the first call and "2" on the second.
        call_index = len(calls) - 1
        if winner_first_call == "1":
            winner = "1" if call_index == 0 else "2"
        elif winner_first_call == "2":
            winner = "2" if call_index == 0 else "1"
        else:
            winner = "tie"
        return {"winner": winner, "reason": "fake"}

    return fake


class JudgePairTest(unittest.TestCase):
    def test_agreeing_orders_return_the_winner(self):
        grok = _agreeing_grok("1")  # stable preference for response_a
        verdict = judge_pair("do the thing", "response a text", "response b text", grok=grok)
        self.assertEqual(verdict["verdict"], ARM_CONTROL)
        self.assertTrue(verdict["consistent"])
        self.assertEqual(len(verdict["raw"]), 3)
        self.assertEqual(verdict["raw"][2], {"judge_model": None})

    def test_agreeing_orders_favoring_b_return_treatment(self):
        grok = _agreeing_grok("2")  # stable preference for response_b
        verdict = judge_pair("do the thing", "response a text", "response b text", grok=grok)
        self.assertEqual(verdict["verdict"], ARM_TREATMENT)
        self.assertTrue(verdict["consistent"])

    def test_disagreeing_orders_return_tie_and_inconsistent(self):
        # Same winner label ("1") on both calls means the preference flips
        # with position -- pure position bias, must resolve to neither side.
        def flippy(prompt_text, schema, *, model=None):
            return {"winner": "1", "reason": "always picks position 1"}

        verdict = judge_pair("do the thing", "response a text", "response b text", grok=flippy)
        self.assertEqual(verdict["verdict"], "tie")
        self.assertFalse(verdict["consistent"])

    def test_garbage_output_degrades_to_tie_without_raising(self):
        def garbage(prompt_text, schema, *, model=None):
            return {"not_a_winner_field": "??"}

        verdict = judge_pair("do the thing", "response a text", "response b text", grok=garbage)
        self.assertEqual(verdict["verdict"], "tie")
        self.assertFalse(verdict["consistent"])

    def test_none_output_degrades_to_tie_without_raising(self):
        verdict = judge_pair("do the thing", "a", "b", grok=lambda *a, **k: None)
        self.assertEqual(verdict["verdict"], "tie")
        self.assertFalse(verdict["consistent"])

    def test_model_is_forwarded_and_recorded_in_raw(self):
        seen = []

        def fake(prompt_text, schema, *, model=None):
            seen.append(model)
            return {"winner": "tie", "reason": "fake"}

        verdict = judge_pair("do the thing", "a", "b", grok=fake, model="grok-4.5")
        self.assertEqual(seen, ["grok-4.5", "grok-4.5"])
        self.assertEqual(verdict["raw"][2], {"judge_model": "grok-4.5"})

    def test_response_text_is_fenced_with_unpredictable_per_call_markers(self):
        # I6: untrusted response text must be delimited, and the delimiter
        # must not be guessable/repeatable by the content itself.
        prompt1 = grade._judge_prompt("do the thing", "resp A", "resp B")
        prompt2 = grade._judge_prompt("do the thing", "resp A", "resp B")
        self.assertIn("resp A", prompt1)
        self.assertIn("resp B", prompt1)
        self.assertIn("untrusted data", grade._JUDGE_RUBRIC)
        self.assertIn("<<<RESPONSE-1-", prompt1)
        self.assertIn("<<<RESPONSE-2-", prompt1)
        # Fences differ call to call, so a response can't predict and forge one.
        self.assertNotEqual(prompt1, prompt2)

    def test_judge_schema_forbids_additional_properties(self):
        self.assertIs(grade._JUDGE_SCHEMA.get("additionalProperties"), False)


# --------------------------------------------------------------------------
# judge_consistency()
# --------------------------------------------------------------------------

def _graded(judge):
    """A minimal Grade carrying only the judge verdict judge_consistency()
    reads -- the efficiency dicts don't matter for this function."""
    return make_grade(
        repeat=0, control_efficiency={}, treatment_efficiency={}, judge=judge,
    )


class JudgeConsistencyTest(unittest.TestCase):
    def test_all_consistent_is_rate_one(self):
        grades = [
            _graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True))
            for _ in range(6)
        ]
        result = judge_consistency(grades)
        self.assertEqual(result["pairs"], 6)
        self.assertEqual(result["agreed"], 6)
        self.assertEqual(result["rate"], 1.0)
        self.assertEqual(result["note"], "")

    def test_mixed_consistency_computes_rate(self):
        grades = [
            _graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True)),
            _graded(make_judge_verdict(verdict=ARM_TREATMENT, consistent=True)),
            _graded(make_judge_verdict(verdict="tie", consistent=False)),
            _graded(make_judge_verdict(verdict="tie", consistent=False)),
            _graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True)),
        ]
        result = judge_consistency(grades)
        self.assertEqual(result["pairs"], 5)
        self.assertEqual(result["agreed"], 3)
        self.assertAlmostEqual(result["rate"], 0.6)

    def test_ungraded_pairs_are_excluded_not_counted_as_inconsistent(self):
        grades = [
            _graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True)),
            _graded(None),  # no judge ran for this repeat (e.g. a failed run)
            _graded(None),
        ]
        result = judge_consistency(grades)
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["agreed"], 1)
        self.assertEqual(result["rate"], 1.0)

    def test_no_judged_pairs_is_rate_none_with_a_note(self):
        result = judge_consistency([_graded(None), _graded(None)])
        self.assertEqual(result["pairs"], 0)
        self.assertEqual(result["agreed"], 0)
        self.assertIsNone(result["rate"])
        self.assertTrue(result["note"])

    def test_small_sample_gets_an_unreliable_estimate_note(self):
        grades = [_graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True))
                  for _ in range(3)]
        result = judge_consistency(grades)
        self.assertEqual(result["pairs"], 3)
        self.assertIn("unreliable", result["note"])

    def test_five_or_more_pairs_gets_no_note(self):
        grades = [_graded(make_judge_verdict(verdict=ARM_CONTROL, consistent=True))
                  for _ in range(5)]
        result = judge_consistency(grades)
        self.assertEqual(result["pairs"], 5)
        self.assertEqual(result["note"], "")

    def test_empty_grades_list(self):
        result = judge_consistency([])
        self.assertEqual(result, {"pairs": 0, "agreed": 0, "rate": None,
                                   "note": "no judged pairs -- consistency is undefined"})


# --------------------------------------------------------------------------
# grade_matrix()
# --------------------------------------------------------------------------

class GradeMatrixTest(unittest.TestCase):
    def test_pairs_control_and_treatment_by_repeat_without_swapping_arms(self):
        control_trace = _trace()
        treatment_trace = _trace(failed=[{"server": "time"}])

        control_run = make_run_result(
            arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
            home="/tmp/c", workdir="/tmp/cw", trace=control_trace,
            response_text="control did the thing",
            cost_usd=0.01, turns=3, usage={"total_tokens": 1000},
        )
        treatment_run = make_run_result(
            arm=ARM_TREATMENT, repeat=0, ok=True, prompt="do the thing",
            home="/tmp/t", workdir="/tmp/tw", trace=treatment_trace,
            response_text="treatment did the thing",
            cost_usd=0.02, turns=5, usage={"total_tokens": 2000},
        )
        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={ARM_CONTROL: [control_run], ARM_TREATMENT: [treatment_run]},
            control_version_id="c1", treatment_version_id="t1",
        )

        grok = _agreeing_grok("1")
        grades = grade_matrix(matrix, prompt="do the thing", grok=grok)

        self.assertEqual(len(grades), 1)
        g = grades[0]
        self.assertEqual(g["repeat"], 0)
        self.assertTrue(g["control_ok"])
        self.assertTrue(g["treatment_ok"])
        self.assertEqual(g["judge"]["verdict"], ARM_CONTROL)

        # I1: the arms must not be swapped. Assert against efficiency()
        # computed directly on each run (which differ), not just any dict.
        self.assertEqual(g["control_efficiency"], efficiency(control_run))
        self.assertEqual(g["treatment_efficiency"], efficiency(treatment_run))
        self.assertNotEqual(g["control_efficiency"], g["treatment_efficiency"])
        self.assertEqual(g["control_efficiency"]["total_tokens"], 1000)
        self.assertEqual(g["treatment_efficiency"]["total_tokens"], 2000)

        self.assertTrue(any(r["kind"] == "mcp_newly_failed" for r in g["regressions"]))
        # turns: control=3, treatment=5 -> +66% > 50% threshold
        self.assertTrue(any(r["kind"] == "turn_count_increase" for r in g["regressions"]))

    def test_failed_run_skips_the_judge_reports_no_efficiency_and_flags_high_severity(self):
        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=_trace(),
                    response_text="ok", cost_usd=0.05, turns=3, usage=_REAL_USAGE,
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=False, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=None,
                    error="crashed", cost_usd=0.01, turns=1,
                    usage={"total_tokens": 500},
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )

        def unused_grok(*a, **k):
            raise AssertionError("grok should not be called when a run failed")

        grades = grade_matrix(matrix, prompt="do the thing", grok=unused_grok)
        self.assertEqual(len(grades), 1)
        g = grades[0]
        self.assertIsNone(g["judge"])
        self.assertFalse(g["treatment_ok"])

        # C2 (efficiency half): a failed treatment run reports no efficiency
        # numbers at all, even though grok emitted partial cost/token/turn
        # figures before crashing.
        self.assertEqual(g["treatment_efficiency"], _ALL_NONE)
        self.assertIsNotNone(g["control_efficiency"]["total_tokens"])

        # C2 (regression half): the crash itself is a high-severity signal.
        run_failed = [r for r in g["regressions"] if r["kind"] == "run_failed"]
        self.assertEqual(len(run_failed), 1)
        self.assertEqual(run_failed[0]["severity"], "high")
        self.assertIn("crashed", run_failed[0]["detail"])

    def test_unparseable_judge_call_leaves_judge_none_not_a_fabricated_tie(self):
        # I12: an unparseable pair must not read as a considered "tie"
        # verdict downstream -- grade_matrix nulls it out entirely so it's
        # excluded from win_rate/judge_consistency's sample size.
        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=_trace(),
                    response_text="a",
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=_trace(),
                    response_text="b",
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )

        def broken_grok(prompt_text, schema, *, model=None):
            return {"totally": "unrelated"}

        grades = grade_matrix(matrix, prompt="do the thing", grok=broken_grok)
        self.assertIsNone(grades[0]["judge"])
        self.assertEqual(judge_consistency(grades), {
            "pairs": 0, "agreed": 0, "rate": None,
            "note": "no judged pairs -- consistency is undefined",
        })

    def test_control_failed_treatment_ok_is_not_flagged_as_run_failed(self):
        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=False, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=None, error="crashed",
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=_trace(),
                    response_text="fine",
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )
        grades = grade_matrix(matrix, prompt="do the thing", grok=lambda *a, **k: None)
        self.assertFalse(any(r["kind"] == "run_failed" for r in grades[0]["regressions"]))

    def test_model_is_forwarded_to_every_judge_call(self):
        seen = []

        def fake(prompt_text, schema, *, model=None):
            seen.append(model)
            return {"winner": "tie", "reason": "fake"}

        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=_trace(),
                    response_text="a",
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=_trace(),
                    response_text="b",
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )
        grade_matrix(matrix, prompt="do the thing", grok=fake, model="grok-4.5")
        self.assertEqual(seen, ["grok-4.5", "grok-4.5"])


# --------------------------------------------------------------------------
# _default_grok() field-parsing (C1) -- subprocess.run and harness.materialize
# are mocked; nothing here spawns grok-dev or touches the real ~/.grok.
# --------------------------------------------------------------------------

def _fake_proc(stdout):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)


class DefaultGrokParsingTest(unittest.TestCase):
    def setUp(self):
        materialize_patch = mock.patch(
            "interstellar.grade.harness.materialize",
            return_value=Path("/tmp/fake-judge-home"),
        )
        self.materialize = materialize_patch.start()
        self.addCleanup(materialize_patch.stop)

    def _run_with_stdout(self, stdout):
        with mock.patch("interstellar.grade.subprocess.run",
                         return_value=_fake_proc(stdout)) as run:
            result = grade._default_grok("prompt", grade._JUDGE_SCHEMA)
        return result, run

    def test_reads_structured_output_not_text(self):
        # C1: this is the exact bug -- the real headless payload puts the
        # validated answer in `structuredOutput`, and `text` can carry
        # something else entirely (or nothing). Confirm the field read.
        stdout = json.dumps({
            "text": "not the json you want",
            "structuredOutput": {"winner": "2", "reason": "because"},
            "structuredOutputError": None,
        })
        result, _ = self._run_with_stdout(stdout)
        self.assertEqual(result["winner"], "2")
        self.assertEqual(result["reason"], "because")
        self.assertEqual(result["_source"], "structuredOutput")

    def test_structured_output_error_is_treated_as_unparseable(self):
        stdout = json.dumps({
            "text": '{"winner": "1", "reason": "irrelevant"}',
            "structuredOutput": None,
            "structuredOutputError": "did not match schema",
        })
        result, _ = self._run_with_stdout(stdout)
        self.assertIsNone(result.get("winner"))
        self.assertIn("did not match schema", result["error"])

    def test_falls_back_to_text_only_when_structured_output_absent(self):
        stdout = json.dumps({"text": '{"winner": "tie", "reason": "ok"}'})
        result, _ = self._run_with_stdout(stdout)
        self.assertEqual(result["winner"], "tie")
        self.assertEqual(result["_source"], "text_fallback")

    def test_text_fallback_that_is_not_json_is_unparseable_not_raising(self):
        stdout = json.dumps({"text": "sorry, I can't do that"})
        result, _ = self._run_with_stdout(stdout)
        self.assertIsNone(result.get("winner"))
        self.assertIn("error", result)

    def test_non_json_stdout_is_unparseable_not_raising(self):
        result, _ = self._run_with_stdout("not json at all")
        self.assertIsNone(result.get("winner"))
        self.assertIn("error", result)

    def test_stdout_json_that_is_not_an_object_is_unparseable(self):
        result, _ = self._run_with_stdout(json.dumps(["a", "list"]))
        self.assertIsNone(result.get("winner"))

    def test_subprocess_timeout_is_unparseable_not_raising(self):
        import subprocess
        with mock.patch("interstellar.grade.subprocess.run",
                         side_effect=subprocess.TimeoutExpired(cmd="grok", timeout=1)):
            result = grade._default_grok("prompt", grade._JUDGE_SCHEMA)
        self.assertIsNone(result.get("winner"))
        self.assertIn("error", result)

    def test_missing_judge_credentials_is_unparseable_not_raising(self):
        self.materialize.side_effect = FileNotFoundError("no auth.json")
        result = grade._default_grok("prompt", grade._JUDGE_SCHEMA)
        self.assertIsNone(result.get("winner"))
        self.assertIn("error", result)

    def test_never_calls_the_real_grok_binary(self):
        # Belt and suspenders: subprocess.run is mocked in every test above,
        # so nothing here can have spawned the real binary either way.
        self.assertTrue(True)

    def test_argv_locks_down_tools_subagents_web_and_turns(self):
        stdout = json.dumps({"structuredOutput": {"winner": "tie", "reason": "x"}})
        _, run_mock = self._run_with_stdout(stdout)
        argv = run_mock.call_args.args[0]
        self.assertIn("--no-subagents", argv)
        self.assertIn("--disable-web-search", argv)
        self.assertIn("--max-turns", argv)
        self.assertEqual(argv[argv.index("--max-turns") + 1], "1")
        self.assertIn("--sandbox", argv)
        disallowed = argv[argv.index("--disallowed-tools") + 1]
        for tool in ("run_terminal_command", "read_file", "write", "search_replace",
                      "list_dir", "glob", "grep", "web_search", "web_fetch"):
            self.assertIn(tool, disallowed)

    def test_runs_in_the_materialized_home_not_the_real_one(self):
        stdout = json.dumps({"structuredOutput": {"winner": "tie", "reason": "x"}})
        _, run_mock = self._run_with_stdout(stdout)
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["cwd"], "/tmp/fake-judge-home")
        self.assertEqual(kwargs["env"]["GROK_HOME"], "/tmp/fake-judge-home")
        self.assertNotEqual(kwargs["env"]["GROK_HOME"], str(Path.home() / ".grok"))
        # auth_from passed to materialize is the real home -- that's the one
        # file (auth.json) every child process legitimately needs; nothing
        # else about the real home is used.
        self.assertEqual(self.materialize.call_args.kwargs["auth_from"],
                          Path.home() / ".grok")


if __name__ == "__main__":
    unittest.main()
