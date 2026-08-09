"""Tests for interstellar.grade. No test here spawns grok-dev: judge_pair
takes an injectable `grok` callable and every test supplies a fake one."""
from __future__ import annotations

import glob
import json
import unittest
from pathlib import Path

from interstellar.grade import efficiency, grade_matrix, judge_pair, regressions
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFICIENCY_METRICS,
    make_replay_matrix,
    make_run_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "traces" / "corpus"


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


def _run(*, trace=None, cost_usd=None, turns=None, usage=None, ok=True):
    return make_run_result(
        arm=ARM_CONTROL, repeat=0, ok=ok, prompt="do the thing",
        home="/tmp/h", workdir="/tmp/w",
        trace=trace, cost_usd=cost_usd, turns=turns, usage=usage,
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
        self.assertEqual(eff["duplicate_calls"], len(metrics["duplicate_calls"]))

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

    def test_failed_run_with_no_trace_still_reports_run_level_metrics(self):
        # A run that crashed before producing a trace can still have cost/
        # token/turn accounting from grok's headless stdout.
        run = _run(trace=None, cost_usd=0.01, turns=1, usage=_REAL_USAGE, ok=False)
        eff = efficiency(run)
        self.assertEqual(eff["total_tokens"], 16647)
        self.assertEqual(eff["cost_usd"], 0.01)
        self.assertEqual(eff["turns"], 1)
        self.assertIsNone(eff["wall_ms"])
        self.assertIsNone(eff["tool_calls"])
        self.assertIsNone(eff["duplicate_calls"])

    def test_empty_run_is_all_none(self):
        self.assertEqual(efficiency(None), {name: None for name, _ in EFFICIENCY_METRICS})
        self.assertEqual(efficiency({}), {name: None for name, _ in EFFICIENCY_METRICS})


# --------------------------------------------------------------------------
# regressions()
# --------------------------------------------------------------------------

def _trace(*, failed=None, idle=None, dupes=None, turns=1, status=None, spans=None):
    return {
        "session": {"status": status} if status is not None else {},
        "metrics": {
            "counts": {"turns": turns},
            "failed_mcp_servers": failed or [],
            "idle_mcp_servers": idle or [],
            "duplicate_calls": dupes or [],
        },
        "spans": spans or [],
    }


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

    def test_turn_count_increase_over_50_percent(self):
        control = _trace(turns=2)
        treatment = _trace(turns=4)  # +100%
        regs = regressions(control, treatment)
        self.assertTrue(any(r["kind"] == "turn_count_increase" for r in regs))

    def test_turn_count_increase_under_threshold_is_not_flagged(self):
        control = _trace(turns=4)
        treatment = _trace(turns=5)  # +25%
        regs = regressions(control, treatment)
        self.assertFalse(any(r["kind"] == "turn_count_increase" for r in regs))

    def test_every_regression_has_a_valid_severity(self):
        control = _trace()
        treatment = _trace(
            failed=[{"server": "time"}],
            idle=[{"server": "memory"}],
            dupes=[{"tool": "x", "args": "{}"}],
            turns=10,
        )
        regs = regressions(control, treatment)
        self.assertTrue(regs)
        for r in regs:
            self.assertIn(r["severity"], {"high", "medium", "low"})


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
        self.assertEqual(len(verdict["raw"]), 2)

    def test_agreeing_orders_favoring_b_return_treatment(self):
        grok = _agreeing_grok("2")  # stable preference for response_b
        verdict = judge_pair("do the thing", "response a text", "response b text", grok=grok)
        self.assertEqual(verdict["verdict"], ARM_TREATMENT)
        self.assertTrue(verdict["consistent"])

    def test_disagreeing_orders_return_tie_and_inconsistent(self):
        # Same winner label ("1") on both calls means the preference flips
        # with position -- pure position bias, must resolve to neither side.
        calls = {"n": 0}

        def flippy(prompt_text, schema, *, model=None):
            calls["n"] += 1
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


# --------------------------------------------------------------------------
# grade_matrix()
# --------------------------------------------------------------------------

class GradeMatrixTest(unittest.TestCase):
    def test_pairs_control_and_treatment_by_repeat(self):
        control_trace = _trace(turns=2)
        treatment_trace = _trace(turns=2, failed=[{"server": "time"}])

        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=control_trace,
                    response_text="control did the thing",
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=treatment_trace,
                    response_text="treatment did the thing",
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )

        grok = _agreeing_grok("1")
        grades = grade_matrix(matrix, prompt="do the thing", grok=grok)

        self.assertEqual(len(grades), 1)
        grade = grades[0]
        self.assertEqual(grade["repeat"], 0)
        self.assertTrue(grade["control_ok"])
        self.assertTrue(grade["treatment_ok"])
        self.assertEqual(grade["judge"]["verdict"], ARM_CONTROL)
        self.assertTrue(any(r["kind"] == "mcp_newly_failed" for r in grade["regressions"]))

    def test_failed_run_skips_the_judge_but_still_grades(self):
        matrix = make_replay_matrix(
            prompt="do the thing", k=1,
            arms={
                ARM_CONTROL: [make_run_result(
                    arm=ARM_CONTROL, repeat=0, ok=True, prompt="do the thing",
                    home="/tmp/c", workdir="/tmp/cw", trace=_trace(),
                    response_text="ok",
                )],
                ARM_TREATMENT: [make_run_result(
                    arm=ARM_TREATMENT, repeat=0, ok=False, prompt="do the thing",
                    home="/tmp/t", workdir="/tmp/tw", trace=None,
                    error="crashed",
                )],
            },
            control_version_id="c1", treatment_version_id="t1",
        )

        def unused_grok(*a, **k):
            raise AssertionError("grok should not be called when a run failed")

        grades = grade_matrix(matrix, prompt="do the thing", grok=unused_grok)
        self.assertEqual(len(grades), 1)
        self.assertIsNone(grades[0]["judge"])
        self.assertFalse(grades[0]["treatment_ok"])


if __name__ == "__main__":
    unittest.main()
