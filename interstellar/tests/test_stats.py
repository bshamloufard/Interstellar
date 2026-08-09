"""Tests for interstellar.stats: pure functions, no I/O, no grok-dev.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import random
import unittest

from interstellar import stats
from interstellar.grade import judge_consistency
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFECT_TOOL_CALLS,
    EFFECT_WALL_MS,
    EFFICIENCY_METRICS,
    make_grade,
    make_judge_verdict,
    make_regression,
)


def _grade(repeat, *, control_tokens=None, treatment_tokens=None, verdict=None,
           regressions=None, control_ok=True, treatment_ok=True):
    """Build a make_grade() with only total_tokens populated, for tests that
    don't care about the other nine EFFICIENCY_METRICS."""
    control_eff = {name: None for name, _ in EFFICIENCY_METRICS}
    treatment_eff = {name: None for name, _ in EFFICIENCY_METRICS}
    control_eff["total_tokens"] = control_tokens
    treatment_eff["total_tokens"] = treatment_tokens
    judge = None
    if verdict is not None:
        judge = make_judge_verdict(verdict=verdict, consistent=True, reason="test")
    return make_grade(
        repeat=repeat,
        control_efficiency=control_eff,
        treatment_efficiency=treatment_eff,
        judge=judge,
        regressions=regressions or [],
        control_ok=control_ok,
        treatment_ok=treatment_ok,
    )


# --- paired_bootstrap --------------------------------------------------------

class PairedBootstrapTest(unittest.TestCase):
    def test_insufficient_at_n0(self):
        r = stats.paired_bootstrap([])
        self.assertEqual(r, {"mean": None, "lo": None, "hi": None, "level": 0.95,
                              "n": 0, "note": "insufficient_samples",
                              "distinct_resamples": None})

    def test_insufficient_at_n1(self):
        r = stats.paired_bootstrap([5.0])
        self.assertEqual(r["mean"], 5.0)
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])
        self.assertEqual(r["note"], "insufficient_samples")
        self.assertEqual(r["distinct_resamples"], 1)  # C(1,1) = 1

    def test_insufficient_below_min_samples_for_ci(self):
        # n=9 is still below MIN_SAMPLES_FOR_CI (10) -- no CI at all, even
        # though it's more samples than the old (wrong) threshold of 3.
        deltas = [float(i) for i in range(9)]
        r = stats.paired_bootstrap(deltas, iters=500, seed=1)
        self.assertEqual(r["n"], 9)
        self.assertIsNotNone(r["mean"])
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])
        self.assertEqual(r["note"], "insufficient_samples")
        self.assertEqual(r["distinct_resamples"], stats._distinct_resamples(9))

    def test_low_confidence_band(self):
        # n=12 is in [MIN_SAMPLES_FOR_CI, LOW_CONFIDENCE_CI) -- CI reported,
        # flagged low_confidence.
        deltas = [float(i) for i in range(12)]
        r = stats.paired_bootstrap(deltas, iters=1000, seed=1)
        self.assertEqual(r["n"], 12)
        self.assertIsNotNone(r["lo"])
        self.assertIsNotNone(r["hi"])
        self.assertEqual(r["note"], "low_confidence")

    def test_full_confidence_band_no_note(self):
        deltas = list(range(1, 31))  # n=30 >= LOW_CONFIDENCE_CI, mean = 15.5
        r = stats.paired_bootstrap(deltas, iters=2000, seed=7)
        self.assertIsNone(r["note"])
        self.assertLess(r["lo"], r["mean"])
        self.assertGreater(r["hi"], r["mean"])
        self.assertAlmostEqual(r["mean"], 15.5)

    def test_seed_stability(self):
        deltas = [float(i) for i in range(15)]
        r1 = stats.paired_bootstrap(deltas, iters=500, seed=42)
        r2 = stats.paired_bootstrap(deltas, iters=500, seed=42)
        self.assertEqual(r1, r2)

    def test_bootstrap_coverage_on_known_mean(self):
        """Run many independent synthetic experiments drawn from a
        distribution with a KNOWN population mean (0, from symmetric
        Uniform(-1, 1)), at n=20 (the first n this module treats as
        "reasonably solid" -- LOW_CONFIDENCE_CI), and check that roughly
        95% of the resulting 95% CIs contain that true mean. Statistical,
        not exact: tolerance is generous (80-100%) to avoid flakiness while
        still catching a badly broken implementation."""
        true_mean = 0.0
        gen = random.Random(99)
        trials = 150
        hits = 0
        for i in range(trials):
            deltas = [gen.uniform(-1, 1) for _ in range(20)]
            r = stats.paired_bootstrap(deltas, iters=400, seed=i)
            self.assertIsNone(r["note"])  # n=20 should be full-confidence
            if r["lo"] <= true_mean <= r["hi"]:
                hits += 1
        coverage = hits / trials
        self.assertGreaterEqual(coverage, 0.80, f"coverage too low: {coverage}")
        self.assertLessEqual(coverage, 1.0)


class DistinctResamplesTest(unittest.TestCase):
    def test_matches_hand_computed_values(self):
        # C(2n-1, n): n=3 -> C(5,3)=10, n=5 -> C(9,5)=126 (research-methods.md sec 1)
        self.assertEqual(stats._distinct_resamples(3), 10)
        self.assertEqual(stats._distinct_resamples(5), 126)

    def test_zero_and_negative_is_none(self):
        self.assertIsNone(stats._distinct_resamples(0))


# --- win_rate -----------------------------------------------------------------

class WinRateTest(unittest.TestCase):
    def test_n0_insufficient(self):
        r = stats.win_rate([])
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["note"], "insufficient_samples")
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])

    def test_n1_insufficient_but_rate_reported(self):
        grades = [_grade(0, verdict=ARM_TREATMENT)]
        r = stats.win_rate(grades)
        self.assertEqual(r["n"], 1)
        self.assertEqual(r["wins"], 1)
        self.assertEqual(r["rate"], 1.0)  # real point estimate, not suppressed
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])
        self.assertEqual(r["note"], "insufficient_samples")

    def test_below_min_samples_for_ci_still_insufficient(self):
        # n=8, below MIN_SAMPLES_FOR_CI (10) -- no CI even though > old floor of 3.
        grades = [_grade(i, verdict=ARM_TREATMENT) for i in range(8)]
        r = stats.win_rate(grades, iters=200, seed=0)
        self.assertEqual(r["n"], 8)
        self.assertEqual(r["rate"], 1.0)
        self.assertIsNone(r["lo"])
        self.assertEqual(r["note"], "insufficient_samples")

    def test_grades_without_judge_are_excluded(self):
        grades = [_grade(0, verdict=None, control_ok=False),
                  _grade(1, verdict=ARM_TREATMENT),
                  _grade(2, verdict=ARM_TREATMENT),
                  _grade(3, verdict=ARM_CONTROL)]
        r = stats.win_rate(grades)
        self.assertEqual(r["n"], 3)  # the judge=None grade is dropped

    def test_all_ties_reports_tie_count_not_dropped(self):
        grades = [_grade(i, verdict="tie") for i in range(12)]
        r = stats.win_rate(grades, iters=200, seed=0)
        self.assertEqual(r["n"], 12)
        self.assertEqual(r["ties"], 12)
        self.assertEqual(r["wins"], 0)
        self.assertEqual(r["losses"], 0)
        self.assertIsNone(r["rate"])
        self.assertEqual(r["note"], "no_decided_pairs")

    def test_low_confidence_win_rate_ci(self):
        # n=12 (in the 10-19 band): CI reported, flagged low_confidence.
        grades = ([_grade(i, verdict=ARM_TREATMENT) for i in range(10)]
                  + [_grade(10, verdict=ARM_CONTROL), _grade(11, verdict=ARM_CONTROL)])
        r = stats.win_rate(grades, iters=1000, seed=0)
        self.assertEqual(r["wins"], 10)
        self.assertEqual(r["losses"], 2)
        self.assertAlmostEqual(r["rate"], 10 / 12)
        self.assertIsNotNone(r["lo"])
        self.assertIsNotNone(r["hi"])
        self.assertEqual(r["note"], "low_confidence")
        self.assertLessEqual(r["lo"], r["rate"])
        self.assertGreaterEqual(r["hi"], r["rate"])

    def test_full_confidence_win_rate_ci(self):
        grades = ([_grade(i, verdict=ARM_TREATMENT) for i in range(18)]
                  + [_grade(18, verdict=ARM_CONTROL), _grade(19, verdict=ARM_CONTROL)])
        r = stats.win_rate(grades, iters=1000, seed=0)
        self.assertEqual(r["n"], 20)
        self.assertIsNone(r["note"])


# --- sign_test ------------------------------------------------------------

class SignTestTest(unittest.TestCase):
    def test_hand_computed_p_value(self):
        # deltas: 3 positive, 1 negative, 1 zero (excluded) -> n_nonzero=4.
        # k = min(pos, neg) = min(3, 1) = 1.
        # Exact two-sided binomial p under H0 p=0.5:
        #   P(X<=1) for X~Binomial(4, 0.5) = [C(4,0)+C(4,1)] / 2^4
        #                                   = (1 + 4) / 16 = 5/16 = 0.3125
        #   two-sided p = 2 * 0.3125 = 0.625
        deltas = [2.0, 1.0, 3.0, -1.0, 0.0]
        r = stats.sign_test(deltas)
        self.assertEqual(r["n_nonzero"], 4)
        self.assertEqual(r["direction"], "positive")
        self.assertAlmostEqual(r["p"], 0.625)

    def test_all_zero_is_none(self):
        r = stats.sign_test([0.0, 0.0])
        self.assertEqual(r, {"p": None, "n_nonzero": 0, "direction": "none"})

    def test_symmetric_split_p_is_1(self):
        r = stats.sign_test([1.0, 1.0, -1.0, -1.0])
        self.assertEqual(r["direction"], "tie")
        self.assertAlmostEqual(r["p"], 1.0)


# --- permutation_test -------------------------------------------------------

class PermutationTestTest(unittest.TestCase):
    def test_min_achievable_p_at_n3_equal_magnitude(self):
        # 3 equal-magnitude positive deltas: the all-positive sign pattern
        # is the unique most extreme (and its all-negative mirror), 2 out
        # of 2**3=8 patterns -> exact p = 2/8 = 0.25, matching the sign
        # test's floor and research-methods.md sec 1's stated floor exactly.
        deltas = [1.0, 1.0, 1.0]
        r = stats.permutation_test(deltas, seed=0)
        self.assertTrue(r["exact"])
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["n_nonzero"], 3)
        self.assertAlmostEqual(r["p"], 0.25)
        self.assertAlmostEqual(r["min_achievable_p"], 0.25)
        self.assertEqual(r["direction"], "positive")

    def test_min_achievable_p_at_n5(self):
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = stats.permutation_test(deltas, seed=0)
        self.assertAlmostEqual(r["min_achievable_p"], 2 * (0.5 ** 5))  # 0.0625
        # unanimous direction -> observed IS the most extreme pattern -> p == floor
        self.assertAlmostEqual(r["p"], r["min_achievable_p"])

    def test_neither_n3_nor_n5_can_reach_significance(self):
        for n in (3, 5):
            deltas = [float(i + 1) for i in range(n)]
            r = stats.permutation_test(deltas, seed=0)
            self.assertGreaterEqual(r["p"], 0.05)

    def test_all_zero_is_none(self):
        r = stats.permutation_test([0.0, 0.0, 0.0])
        self.assertEqual(r["p"], 1.0)
        self.assertEqual(r["min_achievable_p"], 1.0)
        self.assertEqual(r["direction"], "none")

    def test_empty_is_none(self):
        r = stats.permutation_test([])
        self.assertIsNone(r["p"])
        self.assertIsNone(r["min_achievable_p"])

    def test_sampled_path_used_above_max_exact_n(self):
        deltas = [1.0] * 8
        r = stats.permutation_test(deltas, seed=0, max_exact_n=5, samples=500)
        self.assertFalse(r["exact"])
        self.assertEqual(r["n"], 8)

    def test_sampled_path_seed_stable(self):
        deltas = [float(i) - 3 for i in range(8)]
        r1 = stats.permutation_test(deltas, seed=7, max_exact_n=5, samples=500)
        r2 = stats.permutation_test(deltas, seed=7, max_exact_n=5, samples=500)
        self.assertEqual(r1, r2)


# --- pass_k -----------------------------------------------------------------

class PassKTest(unittest.TestCase):
    def test_correct_estimator_not_naive_plugin(self):
        # n=3, c=2, k=3 (default k=n): correct C(2,3)/C(3,3) = 0/1 = 0.0,
        # NOT the naive (2/3)**3 ~= 0.296 the old implementation gave.
        r = stats.pass_k([True, True, False])
        self.assertEqual(r["value"], 0.0)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["c"], 2)
        self.assertEqual(r["k"], 3)
        self.assertTrue(r["degenerate"])
        self.assertIn("n_equals_k", r["note"])

    def test_all_success_is_degenerate_one(self):
        r = stats.pass_k([True, True, True])
        self.assertEqual(r["value"], 1.0)
        self.assertTrue(r["degenerate"])

    def test_all_failure_is_degenerate_zero(self):
        r = stats.pass_k([False, False])
        self.assertEqual(r["value"], 0.0)
        self.assertTrue(r["degenerate"])

    def test_explicit_smaller_k_is_not_degenerate(self):
        # n=3, c=2, k=1: C(2,1)/C(3,1) = 2/3, a real (non-degenerate) rate.
        r = stats.pass_k([True, True, False], k=1)
        self.assertAlmostEqual(r["value"], 2 / 3)
        self.assertFalse(r["degenerate"])
        self.assertIsNone(r["note"])

    def test_k_exceeds_n_is_invalid(self):
        r = stats.pass_k([True, True], k=5)
        self.assertIsNone(r["value"])
        self.assertEqual(r["note"], "k_exceeds_n")

    def test_empty_is_insufficient(self):
        r = stats.pass_k([])
        self.assertIsNone(r["value"])
        self.assertEqual(r["note"], "insufficient_samples")


# --- efficiency_deltas --------------------------------------------------------

class EfficiencyDeltasTest(unittest.TestCase):
    def test_n1_every_metric_insufficient_shape(self):
        grades = [_grade(0, control_tokens=100, treatment_tokens=80)]
        r = stats.efficiency_deltas(grades)
        entry = r["total_tokens"]
        self.assertEqual(entry["n"], 1)
        self.assertEqual(entry["control_median"], 100)
        self.assertEqual(entry["treatment_median"], 80)
        self.assertEqual(entry["abs_delta"], 20)
        self.assertAlmostEqual(entry["pct_delta"], 0.2)
        self.assertEqual(entry["bootstrap"]["note"], "insufficient_samples")
        self.assertIsNone(entry["bootstrap"]["lo"])
        # every other metric had no data at all -> n=0, also insufficient
        for name, _ in EFFICIENCY_METRICS:
            if name == "total_tokens":
                continue
            self.assertEqual(r[name]["n"], 0)
            self.assertEqual(r[name]["bootstrap"]["note"], "insufficient_samples")

    def test_zero_control_median_guards_pct_delta(self):
        grades = [_grade(i, control_tokens=0, treatment_tokens=5) for i in range(3)]
        r = stats.efficiency_deltas(grades)
        entry = r["total_tokens"]
        self.assertEqual(entry["control_median"], 0)
        self.assertIsNone(entry["pct_delta"])
        self.assertIsNotNone(entry["abs_delta"])

    def test_missing_one_side_drops_pair_only_for_that_metric(self):
        grades = [
            _grade(0, control_tokens=100, treatment_tokens=90),
            _grade(1, control_tokens=100, treatment_tokens=None),  # dropped
            _grade(2, control_tokens=100, treatment_tokens=80),
        ]
        r = stats.efficiency_deltas(grades)
        self.assertEqual(r["total_tokens"]["n"], 2)

    def test_improvement_has_positive_delta_and_credible_ci(self):
        # n=12 grades -> in the low-confidence-but-real CI band.
        grades = [_grade(i, control_tokens=200, treatment_tokens=100) for i in range(12)]
        r = stats.efficiency_deltas(grades, iters=1000, seed=0)
        entry = r["total_tokens"]
        self.assertEqual(entry["abs_delta"], 100)
        self.assertAlmostEqual(entry["pct_delta"], 0.5)
        self.assertGreater(entry["bootstrap"]["lo"], 0)
        self.assertEqual(entry["bootstrap"]["note"], "low_confidence")

    def test_below_ci_floor_still_has_point_estimates(self):
        grades = [_grade(i, control_tokens=200, treatment_tokens=100) for i in range(4)]
        r = stats.efficiency_deltas(grades)
        entry = r["total_tokens"]
        self.assertEqual(entry["abs_delta"], 100)
        self.assertEqual(entry["bootstrap"]["note"], "insufficient_samples")
        self.assertIsNone(entry["bootstrap"]["lo"])


# --- summarize --------------------------------------------------------------

def _summarize_grade(repeat, *, wall_ms_control=None, wall_ms_treatment=None,
                      verdict=None, regressions=None, control_ok=True, treatment_ok=True):
    """Build a make_grade() with only wall_ms populated -- summarize's
    default test effect is EFFECT_WALL_MS -> "wall_ms"."""
    control_eff = {name: None for name, _ in EFFICIENCY_METRICS}
    treatment_eff = {name: None for name, _ in EFFICIENCY_METRICS}
    control_eff["wall_ms"] = wall_ms_control
    treatment_eff["wall_ms"] = wall_ms_treatment
    judge = None
    if verdict is not None:
        judge = make_judge_verdict(verdict=verdict, consistent=True, reason="test")
    return make_grade(
        repeat=repeat,
        control_efficiency=control_eff,
        treatment_efficiency=treatment_eff,
        judge=judge,
        regressions=regressions or [],
        control_ok=control_ok,
        treatment_ok=treatment_ok,
    )


class SummarizeTest(unittest.TestCase):
    def test_shape_and_field_sourcing(self):
        grades = [
            _summarize_grade(0, wall_ms_control=1000, wall_ms_treatment=800, verdict=ARM_TREATMENT),
            _summarize_grade(1, wall_ms_control=1000, wall_ms_treatment=900, verdict=ARM_TREATMENT),
            _summarize_grade(2, wall_ms_control=1000, wall_ms_treatment=950, verdict="tie",
                              regressions=[make_regression(kind="duplicate_call_new",
                                                            severity="medium", detail="dup")]),
        ]
        summary = stats.summarize(grades, primary_effect=EFFECT_WALL_MS, seed=0)

        self.assertEqual(
            set(summary),
            {"n", "win_rate", "permutation", "efficiency", "pass_k",
             "regressions", "judge_consistency", "primary_effect"},
        )
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["primary_effect"], EFFECT_WALL_MS)

        self.assertEqual(summary["win_rate"], stats.win_rate(grades, seed=0))
        self.assertEqual(summary["efficiency"], stats.efficiency_deltas(grades, seed=0))

        expected_deltas = stats._metric_deltas(grades, "wall_ms")  # EFFECT_WALL_MS -> "wall_ms"
        self.assertEqual(summary["permutation"], stats.permutation_test(expected_deltas, seed=0))

        # pass_k reads treatment_ok, not control_ok -- all three grades ran fine.
        self.assertEqual(summary["pass_k"], stats.pass_k([True, True, True]))

        self.assertEqual(len(summary["regressions"]), 1)
        self.assertEqual(summary["regressions"][0]["kind"], "duplicate_call_new")

        self.assertEqual(summary["judge_consistency"], judge_consistency(grades))

    def test_unknown_effect_raises_key_error(self):
        with self.assertRaises(KeyError):
            stats.summarize([], primary_effect="not_a_real_effect")

    def test_pass_k_uses_treatment_ok_not_control_ok(self):
        grades = [
            _summarize_grade(0, wall_ms_control=100, wall_ms_treatment=90, control_ok=False, treatment_ok=True),
            _summarize_grade(1, wall_ms_control=100, wall_ms_treatment=90, control_ok=True, treatment_ok=False),
        ]
        summary = stats.summarize(grades, primary_effect=EFFECT_WALL_MS, seed=0)
        self.assertEqual(summary["pass_k"], stats.pass_k([True, False]))

    def test_output_feeds_directly_into_gate(self):
        grades = [_summarize_grade(i, wall_ms_control=1000, wall_ms_treatment=700,
                                    verdict=ARM_TREATMENT)
                  for i in range(12)]
        summary = stats.summarize(grades, primary_effect=EFFECT_WALL_MS, seed=0)
        r = stats.gate(summary)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["directional"], "favorable")


# --- gate -----------------------------------------------------------------

def _win_rate_stub(n, wins, losses, ties, *, lo=None, hi=None, note=None):
    rate = wins / (wins + losses) if (wins + losses) > 0 else None
    return {"wins": wins, "ties": ties, "losses": losses, "n": n, "rate": rate,
            "lo": lo, "hi": hi, "note": note}


def _efficiency_stub(primary="wall_ms", mean=30.0, lo=10.0, hi=50.0,
                      pct_delta=0.30, n=12):
    """Build an efficiency_deltas()-shaped dict with `primary` as the
    targeted metric. `total_tokens` is always present too (the gate's
    token-blowup safety net reads it unconditionally) -- when `primary`
    IS "total_tokens" they are the same entry, not two independent ones,
    so the caller's mean/lo/hi/pct_delta actually take effect."""
    result = {
        primary: {
            "lower_is_better": True, "n": n,
            "control_median": 200, "treatment_median": 140,
            "abs_delta": 60, "pct_delta": pct_delta,
            "bootstrap": {"mean": mean, "lo": lo, "hi": hi, "level": 0.95, "n": n, "note": None},
        },
    }
    if primary != "total_tokens":
        result["total_tokens"] = {
            "lower_is_better": True, "n": n,
            "control_median": 1000, "treatment_median": 700,
            "abs_delta": 300, "pct_delta": 0.30,
            "bootstrap": {"mean": 300.0, "lo": 50.0, "hi": 550.0, "level": 0.95, "n": n, "note": None},
        }
    return result


class GateTest(unittest.TestCase):
    def test_accepts_clean_improvement_at_n_ge_10(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": _efficiency_stub(),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertTrue(r["accepted"])
        self.assertFalse(r["provisional"])
        self.assertEqual(r["directional"], "favorable")
        self.assertTrue(r["reasons"])

    def test_rejects_on_high_severity_regression(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [make_regression(kind="mcp_newly_failed", severity="high",
                                             detail="mcp server X failed")],
            "efficiency": _efficiency_stub(),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("high-severity" in reason for reason in r["reasons"]))

    def test_rejects_on_token_blowup_past_threshold(self):
        efficiency = _efficiency_stub()
        efficiency["total_tokens"] = {
            "lower_is_better": True, "n": 12,
            "control_median": 1000, "treatment_median": 1300,
            "abs_delta": -300, "pct_delta": -0.30,
            "bootstrap": {"mean": -300.0, "lo": -500.0, "hi": -100.0, "level": 0.95, "n": 12, "note": None},
        }
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": efficiency,
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s, max_token_regression=0.10)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("total_tokens regressed" in reason for reason in r["reasons"]))

    def test_rejects_when_quality_regresses_at_n_ge_10(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=3, losses=7, ties=2, lo=0.1, hi=0.6),
            "regressions": [],
            "efficiency": _efficiency_stub(mean=-30.0, lo=-50.0, hi=-10.0, pct_delta=-0.15),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["directional"], "unfavorable")

    def test_rejects_when_primary_ci_insufficient_at_n_ge_10(self):
        efficiency = _efficiency_stub()
        efficiency["wall_ms"]["bootstrap"] = {
            "mean": 300.0, "lo": None, "hi": None, "level": 0.95, "n": 12, "note": None,
        }
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": efficiency,
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])

    def test_rejects_when_no_primary_effect_declared(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": _efficiency_stub(),
            "primary_effect": None,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])

    def test_rejects_on_unrecognized_primary_effect(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": _efficiency_stub(),
            "primary_effect": "not_a_real_effect",
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("not a recognized EFFECT_* name" in reason for reason in r["reasons"]))

    def test_primary_effect_kwarg_overrides_summary_value(self):
        efficiency = _efficiency_stub(primary="tool_calls")
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": efficiency,
            "primary_effect": EFFECT_WALL_MS,  # not present in `efficiency` -> would fail if used
        }
        r = stats.gate(s, primary_effect=EFFECT_TOOL_CALLS)
        self.assertTrue(r["accepted"])

    def test_informational_reasons_present_when_available(self):
        s = {
            "win_rate": _win_rate_stub(12, wins=10, losses=0, ties=2),
            "regressions": [],
            "efficiency": _efficiency_stub(),
            "primary_effect": EFFECT_WALL_MS,
            "judge_consistency": {"pairs": 12, "agreed": 10, "rate": 10 / 12, "note": ""},
            "permutation": {"p": 0.25, "n": 3, "n_nonzero": 3, "observed_mean": 30.0,
                             "direction": "positive", "exact": True, "min_achievable_p": 0.25},
        }
        r = stats.gate(s)
        self.assertTrue(any("judge position-consistency: 83%" in reason for reason in r["reasons"]))
        self.assertTrue(any("permutation test on primary effect" in reason for reason in r["reasons"]))
        self.assertTrue(r["accepted"])  # informational reasons never affect accepted

    # --- the n<10 / n<5 sample-size-aware regimes (the point of the rework) ---

    def test_n_below_5_never_accepts_but_reports_directional(self):
        # k=3, unanimous treatment wins and a positive primary-metric point
        # estimate -- this is exactly the headline k=3 demo scenario.
        s = {
            "win_rate": _win_rate_stub(3, wins=3, losses=0, ties=0),
            "regressions": [],
            "efficiency": _efficiency_stub(n=3, mean=30.0, lo=None, hi=None),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertFalse(r["provisional"])
        self.assertEqual(r["directional"], "favorable")
        self.assertTrue(any("insufficient_samples_for_acceptance" in reason for reason in r["reasons"]))

    def test_n_below_5_reports_unfavorable_direction_on_a_loss(self):
        s = {
            "win_rate": _win_rate_stub(3, wins=1, losses=2, ties=0),
            "regressions": [],
            "efficiency": _efficiency_stub(n=3, mean=-10.0, lo=None, hi=None),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["directional"], "unfavorable")

    def test_n_5_to_9_accepts_provisionally_on_zero_losses_and_improvement(self):
        s = {
            "win_rate": _win_rate_stub(5, wins=4, losses=0, ties=1),
            "regressions": [],
            "efficiency": _efficiency_stub(n=5, mean=30.0, lo=None, hi=None),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertTrue(r["accepted"])
        self.assertTrue(r["provisional"])
        self.assertEqual(r["directional"], "favorable")
        self.assertTrue(any("provisional quality pass" in reason for reason in r["reasons"]))

    def test_n_5_to_9_rejects_on_any_loss_no_ci_fallback(self):
        s = {
            "win_rate": _win_rate_stub(7, wins=5, losses=1, ties=1),
            "regressions": [],
            "efficiency": _efficiency_stub(n=7, mean=30.0, lo=None, hi=None),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(r["provisional"])

    def test_n_5_to_9_rejects_when_primary_point_estimate_not_positive(self):
        s = {
            "win_rate": _win_rate_stub(6, wins=6, losses=0, ties=0),
            "regressions": [],
            "efficiency": _efficiency_stub(n=6, mean=-5.0, lo=None, hi=None),
            "primary_effect": EFFECT_WALL_MS,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("did not improve" in reason for reason in r["reasons"]))


if __name__ == "__main__":
    unittest.main()
