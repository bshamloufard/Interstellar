"""Tests for interstellar.stats: pure functions, no I/O, no grok-dev.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import random
import unittest

from interstellar import stats
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
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
                              "n": 0, "note": "insufficient_samples"})

    def test_insufficient_at_n1(self):
        r = stats.paired_bootstrap([5.0])
        self.assertEqual(r["mean"], 5.0)
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])
        self.assertEqual(r["note"], "insufficient_samples")

    def test_insufficient_at_n2(self):
        r = stats.paired_bootstrap([1.0, 3.0])
        self.assertEqual(r["mean"], 2.0)
        self.assertIsNone(r["lo"])
        self.assertIsNone(r["hi"])
        self.assertEqual(r["note"], "insufficient_samples")

    def test_seed_stability(self):
        deltas = [1.0, -2.0, 3.5, 0.0, 4.0, -1.0]
        r1 = stats.paired_bootstrap(deltas, iters=500, seed=42)
        r2 = stats.paired_bootstrap(deltas, iters=500, seed=42)
        self.assertEqual(r1, r2)

    def test_different_seed_may_differ_but_same_mean(self):
        deltas = [1.0, -2.0, 3.5, 0.0, 4.0, -1.0]
        r1 = stats.paired_bootstrap(deltas, iters=500, seed=1)
        r2 = stats.paired_bootstrap(deltas, iters=500, seed=2)
        self.assertEqual(r1["mean"], r2["mean"])  # mean is not resampled
        self.assertIsNotNone(r1["lo"])
        self.assertIsNotNone(r2["lo"])

    def test_ci_brackets_sample_mean(self):
        deltas = list(range(1, 31))  # mean = 15.5
        r = stats.paired_bootstrap(deltas, iters=2000, seed=7)
        self.assertLess(r["lo"], r["mean"])
        self.assertGreater(r["hi"], r["mean"])
        self.assertAlmostEqual(r["mean"], 15.5)

    def test_bootstrap_coverage_on_known_mean(self):
        """Run many independent synthetic experiments drawn from a
        distribution with a KNOWN population mean (0, from symmetric
        Uniform(-1, 1)), and check that roughly 95% of the resulting 95%
        CIs contain that true mean. This is a statistical, not exact,
        check -- percentile bootstrap is known to under-cover somewhat, so
        the tolerance band is generous (80-100%) to avoid a flaky test
        while still catching a badly broken implementation (e.g. one that
        returns a near-zero-width or wildly miscentered interval)."""
        true_mean = 0.0
        gen = random.Random(99)
        trials = 150
        hits = 0
        for i in range(trials):
            deltas = [gen.uniform(-1, 1) for _ in range(20)]
            r = stats.paired_bootstrap(deltas, iters=400, seed=i)
            if r["lo"] <= true_mean <= r["hi"]:
                hits += 1
        coverage = hits / trials
        self.assertGreaterEqual(coverage, 0.80, f"coverage too low: {coverage}")
        self.assertLessEqual(coverage, 1.0)


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

    def test_grades_without_judge_are_excluded(self):
        grades = [_grade(0, verdict=None, control_ok=False),
                  _grade(1, verdict=ARM_TREATMENT),
                  _grade(2, verdict=ARM_TREATMENT),
                  _grade(3, verdict=ARM_CONTROL)]
        r = stats.win_rate(grades)
        self.assertEqual(r["n"], 3)  # the judge=None grade is dropped

    def test_all_ties_reports_tie_count_not_dropped(self):
        grades = [_grade(i, verdict="tie") for i in range(5)]
        r = stats.win_rate(grades)
        self.assertEqual(r["n"], 5)
        self.assertEqual(r["ties"], 5)
        self.assertEqual(r["wins"], 0)
        self.assertEqual(r["losses"], 0)
        self.assertIsNone(r["rate"])
        self.assertEqual(r["note"], "no_decided_pairs")

    def test_clean_treatment_win_rate(self):
        grades = ([_grade(i, verdict=ARM_TREATMENT) for i in range(4)]
                  + [_grade(4, verdict=ARM_CONTROL)])
        r = stats.win_rate(grades, iters=1000, seed=0)
        self.assertEqual(r["wins"], 4)
        self.assertEqual(r["losses"], 1)
        self.assertEqual(r["rate"], 0.8)
        self.assertIsNotNone(r["lo"])
        self.assertIsNotNone(r["hi"])
        self.assertLessEqual(r["lo"], r["rate"])
        self.assertGreaterEqual(r["hi"], r["rate"])


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

    def test_all_positive_is_significant_at_reasonable_n(self):
        r = stats.sign_test([1.0] * 6)
        self.assertEqual(r["direction"], "positive")
        # k = min(6, 0) = 0 -> p = 2 * C(6,0) * 0.5**6 = 2/64 = 0.03125
        self.assertAlmostEqual(r["p"], 2 / 64)


# --- pass_k -----------------------------------------------------------------

class PassKTest(unittest.TestCase):
    def test_all_success(self):
        self.assertAlmostEqual(stats.pass_k([True, True, True]), 1.0)

    def test_all_failure(self):
        self.assertAlmostEqual(stats.pass_k([False, False]), 0.0)

    def test_mixed_defaults_k_to_len(self):
        # p = 2/3, k defaults to len == 3 -> (2/3)**3
        self.assertAlmostEqual(stats.pass_k([True, True, False]), (2 / 3) ** 3)

    def test_explicit_k_override(self):
        self.assertAlmostEqual(stats.pass_k([True, True, False], k=1), 2 / 3)

    def test_empty_is_zero(self):
        self.assertEqual(stats.pass_k([]), 0.0)


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
        grades = [_grade(i, control_tokens=200, treatment_tokens=100) for i in range(6)]
        r = stats.efficiency_deltas(grades, iters=1000, seed=0)
        entry = r["total_tokens"]
        self.assertEqual(entry["abs_delta"], 100)
        self.assertAlmostEqual(entry["pct_delta"], 0.5)
        self.assertGreater(entry["bootstrap"]["lo"], 0)


# --- gate -----------------------------------------------------------------

def _clean_win_rate():
    return {"wins": 5, "ties": 0, "losses": 0, "n": 5, "rate": 1.0,
            "lo": None, "hi": None, "note": None}


def _clean_efficiency(primary="total_tokens", lo=10.0, hi=50.0, pct_delta=0.30):
    return {
        primary: {
            "lower_is_better": True, "n": 5,
            "control_median": 200, "treatment_median": 140,
            "abs_delta": 60, "pct_delta": pct_delta,
            "bootstrap": {"mean": 30.0, "lo": lo, "hi": hi, "level": 0.95, "n": 5, "note": None},
        },
        "total_tokens": {
            "lower_is_better": True, "n": 5,
            "control_median": 1000, "treatment_median": 700,
            "abs_delta": 300, "pct_delta": 0.30,
            "bootstrap": {"mean": 300.0, "lo": 50.0, "hi": 550.0, "level": 0.95, "n": 5, "note": None},
        },
    }


class GateTest(unittest.TestCase):
    def test_accepts_clean_improvement(self):
        s = {
            "win_rate": _clean_win_rate(),
            "regressions": [],
            "efficiency": _clean_efficiency(),
            "primary_metric": "total_tokens",
        }
        r = stats.gate(s)
        self.assertTrue(r["accepted"])
        self.assertTrue(r["reasons"])

    def test_rejects_on_high_severity_regression(self):
        s = {
            "win_rate": _clean_win_rate(),
            "regressions": [make_regression(kind="mcp_newly_failed", severity="high",
                                             detail="mcp server X failed")],
            "efficiency": _clean_efficiency(),
            "primary_metric": "total_tokens",
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("high-severity" in reason for reason in r["reasons"]))

    def test_rejects_on_token_blowup_past_threshold(self):
        efficiency = _clean_efficiency()
        # treatment uses far more tokens: control 1000 -> treatment 1300 (-30%)
        efficiency["total_tokens"] = {
            "lower_is_better": True, "n": 5,
            "control_median": 1000, "treatment_median": 1300,
            "abs_delta": -300, "pct_delta": -0.30,
            "bootstrap": {"mean": -300.0, "lo": -500.0, "hi": -100.0, "level": 0.95, "n": 5, "note": None},
        }
        s = {
            "win_rate": _clean_win_rate(),
            "regressions": [],
            "efficiency": efficiency,
            "primary_metric": "wall_ms",
        }
        r = stats.gate(s, max_token_regression=0.10)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("total_tokens regressed" in reason for reason in r["reasons"]))

    def test_rejects_when_quality_regresses(self):
        s = {
            "win_rate": {"wins": 1, "ties": 1, "losses": 3, "n": 5, "rate": 0.25,
                         "lo": 0.05, "hi": 0.6, "note": None},
            "regressions": [],
            "efficiency": _clean_efficiency(),
            "primary_metric": "total_tokens",
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])

    def test_rejects_when_primary_ci_insufficient(self):
        efficiency = _clean_efficiency()
        efficiency["total_tokens"]["bootstrap"] = {
            "mean": 300.0, "lo": None, "hi": None, "level": 0.95, "n": 1, "note": "insufficient_samples",
        }
        s = {
            "win_rate": _clean_win_rate(),
            "regressions": [],
            "efficiency": efficiency,
            "primary_metric": "total_tokens",
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])
        self.assertTrue(any("insufficient samples" in reason for reason in r["reasons"]))

    def test_rejects_when_no_primary_metric_declared(self):
        s = {
            "win_rate": _clean_win_rate(),
            "regressions": [],
            "efficiency": _clean_efficiency(),
            "primary_metric": None,
        }
        r = stats.gate(s)
        self.assertFalse(r["accepted"])

    def test_accepts_with_losses_zero_even_without_ci(self):
        # n too small for a win-rate CI, but 0 losses -- should still pass
        # check 1 (the "losses == 0" branch does not require lo).
        s = {
            "win_rate": {"wins": 1, "ties": 0, "losses": 0, "n": 1, "rate": 1.0,
                         "lo": None, "hi": None, "note": "insufficient_samples"},
            "regressions": [],
            "efficiency": _clean_efficiency(),
            "primary_metric": "total_tokens",
        }
        r = stats.gate(s)
        self.assertTrue(r["accepted"])


if __name__ == "__main__":
    unittest.main()
