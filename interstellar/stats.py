"""Paired statistics over `Grade` lists, and the acceptance gate.

This is the layer that turns k noisy paired runs (control vs. treatment,
same prompt, same isolation -- see types.make_replay_matrix) into a
defensible claim: "this patch helped, here is the interval, here is why we
believe it or don't." Every function here is pure and takes plain dicts in,
plain dicts out, so it is exhaustively testable without I/O or grok-dev.

The organizing rule, repeated at every function boundary: a confidence
interval computed from too few samples is not a weaker interval, it is a
fabricated one. See .superpowers/sdd/review-loop/research-methods.md for
the literature review this module's thresholds are drawn from; the short
version:

  - A bootstrap resample of n paired deltas has only C(2n-1, n) distinct
    possible values -- 10 at n=3, 126 at n=5. More `iters` reduces
    Monte-Carlo noise in *approximating* that resampling distribution; it
    adds zero information beyond the original n points. Below n=10
    (MIN_SAMPLES_FOR_CI) this module reports NO interval at all, only the
    point estimate. 10 <= n < 20 gets an interval flagged "low_confidence".
    n >= 20 is the regime the applied-stats N>=20 floor is describing.
  - The exact sign test (and the strictly-more-powerful exact permutation
    test) has a hard p-value floor of 2*(0.5)**n_nonzero: at n=3 the best
    possible two-sided p is 0.25, at n=5 it's 0.0625. Neither can ever cross
    p<0.05, which is why the acceptance gate does not lean on a p-value at
    k=3/5 -- it leans on zero-losses-with-a-sample-size-floor instead (see
    `gate`).
  - The naive `(success_rate)**k` is NOT the tau-bench pass^k estimator; the
    correct one is the unbiased combinatorial C(c,k)/C(n,k) (see `pass_k`).

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import math
import random
import statistics

from interstellar.grade import judge_consistency
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFECT_MCP_STARTUP_MS,
    EFFECT_SKILL_TOKENS,
    EFFECT_TOOL_CALLS,
    EFFECT_TOOL_RESULT_TOKENS,
    EFFECT_WALL_MS,
    EFFICIENCY_METRICS,
)

# A patch's claimed effect (types.EFFECT_*, what make_patch's expected_effect
# names) mapped to the EFFICIENCY_METRICS name that measures it. Lives here,
# once, so the CLI and the report reuse the same mapping instead of each
# inventing their own -- see `summarize` and `gate`.
EFFECT_TO_EFFICIENCY_METRIC = {
    EFFECT_SKILL_TOKENS: "skill_tokens_est",
    EFFECT_TOOL_CALLS: "tool_calls",
    EFFECT_MCP_STARTUP_MS: "mcp_startup_ms",
    EFFECT_TOOL_RESULT_TOKENS: "tool_result_tokens_est",
    EFFECT_WALL_MS: "wall_ms",
}

# Below this many paired samples, a percentile bootstrap CI is not reported
# at all -- see the module docstring and research-methods.md sec 1. Point
# estimates (mean, rate, medians) are still reported below this floor; they
# are real numbers, just uninterval-ed.
MIN_SAMPLES_FOR_CI = 10

# [MIN_SAMPLES_FOR_CI, LOW_CONFIDENCE_CI): a CI is reported but flagged
# low-confidence. n >= LOW_CONFIDENCE_CI is the "reasonably solid" regime.
LOW_CONFIDENCE_CI = 20

# n < 5 is a hard floor below which `gate` refuses to accept under any
# circumstance -- see research-methods.md sec 5.1 (zero losses at k=3 has
# ~12.5% false-accept probability for a truly neutral patch; ~3% at k=5).
MIN_SAMPLES_FOR_ACCEPT = 5

CI_LEVEL = 0.95
DEFAULT_ITERS = 10000
DEFAULT_SEED = 0

NOTE_INSUFFICIENT = "insufficient_samples"
NOTE_LOW_CONFIDENCE = "low_confidence"

# Enumerate all 2**n sign-flip patterns up to this n (2**20 ~= 1e6, seconds
# in pure Python); above it, sample patterns with a seeded RNG instead.
DEFAULT_MAX_EXACT_N = 20
DEFAULT_PERMUTATION_SAMPLES = 20000


def _percentile(sorted_values, pct):
    """Linear-interpolation percentile over an already-sorted sequence.

    Same convention as numpy's default ("linear") method, reimplemented here
    because numpy is not a dependency of this module.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _distinct_resamples(n):
    """Count of distinct possible bootstrap resamples of size n drawn with
    replacement from n items: C(2n-1, n) (stars-and-bars / multiset count).
    This is the number research-methods.md sec 1 uses to show that a k=3
    bootstrap has only 10 possible outcomes -- reported alongside every CI
    (or its absence) so a reader never has to take "not enough samples" on
    faith.
    """
    if n <= 0:
        return None
    return math.comb(2 * n - 1, n)


def paired_bootstrap(deltas, *, iters=DEFAULT_ITERS, seed=DEFAULT_SEED):
    """Percentile bootstrap CI on the mean of paired differences.

    Estimator: resample `deltas` with replacement (same size as the input)
    `iters` times using `random.Random(seed)`, take the mean of each
    resample, and report the 2.5th/97.5th percentiles of that distribution
    as a 95% CI (Efron's percentile bootstrap). This is NOT bias-corrected
    (BCa) -- it under-covers when the deltas are skewed or n is small.
    Seeded explicitly so the same input always produces the same interval;
    the report is regenerated and must not shimmer between runs.

    Sample-size gating (research-methods.md sec 1) -- this is the part that
    keeps this function from printing a fabricated interval:
      n == 0             -> mean=None, lo=hi=None, note="insufficient_samples"
      0 < n < MIN_SAMPLES_FOR_CI (10)
                         -> mean is a real computed number; lo=hi=None,
                            note="insufficient_samples". Below n=10 a
                            bootstrap resample has too few distinct outcomes
                            (see distinct_resamples) to characterize
                            coverage -- more `iters` would only manufacture
                            false precision, so no interval is computed at all.
      MIN_SAMPLES_FOR_CI <= n < LOW_CONFIDENCE_CI (10-19)
                         -> interval IS computed and reported, but flagged
                            note="low_confidence": usable, not authoritative.
      n >= LOW_CONFIDENCE_CI (20+)
                         -> full interval, note=None.
    `distinct_resamples` (C(2n-1, n)) is always reported so the "why" behind
    a missing or flagged interval is never just asserted.
    """
    n = len(deltas)
    result = {"mean": None, "lo": None, "hi": None, "level": CI_LEVEL, "n": n,
              "note": None, "distinct_resamples": _distinct_resamples(n)}
    if n == 0:
        result["note"] = NOTE_INSUFFICIENT
        return result

    result["mean"] = statistics.mean(deltas)
    if n < MIN_SAMPLES_FOR_CI:
        result["note"] = NOTE_INSUFFICIENT
        return result

    rng = random.Random(seed)
    resample_means = []
    for _ in range(iters):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.mean(resample))
    resample_means.sort()
    result["lo"] = _percentile(resample_means, 2.5)
    result["hi"] = _percentile(resample_means, 97.5)
    if n < LOW_CONFIDENCE_CI:
        result["note"] = NOTE_LOW_CONFIDENCE
    return result


def win_rate(grades, *, iters=DEFAULT_ITERS, seed=DEFAULT_SEED):
    """Treatment win rate over judged pairs, with a bootstrap CI.

    Reads `grade["judge"]["verdict"]` (ARM_CONTROL, ARM_TREATMENT, or
    "tie" -- see types.make_judge_verdict). Grades whose judge is None
    (one or both arms failed to run, so grade.py never called judge_pair)
    are excluded entirely: there is no pairwise preference to count.

    `n` is the number of judged grades considered. `wins`/`ties`/`losses`
    are counts from the treatment's point of view. `rate` is
    wins / (wins + losses) -- ties are excluded from the denominator on
    purpose: with ties dominant, `rate` can look extreme (or be undefined)
    on very few decided pairs while the tie count stays large, so `ties` is
    always reported alongside `rate`, never dropped.

    CI availability follows the same sample-size floor as paired_bootstrap
    (research-methods.md sec 1), using n = judged pairs -- the resampling
    unit here is the per-pair win/tie/loss label, exactly analogous to a
    paired delta:
      n < MIN_SAMPLES_FOR_CI (10)              -> note="insufficient_samples"
      MIN_SAMPLES_FOR_CI <= n < LOW_CONFIDENCE_CI (10-19)
                                                -> CI reported, note="low_confidence"
      n >= LOW_CONFIDENCE_CI (20+)              -> CI reported, note=None
    `rate` (the point estimate) is still reported below the CI floor -- it
    is a real number, just uninterval-ed. `distinct_resamples` is reported
    alongside, same as paired_bootstrap.

    decided == 0 (all ties) is its own note, "no_decided_pairs" -- there is
    nothing to rate regardless of n, but wins/ties/losses/n stay visible.
    """
    judged = [g for g in grades if g.get("judge")]
    n = len(judged)
    wins = sum(1 for g in judged if g["judge"]["verdict"] == ARM_TREATMENT)
    losses = sum(1 for g in judged if g["judge"]["verdict"] == ARM_CONTROL)
    ties = sum(1 for g in judged if g["judge"]["verdict"] == "tie")
    decided = wins + losses

    result = {
        "wins": wins, "ties": ties, "losses": losses, "n": n,
        "rate": (wins / decided) if decided > 0 else None,
        "lo": None, "hi": None, "note": None,
        "distinct_resamples": _distinct_resamples(n),
    }

    if n == 0:
        result["note"] = NOTE_INSUFFICIENT
        return result
    if n < MIN_SAMPLES_FOR_CI:
        result["note"] = NOTE_INSUFFICIENT
        return result
    if decided == 0:
        result["note"] = "no_decided_pairs"
        return result

    labels = []
    for g in judged:
        v = g["judge"]["verdict"]
        if v == ARM_TREATMENT:
            labels.append(1)
        elif v == ARM_CONTROL:
            labels.append(-1)
        else:
            labels.append(0)

    rng = random.Random(seed)
    resample_rates = []
    for _ in range(iters):
        sample = [labels[rng.randrange(n)] for _ in range(n)]
        w = sample.count(1)
        l = sample.count(-1)
        if w + l > 0:
            resample_rates.append(w / (w + l))

    if len(resample_rates) < 2:
        # Vanishingly unlikely given decided > 0 in the real sample, but a
        # resampling distribution with < 2 points is exactly the case this
        # module refuses to interval -- so refuse it here too.
        result["note"] = NOTE_INSUFFICIENT
        return result

    resample_rates.sort()
    result["lo"] = _percentile(resample_rates, 2.5)
    result["hi"] = _percentile(resample_rates, 97.5)
    if n < LOW_CONFIDENCE_CI:
        result["note"] = NOTE_LOW_CONFIDENCE
    return result


def sign_test(deltas):
    """Exact two-sided sign test on the direction of paired differences.

    Superseded as the primary inferential claim by `permutation_test`
    (strictly more powerful at the same exact finite-sample error control,
    since it uses magnitude, not just direction) -- kept because it's cheap,
    well known, and a useful cross-check. See research-methods.md sec 1.

    Zeros are excluded (no direction to score). Under the null that a
    positive or negative delta is equally likely, n_nonzero is
    Binomial(n_nonzero, 0.5); the two-sided p-value is
    2 * P(X <= min(pos, neg)) computed exactly with math.comb (no normal
    approximation), clamped at 1.0 for the symmetric case.

    Sign convention is the caller's: sign_test does not know whether a
    positive delta means "improved" or "regressed", only which sign is
    more common. `direction` is "positive", "negative", or "tie" (equal
    nonzero counts on both sides, including n_nonzero == 0 -> "none").

    Hard p-value floor: 2*(0.5)**n_nonzero (0.25 at n=3, 0.0625 at n=5) --
    this test structurally cannot reach p<0.05 below n_nonzero=6, which is
    the reason the acceptance gate does not use a p-value threshold at
    k=3/5. Underpowered samples are handled honestly, not specially: a
    small n_nonzero simply yields a p-value near 1, the correct signal.
    """
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"p": None, "n_nonzero": 0, "direction": "none"}

    pos = sum(1 for d in nonzero if d > 0)
    neg = n - pos
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)

    if pos > neg:
        direction = "positive"
    elif neg > pos:
        direction = "negative"
    else:
        direction = "tie"

    return {"p": p, "n_nonzero": n, "direction": direction}


def permutation_test(deltas, *, seed=DEFAULT_SEED, max_exact_n=DEFAULT_MAX_EXACT_N,
                      samples=DEFAULT_PERMUTATION_SAMPLES):
    """Exact sign-flip permutation test on paired differences -- the primary
    small-n inferential test in this module (research-methods.md sec 1),
    strictly more powerful than `sign_test` at the same exact finite-sample
    error control because it uses each delta's magnitude, not just its sign.

    Under the null that each delta's sign is an independent fair coin flip
    (the paired differences are symmetric about zero), every one of the
    2**n sign assignments to `deltas` is equally likely. The test statistic
    is the mean of the (possibly sign-flipped) deltas; the observed
    statistic is the sample mean (the all-original-signs assignment). The
    two-sided p-value is the fraction of the 2**n assignments whose |mean|
    is >= the observed |mean| -- this is exact, not a normal approximation,
    and needs no separate "times 2" step since the 2**n patterns already
    include every assignment's mirror image.

    n <= max_exact_n (default 20, i.e. 2**20 ~ 1e6 patterns): every pattern
    is enumerated, `exact=True`. Above that, `samples` sign patterns are
    drawn with `random.Random(seed)` instead and `exact=False` -- flagged
    explicitly in the result so a Monte-Carlo p-value is never mistaken for
    an exact one.

    `min_achievable_p` = 2*(0.5)**n_nonzero: the smallest two-sided p-value
    this test could possibly report given n_nonzero nonzero deltas, achieved
    only when every nonzero delta points the same direction (the single
    most-extreme sign assignment, plus its mirror, out of 2**n_nonzero
    equally-likely outcomes -- ties in magnitude can only raise this, never
    lower it). At n_nonzero=3 that floor is 0.25, at n_nonzero=5 it's
    0.0625 -- neither can ever cross 0.05, so a k=3/5 result should always
    be read alongside this floor, never as "no effect" on p alone. Zero
    nonzero deltas (all ties) makes significance structurally impossible,
    so min_achievable_p is 1.0 in that case.
    """
    n = len(deltas)
    n_nonzero = sum(1 for d in deltas if d != 0)
    if n == 0:
        return {"p": None, "n": 0, "n_nonzero": 0, "observed_mean": None,
                "direction": "none", "exact": True, "min_achievable_p": None}

    observed = statistics.mean(deltas)
    direction = "positive" if observed > 0 else ("negative" if observed < 0 else "none")
    min_achievable_p = 1.0 if n_nonzero == 0 else min(1.0, 2 * (0.5 ** n_nonzero))

    eps = 1e-9
    threshold = abs(observed) - eps

    if n <= max_exact_n:
        total = 1 << n
        count = 0
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(deltas):
                s += d if (mask >> i) & 1 else -d
            if abs(s / n) >= threshold:
                count += 1
        p = count / total
        exact = True
    else:
        rng = random.Random(seed)
        count = 0
        for _ in range(samples):
            s = 0.0
            for d in deltas:
                s += d if rng.random() < 0.5 else -d
            if abs(s / n) >= threshold:
                count += 1
        p = count / samples
        exact = False

    return {
        "p": min(1.0, p), "n": n, "n_nonzero": n_nonzero,
        "observed_mean": observed, "direction": direction,
        "exact": exact, "min_achievable_p": min_achievable_p,
    }


def pass_k(successes, *, k=None):
    """Unbiased tau-bench pass^k estimator: the probability that all k of k
    trials succeed, given n observed i.i.d. trials with c successes --
    pass^k = C(c, k) / C(n, k) (Yao et al., tau-bench, arXiv:2406.12045
    sec 3; the mirror image of the Codex/HumanEval unbiased pass@k
    estimator). This replaces an earlier `(c/n)**k` plug-in, which is NOT
    the same quantity and is systematically optimistic: e.g. n=3 trials,
    c=2 successes, k=3 -- naive gives (2/3)**3 ~= 0.296, but the honest
    answer is C(2,3)/C(3,3) = 0/1 = 0.0 ("not all 3 succeeded, full stop").

    Returns a dict (not a bare float): {value, n, c, k, degenerate, note}.
    `k` defaults to n (asking "would this same batch have all succeeded").

    `degenerate=True` when n == k: the estimator collapses to a 0/1
    indicator ("did all k of k trials succeed"), not a smooth probability
    -- this is the case for our v1 harness (one prompt, k trials, nothing
    to average over), and a bare 0.0 there is a clean fact, not a
    confident reliability *rate*; `note` says so explicitly so the report
    never mis-renders it as one. The estimator becomes a real rate only
    when aggregated across multiple tasks/prompts with n > k available.

    k > n is invalid (asking for more all-succeed trials than were run) --
    returns value=None, note="k_exceeds_n" rather than raising or dividing
    by a zero binomial coefficient.
    """
    n = len(successes)
    c = sum(1 for s in successes if s)
    if k is None:
        k = n
    result = {"value": None, "n": n, "c": c, "k": k, "degenerate": False, "note": None}
    if n == 0:
        result["note"] = NOTE_INSUFFICIENT
        return result
    if k > n:
        result["note"] = "k_exceeds_n"
        return result

    result["value"] = math.comb(c, k) / math.comb(n, k)
    if n == k:
        result["degenerate"] = True
        result["note"] = "n_equals_k: 0/1 indicator (all-or-nothing outcome), not a probability rate"
    return result


def _metric_pairs(grades, metric_name):
    """(control, treatment) pairs for one EFFICIENCY_METRICS name, only from
    grades where both sides report a non-None value for it -- the single
    pairing rule `efficiency_deltas` and `summarize` both use, so a metric
    is never paired one way for the report and another for the gate."""
    pairs = []
    for g in grades:
        c = (g.get("control_efficiency") or {}).get(metric_name)
        t = (g.get("treatment_efficiency") or {}).get(metric_name)
        if c is None or t is None:
            continue
        pairs.append((c, t))
    return pairs


def _metric_deltas(grades, metric_name):
    """Paired control-treatment deltas (control - treatment) for one
    EFFICIENCY_METRICS name -- see `_metric_pairs` for the pairing rule."""
    return [c - t for c, t in _metric_pairs(grades, metric_name)]


def efficiency_deltas(grades, *, iters=DEFAULT_ITERS, seed=DEFAULT_SEED):
    """Per-metric paired comparison over every name in types.EFFICIENCY_METRICS.

    Returns {metric_name: {...}} for all ten metrics, always -- a metric
    with zero usable pairs still appears, with n=0 and an
    "insufficient_samples" bootstrap note, so a missing metric in the
    report means "not measured", never "silently skipped".

    For each metric, a grade contributes a pair only if BOTH
    control_efficiency[metric] and treatment_efficiency[metric] are not
    None; a grade missing one side is dropped for that metric only; it can
    still contribute to other metrics.

    delta = control - treatment, so a positive delta means the treatment
    used less of the metric. Every metric in EFFICIENCY_METRICS today has
    lower_is_better=True, so positive == improvement; `lower_is_better` is
    carried through per metric from types.py so a future higher-is-better
    metric is not read upside down by a caller relying on sign alone.

    control_median / treatment_median use the ordinary sample median
    (statistics.median). abs_delta is control_median - treatment_median
    (delta of the medians, not the mean of per-pair deltas). pct_delta
    divides by control_median and is None when that median is exactly 0
    -- a percentage of zero is undefined, not reported as 0% or inf.

    `bootstrap` is `paired_bootstrap` called verbatim on the per-pair
    control-treatment deltas, so it inherits that function's sample-size
    gating exactly: no interval below n=10, flagged below n=20 -- see
    paired_bootstrap's docstring and research-methods.md sec 1.
    """
    result = {}
    for name, lower_is_better in EFFICIENCY_METRICS:
        pairs = _metric_pairs(grades, name)

        entry = {
            "lower_is_better": lower_is_better,
            "n": len(pairs),
            "control_median": None,
            "treatment_median": None,
            "abs_delta": None,
            "pct_delta": None,
            "bootstrap": None,
        }

        if not pairs:
            entry["bootstrap"] = paired_bootstrap([], iters=iters, seed=seed)
            result[name] = entry
            continue

        controls = [c for c, _t in pairs]
        treatments = [t for _c, t in pairs]
        deltas = [c - t for c, t in pairs]

        control_median = statistics.median(controls)
        treatment_median = statistics.median(treatments)
        entry["control_median"] = control_median
        entry["treatment_median"] = treatment_median
        entry["abs_delta"] = control_median - treatment_median
        entry["pct_delta"] = (
            (control_median - treatment_median) / control_median
            if control_median != 0 else None
        )
        entry["bootstrap"] = paired_bootstrap(deltas, iters=iters, seed=seed)
        result[name] = entry

    return result


def summarize(grades, *, primary_effect, seed=DEFAULT_SEED):
    """The whole statistics block for one patch. Everything the report
    renders is built from this one call, so the CLI, report.py, and `gate`
    all read the same numbers instead of each re-deriving them with
    slightly different pairing or seeding.

    `primary_effect` is one of types.ALL_EFFECTS -- the patch's
    `expected_effect` (types.make_patch), e.g. types.EFFECT_WALL_MS -- and
    is mapped to an EFFICIENCY_METRICS name via EFFECT_TO_EFFICIENCY_METRIC.
    Raises KeyError if it isn't one of the five known effects: an unmapped
    effect means there is nothing for the gate to check the patch's actual
    claim against, which should fail loudly here rather than silently
    produce an empty permutation test downstream.

    Returns:
      {
        "n": len(grades),   -- the paired replay count k (total repeats).
             NOT the same as win_rate's own "n" (judged pairs), which can
             be smaller when a run failed and grade.py skipped the judge --
             use summary["win_rate"]["n"] for the quality sample size.
        "win_rate": win_rate(grades, seed=seed),
        "permutation": permutation_test(<primary_effect's paired deltas>, seed=seed),
             -- the primary small-n inferential test (research-methods.md
             sec 1), run on the same per-pair control-treatment deltas
             `efficiency`'s bootstrap for that metric uses.
        "efficiency": efficiency_deltas(grades, seed=seed),
             -- {metric_name: {lower_is_better, n, control_median,
             treatment_median, abs_delta, pct_delta, bootstrap}}. Keys are
             abs_delta/pct_delta (not delta_abs/delta_pct); the nested CI
             dict is "bootstrap" (not "ci") -- see efficiency_deltas'
             own docstring for the full per-metric shape.
        "pass_k": pass_k([...]),
             -- reliability of the CANDIDATE (treatment) harness completing
             at all, from each grade's control_ok/treatment_ok
             (types.make_grade); NOT the control's, since pass_k here is
             answering "does the patched harness reliably finish", and k
             defaults to len(grades) (see pass_k's docstring on
             `degenerate` -- true for our v1 one-prompt-per-patch design).
        "regressions": flattened make_regression() dicts across every
             grade's grade["regressions"] for this patch.
        "judge_consistency": grade.judge_consistency(grades) -- position-
             swap agreement rate, published alongside win_rate per
             research-methods.md sec 3 (a win rate is weaker evidence from
             a self-inconsistent judge).
        "primary_effect": primary_effect, carried through unchanged so
             `gate` (and the report) know what was targeted without a
             second parameter to keep in sync.
      }
    """
    metric_name = EFFECT_TO_EFFICIENCY_METRIC[primary_effect]
    primary_deltas = _metric_deltas(grades, metric_name)

    return {
        "n": len(grades),
        "win_rate": win_rate(grades, seed=seed),
        "permutation": permutation_test(primary_deltas, seed=seed),
        "efficiency": efficiency_deltas(grades, seed=seed),
        "pass_k": pass_k([g.get("treatment_ok", True) for g in grades]),
        "regressions": [r for g in grades for r in (g.get("regressions") or [])],
        "judge_consistency": judge_consistency(grades),
        "primary_effect": primary_effect,
    }


def _directional(win_rate_stats, primary_entry):
    """What the (possibly too-thin-to-gate) evidence points toward:
    "favorable", "unfavorable", or "neutral". Used by `gate` to give the
    report something to say even when n is too small to accept (n < 5) --
    a rejection is not the same as "we don't know which way this leans".

    Combines the sign of (wins - losses) with the sign of the primary
    metric's point estimate (paired_bootstrap always reports a real `mean`,
    even below MIN_SAMPLES_FOR_CI). Both signals must agree, or one be
    silent (zero), to call it favorable/unfavorable; outright disagreement
    reads as "neutral" rather than arbitrarily picking a side.
    """
    wins = (win_rate_stats or {}).get("wins", 0)
    losses = (win_rate_stats or {}).get("losses", 0)
    quality_sign = (wins > losses) - (wins < losses)  # 1, -1, or 0

    efficiency_sign = 0
    if primary_entry is not None:
        mean = (primary_entry.get("bootstrap") or {}).get("mean")
        if mean is not None:
            efficiency_sign = (mean > 0) - (mean < 0)

    if quality_sign >= 0 and efficiency_sign >= 0 and (quality_sign or efficiency_sign):
        return "favorable"
    if quality_sign <= 0 and efficiency_sign <= 0 and (quality_sign or efficiency_sign):
        return "unfavorable"
    return "neutral"


def gate(summary, *, primary_effect=None, max_token_regression=0.10):
    """The acceptance gate: the defense against the loop optimizing for its
    own judge. Hard to fool by construction -- thin or missing evidence is
    a reason to reject, not a pass by default. Sample-size-aware per
    research-methods.md secs 1, 4, 5: a k=3/5 replay count cannot support
    the same accept criterion as k>=10, and pretending otherwise (a CI on
    3 points, "zero losses" alone) is exactly the kind of fabricated
    confidence this whole module exists to refuse.

    Takes the output of `summarize` -- see that function's docstring for
    the full shape. `primary_effect` (one of types.ALL_EFFECTS) defaults to
    `summary["primary_effect"]` (the value `summarize` was called with);
    pass it explicitly only to gate on a different claim than the one
    `summarize` was built with. It is mapped to an EFFICIENCY_METRICS name
    via EFFECT_TO_EFFICIENCY_METRIC, same as `summarize`.

    Quality check, keyed on n = win_rate's judged-pair count:
      n < MIN_SAMPLES_FOR_ACCEPT (5)
          -> accepted forced False, reason "insufficient_samples_for_acceptance".
             Even a truly-neutral (50/50) patch shows zero losses in 3 draws
             with probability 0.5**3 = 12.5%; there is no accept criterion
             at this n that isn't mostly luck.
      MIN_SAMPLES_FOR_ACCEPT <= n < MIN_SAMPLES_FOR_CI (5-9)
          -> no bootstrap CI exists at this n (see win_rate). The only
             criterion available is zero losses; if met, `provisional=True`
             and quality passes. Any loss rejects outright -- there is no
             CI to fall back on to say "still probably fine".
      n >= MIN_SAMPLES_FOR_CI (10+)
          -> zero losses, OR win-rate lower CI >= 0.5.

    High-severity regressions: any "high" severity entry in
    summary["regressions"] rejects outright, regardless of n.

    Primary efficiency target, also n-aware (using the metric's own pair
    count, which can differ from win_rate's n if some traces are missing):
      n >= MIN_SAMPLES_FOR_CI (10+) -> require the paired-diff bootstrap CI
          lower bound > 0 (the whole interval says "improved"). A missing
          CI at this n (shouldn't happen, but) does not pass -- the central
          claim of the patch must not be accepted on a point estimate alone.
      n < MIN_SAMPLES_FOR_CI (below 10) -> no CI exists by design; require
          the point estimate (mean) > 0, explicitly labeled directional-only.
          This is what makes the 5-9 zero-losses path reachable at all --
          without it, no patch could ever accept below n=10 CI availability.

    total_tokens regression safety net: independent of whatever the primary
    metric is, total_tokens must not have regressed past
    `max_token_regression` (default 10%) using its point-estimate pct_delta.
    Unmeasured total_tokens does not itself fail the gate -- reported as
    unchecked.

    Returns {accepted, provisional, directional, reasons}:
      accepted    - bool, the final call.
      provisional - bool. True iff the quality check used the small-n (5-9,
                    no-CI) zero-losses fallback rather than a full n>=10
                    CI-backed decision. An accepted:True result with
                    provisional:True should render as a qualified verdict,
                    not a clean pass.
      directional - "favorable" | "unfavorable" | "neutral": what the
                    evidence points toward regardless of whether accepted
                    is True -- always populated, including at n < 5 where
                    acceptance is impossible by construction, so the report
                    always has something better to show than a bare "no".
      reasons     - list[str], one per check, for both accept and reject,
                    so the report can always explain itself. Includes two
                    informational lines (judge position-consistency, the
                    primary effect's permutation-test p-value) that do NOT
                    affect accepted/provisional/directional -- see the end
                    of the function -- because neither is part of the
                    pinned accept criteria, but both are cheap context a
                    reader needs to calibrate the rest of the reasons.
    """
    reasons = []
    accepted = True
    provisional = False

    effect = primary_effect if primary_effect is not None else summary.get("primary_effect")
    primary_metric = EFFECT_TO_EFFICIENCY_METRIC.get(effect) if effect else None

    wr = summary.get("win_rate") or {}
    n = wr.get("n", 0)
    wins = wr.get("wins", 0)
    losses = wr.get("losses", 0)
    lo = wr.get("lo")

    efficiency = summary.get("efficiency") or {}
    primary_entry = efficiency.get(primary_metric) if primary_metric else None
    directional = _directional(wr, primary_entry)

    # --- 1. quality, sample-size-aware ---
    if n < MIN_SAMPLES_FOR_ACCEPT:
        accepted = False
        reasons.append(
            f"insufficient_samples_for_acceptance: only {n} judged pair(s) -- "
            f"k<{MIN_SAMPLES_FOR_ACCEPT} cannot support an accept decision at "
            "any confidence (even zero losses has ~12.5% false-accept risk "
            "at k=3; see research-methods.md sec 5.1)"
        )
    elif n < MIN_SAMPLES_FOR_CI:
        provisional = True
        if losses == 0:
            reasons.append(
                f"provisional quality pass: 0 losses out of {n} judged pairs "
                f"(n<{MIN_SAMPLES_FOR_CI}, no bootstrap CI available -- "
                "see research-methods.md sec 1)"
            )
        else:
            accepted = False
            reasons.append(
                f"quality: {losses} loss(es) out of {n} judged pairs, and "
                f"n<{MIN_SAMPLES_FOR_CI} means no CI branch is available -- "
                "cannot accept without zero losses at this sample size"
            )
    else:
        if losses == 0:
            reasons.append(f"quality: 0 losses out of {n} judged pairs")
        elif lo is not None and lo >= 0.5:
            reasons.append(f"quality: win-rate lower CI {lo:.2f} >= 0.5")
        else:
            accepted = False
            reasons.append(
                f"quality regression: {losses} loss(es), win-rate lower CI "
                f"{lo!r} does not clear 0.5"
            )

    # --- 2. high-severity regressions ---
    all_regressions = summary.get("regressions") or []
    high = [r for r in all_regressions if r.get("severity") == "high"]
    if high:
        accepted = False
        details = "; ".join(r.get("detail", "") for r in high)
        reasons.append(f"{len(high)} high-severity regression(s): {details}")
    else:
        reasons.append("no high-severity regressions")

    # --- 3. primary efficiency target improves, sample-size-aware ---
    if not effect:
        accepted = False
        reasons.append("no primary_effect declared: cannot verify the claimed effect improved")
    elif primary_metric is None:
        accepted = False
        reasons.append(f"primary_effect {effect!r} is not a recognized EFFECT_* name (see EFFECT_TO_EFFICIENCY_METRIC)")
    elif primary_entry is None:
        accepted = False
        reasons.append(f"primary metric {primary_metric!r} (from effect {effect!r}) not present in efficiency stats")
    else:
        boot = primary_entry.get("bootstrap") or {}
        p_n = boot.get("n", 0)
        p_mean = boot.get("mean")
        p_lo, p_hi = boot.get("lo"), boot.get("hi")
        if p_mean is None:
            accepted = False
            reasons.append(f"primary metric {primary_metric!r}: no paired data available")
        elif p_n >= MIN_SAMPLES_FOR_CI:
            if p_lo is not None and p_lo > 0:
                reasons.append(
                    f"primary metric {primary_metric!r} improved: paired-diff CI "
                    f"[{p_lo:.3g}, {p_hi:.3g}] entirely > 0"
                )
            else:
                accepted = False
                reasons.append(
                    f"primary metric {primary_metric!r} did not credibly improve: "
                    f"paired-diff CI [{p_lo!r}, {p_hi!r}] includes zero or below"
                )
        else:
            if p_mean > 0:
                reasons.append(
                    f"primary metric {primary_metric!r} improved directionally: point "
                    f"estimate {p_mean:.3g} (n={p_n} < {MIN_SAMPLES_FOR_CI}, no CI "
                    "-- directional signal only, see research-methods.md sec 1)"
                )
            else:
                accepted = False
                reasons.append(
                    f"primary metric {primary_metric!r} did not improve: point estimate "
                    f"{p_mean:.3g} <= 0 (n={p_n} < {MIN_SAMPLES_FOR_CI}, directional read only)"
                )

    # --- 4. total_tokens regression safety net ---
    tokens = efficiency.get("total_tokens")
    pct = tokens.get("pct_delta") if tokens else None
    if pct is None:
        reasons.append("total_tokens delta unavailable, regression threshold not checked")
    elif pct < -max_token_regression:
        accepted = False
        reasons.append(
            f"total_tokens regressed {-pct:.1%} (exceeds {max_token_regression:.0%} "
            f"threshold): control median {tokens['control_median']}, "
            f"treatment median {tokens['treatment_median']}"
        )
    else:
        reasons.append(
            f"total_tokens within threshold ({pct:+.1%}, max regression {max_token_regression:.0%})"
        )

    # --- informational only: neither affects accepted/provisional/directional ---
    jc = summary.get("judge_consistency") or {}
    if jc.get("rate") is not None:
        reasons.append(
            f"judge position-consistency: {jc['rate']:.0%} over {jc['pairs']} judged "
            "pair(s) (informational -- a low rate means win_rate carries more judge "
            "noise; research-methods.md sec 3)"
        )
    elif jc.get("note"):
        reasons.append(f"judge position-consistency: {jc['note']}")

    perm = summary.get("permutation") or {}
    if perm.get("p") is not None:
        reasons.append(
            f"permutation test on primary effect: p={perm['p']:.3g} "
            f"(min achievable at n={perm['n']} is {perm['min_achievable_p']:.3g}) "
            "-- informational, not part of the accept decision; see "
            "research-methods.md sec 1 for why a k=3/5 p-value can't gate anything"
        )

    return {"accepted": accepted, "provisional": provisional,
            "directional": directional, "reasons": reasons}
