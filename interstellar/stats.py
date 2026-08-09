"""Paired statistics over `Grade` lists, and the acceptance gate.

This is the layer that turns k noisy paired runs (control vs. treatment,
same prompt, same isolation -- see types.make_replay_matrix) into a
defensible claim: "this patch helped, here is the interval, here is why we
believe it or don't." Every function here is pure and takes plain dicts in,
plain dicts out, so it is exhaustively testable without I/O or grok-dev.

The organizing rule, repeated at every function boundary: a confidence
interval computed from too few samples is not a weaker interval, it is a
fabricated one. With k=3 (the typical replay count) a percentile bootstrap
is already barely meaningful; below that (n <= 2) this module refuses to
print lo/hi at all and says so explicitly via a "note" field, rather than
rendering a confident-looking range. See MIN_SAMPLES_FOR_CI below.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import math
import random
import statistics

from interstellar.types import ARM_CONTROL, ARM_TREATMENT, EFFICIENCY_METRICS

# Below this many paired samples, a percentile bootstrap's resampling
# distribution is too coarse to mean anything (e.g. at n=2 there are only 4
# distinct equally-likely resamples). Point estimates (mean, rate, medians)
# are still reported at n <= 2 -- they are real numbers, just uninterval-ed.
MIN_SAMPLES_FOR_CI = 3

CI_LEVEL = 0.95
DEFAULT_ITERS = 10000
DEFAULT_SEED = 0

NOTE_INSUFFICIENT = "insufficient_samples"


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


def paired_bootstrap(deltas, *, iters=DEFAULT_ITERS, seed=DEFAULT_SEED):
    """Percentile bootstrap CI on the mean of paired differences.

    Estimator: resample `deltas` with replacement (same size as the input)
    `iters` times using `random.Random(seed)`, take the mean of each
    resample, and report the 2.5th/97.5th percentiles of that distribution
    as a 95% CI (Efron's percentile bootstrap). This is NOT bias-corrected
    (BCa) -- it under-covers when the deltas are skewed or n is small, and
    it says nothing about the underlying distribution beyond what the
    sample already shows. Seeded explicitly so the same input always
    produces the same interval; the report is regenerated and must not
    shimmer between runs.

    n <= 2 (see MIN_SAMPLES_FOR_CI): the sample mean is still reported (it
    is a real, computed number) but lo/hi are None and note is set --
    there is no meaningful resampling distribution with one or two points.
    n == 0: mean is also None.
    """
    n = len(deltas)
    result = {"mean": None, "lo": None, "hi": None, "level": CI_LEVEL, "n": n, "note": None}
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
    purpose, per the brief: with ties dominant, `rate` can look extreme
    (or be undefined) on very few decided pairs while the tie count stays
    large, so `ties` is always reported alongside `rate`, never dropped.

    CI: percentile bootstrap over the per-pair win/tie/loss labels
    (resample with replacement, recompute rate on each resample, keep only
    resamples that contain at least one decided pair). Same caveats as
    paired_bootstrap.

    Edge cases, each with its own note so the report can say exactly why
    there is no rate/interval rather than printing 0.0 or a fake range:
      n == 0                      -> note="insufficient_samples"
      n in (1, 2)                 -> note="insufficient_samples" (rate is
                                      still computed if decided > 0)
      n >= 3 but decided == 0     -> note="no_decided_pairs" (all ties;
                                      wins/ties/losses/n remain visible)
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
    return result


def sign_test(deltas):
    """Exact two-sided sign test on the direction of paired differences.

    Zeros are excluded (no direction to score). Under the null that a
    positive or negative delta is equally likely, n_nonzero is
    Binomial(n_nonzero, 0.5); the two-sided p-value is
    2 * P(X <= min(pos, neg)) computed exactly with math.comb (no normal
    approximation), clamped at 1.0 for the symmetric case. This tests
    direction only, not magnitude -- pair it with paired_bootstrap for
    effect size.

    Sign convention is the caller's: sign_test does not know whether a
    positive delta means "improved" or "regressed", only which sign is
    more common. `direction` is "positive", "negative", or "tie" (equal
    nonzero counts on both sides, including n_nonzero == 0 -> "none").

    Underpowered samples are handled honestly rather than specially: with
    few nonzero deltas the exact binomial p-value is naturally close to 1,
    which is the correct signal (no evidence of a consistent direction),
    not a case requiring a fabricated-CI-style guard.
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


def pass_k(successes, *, k=None):
    """Plug-in estimate of "would all k of these repeats have succeeded"
    (the tau-bench pass^k reliability reading), NOT the unbiased
    combinatorial pass@k estimator used in Codex-style code eval (which
    needs n > k samples-without-replacement and a hypergeometric formula).

    Estimator: p = (# successes) / len(successes); return p ** k. This
    treats each repeat as an i.i.d. Bernoulli(p) trial and asks "if I drew
    k such trials at that empirical rate, what is the chance all k
    succeed". It is a biased, high-variance point estimate at small n --
    e.g. 3 repeats with 1 failure gives p = 2/3, pass_k = (2/3)**3 ~= 0.30,
    which swings hard on a single outcome. Returns a bare float, not a
    dict: there is no CI to compute or fake here (see paired_bootstrap /
    sign_test on the same successes list for uncertainty).

    k defaults to len(successes) -- "would this same batch of k repeats
    have all succeeded" -- but can be overridden to ask about a
    hypothetical batch of a different size.
    """
    n = len(successes)
    if n == 0:
        return 0.0
    if k is None:
        k = n
    p = sum(1 for s in successes if s) / n
    return p ** k


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

    The bootstrap CI is on the paired difference (the same per-pair deltas
    used for abs_delta's sign), computed by paired_bootstrap verbatim, so
    it inherits that function's n <= 2 handling exactly.
    """
    result = {}
    for name, lower_is_better in EFFICIENCY_METRICS:
        pairs = []
        for g in grades:
            c = (g.get("control_efficiency") or {}).get(name)
            t = (g.get("treatment_efficiency") or {}).get(name)
            if c is None or t is None:
                continue
            pairs.append((c, t))

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


def gate(stats, *, max_token_regression=0.10):
    """The acceptance gate: the defense against the loop optimizing for its
    own judge. Designed to be hard to fool -- thin or missing evidence is
    treated as a reason to reject, not as a pass by default.

    `stats` is expected to be a dict assembled by the caller from this
    module's other functions, shaped:
      stats["win_rate"]       -> win_rate(grades) output
      stats["regressions"]    -> flat list of make_regression() dicts,
                                  concatenated across every grade's
                                  grade["regressions"] for this patch
      stats["efficiency"]     -> efficiency_deltas(grades) output
      stats["primary_metric"] -> name (str) of the EFFICIENCY_METRICS entry
                                  that is this patch's claimed primary
                                  effect, e.g. "total_tokens" -- required so
                                  the gate checks the one metric the patch
                                  claims to move, not whichever metric
                                  happens to look best after the fact

    Accepts iff ALL of:
      1. Quality does not regress: win_rate's losses == 0 (covers the
         all-ties / too-few-judged-pairs case, where there may be no lower
         CI bound at all but also no evidence of a loss), OR win_rate's
         lower CI bound (lo) >= 0.5.
      2. No "high" severity regression appears anywhere in
         stats["regressions"].
      3. The primary efficiency target credibly improves: its paired-diff
         bootstrap CI lower bound (lo) is > 0, i.e. the entire 95% interval
         says the treatment used less of that metric. A missing CI
         (n <= 2, lo is None) does NOT pass this check -- the central claim
         of the patch is exactly what must not be accepted on a point
         estimate alone.
      4. total_tokens has not regressed by more than max_token_regression
         (default 10%), independent of what the primary metric is -- a
         safety net so a patch cannot win by improving one metric while
         quietly blowing up total cost. Unmeasured total_tokens (no pairs)
         does not by itself fail the gate; it is reported as unchecked.

    Returns {accepted: bool, reasons: [str, ...]}. `reasons` is populated
    for both accept and reject, and for every check independently, so the
    report can always explain itself instead of asserting a verdict.
    """
    reasons = []
    accepted = True

    wr = stats.get("win_rate") or {}
    losses = wr.get("losses", 0)
    lo = wr.get("lo")
    if losses == 0:
        reasons.append(f"quality: 0 losses out of {wr.get('n', 0)} judged pairs")
    elif lo is not None and lo >= 0.5:
        reasons.append(f"quality: win-rate lower CI {lo:.2f} >= 0.5")
    else:
        accepted = False
        reasons.append(
            f"quality regression: {losses} loss(es) and win-rate lower CI "
            f"{lo!r} does not clear 0.5"
        )

    all_regressions = stats.get("regressions") or []
    high = [r for r in all_regressions if r.get("severity") == "high"]
    if high:
        accepted = False
        details = "; ".join(r.get("detail", "") for r in high)
        reasons.append(f"{len(high)} high-severity regression(s): {details}")
    else:
        reasons.append("no high-severity regressions")

    efficiency = stats.get("efficiency") or {}
    primary = stats.get("primary_metric")
    if not primary:
        accepted = False
        reasons.append("no primary_metric declared: cannot verify the claimed effect improved")
    else:
        primary_entry = efficiency.get(primary)
        if primary_entry is None:
            accepted = False
            reasons.append(f"primary metric {primary!r} not present in efficiency stats")
        else:
            boot = primary_entry.get("bootstrap") or {}
            p_lo, p_hi = boot.get("lo"), boot.get("hi")
            if p_lo is None:
                accepted = False
                reasons.append(
                    f"primary metric {primary!r}: insufficient samples for a "
                    "credible CI, not accepted on a point estimate alone"
                )
            elif p_lo > 0:
                reasons.append(
                    f"primary metric {primary!r} improved: paired-diff CI "
                    f"[{p_lo:.3g}, {p_hi:.3g}] entirely > 0"
                )
            else:
                accepted = False
                reasons.append(
                    f"primary metric {primary!r} did not credibly improve: "
                    f"paired-diff CI [{p_lo:.3g}, {p_hi:.3g}] includes zero or below"
                )

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
        reasons.append(f"total_tokens within threshold ({pct:+.1%}, max regression {max_token_regression:.0%})")

    return {"accepted": accepted, "reasons": reasons}
