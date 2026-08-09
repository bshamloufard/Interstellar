# Methodology review: harness-patch evaluation stats & judge design

Checked against the fixed design: paired k-repeat runs (k=3 or 5), pairwise position-swapped
LLM judge with tie-on-disagreement, deterministic efficiency meters, percentile bootstrap CI +
win/tie/loss + sign test + pass^k, gate = (win-rate lower CI ≥ 0.5 OR zero losses) AND no
high-severity regression AND targeted efficiency metric improves.

---

## 1. Small-n honesty (k=3, k=5)

**Verdict: the percentile bootstrap CI at k=3 is not defensible. Fix it.**

- A bootstrap resample of size n drawn with replacement from n paired deltas has only
  `C(2n-1, n)` distinct possible values: **n=3 → 10 distinct resamples, n=5 → 126**. At n=3 your
  "confidence interval" is built from 10 possible outcomes — it cannot have characterized
  coverage, and adding more bootstrap iterations (B=1000, 10000, …) does not fix this: B controls
  Monte-Carlo noise in *approximating* the resampling distribution, it adds zero real information
  beyond the original n points. This is a basic property of the percentile bootstrap (Efron &
  Tibshirani, *An Introduction to the Bootstrap*, 1993); empirical accuracy studies of percentile-
  bootstrap CIs consistently require samples far larger than 3–5 before coverage is trustworthy —
  cited minimums in the applied-stats literature run from N≥20 (rough floor, and only for
  well-behaved statistics) up to N≥60 for skewed/tailed cases. [ScienceDirect: Percentile
  Bootstrap overview](https://www.sciencedirect.com/topics/mathematics/percentile-bootstrap),
  [CI of percentiles in skewed distributions, PMC6784425](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6784425/).
- Independently, the **exact sign test has a hard p-value floor that makes k=3 structurally unable
  to ever report significance**: two-sided p = 2·(0.5)^n for the all-same-direction case. At n=3,
  the best possible p-value is **2·(0.5)³ = 0.25**; at n=5 it's **2·(0.5)⁵ = 0.0625**. Neither can
  cross p<0.05 even if every single paired run favors treatment. n=6 is the first n where a
  unanimous result can cross 0.05 (2·(0.5)⁶=0.03125). This is elementary binomial arithmetic, not
  a debated point — confirmed against sign-test references
  ([Some Finite Sample Properties of the Sign Test, arXiv:2103.01412](https://arxiv.org/pdf/2103.01412),
  [Minimax Optimality of Sign Test, arXiv:1801.04005](https://arxiv.org/pdf/1801.04005)).
- **Exact permutation/sign-flip test over 2^n patterns at n≤5 is the right call, and it's exactly
  the same object as the exact sign test's superset**: with paired deltas you can sign-flip each
  of the n deltas independently (2^n patterns, fully enumerable for n≤~20 in practice, trivially
  so at n≤5), recompute the mean (or median) statistic under each pattern, and get an exact
  two-sided p-value. This exact permutation test on the deltas' *magnitudes* has strictly more
  power than the plain sign test (which discards magnitude and only asks better-or-worse) while
  keeping the same exact, finite-sample type-I-error control. `scipy.stats.permutation_test` does
  exactly this and switches to exact enumeration automatically at small n
  ([SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)).
  It still inherits the same floor as the sign test at k=3 (min two-sided p ≈ 0.25 if the
  direction is unanimous but magnitudes don't add extra extremity beyond that), so it doesn't
  rescue "significance" at k=3 — but it's the more honest, strictly-better-powered exact test to
  report instead of a sign test alone.

**Concrete recommendation with thresholds:**

| n (paired repeats) | What to report |
|---|---|
| n < 10 (covers your k=3, k=5) | **No bootstrap CI.** Report: the raw n paired deltas (not compressed), the exact sign-flip permutation p-value (2^n enumeration), and the win/tie/loss counts as raw counts, not rates with an interval. Label the result explicitly as "directional signal only, underpowered for a confidence claim" — at k=3 you literally cannot reach p<0.05 by construction, so don't let a computed bootstrap interval imply otherwise. |
| 10 ≤ n < 20 | Bootstrap CI permitted but flagged low-confidence (state distinct-resample count `C(2n-1,n)`); exact permutation test still preferred as the primary inferential claim. |
| n ≥ 20 | Percentile bootstrap CI on paired deltas is on reasonably solid ground; this is the regime the "N≥20" floor from the applied literature is describing. |

Practical implication for the acceptance gate: **at k=3/k=5, the gate should not be phrased in
terms of a bootstrap CI on win rate at all** — replace "win-rate lower CI ≥ 0.5" with something
the sample size can actually support, e.g. "zero losses AND the exact sign-flip p-value is at
least directionally consistent," and treat any such k=3/5 result as provisional pending
accumulation to n≥10 before it's allowed to gate irreversible decisions (e.g. deleting the
baseline harness version). See the "zero losses" trap in §5 — it's weaker than it sounds at k=3.

---

## 2. pass^k — your estimator is wrong; here's the correct one

**Verdict: `(fraction of successful runs)^k` is not the tau-bench estimator and will give
materially different (and biased/overconfident) numbers. Replace it.**

τ-bench (Yao et al., [arXiv:2406.12045](https://arxiv.org/abs/2406.12045), §3 "Pass^k metric",
confirmed by direct PDF extraction) defines pass^k as **the chance that all k i.i.d. trials of a
task succeed, averaged across tasks**, and gives the unbiased combinatorial estimator explicitly:

```
pass^k = E_task[ C(c, k) / C(n, k) ]
```

where a task is run for **n** trials, **c** of which succeed (reward r=1), and `C(·,·)` is the
binomial coefficient — the mirror image of the Chen et al. (Codex/HumanEval) unbiased pass@k
estimator `pass@k = E_task[1 − C(n−c, k)/C(n, k)]`. τ²-bench
([arXiv:2506.07982](https://arxiv.org/pdf/2506.07982)) reuses this same definition verbatim
without modification (confirmed by direct PDF extraction — it just cites and applies it across
domains).

**Why the naive `(success rate)^k` estimator is actually wrong, not just approximate:** it isn't
computing the same quantity. Concrete example: a task run n=3 times with c=2 successes. Naive:
`(2/3)^3 = 0.296`. Correct (n=k=3): `C(2,3)/C(3,3) = 0/1 = 0`, because you cannot choose 3
successes out of only 2 — i.e. the honest answer is "not all 3 trials succeeded, so pass^3 for
this task is 0," full stop. The naive formula manufactures a smoothed, optimistic fractional
value for something that was in fact a clean failure. This is a systematic bias, not noise, and
it gets worse as c/n moves away from 0 or 1.

**Where the estimator earns its keep — and a design implication for you:** the combinatorial form
lets you run **n > k** trials once and reuse the *same* batch to compute pass^1, pass^2, … pass^n
all at once (that's the whole point of the "Chen et al. construction" — avoid re-running fresh
batches per k). If your harness-eval design genuinely only ever runs exactly k trials (n=k), the
estimator above collapses to a 0/1 indicator per task ("did all k runs succeed"), which is still
correct but is not a smooth reliability curve — you get one bit of information per task, and it
only becomes a meaningful *rate* once averaged over multiple tasks/prompts. If your eval suite has
only one prompt per patch, pass^k is degenerate (a single 0/1). **Recommendation:** either (a)
consider running n slightly larger than the largest k you report (e.g. n=5 trials, report pass^1
through pass^5 all from that one batch via the formula above — cheaper than separate reruns), or
(b) if you truly test one prompt at a time, be explicit that pass^k there is a binary "did all k
agree on success" flag, not a probability estimate, and only quote a real pass^k rate when
aggregating across the prompt suite.

---

## 3. Judge position bias

**Verdict: your tie-on-disagreement rule is standard practice and correct. The gap is you're not
publishing the consistency rate as a first-class number alongside the win rate — fix that.**

- Treating a position-swap disagreement as a tie is exactly the "conservative" approach documented
  in the MT-Bench / LLM-as-judge line of work: call the judge twice with swapped order, declare a
  win only if the same side wins both times, and label disagreement a tie (Zheng et al., "Judging
  LLM-as-a-Judge with MT-Bench and Chatbot Arena," [arXiv:2306.05685](https://arxiv.org/pdf/2306.05685);
  same rule independently adopted by PandaLM). This is the field's default, not a shortcut you
  invented — good.
- What's missing: publish **position consistency** (fraction of pairs where the two orderings
  agree, wins+ties-as-agreement over total) as a standalone diagnostic next to the win rate, not
  just folded silently into the tie bucket. Reported baseline consistency rates across judge models
  run from ~70–77% (GPT-3.5-Turbo to Gemini-Pro; flip rates 25–50% across judges) — if your
  consistency rate is down in that range or lower, that's a signal your win-rate number is noisy
  *because of the judge*, not because the harness patches are actually tied, and readers need that
  context to calibrate trust. Source: "Judging the Judges: A Systematic Investigation of Position
  Bias in Pairwise Comparative Assessments by LLMs,"
  [arXiv:2406.07791](https://arxiv.org/html/2406.07791v4) — this paper's specific proposal is to
  report **Position Consistency (PC)** and a signed **Preference Fairness (PF)** score (−1..+1,
  0=fair) alongside win rate; worth adopting PC at minimum since you're already computing the raw
  ingredient (you already run both orderings).
- Cheap, already-validated additions from the same literature, in order of cost/benefit:
  1. **Reference-guided grading** where a ground-truth or reference answer exists (MT-Bench: this
     and chain-of-thought self-solving before grading "significantly reduces grading errors,"
     particularly for tasks with a checkable right answer — less relevant if your judge is
     comparing open-ended agent transcripts with no single correct answer, but relevant for any
     efficiency/correctness sub-claims the judge is asked to verify).
  2. Randomize position assignment at the population level (not just paired-swap) if you ever move
     beyond exhaustive 2-way swap — not needed given you already do exhaustive swapping.
  3. Explicit, narrow rubric criteria reduce ambiguous-tie inflation (arXiv:2406.07791 finding:
     disagreement correlates with how close/subjective the two candidates are — tightening the
     rubric reduces spurious ties, it doesn't just relabel them).

---

## 4. Multiple patches in one cycle

**Verdict: not addressed in your design at all currently, and it will matter as soon as you test
more than a couple of patches per cycle against the same baseline — but the standard fix is
secondary to the small-n problem in §1.**

- Standard framing: each patch's accept/reject decision is one hypothesis test against the shared
  baseline; running m patches in a cycle makes m simultaneous accept/reject decisions a "family."
  Two standard corrections, pick based on cost of a false accept:
  - **Family-wise (zero-tolerance for any false accept):** Holm-Bonferroni step-down procedure —
    strictly more power than plain Bonferroni for the same FWER guarantee, trivial to implement
    (sort p-values, compare `p_(i)` to `α/(m−i+1)`). Standard reference: Holm (1979).
  - **False-discovery-rate control (tolerate an occasional false accept, maximize power):**
    Benjamini-Hochberg — appropriate if a wrongly-accepted patch is cheap to catch/revert later
    (e.g. it'll show up in the next cycle's aggregate metrics anyway) and you'd rather not veto
    good patches to protect against rare false positives.
  ([Statsig: multiple comparison corrections in A/B testing](https://www.statsig.com/blog/multiple-comparison-corrections-in-a-b),
  [Holm-Bonferroni explainer](https://mcpanalytics.ai/articles/holm-bonferroni-method-practical-guide-for-data-driven-decisions))
- **When it starts to matter, concretely:** at uncorrected α=0.05 per patch, testing m=10 patches
  in a cycle gives P(at least one false accept) ≈ 1−0.95^10 ≈ 40%. That's already a large,
  non-ignorable rate at a cycle size many teams would consider modest. Correction starts mattering
  once you're running roughly **m≥3–5 patches per cycle** against the same baseline; below that,
  uncorrected per-patch testing is a defensible simplification.
- **Important interaction with §1:** at k=3/5 your acceptance gate isn't really running on
  calibrated p-values in the first place (the sign test can't reach p<0.05 at all — see §1), so a
  Bonferroni/BH correction computed on top of an already-uninterpretable small-n statistic is
  cosmetic. Fix sample size first; multiplicity correction becomes meaningful once individual
  patch decisions are themselves statistically meaningful (n≥10, per §1's table).

---

## 5. Other methodological traps in the design

1. **"Zero losses" is a much weaker fallback than it sounds at k=3.** Even a patch with a *true*
   50/50 win/loss rate (i.e. genuinely no effect) will show zero losses in 3 draws with probability
   0.5³ = 0.125 — about 1 in 8 truly-neutral patches will pass this arm of the gate purely by
   chance. At k=5 it's 0.5⁵ = 0.03125 (~1 in 32), materially safer. **Recommendation:** either
   require k≥5 before "zero losses" is allowed to gate acceptance on its own, or require it in
   conjunction with the exact sign-flip p-value from §1 rather than as an independent OR-branch.
2. **Pass^k with a one-prompt-per-patch design degenerates to a coin flip, see §2** — worth
   deciding explicitly whether pass^k is being computed per-prompt (binary) or aggregated across a
   prompt suite (a real rate) before it's put in front of the acceptance gate.
3. **Non-independence across repeats isn't stated as an assumption anywhere in the design.** Paired
   bootstrap/permutation tests assume the k pairs are exchangeable draws. If repeats for a prompt
   are run back-to-back against a model endpoint that has session/cache state, or if a model
   version silently drifts between the k control runs and the k treatment runs (they should be
   temporally interleaved, not control-block-then-treatment-block), that correlation will make
   every one of the above tests overconfident. Worth a one-line explicit statement in the write-up
   that runs are interleaved/randomized in execution order and use fresh sessions.
4. **The efficiency-metric gate ("the targeted efficiency metric improves") is fine as stated** —
   pre-registering one targeted metric per patch avoids the multiple-comparisons problem on the
   efficiency side entirely, since you're not scanning many efficiency metrics and picking
   whichever moved. No fix needed there, just don't let it silently expand to "any efficiency
   metric improves" later without revisiting §4.

---

## Sources

- Yao et al., τ-bench, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) — pass^k definition (§3), read directly from PDF.
- τ²-Bench, [arXiv:2506.07982](https://arxiv.org/pdf/2506.07982) — reuses τ-bench's pass^k unchanged, read directly from PDF.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," [arXiv:2306.05685](https://arxiv.org/pdf/2306.05685) — swap-and-tie protocol, reference-guided/CoT judge.
- "Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs," [arXiv:2406.07791](https://arxiv.org/html/2406.07791v4) — Position Consistency / Preference Fairness metrics.
- Percentile bootstrap small-sample limitations: [ScienceDirect overview](https://www.sciencedirect.com/topics/mathematics/percentile-bootstrap), [PMC6784425](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6784425/).
- Exact sign test / permutation test: [scipy.stats.permutation_test docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [Some Finite Sample Properties of the Sign Test, arXiv:2103.01412](https://arxiv.org/pdf/2103.01412), [Minimax Optimality of Sign Test, arXiv:1801.04005](https://arxiv.org/pdf/1801.04005).
- Multiple comparisons: [Statsig — multiple comparison corrections in A/B testing](https://www.statsig.com/blog/multiple-comparison-corrections-in-a-b), [Holm-Bonferroni explainer](https://mcpanalytics.ai/articles/holm-bonferroni-method-practical-guide-for-data-driven-decisions).
