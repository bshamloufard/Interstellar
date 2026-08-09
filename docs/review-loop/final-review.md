# Final whole-branch review — `interstellar-review-loop`

Reviewer: whole-branch integration pass (the gaps the per-module reviews structurally
could not see). Read-only; nothing in the repo was modified.

Scope reviewed: `interstellar/{types,harness,patches,replay,grade,stats,report,cli}.py`,
`interstellar/tests/`, `docs/plans/review-loop.md`, the branch diff
`review-322f359..6bcd070.diff`, task reports 1–7, the four prior module reviews,
`research-methods.md`, `progress.md`, and the real outputs in
`runs/headline/`, `runs/control/`, `runs/final/`.

Verification actually performed (not assumed):

- `python3 -m unittest discover interstellar/tests` → **Ran 407 tests, OK**.
- `--dry-run --use-cached-analysis` over all 14 corpus traces → 14/14 clean, each
  selected 3 patches, no crash, no spawn.
- Reproduced Critical 1 twice: directly against `replay.isolated_workdir`, and
  end-to-end through `replay.run_matrix` with a runner that asserts if it is ever
  called (it never is — the failure happens before any spawn).
- Read the real numbers out of `runs/headline/report.json` and the three
  `patches/*/matrix.json` files and reconciled them against the rendered
  `index.html`.
- Grepped all 18 headline `response_text` bodies for arm-name leakage.
- Import sweep across `interstellar/` for non-stdlib dependencies (Constraint 1).
- Grep sweep of `interstellar/tests/` for grok-dev invocation (Constraint 8).

**One caveat on this review's own freshness:** `interstellar/cli.py`,
`interstellar/report.py`, and `interstellar/tests/test_report.py` were being edited
by another agent *while this review was running* (mtimes inside the review window).
All line numbers below were re-verified against the working tree at the time of
writing, but they will drift. See "Uncommitted work in the tree" at the end — the
in-flight change has a Constraint-2 problem of its own.

---

## Critical

### C1 — `isolated_workdir` copies the workspace into itself whenever `--out` lives inside the replayed workspace, which is the documented default

`interstellar/replay.py:201-220` (`isolated_workdir`), called from
`interstellar/replay.py:307`, with `scratch` derived from `out_dir` at
`interstellar/cli.py:1109` / `1126`, and `out_dir` defaulting to `runs/<timestamp>`
resolved against the invocation cwd at `interstellar/cli.py:1347`.

`isolated_workdir` is an unconditional `shutil.copytree(src, dest, ignore=_COPY_IGNORE)`.
`_COPY_IGNORE` (`replay.py:197-198`) excludes `.git`, `target`, `node_modules`,
`__pycache__`, `.venv` — it does **not** exclude the cycle's own output directory.
`workspace` is the trace's recorded `session.cwd` (`cli.py:988`). Nothing anywhere
checks whether `out_dir` is a descendant of `workspace`.

**Failure scenario, verified.** The stated purpose of this system is "take a captured
grok-build agent session" — i.e. a session the user recorded while working in their
repo, so `session.cwd` is the repo. Run the documented command from that repo:

```
python3 -m interstellar review traces/<my-session>.json --k 3
```

`--out` defaults to `runs/<ts>` → `<repo>/runs/<ts>`, which is inside `workspace`.
`copytree` then recurses into `runs/` and copies the destination subtree into itself
until it dies on path length. Reproduced through the real `run_matrix`:

```
control   ok=False | setup failed: Error: [('/…/workspace/runs/ts/scratch/p1/runs/
                     control-0-workdir/runs/ts/scratch/p1/runs/control-0-workdir/…
treatment ok=False | setup failed: Error: [(…same…
```

Every one of the `2*k` runs, for every patch, becomes `ok=False`. Downstream this is
handled *gracefully and wrongly*: `stats.gate`'s zero-usable-pairs short-circuit
fires, the badge renders **NO DATA**, and `run_review` exits with

> "all N replay run(s) failed this cycle — no usable data was produced. … **a common
> cause is an auth/environment failure.**"

That message points the user at their login. The actual bug is a path bug. This is
precisely the failure shape of the already-fixed relative-`GROK_HOME` Critical
(progress.md, fix round 3): a path mistake that presents itself as an auth problem
and burns the preflight call (real money) before failing.

Why no prior review or live cycle caught it: every corpus trace records
`cwd = /Users/sentorcc/Interstellar/synth/workspace`, a subdirectory that happens not
to contain `runs/`. The three live cycles wrote to `<repo>/runs/…`, outside
`synth/workspace`. The bug is invisible to the corpus and fires on the primary
use case.

Secondary consequence even when it does not recurse: the copy is unbounded. Each run
copies the whole workspace, and `runs/*/scratch` is already 650 MB across three
cycles — so a repo-rooted workspace pays a multi-hundred-MB copy `2*k` times per patch.

Fix shape (not applied): refuse to start when `out_dir` is inside `workspace`, or add
the resolved `out_dir` to the copy's ignore predicate, or both. Either way this needs
to fail loudly at plan time, not degrade into a misdiagnosed no-data cycle.

---

## Important

### I1 — the reported "cycle cost" is structurally incomplete but rendered as a measurement

`interstellar/cli.py:1168` (`total_cost += _sum_run_costs(matrix)`),
`interstellar/cli.py:152` (`_sum_run_costs` — replay runs only),
`interstellar/grade.py:378` (`_default_grok` parses grok's envelope and discards
`total_cost_usd`), `interstellar/cli.py:859` (`_run_preflight` — a real billed
grok-dev call whose cost is never recorded), rendered at
`interstellar/report.py:871` as `cycle cost $X` with no qualifier.

Verified arithmetic on the real headline run: `report.json`'s `cost_usd` is
`0.824041`, and the sum of `cost_usd` across the 18 replay runs in the three
`matrix.json` files is `0.824041` — identical to the cent. So the figure contains
*only* replay runs. Not counted: the preflight call, and every judge call. The judge
makes two calls per pair (position swap), now with up to one retry each
(`grade.py:_JUDGE_CALL_MAX_ATTEMPTS = 2`), so a k=3 × 3-patch cycle can bill up to 36
judge invocations that contribute $0 to the number the dashboard calls "cycle cost".

The *estimate* path is scrupulous about this — `_cost_estimate`'s note explicitly
says judge and synthesis calls "are not separately measured in this corpus and are
excluded". The *actual* figure inherits none of that honesty. Given Global Constraint
6 ("Every figure in the report traces to a measured value… Estimates are labelled
`_est`"), a headline dollar figure that silently omits two real spend categories is
the one number on the page a reader would most reasonably treat as complete.

Fix shape: capture `total_cost_usd` from the judge subprocess and the preflight into
the total, or rename the pill and add a caveat naming what is excluded.

### I2 — `DIRECTIONAL · FAVORABLE` is rendered with zero decided judge pairs, and the guard that would catch it is unreachable at the default k

`interstellar/report.py:710-714` (`_gate_badge`'s `zero_quality_evidence`),
`interstellar/stats.py:658-692` (`_directional`).

`_gate_badge` correctly refuses to let a bare green **ACCEPTED** stand on an all-ties
result — it downgrades to `ACCEPTED · NO DECIDED PAIRS`. But that guard is applied
*only* to the `accepted` branch (`report.py:714`). The `DIRECTIONAL` branch
(`report.py:719`) gets no equivalent check, and `stats._directional` can return
`"favorable"` from the efficiency point estimate alone when `quality_sign == 0`
(`stats.py:688`).

The interaction is the problem: `stats.MIN_SAMPLES_FOR_ACCEPT = 5` makes `accepted`
**structurally unreachable** at the CLI's default `--k 3`. So the guard fires only in
a state that cannot occur at default settings, while the state that occurs on every
real run is unguarded.

Verified in the actual rendered dashboard, `runs/headline/index.html`:

```
patch-001 skill.truncate strict-audit  →  DIRECTIONAL · FAVORABLE
patch-017 mcp.remove fetch             →  DIRECTIONAL · FAVORABLE
```

Both have `win_rate = {wins: 0, ties: 1, losses: 0, n: 1}` against
`data.pairs_usable = 3` — i.e. **zero decided pairs, and 2 of the 3 pairs were never
judged at all**. A scanning reader sees an amber-blue FAVORABLE badge on a patch with
no quality evidence whatsoever. The efficiency numbers underneath are real and good
(-46.9% skill tokens); the badge's implied quality claim is not backed by anything.

Note the sub-case is now partly self-reporting: current `grade.grade_matrix` records a
`judge_unavailable` low-severity regression with a reason for each unjudged pair
(added in `6bcd070`), which would render in the Regressions list. That explains the
absence; it does not stop the badge from claiming a favorable reading over it.

Fix shape: extend the `zero_quality_evidence` downgrade to the DIRECTIONAL branch, or
have `_directional` return `"neutral"` (not `"favorable"`) when `quality_sign == 0`
and no pair was decided.

### I3 — nothing verifies the harness being patched is the harness the trace recorded

`interstellar/cli.py:_resolve_baseline_home` → `harness.snapshot`, and
`interstellar/patches.py:373-374` (`skill_lines = skill.get("lines")`, validated
against the *snapshot*).

The plan (Task 7) says to snapshot "from the trace's own recorded harness where
possible". What the code actually does is take the *directory path* from the trace's
`session_dir` ancestor and read that directory's **current** contents. There is no
content hash, byte count, or line count compared against what the trace recorded.

`patches._skill_truncate_patch` then validates the analyzer's cited line ranges only
against the current file's length. The evidence to do better already exists and is
already in scope: the analyzer digest carries per-skill `lines` and `tokens_est`
(`analyzer/analyze.py:75-77`), and the trace's `metrics.skill_detail` carries
`tokens_est` / section `start_line`/`end_line`. `patches.from_recommendations`
receives `digest` but uses it only for `installed_skills_never_loaded` and MCP info
(`patches.py:556-558`) — `skills_loaded` is never read.

**Failure scenario:** a user edits `strict-audit/SKILL.md` (adds a paragraph, moves a
section) after the session was recorded but before running the review. The analyzer's
ranges `[[12,20],[34,51],…]` still fall inside the file's length, so validation passes
silently, and `harness.apply` deletes 60 lines of *different* content. Both arms are
still isolated and still differ only by that patch, so the statistics are sound — the
report correctly measures a patch nobody intended, and prints an orphan-fragment count
computed on the wrong text as though it described the recommendation.

This is not a confound; it is a wrong edit measured correctly, which the report has no
way to flag.

### I4 — dropped and lint-rejected patches can disappear from the report entirely, and the purpose-built surface for them is dead code

`interstellar/cli.py:714-722` (`_base_extra_caveats`, `shown = dropped_patches[:8]`)
vs. `interstellar/report.py:176` (`_dropped_patch_caveat`) and
`interstellar/report.py:200` (`build(..., dropped_patches=None, ...)`).

`report.build` has a first-class, non-truncating, per-patch surface for dropped
patches that renders id, kind, target, skip reason, rationale and source
recommendation. **`cli.py` never passes `dropped_patches=`** — grep confirms zero
production callers. Instead the CLI rolls its own single joined caveat line and
truncates it to the first 8, appending "and N more (see progress.json)".

Verified on the real headline run: 20 patches dropped, 8 rendered in the report, 12
reachable only by opening `progress.json`. That list also carries patches rejected
*before* ranking — security-lint hits, invalid line ranges, "skill not found" — mixed
in with ordinary budget cuts, so a lint-rejected patch can fall past the cutoff and
never appear in the report at all.

`types.make_patch`'s own frozen docstring states the requirement being violated:
"`skipped_reason` non-None means the patch was rejected before application … and must
still appear in the report — a silently dropped recommendation is indistinguishable
from one that was never made."

Two independent implementations of the same job, one of them dead and better, is also
exactly the duplication the whole-branch view exists to catch.

### I5 — unknown analyzer recommendation types are dropped with no trace anywhere

`interstellar/patches.py:578-580`.

An unrecognized `rec["type"]` falls off the end of the dispatch chain and produces
*no* Patch — not even a `skipped_reason` one. It therefore never reaches
`_rank_and_select`, never reaches `dropped_patches`, never reaches `progress.json`,
and never reaches the report. The inline comment justifies not *raising*, which is
right; it does not justify not *recording*.

Same class as I4: a recommendation the analyzer made becomes indistinguishable from
one it never made. Low probability (the analyzer's schema is an enum today) but the
comment explicitly anticipates version skew, which is exactly when it would bite.

---

## Minor

- **M1 — `matrix["k"]` is wrong on the fail-fast path.** `interstellar/cli.py:1155`
  sets `matrix["k"] = k` only inside the `elif k > 1:` branch. If the first paired run
  fails on both arms, `k` stays `1` on the persisted matrix while the report header
  and `_small_n_caveat` say `k=3`. The `fail_fast_reason` text on the verdict line
  covers it in practice, but two different k values ship in the same document.

- **M2 — dead step in `run_matrix`.** `interstellar/replay.py:308-310` deletes
  `<workdir>/.grok/skills` before calling `materialize`, and
  `harness.materialize` (`harness.py:413-415`) already wipes that exact directory
  unconditionally, "even when this version has zero project skills". The replay-side
  rmtree can never observe anything the materialize would not have removed.

- **M3 — `materialize` mutates its caller's version dict from worker threads.**
  `interstellar/harness.py:431` and `:438` append to `version["warnings"]` in place,
  guarded by `if warning not in version["warnings"]`. `run_matrix` calls it `2*k`
  times against the *same* `baseline_version` / `treatment_version` objects, from up
  to `max_parallel` threads. Check-then-append is not atomic, so duplicate caveats are
  possible; more structurally, a "pure derive a version" module writing into an object
  the orchestrator later reads for its caveats is surprising action-at-a-distance.

- **M4 — `report._interval_dict`'s docstring is inverted.**
  `interstellar/report.py:297-307` claims the pinned shape carries `lo`/`hi`/`note`
  flat on the metric dict and that nested `"bootstrap"` is a defensive fallback.
  `stats.efficiency_deltas` (`stats.py:501-509`) emits `{lower_is_better, n,
  control_median, treatment_median, abs_delta, pct_delta, bootstrap}` — there is no
  flat `lo`/`hi` and never has been. The "fallback" is the only branch ever taken; the
  documented primary branch is dead.

- **M5 — other dead code across module boundaries** (only visible from here):
  `stats.sign_test` has no caller (superseded by `permutation_test`; `report.py:538`
  reads a `"sign_test"` key nothing ever writes); `grade.judge_pair` is exercised only
  by tests (`grade_matrix` uses `_judge_pair_impl`); `report._fmt_pass_k`'s
  `{control, treatment}` branch and `report._pass_k_line`'s matching legacy branch
  (`report.py:572`) handle a shape `stats.pass_k` cannot produce.

- **M6 — scratch is never cleaned, and arm names are visible to the agent under test.**
  `runs/*/scratch` currently holds **43 copies of `auth.json`** and ~650 MB across
  three cycles. Mode is correct (`0600`, `harness.py:451`) and `report.serve` is
  correctly a two-file allow-list rather than a directory root, so this is hygiene,
  not exposure. Separately, each run's cwd is literally named
  `control-<n>-workdir` / `treatment-<n>-workdir` (`replay.py:304-305`) and is visible
  to the agent under test — a response that cites its cwd would leak the arm to the
  judge, defeating the position-swap blinding. **I checked all 18 headline responses:
  no leak occurred.** The mechanism exists; the exposure did not materialize.

- **M7 — over-stated advice in the cycle summary.** `interstellar/cli.py`'s
  `_print_cycle_summary` says k is "below the {MIN_SAMPLES_FOR_CI} paired repeats the
  acceptance gate requires for a real accept/reject call". The gate can reach a
  provisional accept at `n >= MIN_SAMPLES_FOR_ACCEPT` (5), per `stats.gate`'s own 5–9
  branch.

---

## What held up (checked, no finding)

These are the things most likely to be silently broken in a system like this. I
verified them rather than assuming the module reviews covered them:

- **Constraint 4 — the experiment is valid.** `cli.py` always passes
  `baseline_version` as `control=` and `harness.apply(baseline_version, [patch])` as
  `treatment=` to `run_matrix`; the user's recorded session appears only as a caveat
  string in `_base_extra_caveats`, never as an arm. Both arms are materialized by the
  same `materialize` call path, each into its own `<arm>-<repeat>-home`, each with its
  own workdir copy, and both are sealed identically. The only asymmetries between the
  two argv are `--rules` (which *is* the patch for `rules.append`) and the
  home/workdir path names (M6). Dispatch is interleaved control/treatment per repeat.
  I could not find a shared `sessions/`, `config.toml`, or `.grok/skills/` between
  arms, nor a skill root reachable by one arm and not the other.

- **Constraint 6 — missing measurement is `None`.** Traced end to end.
  `efficiency()` returns all-`None` for a failed run rather than partial figures;
  `_metric_pairs` drops a metric-pair when either side is `None`; `paired_bootstrap`
  returns `lo=hi=None` below n=10 rather than a fabricated interval; `_metric_row`
  intercepts the both-medians-`None` case with an explicit "no paired data" cell
  before `_fmt_pct` can be reached; `_fmt_pct(None)` is reserved for the genuinely
  different "control median is exactly 0" case; `_esc(None)` renders `""` rather than
  a dash so no statistic can inherit a silent em-dash. `stats.gate`'s no-data
  short-circuit correctly distinguishes `"unknown"` (nothing measured) from
  `"neutral"` (measured, came out even), and `report._gate_badge` renders those as
  distinct states. The one place a figure is presented as complete when it is not is
  I1 (cost).

- **Constraint 1 — stdlib only.** Import sweep across `interstellar/` and its tests
  found nothing outside the standard library.

- **Constraint 8 — tests never call grok-dev.** Grep sweep confirms: the only real
  subprocess calls in the suite are `git init`/`git config` in `test_replay.py`, and
  `test_cli.py` installs a `subprocess.run` mock that *raises* if anything tries to
  spawn.

- **Self-contained report.** `runs/headline/index.html` contains zero `http://` or
  `https://` resource references.

---

## Task 8 (end-to-end verification) status

| # | Requirement | Status |
|---|---|---|
| 1 | Full suite green | **Yes** — 407 tests, OK |
| 2 | `--dry-run` over every corpus trace, no crash | **Yes** — 14/14, each produced patches |
| 3 | One real `--k 3` cycle on `skill_bloat` | **Ran, but the artifact is stale** |
| 4 | One negative control on `baseline_clean` | **Yes** — `runs/control/`, 2 patches, neutral/unfavorable, nothing accepted, no manufactured improvement |
| 5 | Report renders and serves; numbers captured | **Renders** (verified); serve is loopback-only + two-file allow-list |

Two gaps worth naming:

- **Item 3's artifact predates the code.** `runs/headline/report.json` carries verdict
  lines reading `REJECTED (trends favorable)` — wording that commit `6bcd070` replaced
  with `DIRECTIONAL: FAVORABLE` — and carries no `judge_unavailable` regressions,
  which the same commit added. So the headline proof was produced by an earlier build
  and is not evidence for the current code. `runs/final/` looks like the intended
  re-run and contains `scratch/` + `progress.json` but **no `report.json` and no
  `index.html`** — that cycle did not complete. Item 3 should be re-run against the
  merge candidate.
- **Item 2's "or an explicit 'no actionable recommendation'" branch is never
  exercised.** `baseline_clean` — the trace designed to have nothing wrong with it —
  still yields 3 selected patches (removals of never-loaded skills / unused MCP
  servers, all scoring `impact_score=0`). The gate does not manufacture an improvement
  from them (that part of item 4 holds), but the CLI's `if not selected_patches:`
  path has no trace in the corpus that reaches it.

---

## Uncommitted work in the tree

At review time the working tree contains uncommitted changes to `interstellar/cli.py`,
`interstellar/report.py`, and `interstellar/tests/test_report.py` adding a
prompt-cache-warmth caveat. It is **half-wired**, and one part violates a Global
Constraint:

1. `_cache_warmth_caveat` and `_cache_read_range` are defined and have no caller.
2. `rpt["cache_sensitive_metrics"] = CACHE_SENSITIVE_METRICS` is set on the
   **no-patches** report path only (`cli.py:1076`), not on the main path that every
   real cycle takes.
3. That assignment **adds a key `types.make_report` does not declare**. Global
   Constraint 2: "`interstellar/types.py` is the interface contract and is frozen…
   Do not add, rename, or repurpose keys. If a shape is genuinely wrong, stop and
   report it — do not unilaterally change it."
4. `report.py` gained CSS for a `.tag.cache-tag` / `.cache-legend` that no renderer
   emits.

The suite is still green with these changes in place. The idea is sound and the
concern is real — cache warmth genuinely does move `total_tokens`/`cost_usd`
independently of the patch — but it should land completely (through a
contract-respecting surface, e.g. an `extra_caveats` string like every other caveat
in the system) or be reverted before merge.

---

## Overall assessment

**Not ready to merge as-is. One thing blocks it.**

The engineering quality here is high and the honesty discipline is unusually good —
the gate refuses to accept at k=3 by construction, the no-data state is genuinely
distinguished from a measured null, `pass^k` uses the correct combinatorial estimator,
the permutation test publishes its own achievable floor, the caveats block is rendered
first and cannot be collapsed, and the position-swap blinding is real. The four prior
module reviews plus three fix rounds clearly did their job: I could not find a
recurrence of anything already ruled on.

**What blocks merge: C1.** The system's headline use case — review a session you
captured while working in your own repo — fails 100% of its runs with the documented
default `--out`, and then tells the user to check their auth. It is unreproducible from
the corpus (whose workspace is a subdirectory that dodges the collision), it costs a
real preflight call before failing, and it is the same misdiagnosis-as-auth shape as
the relative-`GROK_HOME` Critical that fix round 3 already had to chase. It is also
cheap to fix: refuse at plan time when `out_dir` is inside `workspace`, and/or ignore
the resolved `out_dir` during the copy.

**Strongly recommended before merge, because they undercut the one thing this system
sells — trustworthy numbers:**

- **I1** — a headline dollar figure labelled "cycle cost" that provably omits every
  judge call and the preflight.
- **I2** — a `FAVORABLE` badge on a patch with zero decided judge pairs, in the exact
  state (`k=3`) that every default run lands in, while the guard that exists for this
  can only fire in a state `k=3` cannot reach.

**Should be scheduled, not necessarily blocking:** I3 (stale-harness line ranges — a
correctness risk, not a statistics risk), I4/I5 (dropped recommendations that leave no
trace, plus the dead `report.build(dropped_patches=)` path that was built to solve
exactly that).

Also required before calling the branch verified: land or revert the uncommitted
cache-warmth work (it currently violates Constraint 2 and is only half-connected), and
**re-run the Task 8 item 3 headline cycle** — the committed artifact was produced by
an older build and `runs/final/` never finished.
