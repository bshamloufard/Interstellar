# Interstellar Review Loop — implementation plan

Close the loop on a captured grok-build session: **analyze → recommend → apply in
isolation → replay → grade → prove**.

Stages A (normalize) and B-lite (analyzer recommendations) already exist
(`normalizer/`, `analyzer/`, `synth/`, `traces/corpus/`). This plan builds
stages C, D, E and the report:

```
trace.json ──> analyzer (exists) ──> recommendations
                                          │
                            T2 patches.py │ typed, bounded, linted patches
                                          ▼
      T1 harness.py: baseline home ──> candidate home (patch applied)
                                          │
                      T3 replay.py  ──────┼──────  k paired runs of the ORIGINAL prompt
                       control (baseline) │ treatment (candidate)
                                          ▼
                      T4 grade.py: efficiency meters + position-swapped
                                   pairwise judge + regression detectors
                                          ▼
                      T5 stats.py:  paired bootstrap CI, win-rate, pass^k, gate
                                          ▼
                      T6 report.py + web/: scorecard with proof
                                          ▲
                      T7 cli.py: `interstellar review <trace>` drives all of it
```

## Global Constraints

These bind every task. A reviewer checks against them verbatim.

1. **Python 3.12, standard library only.** No pip installs, no numpy/scipy/pydantic.
   Match the existing code's style: plain dicts, module-level functions, no classes
   unless they carry real state.
2. **`interstellar/types.py` is the interface contract and is frozen.** Every task
   consumes and produces exactly the shapes it declares. Do not add, rename, or
   repurpose keys. If a shape is genuinely wrong, stop and report it — do not
   unilaterally change it.
3. **Isolation is `GROK_HOME`.** Every replay sets `GROK_HOME` to a materialized
   candidate home. Verified working: skills, `config.toml`, and `sessions/` all
   resolve from it, and `auth.json` copied in gives inherited login. Never mutate
   the user's real `~/.grok`.
4. **The comparison is control-vs-treatment, both isolated.** Never compare a
   patched isolated run against the user's original recorded run — that confounds
   the patch with the environment change. The baseline harness is re-run under the
   same isolation as the candidate. The original recorded run is reported as
   context only, never as the statistical control.
5. **Every model call is a child `grok-dev` process** at
   `~/.local/bin/grok-dev`, with `--output-format json` and `--json-schema` for
   structured output. No direct API calls, never read `auth.json` contents.
6. **No fabricated numbers.** Every figure in the report traces to a measured
   value in a normalized trace or a computed statistic. Estimates are labelled
   `_est`. When k is too small for a CI, say so instead of printing one.
7. **Bounded patches.** A patch may touch only its declared target path, and may
   change at most `MAX_PATCH_LINES` (60) lines. Generated skill text passes the
   security lint before it is written.
8. **Tests are stdlib `unittest`, live in `interstellar/tests/`, and must not
   call `grok-dev`.** LLM-dependent paths are tested against recorded fixtures.
   `python3 -m unittest discover interstellar/tests` is the suite command.

---

## Task 1 — `interstellar/harness.py`: snapshot & materialize

**Goal:** turn a real grok home + project dir into a content-addressed
`HarnessVersion`, and materialize any version as a runnable isolated `GROK_HOME`.

**Implement:**

- `snapshot(grok_home: Path, project_dir: Path|None = None) -> HarnessVersion`
  - Skills: every `<grok_home>/skills/*/SKILL.md` → `{name, path, bytes, lines,
    sha256, frontmatter_description}`. Also read project `.grok/skills/*` when
    `project_dir` is given, marked `scope: "project"` vs `"user"`.
  - MCP servers: parse `<grok_home>/config.toml` (stdlib `tomllib`) →
    `{name, command, args, enabled}`. Tolerate a missing/!unparseable config
    by returning `[]` plus a warning, never by raising.
  - Config: capture the raw `config.toml` text and its sha256.
  - `version_id` = sha256 over the sorted `(name, sha256)` skill pairs + config
    sha, truncated to 12 hex chars. Two identical harnesses must produce the
    same id; any skill byte change must change it.
- `materialize(version: HarnessVersion, dest: Path, *, auth_from: Path) -> Path`
  - Builds a fresh `GROK_HOME` at `dest`: `skills/<name>/SKILL.md` written from
    the version's recorded content, `config.toml` written from recorded text,
    `auth.json` **copied** (not symlinked — grok rewrites it on refresh) from
    `auth_from/auth.json`. Creates `sessions/`.
  - Returns `dest`. Idempotent: re-materializing over an existing dest replaces
    skills/config wholesale so no stale skill survives.
- `apply(version: HarnessVersion, patches: list[Patch]) -> tuple[HarnessVersion, list[PatchApplication]]`
  - Pure: derives a new `HarnessVersion` with patch edits applied to the recorded
    skill/config content. Never touches disk. Each `PatchApplication` records
    `applied: bool`, `reason` when skipped, and the resulting unified diff.
  - Patch kinds it must handle: `skill.truncate` (delete line ranges),
    `skill.remove` (drop the skill), `skill.insert` (insert lines at an anchor),
    `mcp.remove` (drop a `[mcp_servers.X]` table from config.toml text),
    `rules.append` (append a line to the version's `extra_rules` list).
  - Line ranges are 1-based inclusive, must be validated against the actual file
    length, and must be applied high-to-low so earlier deletions don't shift
    later ranges. An out-of-range range is a skipped patch with a reason, not a
    crash.

**Tests** (`interstellar/tests/test_harness.py`): snapshot determinism; version
id changes on a skill byte change; truncate deletes exactly the cited lines and
nothing else; overlapping and descending ranges; out-of-range is skipped not
raised; `mcp.remove` removes only the named table; materialize produces a home
whose skills match the version and contains no skill the version dropped.

---

## Task 2 — `interstellar/patches.py`: recommendations → typed patches

**Goal:** convert analyzer recommendations (the existing
`analyzer/analyze.py` output shape) into `Patch` objects with real, bounded,
lint-clean edits.

**Implement:**

- `MAX_PATCH_LINES = 60`.
- `from_recommendations(recs: list[dict], version: HarnessVersion, *,
  digest: dict) -> list[Patch]` — maps analyzer types onto patch kinds:

  | analyzer `type` | patch kind | how |
  |---|---|---|
  | `skill_truncate` | `skill.truncate` | line ranges taken **verbatim** from the recommendation; drop the patch if it cites none |
  | `skill_remove` | `skill.remove` | one patch per named skill (the analyzer collapses these into one recommendation naming several — split them) |
  | `skill_add_lines` | `skill.insert` | needs generated text — see `synthesize_insert` |
  | `mcp_remove` | `mcp.remove` | split a multi-server recommendation into one patch per server |
  | `tool_redundant`, `tool_output_truncate`, `subagent_inline`, `permission_friction` | `rules.append` | one imperative line appended to the harness rules |
  | `mcp_latency`, `mcp_fix_broken` | `mcp.remove` | only when the recommendation's own action says to remove/disable; otherwise skip with a reason |
  | `skill_keep` | — | never produces a patch |

  Every produced patch carries `source_recommendation` (the whole rec dict),
  `expected_effect` (which measured metric should move: one of
  `skill_tokens`, `tool_calls`, `mcp_startup_ms`, `tool_result_tokens`,
  `wall_ms`), and `rationale`.
- `synthesize_insert(rec, skill_text, *, grok=...) -> str|None` — one
  `grok-dev --json-schema` call producing ≤4 lines of skill text plus an anchor
  line number. The `grok` argument is an injectable callable so tests can pass a
  fake; the default implementation shells out.
- `security_lint(text: str) -> list[str]` — returns violation strings for
  generated skill/rule content: URLs, `curl`/`wget`/`nc`, base64 blobs, prompt
  injection markers (`ignore previous`, `disregard the above`, `system:`),
  instructions to disable safety/permissions, credential/env exfiltration
  (`auth.json`, `XAI_API_KEY`, `~/.ssh`). A patch with any violation is dropped
  with the violations recorded in `skipped_reason`.
- `dedup(patches) -> list[Patch]` — collapse patches with identical
  `(kind, target, diff_key)`; keep the first, record the merged count.

**Tests** (`test_patches.py`): each mapping row; a `skill_truncate` with no
ranges is dropped; a multi-server `mcp_remove` splits; the lint catches each
violation family and passes clean text; a >60-line patch is rejected; dedup;
`synthesize_insert` with a fake grok callable (both a good response and a
lint-failing one).

---

## Task 3 — `interstellar/replay.py`: isolated paired execution

**Goal:** run one prompt under a materialized harness in an isolated sandbox,
capture the session, normalize it, and return a `RunResult`. Then run the
control/treatment k-repeat matrix.

**Implement:**

- `run_once(prompt, home: Path, *, workdir: Path, timeout=600, model=None,
  extra_rules: list[str]) -> RunResult`
  - `subprocess.run([GROK, "-p", prompt, "--output-format", "json",
    "--permission-mode", "bypassPermissions", ...])` with
    `env={**os.environ, "GROK_HOME": str(home)}` and `cwd=workdir`.
  - `extra_rules` are passed with `--rules`.
  - Locate the new session dir under `<home>/sessions/`, normalize it with
    `normalizer.grok_normalize.normalize_session`, and attach the normalized
    trace. Session discovery must diff the sessions dir before/after — never
    assume the newest mtime, since repeats run concurrently.
  - Record `ok=False` with `error` on timeout or unparseable stdout; never raise.
- `isolated_workdir(src: Path, dest: Path) -> Path` — copy the project workspace
  into a throwaway dir so file mutations from one run cannot leak into another.
  `git worktree` when `src` is a repo, plain copy otherwise.
- `run_matrix(prompt, *, control: Path, treatment: Path, k: int, workspace: Path,
  scratch: Path, max_parallel: int = 3, ...) -> ReplayMatrix`
  - k repeats of each arm, every run in its own workdir and its own copy of the
    home directory (so concurrent runs never share a `sessions/` dir or a
    `config.toml`). `concurrent.futures.ThreadPoolExecutor` for parallelism.
  - Returns arms keyed `"control"` / `"treatment"`, each a list of `RunResult`.

**Tests** (`test_replay.py`): the grok invocation is built through an injectable
`runner` callable, so tests assert the argv and env (notably `GROK_HOME`) without
spawning anything; session-dir diffing picks the new dir and ignores pre-existing
ones; a timeout produces `ok=False`; `run_matrix` produces `2*k` results and
tolerates one arm's run failing.

---

## Task 4 — `interstellar/grade.py`: measure and judge

**Goal:** turn a `ReplayMatrix` into per-run `Grade` records.

**Implement:**

- `efficiency(trace) -> dict` — pulled straight from the normalized trace, no
  recomputation: `total_tokens`, `cost_usd`, `wall_ms`, `tool_calls`,
  `skill_tokens_est`, `skill_wasted_tokens_est`, `tool_result_tokens_est`,
  `mcp_startup_ms`, `duplicate_calls`, `turns`.
- `regressions(control_trace, treatment_trace) -> list[dict]` — detector signals
  present in treatment but not control: new failed/idle MCP servers, new
  duplicate call pairs, a non-`ok` session status, new error statuses on spans,
  a turn count increase >50%. Each with `severity` in `{high, medium, low}`.
- `judge_pair(prompt, response_a, response_b, *, grok=..., model=None) -> dict`
  - **Position-swapped**: two calls, (A,B) then (B,A). Tie is a legitimate
    verdict. Rubric anchors: correctness w.r.t. the prompt, constraint
    adherence, tool-use quality, completeness; efficiency breaks ties only at
    equal correctness; any new failure behavior loses outright.
  - Returns `{verdict: "control"|"treatment"|"tie", consistent: bool,
    raw: [both call results]}`. `consistent=False` when the two orders disagree
    — and an inconsistent pair is reported as a **tie**, never silently resolved
    in either direction.
  - `grok` is injectable for tests.
- `grade_matrix(matrix, *, prompt, grok=...) -> list[Grade]` — pairs
  `control[i]` with `treatment[i]`, computes efficiency for both, judges the
  pair, and collects regressions.

**Tests** (`test_grade.py`): efficiency extraction against a corpus trace
fixture from `traces/corpus/`; regression detection finds a newly failed MCP
server and ignores one present in both; `judge_pair` with a fake grok returns
`tie` + `consistent=False` when the two orders disagree, and the winner when
they agree; a fake grok returning garbage degrades to a tie rather than raising.

---

## Task 5 — `interstellar/stats.py`: paired statistics and the gate

**Goal:** pure-Python paired statistics over `Grade` lists. No dependencies.

**Implement:**

- `paired_bootstrap(deltas: list[float], *, iters=10000, seed=0) -> dict` —
  `{mean, lo, hi, level: 0.95}` via percentile bootstrap using
  `random.Random(seed)` so results are reproducible.
- `win_rate(grades) -> dict` — `{wins, ties, losses, n, rate, lo, hi}` where
  `rate` counts treatment wins over decided pairs, CI by bootstrap over pairs.
  With ties dominant, the rate must still be reported with its tie count, not
  silently dropped.
- `sign_test(deltas) -> dict` — exact two-sided binomial p-value computed with
  `math.comb`, zeros excluded. `{p, n_nonzero, direction}`.
- `pass_k(successes: list[bool]) -> float` — fraction of runs that succeeded,
  raised to k, i.e. the τ-bench reliability reading; document the estimator used
  in the docstring.
- `efficiency_deltas(grades) -> dict` — per metric: control median, treatment
  median, absolute and percent delta, and a bootstrap CI on the paired
  difference. Percent deltas guard against a zero control median.
- `gate(stats, *, max_token_regression=0.10) -> dict` — accept iff:
  quality does not regress (win-rate lower CI ≥ 0.5 **or** losses == 0), no
  `high`-severity regression in any pair, and the primary efficiency target
  improves. Returns `{accepted: bool, reasons: [...]}` — reasons are populated
  for both accept and reject so the report can always explain itself.
- Every function must handle `n <= 2` by returning `lo=hi=None` and a
  `"insufficient_samples"` note rather than a fake CI.

**Tests** (`test_stats.py`): bootstrap CI covers a known mean and is
seed-stable; `sign_test` matches a hand-computed binomial p for a small case;
`win_rate` with all ties; `pass_k`; gate accepts a clean improvement, rejects on
a high-severity regression, rejects on a token blowup past the threshold; every
function on `n=1` returns the insufficient-samples shape.

---

## Task 6 — `interstellar/report.py` + `interstellar/web/`: the proof

**Goal:** assemble a `Report` and render it as a local dashboard.

**Implement:**

- `build(...) -> Report` per the `types.py` shape: session context, harness
  diff, each candidate patch with its diff, its measured stats, its gate
  decision, the caveats block (k, isolation note, judge consistency rate,
  insufficient-sample flags), and a `verdict` line per patch in plain language.
- `write(report, out_dir)` — `report.json` plus a **self-contained**
  `index.html` (inline CSS/JS, no CDN, no build step) that renders:
  1. header: session, harness version ids, total cost of the cycle, k;
  2. per patch: the unified diff, the recommendation it came from, a
     before/after metric table with deltas and CIs, the win/tie/loss bar, gate
     badge (accept / reject) and the reasons;
  3. caveats block, verbatim from the report — never omitted.
  Dark and light both readable; no horizontal page scroll; diffs and tables
  scroll inside their own containers.
- `serve(out_dir, port=4242)` — `http.server` bound to **127.0.0.1 only**.

**Tests** (`test_report.py`): `build` on a fixture produces every documented key;
`write` emits html containing the patch diff text and the caveats; the html has
no `http://` or `https://` external resource references; `serve` binds loopback.

---

## Task 7 — `interstellar/cli.py`: the orchestrator

**Goal:** one command, end to end.

```
python3 -m interstellar review traces/corpus/skill_bloat_*.json \
    --k 3 --max-patches 2 --out runs/<ts>/ [--dry-run] [--serve]
```

Steps: load trace → run the existing `analyzer.analyze` (import it, do not
re-implement) → `patches.from_recommendations` → snapshot the baseline harness
**from the trace's own recorded harness where possible**, else the live
`~/.grok` → for each patch: `harness.apply`, materialize control and treatment
homes, `replay.run_matrix` on the trace's original prompt, `grade.grade_matrix`,
`stats` → `report.build/write` → optionally `serve`.

Requirements: `--dry-run` performs every step except spawning grok and prints
the patches and the planned run matrix with a **cost estimate** (runs × observed
mean cost from the corpus); a `--budget-usd` cap that refuses to start a cycle
whose estimate exceeds it; journaled progress to `out/progress.json` after each
patch so an interrupted cycle can be inspected; and it must never write outside
`--out` and the scratch dir.

**Tests** (`test_cli.py`): argument parsing; `--dry-run` on a real corpus trace
produces patches and a cost estimate without spawning anything (inject fakes);
`--budget-usd 0` refuses; progress journaling after each patch.

---

## Task 8 — End-to-end verification

Not a code task — a verification run, executed by the controller.

1. Full suite green: `python3 -m unittest discover interstellar/tests -v`.
2. `--dry-run` over every trace in `traces/corpus/` — every trace must produce
   patches or an explicit "no actionable recommendation", and no crash.
3. One **real** cycle with `--k 3` on `skill_bloat` (ground truth: `strict-audit`
   loads 5.8 KB of which only the 6-line Procedure section drives behavior, so a
   `skill.truncate` should cut ~1.2k tokens with no quality loss). This is the
   headline proof: measured token reduction, judge verdict not a loss, gate
   accepts.
4. One **negative control**: a cycle on `baseline_clean`, where the analyzer
   should propose nothing session-specific and the gate should not manufacture
   an improvement.
5. Report renders and serves; screenshots/verbatim numbers captured for the
   summary.
