"""interstellar/cli.py — the review-loop orchestrator.

One command, end to end: load a captured grok-build session trace, get fix
recommendations from the analyzer, turn them into bounded harness patches,
apply each patch to an isolated copy of the harness, replay the trace's
original prompt under both the unpatched (control) and patched (treatment)
harness, grade the paired runs, run the statistics and acceptance gate, and
write a self-contained report that shows whether the patch helped.

    python3 -m interstellar review traces/corpus/skill_bloat_*.json \\
        --k 3 --max-patches 3 --out runs/<ts>/ [--dry-run] [--serve]

Design points worth stating up front, because they are easy to get wrong
silently:

  * The comparison is control-vs-treatment, BOTH re-run fresh under this
    cycle's isolation. The user's original recorded session is reported as
    context only (see `_base_extra_caveats`) — it ran in a different
    environment and is never the statistical baseline.
  * `--max-patches` (default 3) is load-bearing, not a nicety: a handful of
    analyzer recommendations can expand into dozens of individual patches
    (one per skill/server named in a comma-joined recommendation target),
    and each tested patch costs `2 * k` grok-dev sessions. Patches are
    ranked by a real, trace-measured impact score (`_patch_impact_score`)
    within same-unit classes (`_patch_impact_unit` -- a millisecond score
    is never ranked against a token score) and interleaved round-robin
    across classes (`_rank_and_select`), so the selection spreads across
    the distinct kinds of evidence present instead of favoring whichever
    unit happens to produce the largest raw numbers. Every dropped patch
    is logged with why, impact score and unit included.
  * `--dry-run` and a live cycle both print a compact, human-readable
    summary as they go (prompt/baseline/selected+dropped patches/cost
    estimate for a dry run; a line per patch plus the report path for a
    live cycle) -- the JSON artifacts are the full record, but a command
    someone is about to spend real money on must say something on stdout,
    not run silently to a 0 exit code.
  * `--dry-run` performs analysis, patch construction/ranking, and the pure
    `harness.apply` diff preview, then stops — it never calls
    `harness.materialize`, `replay.run_matrix`, or spawns grok-dev. A
    `skill.insert` patch's synthesis call is also skipped in dry-run (via a
    no-op `grok` stub passed to `patches.from_recommendations`), so the
    whole path is guaranteed zero-cost regardless of what a live cycle
    would need.
  * Progress is journaled to `<out>/progress.json` after every patch (and
    at start/plan/finish) so an interrupted cycle can be inspected without
    re-running anything.
  * Every dollar figure is either a real, measured number (grok's own
    `total_cost_usd` on a run, or `meta.cost_usd` recorded by a past
    analyzer run) or is clearly labelled as an estimate and states which
    measurements it was derived from. Nothing is invented.
  * `total_tokens` and `cost_usd` are confounded by prompt-cache warmth
    (`cache_read_input_tokens` swings run to run independently of the
    harness -- interleaved dispatch blunts but does not eliminate it); the
    report says so with THIS cycle's own observed range
    (`_cache_warmth_caveat`), never a canned figure, and every report also
    carries `cache_sensitive_metrics` (`CACHE_SENSITIVE_METRICS`) naming
    which of the ten efficiency metrics are cache-driven versus counted
    straight off the trace/harness.
  * Before spending a cycle, one cheap grok-dev call runs through the
    baseline home (`_run_preflight`, skipped by `--dry-run` or
    `--no-preflight`) so an auth/environment failure is caught once instead
    of being rediscovered by every one of `2 * k` runs per patch. Within a
    patch, the first control/treatment pair always runs alone; if BOTH
    arms fail on it, the remaining `k - 1` repeats are skipped rather than
    re-discovering the same failure repeatedly, and the report says so. If
    every run across the whole cycle fails, `run_review` raises
    `SystemExit` after writing the report -- a cycle that produced no
    usable data is not a success, and exiting 0 would say otherwise.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from interstellar import grade, harness, patches, replay, report, stats
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
    PATCH_SKILL_INSERT,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_TRUNCATE,
    make_patch_result,
    make_replay_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_K = 3
DEFAULT_MAX_PATCHES = 3
DEFAULT_MAX_PARALLEL = 3
DEFAULT_TIMEOUT = 600
DEFAULT_PORT = 4242
DEFAULT_MAX_TOKEN_REGRESSION = 0.10


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------

def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path, data):
    """Write `data` as pretty JSON, creating parent dirs. Not atomic-fsync
    durable, but write-then-replace so a reader never sees a half-written
    file mid-journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _noop_patches_grok(prompt, schema):
    """`grok` stub for `patches.from_recommendations` that never spawns
    grok-dev. Used for --dry-run: a `skill.insert` recommendation simply
    shows up as a skipped patch ("produced no usable text") in the plan,
    which is the honest thing to show for a call that was never made."""
    return {}


def _default_stats_summarize(grades, *, primary_effect, seed=0):
    return stats.summarize(grades, primary_effect=primary_effect, seed=seed)


def _default_stats_gate(summary, *, primary_effect=None,
                        max_token_regression=DEFAULT_MAX_TOKEN_REGRESSION):
    return stats.gate(summary, primary_effect=primary_effect,
                      max_token_regression=max_token_regression)


def _strip_traces(matrix):
    """ReplayMatrix minus the full per-run traces (types.make_patch_result's
    `matrix_summary` contract) -- the full matrix, traces included, is
    persisted separately to `<out>/patches/<patch_id>/matrix.json`."""
    def _strip_run(run):
        run = dict(run)
        run["trace"] = None
        return run

    out = dict(matrix)
    out["arms"] = {arm: [_strip_run(r) for r in runs]
                   for arm, runs in (matrix.get("arms") or {}).items()}
    return out


def _sum_run_costs(matrix):
    total = 0.0
    for runs in (matrix.get("arms") or {}).values():
        for r in runs:
            c = r.get("cost_usd")
            if c is not None:
                total += c
    return total


def _count_runs(patch_results):
    """(total attempted runs, total ok runs) across every patch's matrix --
    used to detect a cycle where every single run failed (an auth/
    environment problem, most likely), which must not exit 0."""
    total = ok = 0
    for pr in patch_results:
        for runs in ((pr.get("matrix") or {}).get("arms") or {}).values():
            for r in runs:
                total += 1
                if r.get("ok"):
                    ok += 1
    return total, ok


# --------------------------------------------------------------------------
# prompt / harness-source resolution
# --------------------------------------------------------------------------

_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


def _message_text(entry):
    content = entry.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return ""


def _first_user_query(session_dir):
    """Recover the literal prompt grok-build was given: the text inside the
    first `<user_query>` tag in the raw session's `chat_history.jsonl`. This
    is the harness's own wrapper for the real user input (every grok-build
    system prompt names it: "denoted within the <user_query> tag") -- unlike
    the earlier synthetic `<user_info>`/system-reminder turns grok-dev
    injects before it, so matching the tag rather than "first user message"
    is what keeps this from replaying injected context instead of the
    actual ask. Returns None (never raises) when no such turn exists so the
    caller can fail loudly with full context instead of guessing."""
    path = Path(session_dir) / "chat_history.jsonl"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user":
            continue
        m = _USER_QUERY_RE.search(_message_text(entry))
        if m:
            return m.group(1).strip()
    return None


def _resolve_prompt(trace):
    """The prompt to replay. Corpus traces are documented to carry it under
    `ground_truth.prompt`; real traces will not, so fall back to the first
    `<user_query>` recorded in the trace's own session directory. Fails
    loudly rather than guessing -- replaying the wrong prompt would
    invalidate everything downstream. Returns (prompt, source_description).
    """
    ground_truth = trace.get("ground_truth") or {}
    prompt = ground_truth.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip(), "ground_truth.prompt"

    session = trace.get("session") or {}
    session_dir = session.get("session_dir")
    if session_dir:
        found = _first_user_query(session_dir)
        if found:
            return found, f"first <user_query> in {session_dir}/chat_history.jsonl"

    raise SystemExit(
        "error: could not determine the original prompt to replay -- this "
        "trace has no ground_truth.prompt and no <user_query> could be "
        f"recovered from session_dir={session_dir!r}. Refusing to guess: "
        "replaying the wrong prompt would invalidate every downstream "
        "measurement."
    )


def _resolve_baseline_home(trace, override):
    """Where to snapshot the baseline harness from: an explicit override,
    else the trace's own recorded home (derived from its session_dir --
    `<home>/sessions/<cwd-encoded>/<session-uuid>`, so the home is three
    parents up), else the live `~/.grok`. Returns (path, description).

    Always resolved to an absolute path: this home ultimately becomes a
    `GROK_HOME` env value handed to a grok-dev subprocess, and grok
    resolves a relative `GROK_HOME` against ITS OWN cwd, not this
    process's -- a relative path here reads back as a broken/empty home
    (misreported as "not signed in"), not the path bug it actually is."""
    if override is not None:
        resolved = Path(override).resolve()
        return resolved, f"--grok-home override ({resolved})"

    session_dir = (trace.get("session") or {}).get("session_dir")
    if session_dir:
        candidate = Path(session_dir).parent.parent.parent.resolve()
        if (candidate / "skills").is_dir() and (candidate / "config.toml").is_file():
            return candidate, f"trace's recorded session_dir ancestor ({candidate})"

    return Path.home() / ".grok", "live ~/.grok (trace's own home was unavailable)"


def _resolve_cwd(trace):
    cwd = (trace.get("session") or {}).get("cwd")
    return cwd


# --------------------------------------------------------------------------
# analyzer (imported, not reimplemented)
# --------------------------------------------------------------------------

def _analyzer_functions(repo_root):
    """Import analyzer/analyze.py's digest()/analyze() -- same sys.path
    dance synth/build_corpus.py uses for the normalizer, since neither
    top-level dir is an installed package."""
    analyzer_dir = str(repo_root / "analyzer")
    if analyzer_dir not in sys.path:
        sys.path.insert(0, analyzer_dir)
    from analyze import analyze, digest  # noqa: E402
    return digest, analyze


def _load_cached_analysis(trace_path, results_dir):
    """Load a pre-computed analyzer/results/<stem>.result.json instead of
    spending a model call -- makes --dry-run free and the tests hermetic."""
    result_path = Path(results_dir) / f"{Path(trace_path).stem}.result.json"
    if not result_path.is_file():
        raise SystemExit(
            f"error: --use-cached-analysis given but no cached result at "
            f"{result_path} -- run analyzer/analyze.py on this trace first, "
            "or drop --use-cached-analysis to call the analyzer live"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    recs = (payload.get("result") or {}).get("recommendations", [])
    return payload.get("digest") or {}, recs, payload.get("meta") or {}


# --------------------------------------------------------------------------
# patch ranking (the --max-patches budget)
# --------------------------------------------------------------------------

def _patch_impact_score(patch, dig):
    """A real, trace-measured number estimating what this one patch would
    actually have saved IN THIS SESSION -- never a fabricated constant.
    Always denominated in the unit `_patch_impact_unit` names for this same
    patch's kind; the two functions are a pair and must be read together.

    `skill.remove` against a skill this session never loaded scores 0 on
    purpose, not by accident: removing dead catalog weight is real hygiene,
    but this trace has no evidence it cost anything HERE, so it is right
    for a patch with real per-session evidence to always outrank it -- see
    the brief's own example, a skill.truncate against a skill with high
    measured `unreferenced_tokens_est` outranking a skill.remove for a
    never-loaded skill (score 0, even though it costs real tokens in the
    install catalog)."""
    kind, target = patch["kind"], patch["target"]

    if kind == PATCH_SKILL_TRUNCATE:
        for s in dig.get("skills_loaded", []) or []:
            if s.get("skill") == target:
                return float(s.get("unreferenced_tokens_est", 0) or 0)
        return 0.0

    if kind == PATCH_SKILL_REMOVE:
        for s in dig.get("skills_loaded", []) or []:
            if s.get("skill") == target:
                return float(s.get("tokens_est", 0) or 0)
        return 0.0  # never loaded this session -> zero measured impact here

    if kind == PATCH_SKILL_INSERT:
        for d in dig.get("discovery_after_skill_load", []) or []:
            if d.get("skill") == target:
                return float(d.get("total_ms", 0) or 0)
        return 0.0

    if kind == PATCH_MCP_REMOVE:
        for f in dig.get("failed_mcp_servers", []) or []:
            if f.get("server") == target:
                return float(f.get("wasted_ms", 0) or 0)
        info = (dig.get("mcp_servers") or {}).get(target)
        return float(info.get("connect_ms", 0) or 0) if info else 0.0

    if kind == PATCH_RULES_APPEND:
        return 0.0  # no measured-in-trace evidence for a rules-append ask

    return 0.0


def _patch_impact_unit(patch):
    """The physical unit `_patch_impact_score` is denominated in for this
    patch's kind -- the grouping key `_rank_and_select` uses so a score is
    never ranked against a score measuring something else.

    Deliberately keyed on the unit itself, not on `patch["expected_effect"]`:
    `skill.truncate`/`skill.remove`/`skill.insert` all share
    `EFFECT_SKILL_TOKENS`, but `skill.insert`'s only available trace-measured
    evidence is milliseconds of discovery calls after the skill load (there
    is no token count for guidance that was never written), not a token
    count like the other two skill kinds -- grouping by `expected_effect`
    alone would still silently rank an ms score against a token score
    within that one family, which is exactly the bug this function exists
    to prevent."""
    kind = patch["kind"]
    if kind in (PATCH_SKILL_TRUNCATE, PATCH_SKILL_REMOVE):
        return "tokens_est"
    if kind in (PATCH_SKILL_INSERT, PATCH_MCP_REMOVE):
        return "ms"
    return "unmeasured"  # rules.append: _patch_impact_score is always 0.0


def _rank_and_select(all_patches, dig, max_patches):
    """Rank appliable patches WITHIN their `_patch_impact_unit` class (a
    millisecond score is never compared against a token score), then
    interleave the per-class ranked lists round-robin -- one candidate from
    each class present, repeated -- so `--max-patches` yields a spread
    across the distinct measured units in play rather than N patches from
    whichever unit happens to produce the largest raw numbers.

    Class order is first-seen order among the ranked patches, not sorted by
    name or magnitude, so which class gets first pick in the round-robin is
    an accident of recommendation order, not a re-introduction of the same
    scale bias one level up.

    Returns (selected, dropped). `dropped` is a list of full Patch dicts
    (types.make_patch shape, each a copy of the original) -- directly
    usable as `report.build(dropped_patches=dropped)`, per
    final-review.md I4: a patch that was ranked out or rejected before
    ranking must still appear in the report itself, not only in
    progress.json/stdout. Each carries two extra keys beyond the frozen
    Patch shape (`impact_score`/`impact_unit`) for the CLI's own ranking
    transparency; report.py's `_dropped_patch_caveat` only reads the
    frozen fields, so the extras are harmless baggage to it.

    A patch cut for budget (appliable, just outranked) arrives with
    `skipped_reason=None` -- patches.py never rejected it -- so one is
    synthesized here naming the rank/class/score, letting it render
    through the exact same `report.build` surface as a genuinely
    lint-rejected patch instead of needing a second, parallel rendering
    path (the same "two implementations of one job" duplication
    final-review.md calls out elsewhere). An already-skipped patch keeps
    its own `skipped_reason` from `patches.py` untouched."""
    appliable = [p for p in all_patches if not p.get("skipped_reason")]
    already_skipped = [p for p in all_patches if p.get("skipped_reason")]

    classes = {}
    for p in appliable:
        classes.setdefault(_patch_impact_unit(p), []).append(p)
    for group in classes.values():
        group.sort(key=lambda p: (-_patch_impact_score(p, dig), p["patch_id"]))
    class_order = list(classes.keys())  # first-seen order, see docstring

    selected = []
    cursor = {unit: 0 for unit in class_order}
    while len(selected) < max_patches and any(
            cursor[u] < len(classes[u]) for u in class_order):
        for unit in class_order:
            if len(selected) >= max_patches:
                break
            i = cursor[unit]
            if i < len(classes[unit]):
                selected.append(classes[unit][i])
                cursor[unit] += 1
    selected_ids = {p["patch_id"] for p in selected}

    dropped = []
    for unit in class_order:
        group = classes[unit]
        for rank, p in enumerate(group, start=1):
            if p["patch_id"] in selected_ids:
                continue
            score = _patch_impact_score(p, dig)
            entry = dict(p)
            entry["impact_score"] = score
            entry["impact_unit"] = unit
            entry["skipped_reason"] = (
                f"budget: ranked {rank} of {len(group)} in the "
                f"{unit!r} impact class (impact_score={score:g} "
                f"{unit}); round-robin across {len(class_order)} "
                f"class(es) filled --max-patches={max_patches} "
                "before reaching it"
            )
            dropped.append(entry)
    for p in already_skipped:
        entry = dict(p)
        entry["impact_score"] = None
        entry["impact_unit"] = None
        dropped.append(entry)
    return selected, dropped


def _patch_preview(patch, k, dig):
    preview = dict(patch)
    preview["planned_runs"] = 2 * k
    preview["impact_score"] = _patch_impact_score(patch, dig)
    preview["impact_unit"] = _patch_impact_unit(patch)
    return preview


# --------------------------------------------------------------------------
# cost estimate (real, measured numbers only)
# --------------------------------------------------------------------------

def _mean_cost(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return statistics.mean(values), len(values)


def _corpus_mean_session_cost(repo_root):
    manifest_path = repo_root / "synth" / "manifest.json"
    if not manifest_path.is_file():
        return None, 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _mean_cost(s.get("cost_usd") for s in manifest.get("sessions", []))


def _analyzer_mean_cost(repo_root):
    results_dir = repo_root / "analyzer" / "results"
    if not results_dir.is_dir():
        return None, 0
    costs = []
    for f in sorted(results_dir.glob("*.result.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        costs.append((payload.get("meta") or {}).get("cost_usd"))
    return _mean_cost(costs)


def _cost_estimate(selected_patches, *, k, use_cached_analysis, repo_root):
    """runs x observed mean cost from the corpus (brief's formula), using
    real measured numbers: synth/manifest.json's per-session `cost_usd` for
    the dominant replay cost, and analyzer/results/*.result.json's own
    `meta.cost_usd` for the analyzer call. Judge and skill.insert-synthesis
    calls are not separately measured anywhere in this corpus, so they are
    named as excluded rather than guessed at."""
    mean_session_cost, n_sessions = _corpus_mean_session_cost(repo_root)
    mean_analyzer_cost, n_analyzer = _analyzer_mean_cost(repo_root)

    planned_runs = len(selected_patches) * 2 * k
    replay_cost = (planned_runs * mean_session_cost) if mean_session_cost is not None else None
    analyzer_cost = 0.0 if use_cached_analysis else mean_analyzer_cost
    total = (replay_cost + (analyzer_cost or 0.0)) if replay_cost is not None else None

    return {
        "planned_replay_runs": planned_runs,
        "mean_session_cost_usd": mean_session_cost,
        "mean_session_cost_sample_n": n_sessions,
        "replay_cost_estimate_usd": replay_cost,
        "analyzer_cost_estimate_usd": analyzer_cost,
        "analyzer_cost_sample_n": n_analyzer,
        "estimated_total_usd": total,
        "note": (
            "Estimate = planned replay runs (control + treatment, k repeats "
            "each, per selected patch) x the mean cost_usd observed across "
            f"{n_sessions} real synth/manifest.json session(s), plus the "
            f"mean analyzer cost_usd observed across {n_analyzer} "
            "analyzer/results/*.result.json run(s) (0 when "
            "--use-cached-analysis is set). Judge and skill.insert "
            "synthesis calls are not separately measured in this corpus and "
            "are excluded, so the real total will run somewhat higher."
        ),
    }


def _check_workspace_out_collision(workspace, out_dir):
    """Refuse when `out_dir` and `workspace` are nested inside each other.

    `replay.isolated_workdir` copies `workspace` into a directory under
    `out_dir`'s own scratch tree for every run; if `out_dir` (or that
    scratch tree) is itself inside `workspace`, that copy recurses into
    its own destination -- final-review.md C1. This is not a rare
    misconfiguration: the documented default (`--out` unset ->
    `runs/<timestamp>`, resolved against the invocation cwd) collides with
    the headline use case, "review a session captured while working in
    your own repo" -- there `workspace` (the trace's recorded
    `session.cwd`) IS that repo, and `--out`'s default lands right inside
    it. Every run then fails deep in the copy with an OSError, which
    `replay.py`'s never-raise design turns into an ordinary `ok=False` on
    every run, which the all-runs-failed check then (correctly, but
    misleadingly) reports as an auth/environment failure -- so this must
    be caught here, at plan time, before the preflight call spends
    anything on a cycle that was always going to fail.

    Both directions are checked (`out_dir` inside `workspace`, or the
    reverse) and exact equality too; `workspace is None` (no recorded cwd)
    is not this function's problem -- the later `workspace.is_dir()` check
    covers that."""
    if workspace is None:
        return
    workspace = Path(workspace).resolve()
    out_dir = Path(out_dir).resolve()
    nested = (out_dir == workspace or workspace in out_dir.parents
             or out_dir in workspace.parents)
    if nested:
        raise SystemExit(
            f"error: --out ({out_dir}) and the trace's recorded workspace "
            f"({workspace}) are nested inside each other. Replaying copies "
            "the workspace into a directory under --out for every run; a "
            "nested --out would copy that destination into itself and "
            "every run would fail. Pass an --out path outside the "
            "workspace (e.g. a sibling directory, or outside the repo "
            "entirely)."
        )


def _check_budget(estimate, budget_usd):
    if budget_usd is None:
        return
    total = estimate.get("estimated_total_usd")
    if total is None:
        raise SystemExit(
            "error: --budget-usd was given but no cost estimate could be "
            "computed (synth/manifest.json missing or has no cost_usd "
            "figures) -- refusing to start without a basis for the check"
        )
    if total > budget_usd:
        raise SystemExit(
            f"error: estimated cost ${total:.4f} exceeds --budget-usd "
            f"${budget_usd:.4f}; refusing to start this cycle. Lower "
            "--max-patches or --k, raise --budget-usd, or use --dry-run to "
            "inspect the plan without spending anything."
        )


# --------------------------------------------------------------------------
# report assembly helpers
# --------------------------------------------------------------------------

def _verdict_line(patch, gate_result):
    """Render the gate's three-state truth, by name, matching report.py's
    own vocabulary (`_gate_badge`) exactly -- this used to independently
    guess at wording and got it wrong: a k=3 patch that measured a real
    47% token reduction with a judged tie and zero regressions printed as
    "REJECTED (trends favorable)", which reads as "we tested this and it
    lost" for a patch that, in fact, simply could not be accepted OR
    rejected at this sample size. `directional` always says which way the
    evidence leans regardless of whether a call could be made; `accepted`/
    `provisional`/`reasons` say whether one could.

    REJECTED is reserved for a patch that stats.gate() actually had enough
    evidence to reject -- a real judge-loss majority, a metric that
    credibly got worse, or a high-severity regression -- detected the same
    way report.py's badge does: `"insufficient_samples_for_acceptance"` in
    `reasons` means n < stats.MIN_SAMPLES_FOR_ACCEPT (5), the zone where NO
    accept/reject call is possible by construction (every k=3/k=5 cycle
    lands here), so that must render as a directional reading instead.

    `directional == "unknown"` (zero usable pairs -- every run failed on
    at least one arm, or the patch was never applied) is checked first and
    always renders as DIRECTIONAL: UNKNOWN: "nothing could be measured" is
    a different claim from "measured and came out even" (a real "neutral"
    reading) and must never collapse into either NEUTRAL or a false
    REJECTED."""
    kind, target = patch["kind"], patch["target"]
    reasons = gate_result.get("reasons") or []
    headline = reasons[0] if reasons else "no reasons recorded"
    directional = (gate_result.get("directional") or "unknown").upper()

    if gate_result.get("directional") == "unknown":
        status = "DIRECTIONAL: UNKNOWN"
    elif gate_result.get("accepted"):
        status = "ACCEPTED"
    elif gate_result.get("provisional"):
        status = "PROVISIONAL"
    else:
        too_few_for_any_call = any(
            "insufficient_samples_for_acceptance" in r for r in reasons)
        status = f"DIRECTIONAL: {directional}" if too_few_for_any_call else "REJECTED"

    return f"{status}: {kind} on {target!r} -- {headline}"


def _baseline_summary(baseline_version):
    return {
        "version_id": baseline_version["version_id"],
        "skills": len(baseline_version["skills"]),
        "mcp": len(baseline_version["mcp_servers"]),
    }


def _session_info(trace, prompt, trace_path):
    session = trace.get("session") or {}
    return {
        "session_id": session.get("session_id"),
        "title": session.get("title"),
        "prompt": prompt,
        "trace_file": str(trace_path),
    }


# Which of the ten EFFICIENCY_METRICS (types.py) are driven by grok's own
# token/cost accounting -- and therefore carry prompt-cache-warmth noise --
# versus counted straight off the trace/harness, independent of cache
# state. Only the metrics explicitly classified (7 of 10) appear here;
# `wall_ms`, `tool_result_tokens_est`, and `mcp_startup_ms` are
# deliberately left out rather than guessed at -- report.py's renderer
# should treat a metric with no entry as unclassified, not as "confirmed
# not cache-sensitive".
CACHE_SENSITIVE_METRICS = {
    "total_tokens": True,
    "cost_usd": True,
    "skill_tokens_est": False,
    "skill_wasted_tokens_est": False,
    "tool_calls": False,
    "turns": False,
    "duplicate_calls": False,
}


def _cache_read_range(patch_results):
    """(min, max) of `usage.cache_read_input_tokens` observed across every
    run in every patch's matrix this cycle -- real numbers, never a canned
    figure, so `_cache_warmth_caveat` can state THIS cycle's actual spread
    rather than repeating someone else's. None when no run has usage data
    to draw a range from (e.g. every run failed before grok reported
    usage)."""
    values = []
    for pr in patch_results:
        for runs in ((pr.get("matrix") or {}).get("arms") or {}).values():
            for r in runs:
                v = (r.get("usage") or {}).get("cache_read_input_tokens")
                if v is not None:
                    values.append(v)
    if not values:
        return None
    return min(values), max(values)


def _cache_warmth_caveat(patch_results):
    """Prompt-cache state (how much of the prompt was served from cache vs.
    recomputed) is a function of run history and timing, not of the
    harness -- interleaved dispatch (replay.run_matrix's control/treatment
    interleaving) helps but does not eliminate it. `total_tokens` and
    `cost_usd` are both read straight off grok's own token/cost accounting
    (types.make_run_result's `usage`/`cost_usd`), which is exactly what
    `cache_read_input_tokens` swings move -- so both carry this noise. The
    deterministic metrics (skill tokens loaded, tool calls, turns, ...) are
    counted off the trace/harness directly, not derived from token
    accounting, so they are not cache-sensitive. Returns None (caveat
    omitted, not a fabricated generic warning) when this cycle has no
    usage data to show a real range from."""
    cache_range = _cache_read_range(patch_results)
    if cache_range is None:
        return None
    lo, hi = cache_range
    return (
        "Token and cost figures are affected by prompt-cache warmth, which "
        "varies per run independently of the harness (this cycle observed "
        f"cache_read_input_tokens ranging {lo:,} to {hi:,} across runs). "
        "Medians are reported to blunt this, but cost_usd in particular "
        "should be read as indicative, not as a measurement of the patch. "
        "The deterministic metrics -- skill tokens loaded, tool calls, "
        "turns -- are not cache-sensitive."
    )


# The cost figure omits judge (LLM-as-judge grading) calls: grade.py's own
# default judge implementation does not surface `total_cost_usd` anywhere
# this module can read without reimplementing its call/retry/isolation
# logic (final-review.md I1) -- doing that here would be exactly the "two
# implementations of the same job" duplication final-review.md separately
# flags elsewhere, so the honest fix on this side is disclosure, not a
# parallel judge-calling path. Replay runs, the analyzer call, and the
# preflight check ARE all folded into the total.
JUDGE_COST_CAVEAT = (
    "Reported cycle cost includes replay runs, the analyzer call (when not "
    "--use-cached-analysis), and the preflight check -- it does NOT "
    "include judge (LLM-as-judge grading) call costs, which are not "
    "currently surfaced by the grading module. The true spend for this "
    "cycle is higher than the figure shown."
)


def _base_extra_caveats(trace, baseline_source, baseline_version, estimate):
    caveats = [
        f"Baseline harness snapshotted from: {baseline_source} "
        f"(version {baseline_version['version_id']})."
    ]
    caveats.extend(baseline_version.get("warnings") or [])

    ground_truth = trace.get("ground_truth") or {}
    orig_cost, orig_turns = ground_truth.get("cost_usd"), ground_truth.get("turns")
    if orig_cost is not None or orig_turns is not None:
        caveats.append(
            "The user's originally recorded session (context only, NOT the "
            f"statistical control) cost "
            f"${orig_cost if orig_cost is not None else 'unknown'} over "
            f"{orig_turns if orig_turns is not None else 'an unknown number of'} "
            "turn(s), measured in a different environment; the control arm "
            "in this report is a fresh, isolated re-run of the unpatched "
            "harness, not that number."
        )
    else:
        caveats.append(
            "The user's originally recorded session cost/turns were not "
            "available on this trace; the freshly re-run control arm in "
            "this report is the only baseline used for grading."
        )

    total = estimate.get("estimated_total_usd")
    caveats.append(
        "Pre-flight cost estimate for this cycle: "
        + (f"${total:.4f}. " if total is not None else "unavailable. ")
        + estimate.get("note", "")
    )

    return caveats


# --------------------------------------------------------------------------
# stdout summaries
# --------------------------------------------------------------------------
#
# Silence on success is a fine convention for a library call and a bad one
# for a command someone is about to spend real money on: every path below
# that can spend a dollar or more (a live analyzer call, a replay matrix)
# prints something a human can read before/while it happens, in addition to
# the always-written JSON artifacts. Nothing here is read back by any other
# part of this module -- it is output, not state.

def _truncate(text, n=200):
    text = text or ""
    return text if len(text) <= n else text[: n - 3] + "..."


def _print_dropped(dropped_patches, limit=8):
    shown = dropped_patches[:limit]
    for d in shown:
        print(f"    [{d['patch_id']}] {d['kind']}/{d['target']}: {d['skipped_reason']}")
    more = len(dropped_patches) - len(shown)
    if more > 0:
        print(f"    ... and {more} more (see plan.json / progress.json / the report)")


def _print_cost_estimate(estimate, budget_usd):
    total = estimate.get("estimated_total_usd")
    print(f"  planned replay runs: {estimate['planned_replay_runs']}")
    if total is not None:
        print(
            f"  estimated cost: ${total:.4f} "
            f"(replay ${estimate['replay_cost_estimate_usd']:.4f}"
            f" + analyzer ${estimate['analyzer_cost_estimate_usd'] or 0.0:.4f})"
        )
    else:
        print("  estimated cost: unavailable (no manifest cost data)")
    print(f"  {estimate['note']}")
    if budget_usd is not None:
        print(f"  budget cap: ${budget_usd:.4f}")


def _print_dry_run_summary(plan, plan_path):
    b = plan["baseline"]
    print(f"=== dry run: {plan['trace_file']} ===")
    print(f"prompt (source: {plan['prompt_source']}):")
    print(f"  {_truncate(plan['prompt'])}")
    print(
        f"baseline: {b['version_id']} from {b['source']} "
        f"({b['skills']} skill(s), {b['mcp_servers']} mcp server(s))"
    )
    for w in b.get("warnings") or []:
        print(f"  ! {w}")
    print(f"k={plan['k']} paired repeats per patch")

    selected = plan["selected_patches"]
    dropped = plan["dropped_patches"]
    print(f"selected {len(selected)} patch(es), {len(dropped)} candidate(s) dropped:")
    for p in selected:
        print(
            f"    [{p['patch_id']}] {p['kind']} on {p['target']!r} -> "
            f"{p['expected_effect']} (impact_score={p['impact_score']:g} "
            f"{p['impact_unit']}, {p['planned_runs']} planned run(s))"
        )
    if dropped:
        print(f"  dropped ({len(dropped)}):")
        _print_dropped(dropped)

    print("cost estimate:")
    _print_cost_estimate(plan["cost_estimate"], plan["budget_usd"])
    print(f"full plan: {plan_path}")


def _run_health(matrix_summary, k):
    """{arm: (ok_count, k)} read straight off a (possibly trace-stripped)
    ReplayMatrix's `ok` flags -- k is always the ORIGINALLY REQUESTED repeat
    count, not how many actually ran, so "1/3 ok" still reads correctly when
    fail-fast stopped a patch after just the first pair: the shortfall
    itself is the signal, not just the failures within what did run."""
    arms = (matrix_summary or {}).get("arms") or {}
    health = {}
    for arm in (ARM_CONTROL, ARM_TREATMENT):
        runs = arms.get(arm) or []
        health[arm] = (sum(1 for r in runs if r.get("ok")), k)
    return health


def _print_patch_result(patch, gate_result, verdict, health):
    control_ok, control_k = health[ARM_CONTROL]
    treatment_ok, treatment_k = health[ARM_TREATMENT]
    print(f"  [{patch['patch_id']}] {patch['kind']} on {patch['target']!r}: {verdict}")
    print(f"      control {control_ok}/{control_k} ok, "
          f"treatment {treatment_ok}/{treatment_k} ok")


def _all_directional_for_sample_size(patch_results):
    """True iff every patch's gate decision was blocked purely by sample
    size -- stats.gate()'s own "insufficient_samples_for_acceptance"
    reason (n < stats.MIN_SAMPLES_FOR_ACCEPT), the same signal
    `_verdict_line` uses to render DIRECTIONAL instead of a false
    REJECTED -- and not by an actual loss, a high-severity regression, a
    no-data cycle, or a patch that was never applied. An empty
    `patch_results` is not this case (there is nothing to re-run bigger)."""
    if not patch_results:
        return False
    for pr in patch_results:
        gate_result = pr.get("gate") or {}
        if gate_result.get("accepted") or gate_result.get("provisional"):
            return False
        reasons = gate_result.get("reasons") or []
        if not any("insufficient_samples_for_acceptance" in r for r in reasons):
            return False
    return True


def _print_cycle_summary(rpt, out_dir, total_cost):
    print(f"=== cycle done: {len(rpt['patch_results'])} patch(es) tested "
          f"(spent ${total_cost:.4f}) ===")
    print(f"report: {Path(out_dir) / 'index.html'}")
    if _all_directional_for_sample_size(rpt.get("patch_results") or []):
        print(
            f"note: every patch's result is directional-only -- k={rpt.get('k')} "
            f"is below the {stats.MIN_SAMPLES_FOR_CI} paired repeats the "
            f"acceptance gate requires for a real accept/reject call. "
            f"Re-run with --k {stats.MIN_SAMPLES_FOR_CI} to reach the "
            "sample size the gate requires."
        )


def _preflight_prompt():
    return "Reply with the single word OK."


def _run_preflight(baseline_version, *, out_dir, auth_from, model, timeout, runner):
    """One cheap grok-dev call through the baseline home before spending a
    whole cycle on it: an expired token, a missing XAI_API_KEY, or a broken
    grok-dev install would otherwise be discovered independently by every
    one of `2 * k` runs per patch, for every patch, before the cycle gives
    up on its own. Raises SystemExit with the failure verbatim -- never
    silently continues past a broken environment.

    Returns the call's own `total_cost_usd` (a real, billed grok-dev call
    -- final-review.md I1) so the caller can fold it into the cycle's
    reported cost instead of it vanishing the way it used to.

    Uses a shallow copy of `baseline_version` with its own `warnings` list,
    not the real one: `harness.materialize` can append a warning in place
    (e.g. "project skill not materialized" when no `project_dest` is given,
    which this throwaway connectivity check has no use for and no business
    leaking into the real cycle's report caveats).

    Asserts the `GROK_HOME` it is about to hand to `runner` is absolute
    before making the call: grok resolves a relative `GROK_HOME` against
    ITS OWN cwd, not this process's, so a relative path here would silently
    point at an empty/nonexistent home and come back reading as "not
    signed in" -- a path bug in the caller, misdiagnosed as an auth
    failure. The whole point of a preflight is a diagnosis worth trusting;
    this must raise as the internal bug it is, not surface as a confusing,
    wrong auth error."""
    probe_version = dict(baseline_version)
    probe_version["warnings"] = list(baseline_version.get("warnings") or [])
    home = harness.materialize(
        probe_version, Path(out_dir) / "scratch" / "preflight-home",
        auth_from=auth_from)
    if not Path(home).is_absolute():
        raise RuntimeError(
            f"internal error: preflight GROK_HOME is not absolute ({home!r}) "
            "-- this is a path bug in the caller (out_dir was not resolved "
            "to an absolute path before reaching _run_preflight), not an "
            "auth problem. Fix the caller; do not chase this as a login issue."
        )

    argv = [str(replay.GROK), "-p", _preflight_prompt(),
            "--output-format", "json", "--permission-mode", "bypassPermissions"]
    if model:
        argv += ["--model", model]
    env = {**os.environ, "GROK_HOME": str(home)}

    try:
        proc = runner(argv, cwd=str(home), env=env,
                      capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        raise SystemExit(
            f"error: preflight check could not run grok-dev at all: "
            f"{type(exc).__name__}: {exc}\nAborting before spending a "
            "cycle. Pass --no-preflight to skip this check."
        )

    error = None
    out = {}
    try:
        out = json.loads(proc.stdout)
        if not isinstance(out, dict):
            raise TypeError("stdout json is not an object")
    except (json.JSONDecodeError, TypeError) as exc:
        error = (f"unparseable stdout ({exc}); exit={proc.returncode} "
                 f"stderr={(proc.stderr or '')[-500:]!r}")
    else:
        if proc.returncode != 0:
            error = (f"grok-dev exited {proc.returncode}: "
                     f"{(proc.stderr or out.get('text') or '')[-500:]}")

    if error:
        raise SystemExit(
            "error: preflight check failed -- aborting before spending a "
            f"cycle.\n{error}\nIf this is an auth problem, grok's own "
            "error above usually names the fix (commonly `grok login "
            "--device-code`, or setting XAI_API_KEY). Pass --no-preflight "
            "to skip this check."
        )

    return out.get("total_cost_usd")


# --------------------------------------------------------------------------
# the orchestrator
# --------------------------------------------------------------------------

def run_review(trace_path, *, out_dir, k=DEFAULT_K, max_patches=DEFAULT_MAX_PATCHES,
               budget_usd=None, use_cached_analysis=False, dry_run=False,
               serve=False, port=DEFAULT_PORT, model=None,
               max_parallel=DEFAULT_MAX_PARALLEL, timeout=DEFAULT_TIMEOUT, seed=0,
               max_token_regression=DEFAULT_MAX_TOKEN_REGRESSION,
               grok_home_override=None, auth_from=None, repo_root=None,
               digest_fn=None, analyze_fn=None, patches_grok=None,
               judge_grok=None, runner=None, stats_summarize=None,
               stats_gate=None, no_preflight=False, now=None):
    """Run one review cycle end to end. Every dependency that would spawn a
    process or call a model has an injection point (defaulting to the real
    implementation) so tests can exercise the whole orchestration without
    ever spawning grok-dev, per the project's global test constraint."""
    # Resolved to absolute here, at the top, before anything downstream
    # derives a path from them: `out_dir` in particular ends up inside a
    # `GROK_HOME` handed to a grok-dev subprocess (via `_run_preflight` and
    # `replay.run_matrix`'s materialize calls), and grok resolves a
    # relative `GROK_HOME` against ITS OWN cwd, not this process's -- a
    # relative `--out` would silently point grok at an empty/nonexistent
    # home and misreport a path bug as "not signed in". Every home,
    # workdir, scratch, and report path built from `out_dir` below is
    # absolute by construction as a result. `cmd_review` also resolves
    # `--out`/`--grok-home` at parse time so this is defense in depth, not
    # the only place it happens -- a direct `run_review(...)` caller (e.g.
    # a test) gets the same guarantee without going through argparse.
    trace_path = Path(trace_path).resolve()
    out_dir = Path(out_dir).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else REPO_ROOT
    now = now or (lambda: datetime.now(timezone.utc))
    runner = runner or subprocess.run
    real_grok_home = (Path(auth_from).resolve() if auth_from
                      else (Path.home() / ".grok"))

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    state = {
        "trace_file": str(trace_path), "out_dir": str(out_dir),
        "k": k, "max_patches": max_patches, "seed": seed, "dry_run": dry_run,
        "status": "starting",
        "started_at": now().isoformat(), "updated_at": now().isoformat(),
    }
    _write_json(progress_path, state)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    prompt, prompt_source = _resolve_prompt(trace)

    baseline_home, baseline_source = _resolve_baseline_home(trace, grok_home_override)
    cwd = _resolve_cwd(trace)
    project_dir = Path(cwd) if cwd and Path(cwd).is_dir() else None
    workspace = Path(cwd) if cwd else None
    _check_workspace_out_collision(workspace, out_dir)

    baseline_version = harness.snapshot(baseline_home, project_dir=project_dir)

    if use_cached_analysis:
        dig, recs, analysis_meta = _load_cached_analysis(
            trace_path, repo_root / "analyzer" / "results")
    else:
        d_fn, a_fn = digest_fn, analyze_fn
        if d_fn is None or a_fn is None:
            live_digest, live_analyze = _analyzer_functions(repo_root)
            d_fn = d_fn or live_digest
            a_fn = a_fn or live_analyze
        dig = d_fn(trace)
        result, analysis_meta = a_fn(dig, model=model)
        if result is None:
            raise SystemExit(f"error: analyzer failed: {analysis_meta}")
        recs = result.get("recommendations", [])
        analysis_meta = analysis_meta or {}

    effective_patches_grok = patches_grok
    if effective_patches_grok is None and dry_run:
        effective_patches_grok = _noop_patches_grok
    patch_kwargs = {"grok": effective_patches_grok} if effective_patches_grok else {}

    all_patches = patches.from_recommendations(
        recs, baseline_version, digest=dig, **patch_kwargs)
    all_patches = patches.dedup(all_patches)

    selected_patches, dropped_patches = _rank_and_select(all_patches, dig, max_patches)
    estimate = _cost_estimate(selected_patches, k=k,
                              use_cached_analysis=use_cached_analysis,
                              repo_root=repo_root)

    state.update({
        "status": "planned",
        "baseline_version_id": baseline_version["version_id"],
        "baseline_source": baseline_source,
        "prompt_source": prompt_source,
        "patches_total": len(all_patches),
        "patches_selected": [p["patch_id"] for p in selected_patches],
        "patches_dropped": dropped_patches,
        "cost_estimate": estimate,
        "updated_at": now().isoformat(),
    })
    _write_json(progress_path, state)

    _check_budget(estimate, budget_usd)

    if dry_run:
        plan = {
            "trace_file": str(trace_path),
            "prompt": prompt, "prompt_source": prompt_source,
            "baseline": {
                "version_id": baseline_version["version_id"],
                "source": baseline_source,
                "skills": len(baseline_version["skills"]),
                "mcp_servers": len(baseline_version["mcp_servers"]),
                "warnings": baseline_version["warnings"],
            },
            "k": k,
            "selected_patches": [_patch_preview(p, k, dig) for p in selected_patches],
            "dropped_patches": dropped_patches,
            "cost_estimate": estimate,
            "budget_usd": budget_usd,
        }
        plan_path = out_dir / "plan.json"
        _write_json(plan_path, plan)
        state["status"] = "dry_run_complete"
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)
        _print_dry_run_summary(plan, plan_path)
        if serve:
            print("note: --serve ignored under --dry-run (no report to serve)",
                  file=sys.stderr)
        return plan

    caveats_args = (baseline_source, baseline_version, estimate)

    if not selected_patches:
        state["status"] = "no_patches"
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)
        rpt = report.build(
            session=_session_info(trace, prompt, trace_path),
            baseline_version=_baseline_summary(baseline_version),
            patch_results=[], k=k, cost_usd=0.0,
            dropped_patches=dropped_patches,
            extra_caveats=_base_extra_caveats(trace, *caveats_args))
        rpt["cache_sensitive_metrics"] = CACHE_SENSITIVE_METRICS
        report.write(rpt, out_dir)
        print("=== cycle done: no appliable candidate patches this cycle ===")
        print(f"report: {out_dir / 'index.html'}")
        if serve:
            report.serve(out_dir, port=port)
        return rpt

    if workspace is None or not workspace.is_dir():
        raise SystemExit(
            f"error: trace's recorded workspace ({cwd!r}) does not exist on "
            "this machine -- cannot safely replay the original prompt"
        )

    total_cost = 0.0
    if not use_cached_analysis and analysis_meta.get("cost_usd"):
        total_cost += analysis_meta["cost_usd"]

    if not no_preflight:
        preflight_cost = _run_preflight(
            baseline_version, out_dir=out_dir, auth_from=real_grok_home,
            model=model, timeout=timeout, runner=runner)
        if preflight_cost is not None:
            total_cost += preflight_cost

    patch_results = []

    print(f"=== running review cycle: {len(selected_patches)} patch(es) selected, "
          f"k={k} ===")
    state["status"] = "running"
    state["completed_patches"] = []
    for patch in selected_patches:
        state["current_patch"] = patch["patch_id"]
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)

        treatment_version, applications = harness.apply(baseline_version, [patch])
        application = applications[0]
        patch_scratch = out_dir / "scratch" / patch["patch_id"]

        fail_fast_reason = None
        if application["applied"]:
            # run_matrix materializes both arms itself, once per run (see
            # its docstring): a shared pre-built home would carry the
            # workdir's original, unpatched project-scope .grok/skills/,
            # which outranks $GROK_HOME/skills/ in grok's own precedence and
            # would silently shadow a patched project skill. control/
            # treatment are passed as HarnessVersion dicts, not paths.
            #
            # The first pair is always run alone (k=1) before committing to
            # the rest: if it fails on BOTH arms, that is almost always a
            # systemic problem (auth expiring mid-cycle, the endpoint down)
            # rather than anything specific to this patch, and running the
            # remaining k-1 repeats would just pay to rediscover the same
            # failure k-1 more times. Only one arm failing is left alone --
            # that is real signal (a regression), not a reason to stop.
            matrix = replay.run_matrix(
                prompt, control=baseline_version, treatment=treatment_version,
                k=1, workspace=workspace, scratch=patch_scratch / "runs",
                auth_from=real_grok_home,
                max_parallel=max_parallel, timeout=timeout, model=model,
                patch_id=patch["patch_id"], runner=runner)
            probe_control = matrix["arms"][ARM_CONTROL][0]
            probe_treatment = matrix["arms"][ARM_TREATMENT][0]

            if k > 1 and not probe_control.get("ok") and not probe_treatment.get("ok"):
                fail_fast_reason = (
                    "stopped after the first paired run failed on both arms "
                    f"(control error: {probe_control.get('error')!r}; "
                    f"treatment error: {probe_treatment.get('error')!r}); "
                    f"not spending the remaining {k - 1} repeat(s)"
                )
            elif k > 1:
                rest = replay.run_matrix(
                    prompt, control=baseline_version, treatment=treatment_version,
                    k=k - 1, workspace=workspace, scratch=patch_scratch / "runs-rest",
                    auth_from=real_grok_home,
                    max_parallel=max_parallel, timeout=timeout, model=model,
                    patch_id=patch["patch_id"], runner=runner)
                for arm in (ARM_CONTROL, ARM_TREATMENT):
                    for r in rest["arms"][arm]:
                        r["repeat"] = r.get("repeat", 0) + 1
                    matrix["arms"][arm].extend(rest["arms"][arm])
                matrix["k"] = k

            grades = grade.grade_matrix(matrix, prompt=prompt, grok=judge_grok)

            summarize_fn = stats_summarize or _default_stats_summarize
            gate_fn = stats_gate or _default_stats_gate
            effect = patch["expected_effect"]
            summary = summarize_fn(grades, primary_effect=effect, seed=seed)
            gate_result = gate_fn(summary, primary_effect=effect,
                                  max_token_regression=max_token_regression)

            _write_json(out_dir / "patches" / patch["patch_id"] / "matrix.json", matrix)
            matrix_summary = _strip_traces(matrix)
            total_cost += _sum_run_costs(matrix)
        else:
            grades = []
            summary = {}
            # directional="unknown", not "neutral": a patch that was never
            # applied measured nothing, which is a different claim from
            # "measured and came out even" -- see _verdict_line.
            gate_result = {
                "accepted": False, "provisional": False, "directional": "unknown",
                "reasons": [f"patch not applied: {application['reason']}"],
            }
            matrix_summary = make_replay_matrix(
                prompt=prompt, k=0, arms={ARM_CONTROL: [], ARM_TREATMENT: []},
                control_version_id=baseline_version["version_id"],
                treatment_version_id=treatment_version["version_id"],
                patch_id=patch["patch_id"])

        verdict = _verdict_line(patch, gate_result)
        if fail_fast_reason:
            verdict += f" [{fail_fast_reason}]"
        patch_results.append(make_patch_result(
            patch=patch, application=application, matrix_summary=matrix_summary,
            grades=grades, statistics=summary, gate=gate_result,
            verdict_line=verdict))
        health = _run_health(matrix_summary, k)
        _print_patch_result(patch, gate_result, verdict, health)

        state["completed_patches"].append({
            "patch_id": patch["patch_id"], "kind": patch["kind"],
            "target": patch["target"], "applied": application["applied"],
            "accepted": gate_result.get("accepted"),
            "provisional": gate_result.get("provisional"),
            "directional": gate_result.get("directional"),
            "verdict_line": verdict,
            "control_ok": health[ARM_CONTROL][0], "treatment_ok": health[ARM_TREATMENT][0],
            "fail_fast_reason": fail_fast_reason,
        })
        state["current_patch"] = None
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)

    extra_caveats = _base_extra_caveats(trace, *caveats_args) + [JUDGE_COST_CAVEAT]
    cache_caveat = _cache_warmth_caveat(patch_results)
    if cache_caveat:
        extra_caveats = extra_caveats + [cache_caveat]

    rpt = report.build(
        session=_session_info(trace, prompt, trace_path),
        baseline_version=_baseline_summary(baseline_version),
        patch_results=patch_results, k=k, cost_usd=total_cost,
        dropped_patches=dropped_patches,
        extra_caveats=extra_caveats)
    rpt["cache_sensitive_metrics"] = CACHE_SENSITIVE_METRICS
    report.write(rpt, out_dir)
    _print_cycle_summary(rpt, out_dir, total_cost)

    total_runs, total_ok = _count_runs(patch_results)
    all_failed = total_runs > 0 and total_ok == 0
    state["status"] = "failed_no_data" if all_failed else "done"
    state["report_cost_usd"] = total_cost
    state["updated_at"] = now().isoformat()
    _write_json(progress_path, state)

    if serve:
        report.serve(out_dir, port=port)

    if all_failed:
        raise SystemExit(
            f"error: all {total_runs} replay run(s) across {len(patch_results)} "
            "patch(es) failed this cycle -- no usable data was produced. A "
            f"report was still written to {out_dir / 'index.html'} for "
            "debugging, but this is not a successful cycle. Check the "
            "per-run `error` fields in progress.json / patches/*/matrix.json "
            "-- a common cause is an auth/environment failure."
        )
    return rpt


# --------------------------------------------------------------------------
# argument parsing / entry point
# --------------------------------------------------------------------------

def _resolve_trace_paths(patterns):
    """Resolved to absolute at argument-parse time, same reasoning as
    `--out`/`--grok-home` (see `run_review`'s top and `cmd_review`): every
    user-supplied path this CLI touches becomes absolute before anything
    downstream derives another path from it."""
    paths = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(Path(p).resolve() for p in glob.glob(pattern))
            if not matches:
                raise SystemExit(f"error: no files matched {pattern!r}")
            paths.extend(matches)
        else:
            p = Path(pattern)
            if not p.is_file():
                raise SystemExit(f"error: trace file not found: {p}")
            paths.append(p.resolve())
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="interstellar",
        description="Review-loop orchestrator: replay a grok-build trace "
                    "control-vs-treatment under a candidate harness patch.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser(
        "review", help="run one review cycle over a captured trace")
    review.add_argument("trace", nargs="+",
                        help="path(s) or glob pattern(s) to a normalized "
                             "trace JSON file; must resolve to exactly one file")
    review.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"paired repeats per arm per patch (default {DEFAULT_K})")
    review.add_argument("--max-patches", type=int, default=DEFAULT_MAX_PATCHES,
                        help="cap on how many ranked candidate patches to "
                             f"actually test this cycle (default {DEFAULT_MAX_PATCHES})")
    review.add_argument("--out", type=Path, default=None,
                        help="output directory (default runs/<UTC timestamp>/)")
    review.add_argument("--dry-run", action="store_true",
                        help="plan the cycle and print the cost estimate; "
                             "never spawns grok-dev")
    review.add_argument("--serve", action="store_true",
                        help="serve the finished report at 127.0.0.1 after writing it")
    review.add_argument("--port", type=int, default=DEFAULT_PORT)
    review.add_argument("--budget-usd", type=float, default=None,
                        help="refuse to start a cycle whose cost estimate exceeds this")
    review.add_argument("--use-cached-analysis", action="store_true",
                        help="load analyzer/results/<trace-stem>.result.json "
                             "instead of calling the analyzer live")
    review.add_argument("--model", default=None,
                        help="grok model override, forwarded to the "
                             "analyzer, replay, and judge calls")
    review.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    review.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    review.add_argument("--seed", type=int, default=0,
                        help="seed for the statistics bootstrap (reproducible reports)")
    review.add_argument("--max-token-regression", type=float,
                        default=DEFAULT_MAX_TOKEN_REGRESSION,
                        # argparse re-expands "%"-style specifiers in help
                        # strings (e.g. %(prog)s), so a literal "%" from the
                        # :.0% format spec must be escaped to "%%" or
                        # print_help() raises ValueError.
                        help="acceptance-gate ceiling on a total_tokens "
                             "regression (default "
                             f"{DEFAULT_MAX_TOKEN_REGRESSION:.0%}".replace("%", "%%")
                             + ")")
    review.add_argument("--grok-home", type=Path, default=None,
                        help="override the harness snapshot source instead "
                             "of the trace's recorded home / live ~/.grok")
    review.add_argument("--no-preflight", action="store_true",
                        help="skip the one-call auth/environment check "
                             "normally run before spending a cycle")

    return parser


def cmd_review(args):
    # Every user-supplied path is resolved to absolute right here, at
    # argument-parse time, before anything derives another path from it --
    # `run_review` resolves them again defensively (a direct caller doesn't
    # go through this function), but the CLI's own `--out`/`--grok-home`
    # must never reach it relative in the first place. A relative `--out`
    # (the default shape, `runs/<timestamp>`, is itself relative) ends up
    # inside a `GROK_HOME` handed to a grok-dev subprocess; grok resolves a
    # relative `GROK_HOME` against ITS OWN cwd, not this process's, so this
    # is a correctness bug, not a style preference -- see `_run_preflight`
    # and `run_review`'s docstring comment for the failure mode this
    # produces if it slips through (a path bug misreported as "not signed
    # in").
    trace_paths = _resolve_trace_paths(args.trace)
    if len(trace_paths) != 1:
        raise SystemExit(
            "error: `review` operates on exactly one trace per invocation; "
            f"{len(trace_paths)} file(s) matched: "
            + ", ".join(str(p) for p in trace_paths)
        )
    out_dir = (args.out or (Path("runs") / _timestamp())).resolve()
    grok_home = args.grok_home.resolve() if args.grok_home else None

    run_review(
        trace_paths[0], out_dir=out_dir, k=args.k, max_patches=args.max_patches,
        budget_usd=args.budget_usd, use_cached_analysis=args.use_cached_analysis,
        dry_run=args.dry_run, serve=args.serve, port=args.port, model=args.model,
        max_parallel=args.max_parallel, timeout=args.timeout, seed=args.seed,
        max_token_regression=args.max_token_regression,
        grok_home_override=grok_home, no_preflight=args.no_preflight,
    )
    return 0


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        return cmd_review(args)
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
