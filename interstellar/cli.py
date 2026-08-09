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
    ranked by a real, trace-measured impact score before truncating
    (`_patch_impact_score`), and every dropped patch is logged with why.
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

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import argparse
import glob
import json
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
    parents up), else the live `~/.grok`. Returns (path, description)."""
    if override is not None:
        return Path(override), f"--grok-home override ({override})"

    session_dir = (trace.get("session") or {}).get("session_dir")
    if session_dir:
        candidate = Path(session_dir).parent.parent.parent
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

    This is deliberately not cross-kind-normalized (skill patches score in
    est. tokens, mcp patches score in ms): comparing tokens to milliseconds
    has no principled exchange rate, so ranking is only claimed to be sound
    within the same expected_effect family. What it does buy: within a
    family it distinguishes a patch with real evidence in this trace from
    one with none, which is exactly the brief's example -- a skill.truncate
    against a skill with high measured `unreferenced_tokens_est` outranks a
    skill.remove for a skill that was never loaded this session at all
    (score 0, even though it costs real tokens in the install catalog)."""
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


def _rank_and_select(all_patches, dig, max_patches):
    """Rank appliable patches by `_patch_impact_score` (desc, patch_id as a
    stable tiebreak) and keep the top `max_patches`. Every patch that is
    NOT selected is logged with a concrete reason -- either it was already
    unappliable (lint/range/model-call failure from patches.py) or it lost
    on rank -- so a run never reads as "we silently tested a top-N and
    called it everything"."""
    appliable = [p for p in all_patches if not p.get("skipped_reason")]
    already_skipped = [p for p in all_patches if p.get("skipped_reason")]

    ranked = sorted(appliable,
                    key=lambda p: (-_patch_impact_score(p, dig), p["patch_id"]))
    selected = ranked[:max_patches]
    selected_ids = {p["patch_id"] for p in selected}

    dropped = []
    for rank, p in enumerate(ranked, start=1):
        if p["patch_id"] in selected_ids:
            continue
        score = _patch_impact_score(p, dig)
        dropped.append({
            "patch_id": p["patch_id"], "kind": p["kind"], "target": p["target"],
            "reason": (f"budget: ranked {rank} of {len(ranked)} candidate "
                       f"patches by impact_score={score:g} "
                       f"(--max-patches={max_patches})"),
        })
    for p in already_skipped:
        dropped.append({
            "patch_id": p["patch_id"], "kind": p["kind"], "target": p["target"],
            "reason": f"not appliable before ranking: {p['skipped_reason']}",
        })
    return selected, dropped


def _patch_preview(patch, k):
    preview = dict(patch)
    preview["planned_runs"] = 2 * k
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
    kind, target = patch["kind"], patch["target"]
    reasons = gate_result.get("reasons") or []
    headline = reasons[0] if reasons else "no reasons recorded"
    if gate_result.get("accepted"):
        status = "ACCEPTED"
    elif gate_result.get("provisional"):
        status = "PROVISIONAL"
    else:
        status = "REJECTED"
    direction = {
        "favorable": "trends favorable", "unfavorable": "trends unfavorable",
        "neutral": "trends neutral",
    }.get(gate_result.get("directional"), "direction not determined")
    return f"{status} ({direction}): {kind} on {target!r} -- {headline}"


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


def _base_extra_caveats(trace, baseline_source, baseline_version, estimate,
                        dropped_patches, max_patches):
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

    if dropped_patches:
        shown = dropped_patches[:8]
        more = len(dropped_patches) - len(shown)
        caveats.append(
            f"{len(dropped_patches)} candidate patch(es) were not run this "
            f"cycle (--max-patches={max_patches}): "
            + "; ".join(f"{d['patch_id']} ({d['kind']}/{d['target']}): {d['reason']}"
                       for d in shown)
            + (f"; and {more} more (see progress.json)." if more > 0 else ".")
        )
    return caveats


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
               stats_gate=None, now=None):
    """Run one review cycle end to end. Every dependency that would spawn a
    process or call a model has an injection point (defaulting to the real
    implementation) so tests can exercise the whole orchestration without
    ever spawning grok-dev, per the project's global test constraint."""
    trace_path = Path(trace_path)
    out_dir = Path(out_dir)
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    now = now or (lambda: datetime.now(timezone.utc))
    runner = runner or subprocess.run
    real_grok_home = Path(auth_from) if auth_from else (Path.home() / ".grok")

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
            "selected_patches": [_patch_preview(p, k) for p in selected_patches],
            "dropped_patches": dropped_patches,
            "cost_estimate": estimate,
            "budget_usd": budget_usd,
        }
        _write_json(out_dir / "plan.json", plan)
        state["status"] = "dry_run_complete"
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)
        if serve:
            print("note: --serve ignored under --dry-run (no report to serve)",
                  file=sys.stderr)
        return plan

    caveats_args = (baseline_source, baseline_version, estimate, dropped_patches, max_patches)

    if not selected_patches:
        state["status"] = "no_patches"
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)
        rpt = report.build(
            session=_session_info(trace, prompt, trace_path),
            baseline_version=_baseline_summary(baseline_version),
            patch_results=[], k=k, cost_usd=0.0,
            extra_caveats=_base_extra_caveats(trace, *caveats_args))
        report.write(rpt, out_dir)
        if serve:
            report.serve(out_dir, port=port)
        return rpt

    if workspace is None or not workspace.is_dir():
        raise SystemExit(
            f"error: trace's recorded workspace ({cwd!r}) does not exist on "
            "this machine -- cannot safely replay the original prompt"
        )

    patch_results = []
    total_cost = 0.0
    if not use_cached_analysis and analysis_meta.get("cost_usd"):
        total_cost += analysis_meta["cost_usd"]

    state["status"] = "running"
    state["completed_patches"] = []
    for patch in selected_patches:
        state["current_patch"] = patch["patch_id"]
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)

        treatment_version, applications = harness.apply(baseline_version, [patch])
        application = applications[0]
        patch_scratch = out_dir / "scratch" / patch["patch_id"]

        if application["applied"]:
            # run_matrix materializes both arms itself, once per run (see
            # its docstring): a shared pre-built home would carry the
            # workdir's original, unpatched project-scope .grok/skills/,
            # which outranks $GROK_HOME/skills/ in grok's own precedence and
            # would silently shadow a patched project skill. control/
            # treatment are passed as HarnessVersion dicts, not paths.
            matrix = replay.run_matrix(
                prompt, control=baseline_version, treatment=treatment_version,
                k=k, workspace=workspace, scratch=patch_scratch / "runs",
                auth_from=real_grok_home,
                max_parallel=max_parallel, timeout=timeout, model=model,
                patch_id=patch["patch_id"], runner=runner)
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
            gate_result = {
                "accepted": False, "provisional": False, "directional": "neutral",
                "reasons": [f"patch not applied: {application['reason']}"],
            }
            matrix_summary = make_replay_matrix(
                prompt=prompt, k=0, arms={ARM_CONTROL: [], ARM_TREATMENT: []},
                control_version_id=baseline_version["version_id"],
                treatment_version_id=treatment_version["version_id"],
                patch_id=patch["patch_id"])

        verdict = _verdict_line(patch, gate_result)
        patch_results.append(make_patch_result(
            patch=patch, application=application, matrix_summary=matrix_summary,
            grades=grades, statistics=summary, gate=gate_result,
            verdict_line=verdict))

        state["completed_patches"].append({
            "patch_id": patch["patch_id"], "kind": patch["kind"],
            "target": patch["target"], "applied": application["applied"],
            "accepted": gate_result.get("accepted"),
            "provisional": gate_result.get("provisional"),
            "directional": gate_result.get("directional"),
            "verdict_line": verdict,
        })
        state["current_patch"] = None
        state["updated_at"] = now().isoformat()
        _write_json(progress_path, state)

    rpt = report.build(
        session=_session_info(trace, prompt, trace_path),
        baseline_version=_baseline_summary(baseline_version),
        patch_results=patch_results, k=k, cost_usd=total_cost,
        extra_caveats=_base_extra_caveats(trace, *caveats_args))
    report.write(rpt, out_dir)

    state["status"] = "done"
    state["report_cost_usd"] = total_cost
    state["updated_at"] = now().isoformat()
    _write_json(progress_path, state)

    if serve:
        report.serve(out_dir, port=port)
    return rpt


# --------------------------------------------------------------------------
# argument parsing / entry point
# --------------------------------------------------------------------------

def _resolve_trace_paths(patterns):
    paths = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(Path(p) for p in glob.glob(pattern))
            if not matches:
                raise SystemExit(f"error: no files matched {pattern!r}")
            paths.extend(matches)
        else:
            p = Path(pattern)
            if not p.is_file():
                raise SystemExit(f"error: trace file not found: {p}")
            paths.append(p)
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

    return parser


def cmd_review(args):
    trace_paths = _resolve_trace_paths(args.trace)
    if len(trace_paths) != 1:
        raise SystemExit(
            "error: `review` operates on exactly one trace per invocation; "
            f"{len(trace_paths)} file(s) matched: "
            + ", ".join(str(p) for p in trace_paths)
        )
    out_dir = args.out or (Path("runs") / _timestamp())

    run_review(
        trace_paths[0], out_dir=out_dir, k=args.k, max_patches=args.max_patches,
        budget_usd=args.budget_usd, use_cached_analysis=args.use_cached_analysis,
        dry_run=args.dry_run, serve=args.serve, port=args.port, model=args.model,
        max_parallel=args.max_parallel, timeout=args.timeout, seed=args.seed,
        max_token_regression=args.max_token_regression,
        grok_home_override=args.grok_home,
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
