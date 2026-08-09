"""Measure and judge: turn a `ReplayMatrix` into per-run `Grade` records.

Three jobs, kept apart on purpose:

  efficiency()   - deterministic. Reads the ten EFFICIENCY_METRICS straight off
                   a RunResult (types.make_run_result): total_tokens/cost_usd/
                   turns from grok's own accounting on the run, everything
                   else from the run's normalized trace. Never recomputes or
                   estimates -- a value nothing measured comes back as None.
  regressions()  - deterministic. Diffs two traces for detector signals that
                   are present in treatment but not control.
  judge_pair()   - the only model call. Position-swapped pairwise preference
                   over the two response texts, so an order-dependent verdict
                   collapses to a tie instead of being reported as evidence.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    make_grade,
    make_judge_verdict,
    make_regression,
)

GROK = Path.home() / ".local" / "bin" / "grok-dev"


# --- efficiency --------------------------------------------------------------

def efficiency(run):
    """Pull the EFFICIENCY_METRICS straight off a RunResult.

    `total_tokens`, `cost_usd`, and `turns` come from grok's own accounting on
    the RunResult (types.make_run_result) rather than the trace: the
    normalizer works from the session store, which records no token counts,
    so those three numbers only ever exist on the run, never on `run["trace"]`.
    `turns` in particular prefers `run["turns"]` (grok's own count) over the
    trace's `metrics.counts.turns` (a normalizer recount) for the same reason
    -- it's the authoritative figure, not a derived one.

    Everything else is a direct read of something the normalizer already
    computed onto `run["trace"]` (normalizer/schema.py). A value with no home
    on the run -- a failed run's `trace: None`, an empty `usage` -- comes back
    as None rather than a guess or a 0.
    """
    run = run or {}
    usage = run.get("usage") or {}
    trace = run.get("trace") or {}
    session = trace.get("session") or {}
    metrics = trace.get("metrics") or {}
    counts = metrics.get("counts") or {}
    timing = metrics.get("timing") or {}
    context_cost = metrics.get("context_cost") or {}
    duplicate_calls = metrics.get("duplicate_calls")

    return {
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": run.get("cost_usd"),
        "wall_ms": session.get("duration_ms"),
        "tool_calls": counts.get("tool_calls"),
        "skill_tokens_est": context_cost.get("skill_tokens_est"),
        "skill_wasted_tokens_est": context_cost.get("skill_wasted_tokens_est"),
        "tool_result_tokens_est": context_cost.get("tool_result_tokens_est"),
        "mcp_startup_ms": timing.get("mcp_startup_ms"),
        "duplicate_calls": len(duplicate_calls) if duplicate_calls is not None else None,
        "turns": run.get("turns"),
    }


# --- regressions ---------------------------------------------------------

def _mcp_names(entries):
    return {e.get("server") for e in (entries or [])}


def _duplicate_keys(entries):
    return {(e.get("tool"), e.get("args")) for e in (entries or [])}


def _error_spans(trace):
    """(kind, name) for every non-ok span, as a multiset-safe set of keys."""
    return {(s.get("kind"), s.get("name"))
            for s in (trace or {}).get("spans", [])
            if s.get("status") not in (None, "ok")}


def regressions(control_trace, treatment_trace):
    """Detector signals present in treatment but not control.

    Never resolved toward "the treatment is fine" -- an ambiguous or missing
    trace on either side simply yields fewer detectable regressions, never a
    fabricated clean bill.
    """
    control_trace = control_trace or {}
    treatment_trace = treatment_trace or {}
    control_metrics = control_trace.get("metrics") or {}
    treatment_metrics = treatment_trace.get("metrics") or {}

    out = []

    new_failed = (_mcp_names(treatment_metrics.get("failed_mcp_servers"))
                  - _mcp_names(control_metrics.get("failed_mcp_servers")))
    if new_failed:
        out.append(make_regression(
            kind="mcp_newly_failed",
            severity="high",
            detail=f"MCP server(s) failed handshake only in treatment: {sorted(new_failed)}",
            evidence=json.dumps(treatment_metrics.get("failed_mcp_servers"), default=str),
        ))

    new_idle = (_mcp_names(treatment_metrics.get("idle_mcp_servers"))
                - _mcp_names(control_metrics.get("idle_mcp_servers")))
    if new_idle:
        out.append(make_regression(
            kind="mcp_newly_idle",
            severity="low",
            detail=f"MCP server(s) connected but never called only in treatment: {sorted(new_idle)}",
            evidence=json.dumps(treatment_metrics.get("idle_mcp_servers"), default=str),
        ))

    new_dupes = (_duplicate_keys(treatment_metrics.get("duplicate_calls"))
                 - _duplicate_keys(control_metrics.get("duplicate_calls")))
    if new_dupes:
        out.append(make_regression(
            kind="duplicate_call_new",
            severity="medium",
            detail=f"New duplicate tool-call pair(s) in treatment: {sorted(t for t, _ in new_dupes)}",
            evidence=json.dumps(treatment_metrics.get("duplicate_calls"), default=str),
        ))

    treatment_status = (treatment_trace.get("session") or {}).get("status")
    if treatment_status not in (None, "ok"):
        out.append(make_regression(
            kind="session_status_error",
            severity="high",
            detail=f"Treatment session status is {treatment_status!r}",
            evidence=treatment_status,
        ))

    new_error_spans = _error_spans(treatment_trace) - _error_spans(control_trace)
    if new_error_spans:
        out.append(make_regression(
            kind="span_error_new",
            severity="high",
            detail=f"New error-status span(s) in treatment: {sorted(new_error_spans)}",
            evidence=json.dumps(sorted(str(k) for k in new_error_spans)),
        ))

    control_turns = ((control_metrics.get("counts") or {}).get("turns"))
    treatment_turns = ((treatment_metrics.get("counts") or {}).get("turns"))
    if control_turns and treatment_turns is not None and treatment_turns > control_turns * 1.5:
        out.append(make_regression(
            kind="turn_count_increase",
            severity="medium",
            detail=f"Turns rose from {control_turns} to {treatment_turns} (>50%)",
            evidence=f"control={control_turns} treatment={treatment_turns}",
        ))

    return out


# --- judge -----------------------------------------------------------------

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
}

_JUDGE_RUBRIC = """\
You are grading two candidate agent runs against the same user prompt. Pick
the better one, or "tie" if neither is clearly better.

Rubric, in priority order:
1. Correctness with respect to the user's request.
2. Constraint adherence (the user's explicit requirements were followed).
3. Tool-use quality: the right tool was used, with valid arguments, and its
   result was actually used rather than ignored or re-derived.
4. Completeness: the request was fully addressed, not partially.
5. Efficiency (fewer tokens, calls, turns) breaks ties ONLY when correctness,
   constraints, tool-use, and completeness are equal between the two.

Any new failure-mode behavior (an error, a broken tool call, a claim not
supported by the transcript) loses outright regardless of efficiency.

Latency or timing claims in a response are only trustworthy when the
underlying span's duration_source is "measured" -- do not reward a response
for citing a derived or unknown duration as if it were fact, and do not
penalize a response for declining to make an unsupported timing claim.

Ties are a legitimate, expected verdict on easy prompts. Do not force a
preference where the two responses are substantively equivalent -- an
inflated preference is noise, not signal.

Return ONLY JSON matching the provided schema: winner is "1" if Response 1 is
better, "2" if Response 2 is better, "tie" otherwise.
"""


def _judge_prompt(user_prompt, first, second):
    return (
        _JUDGE_RUBRIC
        + f"\n\n## User prompt\n\n{user_prompt}\n"
        + f"\n## Response 1\n\n{first}\n"
        + f"\n## Response 2\n\n{second}\n"
    )


def _default_grok(prompt_text, schema, *, model=None, timeout=600):
    """One `grok-dev` child process, structured output via --json-schema.

    Returns the parsed judge dict, or None on any parse failure -- judge_pair
    treats None as an unparseable call and degrades the pair to a tie rather
    than raising or guessing.
    """
    cmd = [str(GROK), "-p", prompt_text,
           "--json-schema", json.dumps(schema),
           "--output-format", "json",
           "--permission-mode", "bypassPermissions",
           "--disallowed-tools", "run_terminal_command,read_file,write,search_replace"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    text = out.get("text") or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _call_winner(raw):
    """Map a raw judge result to 'first' / 'second' / 'tie' / None (garbage)."""
    if not isinstance(raw, dict):
        return None
    winner = raw.get("winner")
    if winner == "1":
        return "first"
    if winner == "2":
        return "second"
    if winner == "tie":
        return "tie"
    return None


def judge_pair(prompt, response_a, response_b, *, grok=None, model=None):
    """Position-swapped pairwise verdict between response_a (control) and
    response_b (treatment).

    Two calls are made: (a, b) then (b, a). When the two orderings disagree
    about the winner, consistent=False and the verdict is "tie" -- an
    order-dependent preference is not evidence, and this function never
    resolves that disagreement toward either side. A judge call that returns
    unparseable output degrades the whole pair to a tie the same way.
    """
    grok = grok or _default_grok

    raw1 = grok(_judge_prompt(prompt, response_a, response_b), _JUDGE_SCHEMA, model=model)
    raw2 = grok(_judge_prompt(prompt, response_b, response_a), _JUDGE_SCHEMA, model=model)

    w1 = _call_winner(raw1)  # "first" == a, "second" == b
    w2 = _call_winner(raw2)  # "first" == b, "second" == a

    if w1 is None or w2 is None:
        return make_judge_verdict(
            verdict="tie", consistent=False,
            reason="judge call returned unparseable output",
            raw=[raw1, raw2],
        )

    side1 = {"first": ARM_CONTROL, "second": ARM_TREATMENT, "tie": "tie"}[w1]
    side2 = {"first": ARM_TREATMENT, "second": ARM_CONTROL, "tie": "tie"}[w2]

    if side1 != side2:
        return make_judge_verdict(
            verdict="tie", consistent=False,
            reason="position swap disagreed",
            raw=[raw1, raw2],
        )

    return make_judge_verdict(
        verdict=side1, consistent=True,
        reason=(raw1.get("reason", "") if isinstance(raw1, dict) else ""),
        raw=[raw1, raw2],
    )


# --- matrix ------------------------------------------------------------------

def grade_matrix(matrix, *, prompt, grok=None):
    """Pair control[i] with treatment[i]: efficiency both sides, judge the
    pair, collect regressions. One Grade per repeat."""
    control_runs = matrix["arms"].get(ARM_CONTROL, [])
    treatment_runs = matrix["arms"].get(ARM_TREATMENT, [])

    grades = []
    for i, (control_run, treatment_run) in enumerate(zip(control_runs, treatment_runs)):
        control_trace = control_run.get("trace")
        treatment_trace = treatment_run.get("trace")

        control_eff = efficiency(control_run)
        treatment_eff = efficiency(treatment_run)

        regs = (regressions(control_trace, treatment_trace)
                if control_trace and treatment_trace else [])

        judge = None
        if control_run.get("ok") and treatment_run.get("ok"):
            judge = judge_pair(
                prompt,
                control_run.get("response_text", ""),
                treatment_run.get("response_text", ""),
                grok=grok,
            )

        grades.append(make_grade(
            repeat=control_run.get("repeat", i),
            control_efficiency=control_eff,
            treatment_efficiency=treatment_eff,
            judge=judge,
            regressions=regs,
            control_ok=control_run.get("ok", True),
            treatment_ok=treatment_run.get("ok", True),
        ))

    return grades
