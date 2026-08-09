"""interstellar/report.py — assemble and render the review-loop Report.

`build()` consumes `make_patch_result()` dicts (interstellar/types.py, the
frozen contract) already carrying their `statistics` / `gate` / verdict_line
— those are computed upstream by interstellar/stats.py and the orchestrator.
This module only assembles the top-level `make_report()` document (deriving
the cross-patch caveats block) and renders it to a self-contained dashboard:

    build(...) -> Report                  # dict, types.make_report shape
    write(report, out_dir) -> Path        # report.json + index.html
    serve(out_dir, port=4242)             # http.server, 127.0.0.1 only

The HTML is generated as one plain string — inline CSS/JS, no CDN, no build
step — so `write()`'s output is the whole deliverable: open index.html and
everything (diffs, tables, caveats) is already there.

No fabricated numbers: every figure rendered comes from the report dict. A
statistic that carries an "insufficient samples" note (bootstrap `lo`/`hi`
of None, per interstellar/stats.py) is rendered as that caveat text, never
as a number.
"""

from __future__ import annotations

import functools
import html
import http.server
import json
import re
import socketserver
from datetime import datetime, timezone
from pathlib import Path

from interstellar.stats import EFFECT_TO_EFFICIENCY_METRIC
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFICIENCY_METRICS,
    make_report,
)

# --- cross-patch caveats -----------------------------------------------------
#
# These are real, measured limitations of this experiment, not fine print —
# the report's credibility rests on rendering them prominently. Three are
# cycle-wide (constant regardless of which patches ran) and are generated
# here so they can never be silently dropped; the rest are computed from the
# actual patch_results so they say something true about *this* cycle, not a
# canned disclaimer.

ISOLATION_NOTE = (
    "Isolation is partial: GROK_HOME sandboxing covers $GROK_HOME/skills/ "
    "only. The harness's bundled skills (~22) plus anything under "
    "~/.agents/skills/ remain visible to both arms. They are constant across "
    "control and treatment, so paired deltas stay valid, but the absolute "
    "numbers in the tables below are not a clean-room measurement."
)

CONTROL_IS_RERUN_NOTE = (
    "The control arm is the unpatched harness re-run under this cycle's "
    "isolation, not the user's original recorded session — that original "
    "run ran in a different environment and is shown for context only, "
    "never as the statistical baseline."
)


def _small_n_caveat(k):
    """At k<10 no statistic here can reach conventional significance by
    construction (research-methods.md §1): the exact sign-flip permutation
    test's best possible two-sided p at n=k is 2*(0.5)^k — 0.25 at k=3,
    0.0625 at k=5 — and a percentile bootstrap on k points has too few
    distinct resamples (C(2k-1,k)) to have characterized coverage. State the
    actual floor for this cycle's k rather than a generic warning."""
    k = int(k)
    if k and k < 10:
        floor = 2 * (0.5 ** k)
        return (
            f"k={k} paired repeats per patch. At this sample size no "
            f"statistic below can reach conventional significance by "
            f"construction — the exact sign-flip permutation test's best "
            f"possible two-sided p-value at n={k} is {floor:.3g}, and a "
            f"percentile bootstrap on {k} points has too few distinct "
            f"resamples for characterized coverage. Every number here is a "
            f"directional signal, not proof — that is why gates land on "
            f"PROVISIONAL or DIRECTIONAL rather than ACCEPTED at this k."
        )
    return f"k={k} paired repeats per arm, per patch."


def _walk_insufficient(node) -> bool:
    """True if any nested stats sub-dict flags insufficient_samples or an
    unresolved (lo=hi=None) confidence interval — the stats.py signal for
    "n too small to trust a CI"."""
    if isinstance(node, dict):
        if node.get("note") == "insufficient_samples" or node.get("insufficient_samples"):
            return True
        if "lo" in node and "hi" in node and node["lo"] is None and node["hi"] is None:
            return True
        return any(_walk_insufficient(v) for v in node.values())
    if isinstance(node, list):
        return any(_walk_insufficient(v) for v in node)
    return False


def _judge_consistency(patch_results):
    """Prefer each patch's own statistics["judge_consistency"] (pinned
    shape: {pairs, agreed, rate, note}) so the cycle-wide caveat and the
    per-patch figure are computed from the same numbers; fall back to
    walking grades[].judge directly for a patch_result that predates that
    field."""
    total = consistent = 0
    for pr in patch_results:
        jc = (pr.get("statistics") or {}).get("judge_consistency")
        if isinstance(jc, dict) and jc.get("pairs") is not None:
            total += jc.get("pairs", 0)
            consistent += jc.get("agreed", 0)
            continue
        for grade in pr.get("grades", []) or []:
            judge = grade.get("judge")
            if not judge:
                continue
            total += 1
            if judge.get("consistent"):
                consistent += 1
    return consistent, total


def _insufficient_patch_ids(patch_results):
    ids = []
    for pr in patch_results:
        if _walk_insufficient(pr.get("statistics") or {}):
            ids.append(pr["patch"]["patch_id"])
    return ids


# Published baseline position-consistency for pairwise LLM judges runs
# ~70-77% (research-methods.md §3). Below that, the win rate is noisy
# because of the judge, not because the patches are actually tied.
CONSISTENCY_BASELINE_FLOOR = 0.70


def default_caveats(k, patch_results):
    """The caveats block: k (with the honest small-n floor), the partial-
    isolation note, the control-is-a-rerun note, judge consistency rate,
    and insufficient-sample flags. Always non-empty, always rendered
    prominently by write() — this is the trust layer, not fine print."""
    caveats = [_small_n_caveat(k), ISOLATION_NOTE, CONTROL_IS_RERUN_NOTE]

    consistent, total = _judge_consistency(patch_results)
    if total:
        rate = consistent / total
        low = (
            f" — below the ~{CONSISTENCY_BASELINE_FLOOR:.0%}-77% baseline "
            f"for pairwise LLM judges; discount the win rates accordingly"
            if rate < CONSISTENCY_BASELINE_FLOOR else ""
        )
        caveats.append(
            f"Judge consistency: {consistent}/{total} position-swapped pairs "
            f"agreed on a winner ({rate:.0%}){low}; an inconsistent pair is "
            f"scored as a tie, never resolved toward either side."
        )
    else:
        caveats.append("No judged pairs were available for this cycle.")

    flagged = _insufficient_patch_ids(patch_results)
    if flagged:
        caveats.append(
            "Point estimates only (no credible interval) for: "
            + ", ".join(str(p) for p in flagged)
            + " — shown with their mean/rate rather than omitted, per the "
            "insufficient-samples rule: a small sample is not a zero."
        )
    return caveats


def _dropped_patch_caveat(p):
    """A patch dropped before ranking (types.make_patch's own docstring:
    "must still appear in the report -- a silently dropped recommendation
    is indistinguishable from one that was never made") never gets a card
    -- it has no application/grades/statistics/gate to render one from.
    `caveats` (frozen: list[str]) is the only surface build() has for it
    without inventing a new Report key outside types.make_report's shape,
    so this renders one line per dropped patch carrying everything a bare
    make_patch() dict has: id, kind, target, why it was skipped, and its
    rationale/recommendation lineage -- richer than a single joined
    sentence naming every dropped id at once."""
    bits = [f"Considered but never run: {p.get('kind', '?')} on {p.get('target', '?')} ({p.get('patch_id', '?')})"]
    if p.get("source_recommendation"):
        bits.append(f"from recommendation {p['source_recommendation']}")
    bits.append(f"skipped: {p.get('skipped_reason') or 'no reason recorded'}")
    if p.get("rationale"):
        bits.append(f"rationale: {p['rationale']}")
    return " — ".join(bits)


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(*, session, baseline_version, patch_results, dropped_patches=None,
          k=0, cost_usd=0.0, generated_at=None, extra_caveats=None):
    """Assemble the Report per types.make_report.

    `patch_results` must already be fully-formed make_patch_result() dicts
    (statistics/gate/verdict_line computed upstream). build() assembles the
    document and derives the cross-patch caveats; it does not recompute any
    per-patch statistic.

    `dropped_patches` (optional): make_patch() dicts for recommendations
    that were dropped before ranking (skipped_reason set, never applied or
    measured). types.py has no shape for a patch with no application/
    grades/statistics, so these do not get their own card -- see
    `_dropped_patch_caveat`. Full first-class rendering would need a
    frozen-contract decision this module cannot make unilaterally.
    """
    for pr in patch_results:
        if not pr.get("verdict_line"):
            raise ValueError(
                f"patch_result {pr.get('patch', {}).get('patch_id')!r} is "
                "missing verdict_line"
            )

    caveats = default_caveats(k, patch_results)
    for p in (dropped_patches or []):
        caveats.append(_dropped_patch_caveat(p))
    if extra_caveats:
        caveats = caveats + list(extra_caveats)

    return make_report(
        session=session,
        baseline_version=baseline_version,
        patch_results=patch_results,
        caveats=caveats,
        cost_usd=cost_usd,
        k=k,
        generated_at=generated_at or _now_iso(),
    )


# --- rendering ----------------------------------------------------------------

METRIC_LABELS = {
    "total_tokens": "Total tokens",
    "cost_usd": "Cost (USD)",
    "wall_ms": "Wall time (ms)",
    "tool_calls": "Tool calls",
    "skill_tokens_est": "Skill tokens (est.)",
    "skill_wasted_tokens_est": "Skill tokens wasted (est.)",
    "tool_result_tokens_est": "Tool result tokens (est.)",
    "mcp_startup_ms": "MCP startup (ms)",
    "duplicate_calls": "Duplicate calls",
    "turns": "Turns",
}


def _esc(x) -> str:
    """Escape x for HTML. None renders as an empty string, NOT a dash --
    constraint 6 names the dash as exactly the forbidden rendering for a
    missing statistic, so this module has no silent None-to-"—" default a
    statistic could accidentally inherit. Call sites that want an explicit
    placeholder for genuinely absent (non-statistical) metadata must say so
    themselves, e.g. `_esc(x or "—")` -- see render_html's header/session
    fields."""
    if x is None:
        return ""
    return html.escape(str(x))


def _fmt_num(x, digits=2):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:,.{digits}f}"
    return f"{x:,}"


def _fmt_delta(x, digits=2):
    if x is None:
        return "n/a"
    return f"{x:+,.{digits}f}"


def _fmt_pct(x):
    """Only called with real paired data (see _metric_row's no_pairs
    branch, which intercepts the "no data at all" case before this is
    reached) -- so a None here means exactly one true thing: the control
    median is exactly 0, which makes a percent change undefined. Do not
    reuse this message for "no data"; that is a different, false claim
    about the control value ("≈ 0" implies a real near-zero measurement)."""
    if x is None:
        return "n/a (control median is 0, percent undefined)"
    return f"{x:+.1%}"


def _interval_dict(d):
    """The pinned per-patch `statistics["efficiency"][metric]` shape carries
    lo/hi/note flat on the metric dict itself (no nested "bootstrap"). Read
    that first; fall back to a nested paired_bootstrap()-shaped sub-dict
    (stats.py's own raw shape, key "bootstrap", or the older "ci") in case
    a raw stats.py dict slips through un-flattened."""
    if not d:
        return {}
    if d.get("lo") is not None or d.get("hi") is not None or "lo" in d:
        return d
    return d.get("bootstrap") or d.get("ci") or {}


def _interval_text(d):
    """lo/hi is None whenever note=="insufficient_samples" (n<10) — stats.py
    never fabricates a CI from too few points. The abs_delta/pct_delta point
    estimate is shown in its own column regardless, so this cell only needs
    to say why there's no interval, never an empty cell or a bare dash.
    note=="low_confidence" (10<=n<20) means an interval IS present but
    flagged; distinct_resamples (math.comb(2n-1,n)) explains the "why" for
    either case rather than just asserting it."""
    d = _interval_dict(d)
    lo, hi, note = d.get("lo"), d.get("hi"), d.get("note")
    dr = d.get("distinct_resamples")
    if lo is None or hi is None:
        text = "insufficient samples for a CI"
        if note and note != "insufficient_samples":
            text += f" ({note})"
        if dr:
            text += f" — only {dr} distinct resample{'s' if dr != 1 else ''} at this n"
        return text
    level = d.get("level", 0.95)
    text = f"[{_fmt_num(lo)}, {_fmt_num(hi)}] ({level:.0%} CI)"
    if note == "low_confidence":
        text += " — low confidence" + (f" ({dr} distinct resamples)" if dr else "")
    # A zero-width interval means every paired delta was numerically
    # identical -- a real, computable CI, but a degenerate one (there was
    # no variation to bootstrap over), not the same credibility as a wide
    # sample producing a tight interval. Say so rather than let it read as
    # an unusually confident result.
    if lo == hi:
        text += " — degenerate: every paired delta was identical, no variation to interval over"
    return text


def _diff_html(diff_text):
    if not diff_text or not diff_text.strip():
        return '<p class="muted">No diff.</p>'
    lines = []
    for ln in diff_text.splitlines():
        if ln.startswith(("+++", "---")):
            cls = "hdr"
        elif ln.startswith("@@"):
            cls = "hunk"
        elif ln.startswith("+"):
            cls = "add"
        elif ln.startswith("-"):
            cls = "del"
        else:
            cls = "ctx"
        lines.append(f'<span class="diff-line {cls}">{html.escape(ln) or " "}</span>')
    return '<pre class="diff-wrap"><code>' + "\n".join(lines) + "</code></pre>"


def _metric_row(name, lower_is_better, m, highlight=False):
    if not m:
        return ""
    label = METRIC_LABELS.get(name, name)
    tag = ' <span class="tag">target</span>' if highlight else ""
    hl = "hl" if highlight else ""

    # No run supplied a pair for this metric at all (n=0, both medians
    # None) is a different, stronger absence than "control median happens
    # to be 0" -- must say so plainly, never as an em-dash plus a percent
    # guess about why (constraint 6: a small sample is not a zero, and an
    # absent sample is not a "control ≈ 0").
    if m.get("control_median") is None and m.get("treatment_median") is None:
        row_cls = f"no-data {hl}".strip()
        return (
            f'<tr class="{row_cls}">'
            f"<td>{_esc(label)}{tag}</td>"
            f'<td colspan="4" class="no-data-cell">no paired data — no run reported both '
            f"a control and a treatment value for this metric</td>"
            f"</tr>"
        )

    # Pinned shape (statistics["efficiency"][metric]): abs_delta/pct_delta,
    # not delta_abs/delta_pct. Fall back to the older names defensively.
    delta_abs = m.get("abs_delta", m.get("delta_abs"))
    delta_pct = m.get("pct_delta", m.get("delta_pct"))

    # abs_delta = control - treatment (stats.py's convention): POSITIVE
    # means the treatment used less of the metric. For a lower_is_better
    # metric that is the improvement -- delta_abs > 0 is good, not < 0.
    # Color and wording are derived from the same `improved` boolean on
    # purpose, so they can never disagree with each other the way an
    # independently-computed label could.
    cls = "neutral"
    direction = ""
    if delta_abs is not None and delta_abs != 0:
        improved = (delta_abs > 0) if lower_is_better else (delta_abs < 0)
        cls = "good" if improved else "bad"
        if lower_is_better:
            direction = " saved" if improved else " cost more"
        else:
            direction = " gained" if improved else " lost"

    row_cls = f"{cls} {hl}".strip()
    return (
        f'<tr class="{row_cls}">'
        f"<td>{_esc(label)}{tag}</td>"
        f"<td>{_fmt_num(m.get('control_median'))}</td>"
        f"<td>{_fmt_num(m.get('treatment_median'))}</td>"
        f"<td>{_fmt_delta(delta_abs)}{_esc(direction)}</td>"
        f"<td>{_fmt_pct(delta_pct)}</td>"
        f"<td>{_esc(_interval_text(m))}</td>"
        f"</tr>"
    )


def _metrics_table(efficiency, metric_target):
    rows = "".join(
        _metric_row(name, lower, efficiency.get(name), highlight=(name == metric_target))
        for name, lower in EFFICIENCY_METRICS
        if efficiency.get(name)
    )
    if not rows:
        return '<p class="muted">No efficiency statistics available.</p>'
    return (
        '<div class="table-wrap"><table class="metrics">'
        "<thead><tr><th>Metric</th><th>Control (median)</th>"
        "<th>Treatment (median)</th><th>&Delta;</th><th>&Delta;%</th>"
        "<th>95% CI on &Delta;</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _win_rate_block(win_rate, judge_consistency=None):
    """win/tie/loss bar + rate, with judge_consistency (a SEPARATE key in
    the pinned statistics shape, not nested under win_rate) rendered
    directly beneath it — adjacent to the number it qualifies.

    All-ties (note=="no_decided_pairs") is a normal, common result — the
    judge finding the two harnesses equivalent is real evidence, not
    missing data — and must render as a deliberate reading ("N ties, no
    decided pairs"), not as an empty/n-a widget."""
    win_rate = win_rate or {}
    wins, ties, losses = win_rate.get("wins", 0), win_rate.get("ties", 0), win_rate.get("losses", 0)
    total = wins + ties + losses
    if total == 0:
        return '<p class="muted">No judged pairs.</p>'
    seg = lambda cls, n, label: (
        f'<div class="wtl-seg wtl-{cls}" style="width:{n / total * 100:.2f}%" '
        f'title="{label}: {n}"></div>'
    )
    bar = (
        '<div class="wtl-bar">'
        + seg("win", wins, "treatment wins")
        + seg("tie", ties, "ties")
        + seg("loss", losses, "treatment losses")
        + "</div>"
    )
    rate, lo, hi, note = win_rate.get("rate"), win_rate.get("lo"), win_rate.get("hi"), win_rate.get("note")
    if note == "no_decided_pairs" or (rate is None and wins == 0 and losses == 0 and ties > 0):
        rate_txt = (
            f"{ties} tie{'s' if ties != 1 else ''}, no decided pairs — the judge "
            f"found the two harnesses equivalent on every judged run"
        )
    elif rate is None:
        rate_txt = "win rate: n/a" + (f" ({note})" if note else "")
    else:
        rate_txt = f"win rate over decided pairs: {rate:.0%}"
        if lo is not None and hi is not None:
            rate_txt += f" [{lo:.0%}, {hi:.0%}] 95% CI"
            if note == "low_confidence":
                rate_txt += " (low confidence)"
        else:
            rate_txt += " (insufficient samples for a CI)"
    legend = (
        '<div class="wtl-legend">'
        f'<span class="wtl-dot wtl-win"></span>wins {wins} '
        f'<span class="wtl-dot wtl-tie"></span>ties {ties} '
        f'<span class="wtl-dot wtl-loss"></span>losses {losses}'
        f'<span class="muted"> ({total} judged pair{"s" if total != 1 else ""})</span>'
        f'<span class="muted"> &middot; {_esc(rate_txt)}</span>'
        "</div>"
    )
    # Position consistency: the fraction of position-swapped pairs where the
    # judge agreed on a winner both orders. Published baselines for pairwise
    # LLM judges run ~70-77% — put it right next to the win rate it qualifies
    # (research-methods.md §3), not buried, and flag it visibly when low.
    jc = judge_consistency or {}
    cons_rate = jc.get("rate")
    consistency_html = ""
    if cons_rate is not None:
        low = cons_rate < CONSISTENCY_BASELINE_FLOOR
        pairs, agreed = jc.get("pairs"), jc.get("agreed")
        cons_txt = f"position consistency: {cons_rate:.0%}"
        if pairs is not None and agreed is not None:
            cons_txt += f" ({agreed}/{pairs} pairs)"
        if low:
            cons_txt += " — below the ~70-77% published baseline; discount the win rate above"
        consistency_html = f'<div class="stat-line{" consistency-low" if low else ""}">{_esc(cons_txt)}</div>'
    elif jc.get("note"):
        consistency_html = f'<div class="stat-line muted">position consistency: {_esc(jc["note"])}</div>'
    return bar + legend + consistency_html


def _permutation_line(stats):
    """The honest replacement for a plain sign test at small n
    (research-methods.md §1): carries min_achievable_p, the smallest
    two-sided p this sample size could ever produce. When the observed p
    equals that floor, "not significant" is the wrong reading — it's "as
    extreme as this sample size permits," which is evidence of nothing,
    not evidence of no effect. Pinned key is "permutation"; fall back to
    stats.py's own function-name keys for resilience."""
    perm = stats.get("permutation") or stats.get("permutation_test") or stats.get("sign_test")
    if not isinstance(perm, dict) or perm.get("p") is None:
        return ""
    p = perm["p"]
    min_p = perm.get("min_achievable_p")
    n_nz = perm.get("n_nonzero", perm.get("n"))
    direction = perm.get("direction")
    note = perm.get("note")
    # The floor phrasing exists to stop a small-n p-value from being
    # misread as "no effect" when the test structurally could not have
    # reached significance. At large n the floor is far below 0.05 and
    # this test IS well-powered -- gate the phrasing on the floor itself
    # being underpowered (>0.05), not merely on p equalling it, so a
    # genuinely strong large-n result isn't hedged with language written
    # for the k=3 case.
    if min_p is not None and min_p > 0.05 and abs(p - min_p) < 1e-9:
        p_txt = f"p={p:.3g} (as extreme as n={_esc(n_nz)} permits — not evidence of no effect)"
    else:
        p_txt = f"p={p:.3g}"
    extra = f", {_esc(note)}" if note else ""
    return (
        f'<div class="stat-line">Sign-flip permutation test: {p_txt}, '
        f'direction={_esc(direction)}, n<sub>nonzero</sub>={_esc(n_nz)}{extra}</div>'
    )


def _pass_k_line(stats):
    """Pinned shape is a single flat {value, n, c, k, degenerate, note}
    dict (the treatment's pass^k reliability reading, our v1 one-prompt-
    per-patch design), not a {control, treatment} pair — but a legacy
    per-arm shape is still handled so this never crashes on drift."""
    pk = stats.get("pass_k")
    if not isinstance(pk, dict):
        return ""
    if "control" in pk or "treatment" in pk:
        return (
            f'<div class="stat-line">pass&#94;k: control {_fmt_pass_k(pk.get("control"))} '
            f'&rarr; treatment {_fmt_pass_k(pk.get("treatment"))}</div>'
        )
    if "value" not in pk and "degenerate" not in pk:
        return ""
    if pk.get("n") == 0:
        return '<div class="stat-line">pass&#94;k: no runs completed</div>'
    k = pk.get("k")
    label = f"pass&#94;{_esc(k)}" if k is not None else "pass&#94;k"
    detail = f" (c={_esc(pk.get('c'))}/n={_esc(pk.get('n'))})" if pk.get("c") is not None else ""
    note = pk.get("note")
    # degenerate's own note (e.g. "n_equals_k: 0/1 indicator...") restates
    # what _fmt_pass_k's "(flag, n=k — not a rate)" phrasing already says —
    # skip it there to avoid saying the same thing twice.
    note_txt = f", {_esc(note)}" if note and note != "k_exceeds_n" and not pk.get("degenerate") else ""
    if note == "k_exceeds_n":
        return f'<div class="stat-line">{label}: n/a (k exceeds n, no batch large enough yet)</div>'
    return f'<div class="stat-line">{label}: {_fmt_pass_k(pk)}{detail}{note_txt}</div>'


def _regressions_html(stats):
    """Flat list of make_regression() dicts across the patch's grades. The
    gate cites high-severity ones in its reasons already; showing the full
    list (medium/low included) is the transparency the gate reasons alone
    don't give a reader deciding whether to trust a PROVISIONAL/DIRECTIONAL
    patch."""
    regs = stats.get("regressions") or []
    if not regs:
        return ""
    items = []
    for r in regs:
        severity = r.get("severity", "low")
        evidence = f" &mdash; {_esc(r.get('evidence'))}" if r.get("evidence") else ""
        items.append(
            f'<li class="regression-{_esc(severity)}">'
            f'<span class="regression-sev">{_esc(severity)}</span> '
            f'{_esc(r.get("kind"))}: {_esc(r.get("detail"))}{evidence}</li>'
        )
    return (
        '<div class="section-label">Regressions observed</div>'
        f'<ul class="regressions">{"".join(items)}</ul>'
    )


def _fmt_pass_k(v):
    """pass^k (tau-bench reliability reading) can arrive as a bare float or
    as {value, degenerate}. degenerate=True (n==k) means the number is a 0/1
    "did all k runs succeed" flag, not a probability — label it as a flag,
    never format it as a percentage (research-methods.md §2)."""
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        val = v.get("value", v.get("pass_k"))
        degenerate = bool(v.get("degenerate"))
    else:
        val, degenerate = v, False
    if val is None:
        return "n/a"
    if degenerate:
        return ("all k succeeded" if val >= 1 else "not all k succeeded") + " (flag, n=k — not a rate)"
    return f"{val:.0%}"


def _matrix_summary_line(matrix):
    if not matrix:
        return ""
    arms = matrix.get("arms") or {}
    parts = []
    for arm in (ARM_CONTROL, ARM_TREATMENT):
        runs = arms.get(arm) or []
        ok = sum(1 for r in runs if r.get("ok"))
        parts.append(f"{arm}: {ok}/{len(runs)} runs ok")
    return " · ".join(parts)


def _gate_badge(gate, *, applied=True, high_severity_regressions=0, win_rate=None):
    """The badge is the one thing a scanning reader actually reads, so it
    must never be able to say something that isn't true. Three checks run
    BEFORE the gate is even consulted, in this order, because each one
    contradicts what the gate alone would otherwise imply:

    1. `not applied` -- nothing was ever measured. `stats.gate()` doesn't
       know a patch never ran; cli.py's not-applied gate result still sets
       `directional`/`reasons` fields that would otherwise fall through to
       a red REJECTED ("we measured this and it lost"), when the true
       state is "we couldn't even build it to measure." NOT RUN, gray,
       always wins.
    2. `high_severity_regressions > 0` -- the harness crashed or produced a
       high-severity regression. `stats.gate()` can still compute a
       favorable `directional`/`provisional` reading from win/loss counts
       and the efficiency point estimate alone (see `_directional` in
       stats.py), which never looks at `regressions` -- so a treatment that
       won every judged pair right up until it crashed on one repeat can
       score PROVISIONAL/DIRECTIONAL · FAVORABLE. The team-lead is
       separately having stats.py stop doing this, but this module does
       not rely on that alone: a high-severity regression forces REGRESSED
       here regardless of what accepted/provisional/directional say.
    3. Zero decided pairs backing an ACCEPTED -- `stats.gate()`'s n>=10
       quality check passes on `losses == 0`, which an all-ties result
       (0 wins, 0 losses) satisfies with no quality evidence at all. A
       green ACCEPTED there is technically the gate's own call but is
       exactly the "plausible, not true" failure this dashboard cannot
       have -- downgrade it to a qualified badge, matching the visual
       weight of ACCEPTED · PROVISIONAL, so it cannot be mistaken for the
       same claim as a real win.

    Only past all three does the five-state truth table apply -- see the
    inline states below. Falls back to a legacy REJECT only for a gate
    dict that predates the provisional/directional fields entirely.
    """
    if not applied:
        return '<span class="badge not-run">NOT RUN</span>'
    if high_severity_regressions:
        n = high_severity_regressions
        return f'<span class="badge regressed">REGRESSED &middot; {n} HIGH-SEVERITY</span>'

    accepted = bool(gate.get("accepted"))
    provisional = bool(gate.get("provisional"))
    has_directional = "directional" in gate
    reading = (gate.get("directional") or "neutral").lower()
    reading_cls = {"favorable": "directional-favorable", "unfavorable": "directional-unfavorable"}.get(
        reading, "directional-neutral"
    )

    wr = win_rate or {}
    zero_quality_evidence = (wr.get("wins", 0) + wr.get("losses", 0)) == 0 and wr.get("ties", 0) > 0

    if accepted and provisional:
        return '<span class="badge accept-provisional">ACCEPTED &middot; PROVISIONAL</span>'
    if accepted and zero_quality_evidence:
        return '<span class="badge accept-provisional">ACCEPTED &middot; NO DECIDED PAIRS</span>'
    if accepted:
        return '<span class="badge accept">ACCEPTED</span>'
    if provisional:
        return f'<span class="badge {reading_cls}">PROVISIONAL &middot; {_esc(reading.upper())}</span>'
    if has_directional:
        reasons = gate.get("reasons") or []
        too_few_for_any_call = any("insufficient_samples_for_acceptance" in r for r in reasons)
        if too_few_for_any_call:
            return f'<span class="badge {reading_cls}">DIRECTIONAL &middot; {_esc(reading.upper())}</span>'
        return '<span class="badge reject">REJECTED</span>'
    return '<span class="badge reject">REJECT</span>'


# Caveats generated by default_caveats() are cycle-wide by construction
# (they describe the whole run, not one patch) and several of them
# legitimately name every patch id in the cycle (e.g. the insufficient-
# samples summary) -- matching those against a single patch's target/id
# would staple the same generic warning onto every card (see I-3). Only
# caveats NOT produced by default_caveats() (i.e. extra_caveats the
# orchestrator passed in) are candidates for per-patch surfacing.
_DEFAULT_CAVEAT_PREFIXES = (
    "k=", "Isolation is partial:", "The control arm is the unpatched harness re-run",
    "Judge consistency:", "No judged pairs were available",
    "Point estimates only (no credible interval) for:",
)


def _is_default_caveat(c):
    return any(c.startswith(p) for p in _DEFAULT_CAVEAT_PREFIXES)


def _patch_specific_caveats(extra_caveats, target, patch_id):
    """extra_caveats (see _is_default_caveat) that name this patch's
    target or id -- e.g. "skill.remove is a no-op for <target> because it
    also exists under an uncontrolled root" -- surfaced inline next to
    that patch's own numbers, in addition to the persistent cycle-wide
    panel. Caveats are plain strings (frozen contract), so this is a text
    match, not a structured link; word-boundary matching (not a raw
    substring) so a short/common target name like "rules" doesn't
    false-positive inside unrelated caveat text."""
    needles = [str(n) for n in (target, patch_id) if n]
    hits = []
    for c in extra_caveats or []:
        if any(re.search(r"\b" + re.escape(n) + r"\b", c, re.IGNORECASE) for n in needles):
            hits.append(c)
    return hits


def _patch_card(pr, idx, extra_caveats=None):
    patch = pr.get("patch") or {}
    application = pr.get("application") or {}
    stats = pr.get("statistics") or {}
    gate = pr.get("gate") or {}
    matrix = pr.get("matrix") or {}
    verdict = pr.get("verdict_line") or ""

    patch_id = patch.get("patch_id") or f"patch-{idx}"
    kind, target = patch.get("kind", "—"), patch.get("target", "—")
    expected_effect = patch.get("expected_effect", "")
    source_rec = patch.get("source_recommendation")

    diff_text = application.get("diff") or patch.get("diff") or ""
    applied = application.get("applied", False)
    skip_reason = patch.get("skipped_reason") or (not applied and application.get("reason"))

    high_sev_regressions = sum(
        1 for r in (stats.get("regressions") or []) if r.get("severity") == "high"
    )
    gate_badge = _gate_badge(
        gate, applied=applied, high_severity_regressions=high_sev_regressions,
        win_rate=stats.get("win_rate"),
    )
    reasons = gate.get("reasons") or []
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons) or '<li class="muted">No reasons recorded.</li>'

    diff_block = _diff_html(diff_text)
    if skip_reason:
        diff_block = f'<p class="muted">Not applied: {_esc(skip_reason)}</p>' + diff_block

    # statistics["primary_effect"] is an EFFECT_* name (types.ALL_EFFECTS,
    # e.g. "skill_tokens"), the same value summarize()/gate() were called
    # with — NOT an EFFICIENCY_METRICS name by itself (e.g. "skill_tokens"
    # vs. the metric it actually measures, "skill_tokens_est"). Map it
    # through stats.EFFECT_TO_EFFICIENCY_METRIC, the single canonical
    # mapping the CLI, gate(), and this report all share. Falls back to the
    # patch's own expected_effect (types.make_patch, frozen) if statistics
    # didn't carry primary_effect for some reason.
    primary_effect = stats.get("primary_effect") or expected_effect
    metric_target = EFFECT_TO_EFFICIENCY_METRIC.get(primary_effect)
    table_html = _metrics_table(stats.get("efficiency") or {}, metric_target)
    wtl_html = _win_rate_block(stats.get("win_rate"), stats.get("judge_consistency"))
    extra_html = _permutation_line(stats) + _pass_k_line(stats)
    regressions_html = _regressions_html(stats)
    matrix_line = _matrix_summary_line(matrix)

    meta_bits = [f"expects to move <strong>{_esc(expected_effect)}</strong>"]
    if source_rec:
        meta_bits.append(f"from recommendation <em>{_esc(source_rec)}</em>")
    if stats.get("n") is not None:
        meta_bits.append(f"{_esc(stats.get('n'))} paired repeats run")
    if matrix_line:
        meta_bits.append(_esc(matrix_line))

    patch_caveats = _patch_specific_caveats(extra_caveats, target, patch_id)
    patch_caveats_html = ""
    if patch_caveats:
        items = "".join(f"<li>{_esc(c)}</li>" for c in patch_caveats)
        patch_caveats_html = f'<ul class="patch-caveats">{items}</ul>'

    return f"""
    <article class="patch-card" id="{_esc(patch_id)}">
      <div class="patch-head">
        <div class="patch-head-left">
          <span class="badge kind">{_esc(kind)}</span>
          <span class="patch-target">{_esc(target)}</span>
          {gate_badge}
        </div>
        <div class="patch-verdict">{_esc(verdict)}</div>
      </div>
      <div class="patch-meta muted">{" &middot; ".join(meta_bits)}</div>
      <p class="rationale">{_esc(patch.get("rationale"))}</p>
      {patch_caveats_html}
      <div class="patch-body">
        <div class="diff-col">
          <div class="section-label">Diff</div>
          {diff_block}
        </div>
        <div class="stats-col">
          <div class="section-label">Before / after (k paired repeats)</div>
          {table_html}
          <div class="section-label">Judge outcome (treatment vs. control)</div>
          {wtl_html}
          {extra_html}
          {regressions_html}
          <div class="section-label">Gate</div>
          <ul class="reasons">{reasons_html}</ul>
        </div>
      </div>
    </article>
    """


def render_html(report) -> str:
    session = report.get("session") or {}
    baseline = report.get("baseline_version") or {}
    patch_results = report.get("patch_results") or []
    caveats = report.get("caveats") or []
    k = report.get("k", 0)
    cost = report.get("cost_usd", 0.0) or 0.0
    generated_at = report.get("generated_at") or ""

    title = session.get("title") or session.get("session_id") or "Review Report"

    header = f"""
    <header class="topbar">
      <div class="topbar-left">
        <span class="brand-mark">&#9670;</span>
        <div class="trace-title">
          <div class="trace-label">Review Report</div>
          <h1>{_esc(title)}</h1>
        </div>
      </div>
      <div class="topbar-right">
        <span class="pill">session <strong>{_esc(session.get("session_id") or "—")}</strong></span>
        <span class="pill">baseline <strong>{_esc(baseline.get("version_id") or "—")}</strong></span>
        <span class="pill">k <strong>{_esc(k)}</strong></span>
        <span class="pill">cycle cost <strong>${cost:,.4f}</strong></span>
        <span class="pill">generated <strong>{_esc(generated_at or "—")}</strong></span>
      </div>
    </header>
    """

    session_section = f"""
    <section class="session-card">
      <div class="section-label">Prompt replayed</div>
      <pre class="prompt-block">{_esc(session.get("prompt"))}</pre>
      <div class="kv">
        <dt>Trace file</dt><dd>{_esc(session.get("trace_file") or "—")}</dd>
        <dt>Baseline skills</dt><dd>{_esc(baseline.get("skills") if baseline.get("skills") is not None else "—")}</dd>
        <dt>Baseline MCP servers</dt><dd>{_esc(baseline.get("mcp") if baseline.get("mcp") is not None else "—")}</dd>
      </div>
    </section>
    """

    # Only caveats build() didn't generate itself are candidates for
    # per-patch surfacing (see _is_default_caveat / I-3) -- the generated
    # ones are cycle-wide by construction and several legitimately name
    # every patch id, which would staple an identical warning onto every
    # card otherwise.
    extra_only = [c for c in caveats if not _is_default_caveat(c)]
    cards = "\n".join(_patch_card(pr, i, extra_caveats=extra_only) for i, pr in enumerate(patch_results, 1))
    if not cards:
        cards = '<p class="muted">No candidate patches were produced for this session.</p>'

    # Caveats are the trust layer, not a footnote: rendered first, right
    # under the header, in a visually distinct panel — never collapsed by
    # default, never scrolled past to reach the numbers they qualify.
    caveats_html = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    caveats_section = f"""
    <section class="caveats">
      <div class="caveats-title">&#9888; Read before trusting the numbers below</div>
      <ul>{caveats_html}</ul>
    </section>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
{header}
<main class="content">
{caveats_section}
{session_section}
<section class="patches">
{cards}
</section>
</main>
<script>{_SCRIPT}</script>
</body>
</html>
"""


_STYLE = """
:root {
  --bg: #0b0e14;
  --bg-elev: #11151c;
  --bg-card: #0f131a;
  --border: #1e2633;
  --border-strong: #2a3548;
  --text: #e6edf7;
  --muted: #8b97a8;
  --faint: #5c6b7e;
  --accent: #3b82f6;
  --accent-dim: rgba(59, 130, 246, 0.15);
  --good: #34d399;
  --good-dim: rgba(52, 211, 153, 0.14);
  --bad: #f87171;
  --bad-dim: rgba(248, 113, 113, 0.14);
  --add-bg: rgba(52, 211, 153, 0.12);
  --add-fg: #86efac;
  --del-bg: rgba(248, 113, 113, 0.12);
  --del-fg: #fca5a5;
  --hunk-fg: #93c5fd;
  --warn: #fbbf24;
  --warn-dim: rgba(251, 191, 36, 0.12);
  --provisional: #38bdf8;
  --provisional-dim: rgba(56, 189, 248, 0.14);
  --favorable: #38bdf8;
  --favorable-dim: rgba(56, 189, 248, 0.14);
  --unfavorable: #fb923c;
  --unfavorable-dim: rgba(251, 146, 60, 0.14);
  --neutral-reading: #94a3b8;
  --neutral-reading-dim: rgba(148, 163, 184, 0.12);
  --kind-fg: #93c5fd;
  --kind-border: #334155;
  --not-run: #94a3b8;
  --not-run-dim: rgba(148, 163, 184, 0.12);
  --regressed: #f87171;
  --regressed-dim: rgba(248, 113, 113, 0.16);
  --sev-chip-bg: rgba(0, 0, 0, 0.15);
  --radius: 8px;
  --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f6f7f9;
    --bg-elev: #ffffff;
    --bg-card: #ffffff;
    --border: #dde2ea;
    --border-strong: #c7cedb;
    --text: #17202e;
    --muted: #57626f;
    --faint: #8792a1;
    --accent: #2563eb;
    --accent-dim: rgba(37, 99, 235, 0.10);
    --good: #16a34a;
    --good-dim: rgba(22, 163, 74, 0.10);
    --bad: #dc2626;
    --bad-dim: rgba(220, 38, 38, 0.10);
    --add-bg: rgba(22, 163, 74, 0.10);
    --add-fg: #166534;
    --del-bg: rgba(220, 38, 38, 0.10);
    --del-fg: #991b1b;
    --hunk-fg: #1d4ed8;
    --warn: #b45309;
    --warn-dim: rgba(180, 83, 9, 0.10);
    --provisional: #0369a1;
    --provisional-dim: rgba(3, 105, 161, 0.10);
    --favorable: #0369a1;
    --favorable-dim: rgba(3, 105, 161, 0.10);
    --unfavorable: #c2410c;
    --unfavorable-dim: rgba(194, 65, 12, 0.10);
    --neutral-reading: #64748b;
    --neutral-reading-dim: rgba(100, 116, 139, 0.10);
    --kind-fg: #1d4ed8;
    --kind-border: #93c5fd;
    --not-run: #57626f;
    --not-run-dim: rgba(87, 98, 111, 0.10);
    --regressed: #dc2626;
    --regressed-dim: rgba(220, 38, 38, 0.12);
    --sev-chip-bg: rgba(255, 255, 255, 0.55);
  }
}
* { box-sizing: border-box; }
/* No page-level overflow-x:hidden: it would silently swallow a real
   layout overflow bug and can break the sticky topbar below (overflow on
   an ancestor other than a sticky element's own containing block is a
   known way to defeat position:sticky in some browsers). Every wide
   element (diff, metrics table) already scrolls inside its own
   overflow:auto container, which is the correct mechanism and needs no
   page-level backstop. */
html, body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  max-width: 100%;
}
h1, h2, h3 { margin: 0; }
.muted { color: var(--muted); }

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.75rem 1.1rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
  position: sticky;
  top: 0;
  z-index: 5;
}
.topbar-left { display: flex; align-items: center; gap: 0.65rem; min-width: 0; }
.brand-mark { color: var(--accent); font-size: 1.2rem; }
.trace-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--faint); }
.trace-title h1 { font-size: 15px; font-weight: 600; }
.topbar-right { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: flex-end; }
.pill {
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--muted);
  border-radius: 999px;
  padding: 0.22rem 0.6rem;
  font-size: 11.5px;
  font-family: var(--mono);
  white-space: nowrap;
}
.pill strong { color: var(--text); font-weight: 600; }

.content { max-width: 1180px; margin: 0 auto; padding: 1.1rem 1.1rem 3rem; }

.section-label {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--faint);
  margin: 0.9rem 0 0.45rem;
}
.section-label:first-child { margin-top: 0; }

.session-card, .patch-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.95rem 1.05rem;
  margin-bottom: 1rem;
}

/* Caveats are the trust layer, not a footnote: a visually distinct panel,
   first thing under the header, never collapsed and never muted-gray. */
.caveats {
  background: var(--warn-dim);
  border: 1px solid var(--warn);
  border-radius: var(--radius);
  padding: 0.85rem 1.05rem;
  margin-bottom: 1.1rem;
}
.caveats-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--warn);
  margin-bottom: 0.4rem;
}
.caveats ul { margin: 0; padding-left: 1.2rem; font-size: 13px; }
.caveats li { margin-bottom: 0.4rem; }
.caveats li:last-child { margin-bottom: 0; }

/* Per-patch caveats: the same warning tone, scoped to one card. */
ul.patch-caveats {
  list-style: none;
  margin: 0 0 0.7rem;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
ul.patch-caveats li {
  background: var(--warn-dim);
  border: 1px solid var(--warn);
  border-radius: 6px;
  padding: 0.35rem 0.6rem;
  font-size: 12px;
  color: var(--text);
}
ul.patch-caveats li::before { content: "\26A0  "; color: var(--warn); }
.prompt-block {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--mono);
  font-size: 12.5px;
  line-height: 1.5;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.6rem 0.75rem;
  max-height: 220px;
  overflow: auto;
  margin: 0 0 0.6rem;
}
.kv { display: grid; grid-template-columns: 160px 1fr; gap: 0.3rem 0.5rem; font-size: 12.5px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-family: var(--mono); word-break: break-all; }

.patches { display: flex; flex-direction: column; gap: 1rem; }
.patch-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.patch-head-left { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.patch-target { font-family: var(--mono); font-size: 13px; color: var(--text); }
.patch-verdict { font-size: 13px; color: var(--text); font-style: italic; max-width: 46ch; text-align: right; }
.patch-meta { font-size: 12px; margin: 0.35rem 0; }
.rationale { font-size: 13px; margin: 0.2rem 0 0.6rem; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  border: 1px solid var(--border);
  background: var(--bg);
  border-radius: 6px;
  padding: 0.2rem 0.5rem;
  font-size: 11px;
  font-family: var(--mono);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.badge.kind { color: var(--kind-fg); border-color: var(--kind-border); text-transform: none; }
.badge.accept { color: var(--good); border-color: var(--good); background: var(--good-dim); }
.badge.reject { color: var(--bad); border-color: var(--bad); background: var(--bad-dim); }
/* Nothing was measured -- not a red REJECT ("we tested this and it lost"),
   a distinct neutral state ("we couldn't even build it to measure"). This
   check runs before the gate is even consulted (see _gate_badge). */
.badge.not-run { color: var(--not-run); border-color: var(--not-run); background: var(--not-run-dim); }
/* A high-severity regression overrides whatever accepted/provisional/
   directional say -- the gate alone can't see it (see _gate_badge) and
   this must never render as favorable-looking. */
.badge.regressed { color: var(--regressed); border-color: var(--regressed); background: var(--regressed-dim); }
/* Passed, but only via the n=5-9 zero-CI fallback, or accepted with zero
   decided pairs behind it -- still green-family (it DID pass the check it
   was given) but the dashed border marks it "qualified," not a clean
   accept, so it can't be mistaken for the same claim as a full ACCEPTED. */
.badge.accept-provisional { color: var(--good); border-color: var(--good); background: var(--good-dim); border-style: dashed; }
/* A patch that only has directional/provisional evidence at small k is not
   a failure — it needs one more cycle of repeats, not a red badge implying
   it's bad. Reused for both the n<5 DIRECTIONAL and n=5-9 PROVISIONAL·loss
   states, which share the same favorable/unfavorable/neutral coloring. */
.badge.directional-favorable { color: var(--favorable); border-color: var(--favorable); background: var(--favorable-dim); }
.badge.directional-unfavorable { color: var(--unfavorable); border-color: var(--unfavorable); background: var(--unfavorable-dim); }
.badge.directional-neutral { color: var(--neutral-reading); border-color: var(--neutral-reading); background: var(--neutral-reading-dim); }

.patch-body { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 1.1rem; align-items: start; }
@media (max-width: 860px) { .patch-body { grid-template-columns: 1fr; } }

.diff-wrap {
  margin: 0;
  max-height: 420px;
  overflow: auto;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0;
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.5;
}
.diff-line { display: block; padding: 0 0.75rem; white-space: pre; }
.diff-line.add { background: var(--add-bg); color: var(--add-fg); }
.diff-line.del { background: var(--del-bg); color: var(--del-fg); }
.diff-line.hunk { color: var(--hunk-fg); }
.diff-line.hdr { color: var(--muted); }
.diff-line.ctx { color: var(--muted); }

.table-wrap { overflow-x: auto; max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }
table.metrics { border-collapse: collapse; width: 100%; font-size: 12px; white-space: nowrap; }
table.metrics th, table.metrics td { padding: 0.4rem 0.6rem; text-align: right; border-bottom: 1px solid var(--border); }
table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
table.metrics thead th { color: var(--faint); font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; background: var(--bg-elev); }
table.metrics tbody tr:last-child td { border-bottom: none; }
table.metrics tr.good td:nth-child(4), table.metrics tr.good td:nth-child(5) { color: var(--good); }
table.metrics tr.bad td:nth-child(4), table.metrics tr.bad td:nth-child(5) { color: var(--bad); }
table.metrics tr.hl { background: var(--accent-dim); }
.tag { font-size: 9px; color: var(--accent); border: 1px solid var(--accent); border-radius: 4px; padding: 0 0.3rem; margin-left: 0.3rem; text-transform: uppercase; }

.wtl-bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
.wtl-seg { height: 100%; }
.wtl-seg.wtl-win { background: var(--good); }
.wtl-seg.wtl-tie { background: var(--faint); }
.wtl-seg.wtl-loss { background: var(--bad); }
.wtl-legend { font-size: 11.5px; margin-top: 0.4rem; display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap; }
.wtl-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-left: 0.5rem; }
.wtl-dot:first-child { margin-left: 0; }
.wtl-dot.wtl-win { background: var(--good); }
.wtl-dot.wtl-tie { background: var(--faint); }
.wtl-dot.wtl-loss { background: var(--bad); }

.stat-line { font-size: 12px; color: var(--muted); margin-top: 0.3rem; }
.stat-line.consistency-low { color: var(--warn); font-weight: 600; }

ul.reasons { margin: 0; padding-left: 1.1rem; font-size: 12.5px; }
ul.reasons li { margin-bottom: 0.2rem; }

ul.regressions { list-style: none; margin: 0 0 0.4rem; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
ul.regressions li { border: 1px solid var(--border); border-radius: 6px; padding: 0.35rem 0.55rem; font-size: 12px; }
.regression-sev { text-transform: uppercase; font-size: 10px; font-weight: 700; letter-spacing: 0.03em; margin-right: 0.35rem; padding: 0.05rem 0.35rem; border-radius: 4px; }
li.regression-high { border-color: var(--bad); background: var(--bad-dim); }
li.regression-high .regression-sev { color: var(--bad); background: var(--sev-chip-bg); }
li.regression-medium { border-color: var(--warn); background: var(--warn-dim); }
li.regression-medium .regression-sev { color: var(--warn); background: var(--sev-chip-bg); }
li.regression-low { border-color: var(--border-strong); }
li.regression-low .regression-sev { color: var(--muted); }

/* No paired data at all for a metric -- muted/italic, distinct from a
   normal (good/bad/neutral) row, so it can't be skimmed as "no change". */
tr.no-data td { color: var(--faint); font-style: italic; }
.no-data-cell { text-align: left !important; }
"""

# No click-to-collapse: an earlier version let clicking .patch-head hide
# .patch-body (diff, metrics table, gate reasons) while leaving the badge
# and verdict visible -- an accidental click could hide exactly the
# evidence a reader needs while the headline claim stayed on screen, with
# no chevron/aria affordance signaling it was even interactive. Removed
# rather than made discoverable: nothing in the brief requires it, and the
# risk (trust-critical evidence disappearing silently) outweighs the
# convenience. The page is intentionally static; no JS is required at all.
_SCRIPT = ""


# --- persistence ---------------------------------------------------------------

def write(report, out_dir):
    """Write report.json and index.html into out_dir. Returns out_dir as a
    Path. Both files are self-contained: index.html has no external
    http(s) resource references."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    return out_dir


# --- serving ---------------------------------------------------------------
#
# SECURITY: out_dir is NOT trusted to contain only report assets. cli.py's
# cycle output directory also holds scratch/<patch_id>/*-template/ homes
# materialized for replay, each carrying a copy of the user's real grok
# auth.json (harness.materialize's auth_from) so the treatment/control runs
# can authenticate. A SimpleHTTPRequestHandler rooted at out_dir would serve
# that file (and a directory listing to find it) to anything that can reach
# 127.0.0.1 on this machine. This module only ever writes report.json and
# index.html (see write()), so the server is a strict allow-list of exactly
# those two names -- never a directory root -- regardless of what else out_dir
# contains, now or if cli.py's layout changes later.

_SERVABLE = {"/": "index.html", "/index.html": "index.html", "/report.json": "report.json"}
_SERVABLE_CONTENT_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "report.json": "application/json; charset=utf-8",
}


class _ReportOnlyHandler(http.server.BaseHTTPRequestHandler):
    """Serves only report.json and index.html from out_dir. No directory
    listing, no path traversal (the path is looked up in a fixed dict, never
    joined onto the filesystem), no other file in out_dir is ever reachable."""

    def __init__(self, *args, out_dir, **kwargs):
        self._out_dir = out_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (stdlib handler method name)
        name = _SERVABLE.get(self.path.split("?", 1)[0])
        fp = self._out_dir / name if name else None
        if fp is None or not fp.is_file():
            self.send_error(404)
            return
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _SERVABLE_CONTENT_TYPES[name])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class _LoopbackTCPServer(socketserver.TCPServer):
    """allow_reuse_address as an instance-scoped subclass attribute, not a
    mutation of socketserver.TCPServer itself -- the base class is shared
    process-wide, so setting it there would silently change the behavior of
    every other TCPServer anyone else in this process creates."""

    allow_reuse_address = True


def make_server(out_dir, host="127.0.0.1", port=4242):
    """Build (but do not start) a loopback-only TCP server that serves only
    report.json and index.html from out_dir -- see the module note above.
    Split out from serve() so tests can bind an ephemeral port (port=0) and
    inspect server_address without blocking."""
    out_dir = Path(out_dir).resolve()
    if not (out_dir / "index.html").is_file():
        raise SystemExit(f"error: {out_dir} has no index.html (call write() first)")
    handler = functools.partial(_ReportOnlyHandler, out_dir=out_dir)
    return _LoopbackTCPServer((host, port), handler)


def serve(out_dir, port=4242):
    """Serve report.json + index.html from out_dir over http.server, bound
    to 127.0.0.1 only — never 0.0.0.0 — and never anything else in out_dir."""
    with make_server(out_dir, host="127.0.0.1", port=port) as httpd:
        host, bound_port = httpd.server_address
        print(f"serving {Path(out_dir).resolve()} at http://{host}:{bound_port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
