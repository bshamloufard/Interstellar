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

import difflib
import functools
import html
import http.server
import json
import re
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from interstellar.stats import EFFECT_TO_EFFICIENCY_METRIC
from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFICIENCY_METRICS,
    PATCH_SKILL_TRUNCATE,
    make_report,
)

# --- rerun (Task 2: the dashboard's "run" button) ----------------------------
#
# k=5 is the review loop's default sample size, but at k=5 a verdict can
# still wobble between PROVISIONAL and DIRECTIONAL (small_n's floor covers
# why -- see _small_n_caveat). k=10 is the actual remedy, and k=3 is the
# cheap sanity-check floor -- these three are the only values a reader is
# ever offered, both in the UI's <select> and as the server's own
# allow-list (see _RunManager.start / the /run handler): nothing else is
# ever accepted as `k`, from the request or anywhere else.
ALLOWED_RERUN_K = (3, 5, 10)

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
          k=0, cost_usd=0.0, generated_at=None, extra_caveats=None,
          rerun_args=None, trace_viz_url=None, trace_viz_session_id=None):
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

    `rerun_args` (optional): a flat dict of extra CLI flags (cli.py's own
    keyword names -- "max_patches", "model", "seed", etc., see
    _RERUN_ARG_SPEC) the dashboard's run button should reuse when it
    re-invokes `python3 -m interstellar review` for this session. Stored
    verbatim under the report's own "rerun" key (report.json is this
    module's artifact, not types.make_report's frozen contract, so this key
    is not part of that shape -- see cli.py's own precedent of stamping
    "cache_sensitive_metrics" onto the dict build() returns). `k` is
    deliberately NOT included here: it comes from the request, validated
    against ALLOWED_RERUN_K server-side, never trusted from stored config
    either -- see _RunManager.

    `trace_viz_url`/`trace_viz_session_id` (both optional): the URL of
    Alex's trace visualizer (sa.serve) for the SAME session, and the
    session_id it actually serves. Stored under the report's own
    "trace_viz" key (same precedent as "rerun" above) and rendered as a
    view-switcher link in the topbar -- see _view_switcher_html. The hard
    rule this module holds to: a missing/wrong link is worse than no link
    at all, so:
      - no `trace_viz_url` -> "trace_viz" is {} and nothing is rendered.
      - `trace_viz_url` with a `trace_viz_session_id` that matches this
        report's own session_id -> a normal, confident link.
      - `trace_viz_url` with a `trace_viz_session_id` that does NOT match
        -> still rendered (never silently swapped for nothing), but with
        a loud mismatch warning and the target session id spelled out in
        the label, so it is never mistaken for a same-session link.
      - `trace_viz_url` with no `trace_viz_session_id` at all (match
        can't be checked) -> rendered, visibly marked "unverified" rather
        than presented as confirmed.
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

    rpt = make_report(
        session=session,
        baseline_version=baseline_version,
        patch_results=patch_results,
        caveats=caveats,
        cost_usd=cost_usd,
        k=k,
        generated_at=generated_at or _now_iso(),
    )
    rpt["rerun"] = {
        "trace_file": (session or {}).get("trace_file") or "",
        "k_choices": list(ALLOWED_RERUN_K),
        "args": dict(rerun_args) if rerun_args else {},
    }
    if trace_viz_url:
        own_session_id = (session or {}).get("session_id") or ""
        target_session_id = trace_viz_session_id or ""
        rpt["trace_viz"] = {
            "url": trace_viz_url,
            "session_id": target_session_id,
            "verified_match": bool(target_session_id) and target_session_id == own_session_id,
        }
    else:
        rpt["trace_viz"] = {}
    return rpt


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


def _diff_html_flat(diff_text):
    """The original flat colored-<pre> renderer -- kept as the fallback for
    a diff string that isn't parseable unified-diff output at all (no hunk
    headers), so a malformed/raw diff still renders as *something* instead
    of an empty box. The normal path is _diff_html below."""
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
    return '<pre class="diff-flat"><code>' + "\n".join(lines) + "</code></pre>"


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_WORD_RE = re.compile(r"\s+|\S+")
# Below this SequenceMatcher.ratio() on a del/add pair, the two lines are
# not "the same line, modified" -- they're a coincidental sequence-length
# match (or a genuine pure delete + pure add sitting next to each other).
# Word-highlighting a pair that dissimilar would mark almost the whole line
# as "changed", which is no more informative than the plain add/del rows
# and actively misleading (it implies a closer relationship than exists).
_CLOSE_MATCH_RATIO = 0.5


def _strip_ab_prefix(path):
    """harness._unified_diff() passes the same bare label ("skills/x/
    SKILL.md", no a/ b/ prefix) as both fromfile and tofile, but a
    hand-written or externally sourced diff may use git's a/ b/ convention
    -- strip it if present so the file path in the summary header reads the
    same either way, never "a/skills/x/SKILL.md"."""
    path = path.strip()
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


def _parse_unified_diff(diff_text):
    """Parse unified-diff text (difflib.unified_diff's own output shape,
    lineterm="") into (file_path, hunks). Each hunk is
    {old_start, old_count, new_start, new_count, lines}, lines a list of
    (tag, text) with tag in "ctx"/"add"/"del"/"note" ("note" is difflib's
    "\\ No newline at end of file" marker -- kept visible, never counted
    toward the add/del totals). The "---"/"+++" file-header lines are
    consumed for the path and dropped from `hunks` entirely -- the diff's
    summary header renders the path once, not as a diff row. A line before
    any hunk header, or one that doesn't start with the expected +/-/space
    prefix, is tolerated rather than raising: this renders real, possibly
    slightly irregular diffs, it does not validate them."""
    file_path = ""
    hunks = []
    cur = None
    for raw in diff_text.splitlines():
        if raw.startswith("--- "):
            file_path = file_path or _strip_ab_prefix(raw[4:])
            continue
        if raw.startswith("+++ "):
            stripped = _strip_ab_prefix(raw[4:])
            if stripped and stripped != "/dev/null":
                file_path = stripped
            continue
        m = _HUNK_RE.match(raw)
        if m:
            old_start, old_count, new_start, new_count = m.groups()
            cur = {
                "old_start": int(old_start), "old_count": int(old_count or 1),
                "new_start": int(new_start), "new_count": int(new_count or 1),
                "lines": [],
            }
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if raw.startswith("\\"):
            cur["lines"].append(("note", raw))
        elif raw.startswith("+"):
            cur["lines"].append(("add", raw[1:]))
        elif raw.startswith("-"):
            cur["lines"].append(("del", raw[1:]))
        elif raw.startswith(" "):
            cur["lines"].append(("ctx", raw[1:]))
        elif raw == "":
            cur["lines"].append(("ctx", ""))
        else:
            cur["lines"].append(("ctx", raw))
    return file_path, hunks


def _tokenize_words(s):
    return _WORD_RE.findall(s)


def _is_close_match(a, b):
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= _CLOSE_MATCH_RATIO


def _word_diff_spans(old_line, new_line):
    """Word-level highlighting for a del/add pair that _is_close_match: run
    difflib.SequenceMatcher over word tokens (regex, not str.split, so
    whitespace is its own token and reassembling the opcodes reproduces the
    original text exactly) and wrap the non-equal spans in <mark>. Only
    called for a pair that already passed the closeness check -- this does
    not itself decide whether highlighting is warranted."""
    a, b = _tokenize_words(old_line), _tokenize_words(new_line)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    old_parts, new_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old_seg = html.escape("".join(a[i1:i2]))
        new_seg = html.escape("".join(b[j1:j2]))
        if tag == "equal":
            old_parts.append(old_seg)
            new_parts.append(new_seg)
        else:
            if old_seg:
                old_parts.append(f'<mark class="chg-del">{old_seg}</mark>')
            if new_seg:
                new_parts.append(f'<mark class="chg-add">{new_seg}</mark>')
    return "".join(old_parts), "".join(new_parts)


def _diff_row(cls, old_no, new_no, code_html):
    return (
        f'<tr class="diff-row {cls}">'
        f'<td class="ln old">{old_no}</td><td class="ln new">{new_no}</td>'
        f'<td class="code">{code_html}</td></tr>'
    )


def _hunk_rows_html(h):
    """Render one hunk's lines with old/new gutters. A contiguous del-run
    immediately followed by an add-run -- the exact shape difflib.unified_
    diff produces for a changed block -- is paired index-wise (del[i] with
    add[i]) so each pair can be word-highlighted when close (see
    _is_close_match); leftover dels or adds beyond the shorter run's length
    render plain, as does a del-run or add-run with nothing paired at all
    (a pure deletion or pure insertion)."""
    lines = h["lines"]
    old_no, new_no = h["old_start"], h["new_start"]
    rows = []
    i, n = 0, len(lines)
    while i < n:
        tag, text = lines[i]
        if tag == "ctx":
            rows.append(_diff_row("ctx", old_no, new_no, html.escape(text) or "&nbsp;"))
            old_no += 1
            new_no += 1
            i += 1
        elif tag == "note":
            rows.append(
                f'<tr class="diff-row note"><td class="ln" colspan="2"></td>'
                f'<td class="code">{_esc(text)}</td></tr>'
            )
            i += 1
        elif tag == "del":
            dels = []
            while i < n and lines[i][0] == "del":
                dels.append(lines[i][1])
                i += 1
            adds = []
            while i < n and lines[i][0] == "add":
                adds.append(lines[i][1])
                i += 1
            paired = min(len(dels), len(adds))
            for k in range(paired):
                old_line, new_line = dels[k], adds[k]
                if _is_close_match(old_line, new_line):
                    old_html, new_html = _word_diff_spans(old_line, new_line)
                else:
                    old_html, new_html = html.escape(old_line), html.escape(new_line)
                rows.append(_diff_row("del", old_no, "", old_html or "&nbsp;"))
                old_no += 1
                rows.append(_diff_row("add", "", new_no, new_html or "&nbsp;"))
                new_no += 1
            for old_line in dels[paired:]:
                rows.append(_diff_row("del", old_no, "", html.escape(old_line) or "&nbsp;"))
                old_no += 1
            for new_line in adds[paired:]:
                rows.append(_diff_row("add", "", new_no, html.escape(new_line) or "&nbsp;"))
                new_no += 1
        elif tag == "add":
            rows.append(_diff_row("add", "", new_no, html.escape(text) or "&nbsp;"))
            new_no += 1
            i += 1
        else:
            i += 1
    return rows


def _truncate_fraction_text(patch, application, hunks):
    """For a skill.truncate patch (the common case), state what fraction of
    the file is being removed. Numerator: application["lines_changed"] --
    the same figure the patch card's meta line already reports elsewhere --
    falling back to summing patch["line_ranges"] directly if that field is
    absent. Denominator: the highest old-file line number any hunk's
    context reaches (max old_start+old_count-1 across hunks) -- the most a
    figure derived only from patch/application/the diff text can honestly
    claim as "how long the file is": difflib's default context window (3
    lines) means this equals the file's true length only when the last
    hunk's trailing context reaches EOF, which is the common case for a
    truncate (the cut sections are usually not the file's last 3 lines) but
    not a guarantee -- so this is the file's line-span as revealed by the
    diff, not a value read from a stored total. Returns "" when there is
    nothing to report (not a truncate, or no hunks)."""
    if not patch or patch.get("kind") != PATCH_SKILL_TRUNCATE or not hunks:
        return ""
    removed = (application or {}).get("lines_changed")
    if not removed:
        ranges = patch.get("line_ranges") or []
        removed = sum(int(e) - int(s) + 1 for s, e in ranges) if ranges else None
    if not removed:
        return ""
    extent = max(h["old_start"] + h["old_count"] - 1 for h in hunks)
    return f"removes {removed} of {extent} lines"


def _diff_html(diff_text, *, patch=None, application=None):
    if not diff_text or not diff_text.strip():
        return '<p class="muted">No diff.</p>'
    file_path, hunks = _parse_unified_diff(diff_text)
    if not hunks:
        return _diff_html_flat(diff_text)

    added = sum(1 for h in hunks for tag, _ in h["lines"] if tag == "add")
    removed = sum(1 for h in hunks for tag, _ in h["lines"] if tag == "del")
    kind = (patch or {}).get("kind") or ""

    summary_bits = [
        f'<span class="diff-file">{_esc(file_path or "(no file path in diff)")}</span>',
        '<span class="diff-counts">'
        f'<span class="diff-add-count">+{added}</span> '
        f'<span class="diff-del-count">&minus;{removed}</span></span>',
    ]
    if kind:
        summary_bits.append(f'<span class="diff-kind">{_esc(kind)}</span>')
    fraction = _truncate_fraction_text(patch, application, hunks)
    if fraction:
        summary_bits.append(f'<span class="diff-fraction">{_esc(fraction)}</span>')
    summary_html = f'<div class="diff-summary">{"".join(summary_bits)}</div>'

    rows = []
    for h in hunks:
        rows.append(
            '<tr class="diff-row hunk"><td class="ln" colspan="2"></td>'
            f'<td class="code">@@ -{h["old_start"]},{h["old_count"]} '
            f'+{h["new_start"]},{h["new_count"]} @@</td></tr>'
        )
        rows.extend(_hunk_rows_html(h))
    table_html = (
        '<div class="diff-wrap"><table class="diff-table"><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )
    return summary_html + table_html


CACHE_SENSITIVE_TITLE = (
    "This figure swings with prompt-cache warmth (run history and timing), "
    "not only with this patch — read it as indicative, not as a precise "
    "measurement of what the patch did."
)


def _metric_row(name, lower_is_better, m, highlight=False):
    if not m:
        return ""
    label = METRIC_LABELS.get(name, name)
    tag = ' <span class="tag">target</span>' if highlight else ""
    # Absent on older reports (statistics predating the cache_sensitive
    # flag) -- default to not-flagged rather than crash or over-warn.
    cache_tag = (
        f' <span class="tag cache-tag" title="{_esc(CACHE_SENSITIVE_TITLE)}">cache&#8224;</span>'
        if m.get("cache_sensitive") else ""
    )
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
            f"<td>{_esc(label)}{tag}{cache_tag}</td>"
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
        f"<td>{_esc(label)}{tag}{cache_tag}</td>"
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
    # Only shown when at least one row actually carries the flag (absent on
    # reports predating it, or when the CLI genuinely found nothing
    # cache-sensitive in this patch's metric set) -- the marker's tooltip
    # explains it too, but a hover-only explanation is not legible on a
    # static, printable page, so it's spelled out here as well.
    any_cache_sensitive = any(
        (efficiency.get(name) or {}).get("cache_sensitive") for name, _ in EFFICIENCY_METRICS
    )
    legend = (
        f'<p class="cache-legend muted">cache&#8224; {_esc(CACHE_SENSITIVE_TITLE)}</p>'
        if any_cache_sensitive else ""
    )
    return (
        '<div class="table-wrap"><table class="metrics">'
        "<thead><tr><th>Metric</th><th>Control (median)</th>"
        "<th>Treatment (median)</th><th>&Delta;</th><th>&Delta;%</th>"
        "<th>95% CI on &Delta;</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>{legend}"
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


def run_panel_html(rerun, *, current_k, first_run=False):
    """Task 2: the button that runs (or re-runs) a review cycle for this
    trace. `rerun` is the report["rerun"] dict build() stamped on
    (trace_file, k_choices, args) -- see build()'s docstring -- OR, for
    `first_run=True` (interstellar/combined_serve.py's Review tab when no
    report exists yet at all), an equivalent dict assembled from whatever
    trace_file the combined server was launched with (see _RunManager.
    configure_first_run). With no trace_file at all, there is nothing for
    the server to run: render an explanation, not a dead button."""
    trace_file = rerun.get("trace_file") or ""
    if not trace_file:
        verb = "Run" if first_run else "Re-run"
        return (
            '<div class="run-panel run-panel-disabled">'
            f'<span class="muted">{verb} unavailable: no trace file recorded '
            "for this report.</span></div>"
        )
    choices = rerun.get("k_choices") or list(ALLOWED_RERUN_K)
    default_k = current_k if current_k in choices else (5 if 5 in choices else choices[0])
    options = "".join(
        f'<option value="{c}"{" selected" if c == default_k else ""}>{c}</option>'
        for c in choices
    )
    current_note = (
        f'<span class="run-current muted">current report: k={_esc(current_k)}</span>'
        if current_k and not first_run else ""
    )
    btn_label = "Run review cycle" if first_run else "Re-run review cycle"
    return f"""<div class="run-panel" id="run-panel" data-trace="{_esc(trace_file)}">
      {current_note}
      <label class="run-k-label">k <select id="run-k">{options}</select></label>
      <button id="run-btn" type="button">{btn_label}</button>
      <span id="run-status" class="run-status idle">idle</span>
    </div>"""


def _short_session_id(session_id):
    """The first hyphen-segment of a session id (e.g. "019fe4e7" from
    "019fe4e7-68a7-...") -- long enough to be recognizable, short enough
    for a topbar label. Matches how these ids get abbreviated in practice
    (session_id is a hyphenated UUIDv7)."""
    return (session_id or "").split("-")[0] or session_id or ""


def _view_switcher_html(trace_viz, own_session_id):
    """The Review/Trace view switcher: a link to Alex's trace visualizer
    (sa.serve) for this same session. Renders nothing when no URL is
    configured -- see build()'s docstring for why a missing link is the
    safe default and a wrong one is not. `trace_viz` is report["trace_viz"]
    exactly as build() stamped it."""
    url = (trace_viz or {}).get("url")
    if not url:
        return ""
    target_id = (trace_viz or {}).get("session_id") or ""
    verified = bool((trace_viz or {}).get("verified_match"))
    if not target_id:
        cls = "view-link view-link-unverified"
        label = "Trace"
        title = (
            f"Opens {url} — target session could not be verified "
            "against this report"
        )
    elif verified:
        cls = "view-link"
        label = "Trace"
        title = f"Opens {url} — verified same session ({_esc(own_session_id)})"
    else:
        cls = "view-link view-link-mismatch"
        label = f"Trace &middot; {_esc(_short_session_id(target_id))}"
        title = (
            f"WARNING: this trace is for a DIFFERENT session ({target_id}) "
            f"than this report ({own_session_id})"
        )
    return (
        '<div class="view-switcher" role="group" aria-label="View">'
        '<span class="view-seg view-seg-active">Review</span>'
        f'<a class="{cls}" href="{_esc(url)}" target="_blank" rel="noopener" '
        f'title="{_esc(title)}">{label}</a>'
        "</div>"
    )


def _gate_badge(gate, *, applied=True, high_severity_regressions=0, win_rate=None):
    """The badge is the one thing a scanning reader actually reads, so it
    must never be able to say something that isn't true. Four checks run
    BEFORE the ordinary gate table is even consulted, in this order,
    because each one contradicts what that table alone would otherwise
    imply:

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
       score PROVISIONAL/DIRECTIONAL · FAVORABLE. stats.py separately
       forces its own override for this, but this module does not rely on
       that alone: a high-severity regression forces REGRESSED here
       regardless of what accepted/provisional/directional say.
    3. `directional == "unknown"` -- stats.gate()'s zero-usable-pairs
       short-circuit (every run in the cycle failed on at least one arm):
       accepted/provisional are already False there, but a bare REJECTED
       would still read as "we measured this and it lost" exactly like the
       not-applied case above -- the truth is "nothing could be measured,"
       distinct from both "never applied" (NOT RUN) and "measured and came
       out even" (`directional == "neutral"`, a real reading). NO DATA,
       same neutral gray family as NOT RUN, not a reject color.
    4. Zero decided pairs backing an ACCEPTED -- `stats.gate()`'s n>=10
       quality check passes on `losses == 0`, which an all-ties result
       (0 wins, 0 losses) satisfies with no quality evidence at all. A
       green ACCEPTED there is technically the gate's own call but is
       exactly the "plausible, not true" failure this dashboard cannot
       have -- downgrade it to a qualified badge, matching the visual
       weight of ACCEPTED · PROVISIONAL, so it cannot be mistaken for the
       same claim as a real win.

    Only past all four does the five-state truth table apply -- see the
    inline states below. Falls back to a legacy REJECT only for a gate
    dict that predates the provisional/directional fields entirely.

    A `favorable`/`unfavorable` reading (PROVISIONAL or DIRECTIONAL state)
    also gets `" (efficiency only, no decided quality pairs)"` appended
    when `gate["quality_evidence"]` is present and not `"decided"` -- the
    reading can come purely from the efficiency point estimate with zero
    corroborating win/loss evidence, and that provenance belongs on the
    badge itself, not only in the reasons list. Absent field (older gate
    dicts) renders no suffix.
    """
    if not applied:
        return '<span class="badge not-run">NOT RUN</span>'
    if high_severity_regressions:
        n = high_severity_regressions
        return f'<span class="badge regressed">REGRESSED &middot; {n} HIGH-SEVERITY</span>'
    if gate.get("directional") == "unknown":
        return '<span class="badge no-data">NO DATA</span>'

    accepted = bool(gate.get("accepted"))
    provisional = bool(gate.get("provisional"))
    has_directional = "directional" in gate
    reading = (gate.get("directional") or "neutral").lower()
    reading_cls = {"favorable": "directional-favorable", "unfavorable": "directional-unfavorable"}.get(
        reading, "directional-neutral"
    )
    # stats.gate()'s quality_evidence ("none" | "ties_only" | "decided"):
    # a favorable/unfavorable reading can come purely from the efficiency
    # point estimate with zero corroborating win/loss evidence (e.g. one
    # judged pair, a tie) -- real evidence, not to be thrown away, but its
    # provenance must be visible right on the badge, not just in the
    # reasons list a reader may not scroll to. Absent (older gate dicts)
    # renders no suffix at all -- not every gate has this field yet.
    quality_evidence = gate.get("quality_evidence")
    reading_suffix = (
        " (efficiency only, no decided quality pairs)"
        if reading in ("favorable", "unfavorable") and quality_evidence not in (None, "decided")
        else ""
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
        return f'<span class="badge {reading_cls}">PROVISIONAL &middot; {_esc(reading.upper())}{_esc(reading_suffix)}</span>'
    if has_directional:
        reasons = gate.get("reasons") or []
        too_few_for_any_call = any("insufficient_samples_for_acceptance" in r for r in reasons)
        if too_few_for_any_call:
            return f'<span class="badge {reading_cls}">DIRECTIONAL &middot; {_esc(reading.upper())}{_esc(reading_suffix)}</span>'
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

    diff_block = _diff_html(diff_text, patch=patch, application=application)
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
    # statistics["n"] means usable pairs (both arms succeeded), not total
    # repeats attempted. A partial failure must stay visible as "1 of 3
    # usable" rather than silently reporting n=1 with no context for where
    # the other 2 went -- see statistics["data"] (pairs_total/pairs_usable/
    # control_failures/treatment_failures). Falls back to the bare n when
    # a statistics dict predates the "data" block.
    data = stats.get("data") or {}
    pairs_total, pairs_usable = data.get("pairs_total"), data.get("pairs_usable")
    if pairs_total is not None and pairs_usable is not None and pairs_total != pairs_usable:
        meta_bits.append(
            f"{_esc(pairs_usable)} of {_esc(pairs_total)} paired repeats usable "
            f"(control failed {_esc(data.get('control_failures', 0))}, "
            f"treatment failed {_esc(data.get('treatment_failures', 0))})"
        )
    elif stats.get("n") is not None:
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
        <div class="diff-col-full">
          <div class="section-label">Diff</div>
          {diff_block}
        </div>
        <div class="stats-row">
          <div class="stats-col">
            <div class="section-label">Before / after (k paired repeats)</div>
            {table_html}
            {regressions_html}
          </div>
          <div class="stats-col">
            <div class="section-label">Judge outcome (treatment vs. control)</div>
            {wtl_html}
            {extra_html}
            <div class="section-label">Gate</div>
            <ul class="reasons">{reasons_html}</ul>
          </div>
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
        {_view_switcher_html(report.get("trace_viz") or {}, session.get("session_id") or "")}
        {run_panel_html(report.get("rerun") or {}, current_k=k)}
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

/* View switcher: a link to Alex's trace visualizer (sa.serve) for this
   same session -- two segments, "Review" (this page, always the active/
   filled segment since you're on it) and "Trace" (the link). Renders only
   when report["trace_viz"]["url"] is set (see _view_switcher_html) -- no
   link is safer than a wrong one, so there is no placeholder/disabled
   variant here the way the run panel has one. */
.view-switcher {
  display: inline-flex;
  align-items: stretch;
  border: 1px solid var(--border);
  border-radius: 999px;
  overflow: hidden;
  font-size: 11px;
  font-family: var(--mono);
}
.view-seg, .view-link { padding: 0.2rem 0.55rem; white-space: nowrap; }
.view-seg-active { color: var(--text); background: var(--accent-dim); font-weight: 600; }
.view-link {
  color: var(--muted);
  background: var(--bg);
  border-left: 1px solid var(--border);
  text-decoration: none;
}
.view-link:hover { color: var(--accent); }
/* Session id could not be checked against this report's own id -- shown,
   never hidden, but visibly not vouched for. */
.view-link-unverified { color: var(--warn); border-left-color: var(--warn); background: var(--warn-dim); }
.view-link-unverified:hover { color: var(--warn); filter: brightness(1.15); }
/* Confirmed a DIFFERENT session -- the exact failure mode the brief calls
   out as worse than no link at all if left silent. Loud on purpose: red,
   and the label itself carries the mismatched session id. */
.view-link-mismatch { color: var(--bad); border-left-color: var(--bad); background: var(--bad-dim); font-weight: 600; }
.view-link-mismatch:hover { color: var(--bad); filter: brightness(1.15); }

/* Task 2: the trigger button, in the topbar since it acts on the whole
   session (re-runs the whole cycle), not any one patch. */
.run-panel {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  border-left: 1px solid var(--border);
  padding-left: 0.7rem;
  margin-left: 0.2rem;
}
.run-panel-disabled { border-left: 1px solid var(--border); padding-left: 0.7rem; }
.run-current { font-size: 11px; font-family: var(--mono); }
.run-k-label {
  font-size: 11.5px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
.run-k-label select {
  font-family: var(--mono);
  font-size: 11.5px;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.15rem 0.3rem;
}
#run-btn {
  font-family: var(--font);
  font-size: 11.5px;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-dim);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 0.25rem 0.7rem;
  cursor: pointer;
}
#run-btn:hover:not(:disabled) { filter: brightness(1.08); }
#run-btn:disabled { cursor: not-allowed; opacity: 0.55; }
.run-status {
  font-size: 11px;
  font-family: var(--mono);
  color: var(--muted);
  max-width: 32ch;
}
.run-status.running { color: var(--provisional); }
.run-status.done { color: var(--good); }
.run-status.failed { color: var(--bad); font-weight: 600; }

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
/* stats.gate()'s zero-usable-pairs short-circuit (directional=="unknown"):
   every run failed on at least one arm, so nothing could be measured --
   the same neutral family as NOT RUN (not a reject color), but a distinct
   state: the patch DID apply, it just never produced usable data. */
.badge.no-data { color: var(--not-run); border-color: var(--not-run); background: var(--not-run-dim); }
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

/* Task 1: the diff is the thing the user is deciding on, so it gets the
   card's full width; the metrics table and gate/judge readout sit below
   it, side by side on anything wide enough. */
.patch-body { display: flex; flex-direction: column; gap: 1rem; }
.stats-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.1rem; align-items: start; }
@media (max-width: 780px) { .stats-row { grid-template-columns: 1fr; } }

/* Fallback for a diff string that has no parseable hunk headers (see
   _diff_html_flat) -- the old flat colored-line view. */
.diff-flat {
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

/* The real (parsed) diff view: a summary header (path, +/-, kind, and for
   a skill.truncate the removed-fraction line) above a line-numbered,
   gutter'd table -- see _diff_html / _hunk_rows_html. */
.diff-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.6rem;
  font-size: 12px;
  margin-bottom: 0.4rem;
}
.diff-file { font-family: var(--mono); color: var(--text); font-weight: 600; word-break: break-all; }
.diff-counts { font-family: var(--mono); }
.diff-add-count { color: var(--good); }
.diff-del-count { color: var(--bad); }
.diff-kind { color: var(--muted); font-family: var(--mono); text-transform: none; }
.diff-fraction { color: var(--warn); }

.diff-wrap {
  max-height: 460px;
  overflow: auto;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
}
table.diff-table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.5;
}
table.diff-table td.ln {
  width: 1%;
  min-width: 34px;
  padding: 0 0.5rem;
  text-align: right;
  color: var(--faint);
  user-select: none;
  border-right: 1px solid var(--border);
  white-space: nowrap;
}
table.diff-table td.ln.new { border-right-color: var(--border-strong); }
table.diff-table td.code { padding: 0 0.75rem; white-space: pre; width: 100%; }
tr.diff-row.add { background: var(--add-bg); color: var(--add-fg); }
tr.diff-row.add td.ln { color: var(--add-fg); opacity: 0.7; }
tr.diff-row.del { background: var(--del-bg); color: var(--del-fg); }
tr.diff-row.del td.ln { color: var(--del-fg); opacity: 0.7; }
tr.diff-row.hunk td.code { color: var(--hunk-fg); padding-top: 0.3rem; padding-bottom: 0.3rem; }
tr.diff-row.hunk { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
tr.diff-row.ctx td.code { color: var(--muted); }
tr.diff-row.note td.code { color: var(--faint); font-style: italic; }
/* Word-level highlighting within a close-matching del/add pair (see
   _word_diff_spans) -- a stronger mark than the whole row's tint so the
   reader's eye lands on exactly what changed, not just which line. */
mark.chg-del { background: var(--bad); color: var(--bg-elev); border-radius: 2px; padding: 0 1px; }
mark.chg-add { background: var(--good); color: var(--bg-elev); border-radius: 2px; padding: 0 1px; }

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
/* Flags a metric that swings with prompt-cache warmth (run history/timing)
   rather than only with the patch -- amber, distinct from the accent-blue
   "target" tag, so the two never look like the same kind of claim. */
.tag.cache-tag { color: var(--warn); border-color: var(--warn); cursor: help; }
.cache-legend { font-size: 11px; margin: 0.35rem 0 0; }

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
# convenience.
#
# The one piece of real interactivity this page has is Task 2's run panel:
# POST /run to start a cycle, poll GET /status to reflect it live, reload
# on completion so the reloaded page shows the new recommendations/diffs/
# gate verdicts -- never left stale on "running" or on the pre-run cards.
# Both endpoints are only served over the loopback-only server (see
# make_server) -- opening index.html directly (file://, or a copy someone
# emails around) just shows "failed to start" on click, never a broken
# page. No external script/resource is ever referenced (the self-
# contained-page test asserts this), and nothing here reads or writes
# anything outside these two endpoints.
#
# PENDING_KEY (sessionStorage, this tab only) marks "a run was started
# from THIS page load and hasn't been reloaded-for yet". Without it, the
# server's /status keeps reporting the last run's outcome forever (it has
# no concept of "already shown to the user") -- so the very reload this
# code triggers on completion would load a fresh page whose first poll()
# sees that same "done" status and reloads again, forever. PENDING_KEY is
# cleared right before the reload, so the reloaded page's poll() sees
# "done" with no pending flag and renders it as a settled, final state
# instead of triggering another reload. It's also NOT set on ordinary page
# load, so simply opening/refreshing a report that finished in a previous
# session shows "up to date" immediately, with no surprise reload.
_SCRIPT = """
(function () {
  var panel = document.getElementById("run-panel");
  var btn = document.getElementById("run-btn");
  var statusEl = document.getElementById("run-status");
  var kSelect = document.getElementById("run-k");
  if (!panel || !btn || !statusEl || !kSelect) return;

  var PENDING_KEY = "interstellar-run-pending";
  var pollTimer = null;

  function fmtElapsed(s) {
    if (s === null || s === undefined) return "";
    var m = Math.floor(s / 60), r = Math.floor(s % 60);
    return m > 0 ? (m + "m " + r + "s") : (r + "s");
  }

  function setStatus(cls, text) {
    statusEl.className = "run-status " + cls;
    statusEl.textContent = text;
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function isPending() {
    try { return sessionStorage.getItem(PENDING_KEY) === "1"; } catch (e) { return false; }
  }
  function setPending(v) {
    try {
      if (v) sessionStorage.setItem(PENDING_KEY, "1");
      else sessionStorage.removeItem(PENDING_KEY);
    } catch (e) { /* sessionStorage unavailable -- degrade to no auto-reload loop guard */ }
  }

  function poll() {
    fetch("/status", {cache: "no-store"}).then(function (r) {
      return r.json();
    }).then(function (s) {
      if (s.status === "running") {
        btn.disabled = true;
        setStatus("running", "running k=" + s.k + " \\u2014 " + fmtElapsed(s.elapsed_s) +
          " elapsed (a cycle can take several minutes and spends real API cost)");
      } else if (s.status === "done") {
        stopPolling();
        btn.disabled = false;
        if (isPending()) {
          // This is OUR run finishing -- reload once to pick up the new
          // recommendations/diffs/gate verdicts, then clear the flag so
          // the reloaded page settles instead of reloading again.
          setPending(false);
          setStatus("done", "done in " + fmtElapsed(s.elapsed_s) +
            " \\u2014 loading new recommendations\\u2026");
          setTimeout(function () { window.location.reload(); }, 600);
        } else {
          // A finished run from before this page load (or the reload we
          // already did) -- the page already reflects it; say so plainly
          // rather than reloading again.
          setStatus("done", "up to date \\u2014 last cycle finished in " + fmtElapsed(s.elapsed_s));
        }
      } else if (s.status === "failed") {
        stopPolling();
        setPending(false);
        btn.disabled = false;
        setStatus("failed", "failed: " + (s.error || ("exit code " + s.returncode)));
      } else {
        btn.disabled = false;
        setStatus("idle", "idle");
      }
    }).catch(function () {
      // A dropped poll is not evidence the run stopped -- try again next
      // tick rather than collapsing a running state back to idle.
    });
  }

  btn.addEventListener("click", function () {
    setPending(true);
    btn.disabled = true;
    setStatus("running", "starting\\u2026");
    fetch("/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({k: parseInt(kSelect.value, 10)}),
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (e) {
          throw new Error(e.error || ("HTTP " + r.status));
        });
      }
      return r.json();
    }).then(function () {
      if (!pollTimer) pollTimer = setInterval(poll, 2000);
      poll();
    }).catch(function (err) {
      setPending(false);
      btn.disabled = false;
      setStatus("failed", "failed to start: " + err.message);
    });
  });

  // Pick up a cycle already running from a previous click (e.g. the user
  // reloaded, or opened this report in a second tab) instead of only
  // reacting to a click made in this exact page load.
  pollTimer = setInterval(poll, 2000);
  poll();
})();
"""


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

# --- Task 2: the /run + /status endpoints ------------------------------------
#
# SECURITY: /run is a loopback-only endpoint that spawns a subprocess. The
# ONE thing this module will never do is build that subprocess's argv from
# request-supplied content. `k` is the only value the request contributes,
# and it is checked against ALLOWED_RERUN_K (a fixed tuple of ints, not a
# pattern or a range) before it touches anything -- see _RunManager.start.
# Every other argument -- the trace file, --out, and any extra flags --
# comes from `report["rerun"]`, itself only ever populated by build() at
# report-generation time (see build()'s docstring), never by a request.

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Extra CLI flags a stored rerun config MAY carry (report["rerun"]["args"],
# set via build(rerun_args=...)), mapped to the interstellar-review argv
# flag and a caster. Only these names are ever turned into argv -- an
# unrecognized key in the stored config is dropped, not passed through.
# This is defense in depth, not a defense against the request (the request
# never reaches this dict at all -- see _build_rerun_argv) -- it keeps the
# subprocess's argv fully enumerable from this one table even if
# report["rerun"]["args"] is later populated by a caller this module
# doesn't control.
_RERUN_ARG_SPEC = {
    "max_patches": ("--max-patches", int),
    "budget_usd": ("--budget-usd", float),
    "use_cached_analysis": ("--use-cached-analysis", bool),
    "model": ("--model", str),
    "max_parallel": ("--max-parallel", int),
    "timeout": ("--timeout", int),
    "seed": ("--seed", int),
    "max_token_regression": ("--max-token-regression", float),
    "grok_home": ("--grok-home", str),
    "no_preflight": ("--no-preflight", bool),
}


def _build_rerun_argv(*, trace_file, out_dir, k, args):
    """The exact argv for `python3 -m interstellar review ...`. Caller must
    have already validated `k` against ALLOWED_RERUN_K -- this function
    trusts it as given. `args` is only ever report["rerun"]["args"] (see
    module note above); a key not in _RERUN_ARG_SPEC is dropped."""
    argv = [
        sys.executable, "-m", "interstellar", "review", str(trace_file),
        "--out", str(out_dir), "--k", str(k),
    ]
    for name, value in (args or {}).items():
        spec = _RERUN_ARG_SPEC.get(name)
        if spec is None or value is None:
            continue
        flag, caster = spec
        if caster is bool:
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(caster(value)))
    return argv


def validate_run_k(body):
    """Parse+validate a /run POST body. Returns (k, error_message): k is
    None iff error_message is set. This is the ONLY thing ever read from
    the request body -- see _RunManager.start for what deliberately does
    NOT come from here. Shared by _ReportOnlyHandler.do_POST and
    interstellar/combined_serve.py's handler so both enforce the exact
    same rule from one place."""
    k = body.get("k") if isinstance(body, dict) else None
    # bool is an int subclass in Python -- explicitly excluded so
    # {"k": true} can't sneak past `k in ALLOWED_RERUN_K` on some future
    # allow-list that happened to include 1.
    if not isinstance(k, int) or isinstance(k, bool) or k not in ALLOWED_RERUN_K:
        return None, f"k must be one of {list(ALLOWED_RERUN_K)}"
    return k, None


class _RunManager:
    """Tracks at most one review-cycle subprocess for this server's out_dir.
    A cycle takes minutes and spends real API/model cost, so a second /run
    while one is already in flight is refused (409), never queued and never
    used to kill the first one.

    `argv_builder` defaults to _build_rerun_argv (the real `python3 -m
    interstellar review ...` command) and exists as an injection point so
    tests can exercise this class's threading/status/HTTP behavior against
    a real, fast, free subprocess instead of the real (slow, costly) review
    cycle -- the same dependency-injection pattern cli.py's own
    run_review() uses for grok-dev calls.
    """

    def __init__(self, out_dir, rerun_config, argv_builder=None):
        self._out_dir = out_dir
        self._rerun = rerun_config or {}
        self._argv_builder = argv_builder or _build_rerun_argv
        self._lock = threading.Lock()
        self._status = "idle"
        self._k = None
        self._started_at = None
        self._finished_at = None
        self._returncode = None
        self._error = None

    def snapshot(self):
        with self._lock:
            elapsed = None
            if self._started_at is not None:
                end = self._finished_at if self._finished_at is not None else time.time()
                elapsed = round(end - self._started_at, 1)
            return {
                "status": self._status,
                "k": self._k,
                "elapsed_s": elapsed,
                "returncode": self._returncode,
                "error": self._error,
                "trace_file": self._rerun.get("trace_file") or "",
            }

    def configure_first_run(self, trace_file, args=None):
        """Used when out_dir has no report.json yet -- the very first cycle
        for this directory, which make_run_manager() has nothing to load a
        "rerun" config from. Stamps a trace_file/args directly so /run can
        still work; a real cycle finishing afterward writes report.json
        with its own "rerun" key, and the NEXT make_run_manager() call (a
        fresh server start, or interstellar/combined_serve.py's per-
        request _Sides re-check) picks that up normally instead. Never
        overwrites a trace_file this manager was already given -- only
        fills in a genuinely empty one."""
        with self._lock:
            if not self._rerun.get("trace_file"):
                self._rerun = {
                    "trace_file": str(trace_file),
                    "k_choices": list(ALLOWED_RERUN_K),
                    "args": dict(args) if args else {},
                }

    def rerun_config(self):
        """A copy of the rerun config (trace_file/k_choices/args) currently
        backing this manager -- for rendering (see run_panel_html) when no
        report.json exists yet to read it from directly. start() reads
        self._rerun itself; this is for display only."""
        with self._lock:
            return dict(self._rerun)

    def start(self, k):
        """Validate `k` (the only request-supplied value -- see the module
        note above) and, if nothing is already running, spawn the
        subprocess in a background thread. Returns (ok, error_message)."""
        if k not in ALLOWED_RERUN_K:
            return False, f"k must be one of {list(ALLOWED_RERUN_K)}"
        trace_file = self._rerun.get("trace_file")
        if not trace_file:
            return False, "no trace file recorded for this report; cannot re-run"
        with self._lock:
            if self._status == "running":
                return False, "a review cycle is already running"
            self._status = "running"
            self._k = k
            self._started_at = time.time()
            self._finished_at = None
            self._returncode = None
            self._error = None
        argv = self._argv_builder(
            trace_file=trace_file, out_dir=self._out_dir, k=k,
            args=self._rerun.get("args"),
        )
        threading.Thread(target=self._run, args=(argv,), daemon=True).start()
        return True, None

    def _run(self, argv):
        try:
            proc = subprocess.run(
                argv, cwd=str(_REPO_ROOT), capture_output=True, text=True,
                timeout=3600,
            )
            with self._lock:
                self._finished_at = time.time()
                self._returncode = proc.returncode
                if proc.returncode == 0:
                    self._status = "done"
                else:
                    tail = (proc.stderr or proc.stdout or "").strip()
                    self._status = "failed"
                    self._error = tail[-2000:] or f"exited with code {proc.returncode}"
        except Exception as exc:  # subprocess.TimeoutExpired, spawn failure, ...
            with self._lock:
                self._finished_at = time.time()
                self._status = "failed"
                self._error = str(exc)


class _ReportOnlyHandler(http.server.BaseHTTPRequestHandler):
    """Serves only report.json and index.html from out_dir (GET), plus the
    /run and /status control endpoints (see module note above). No
    directory listing, no path traversal (the path is looked up in a fixed
    dict, never joined onto the filesystem), no other file in out_dir is
    ever reachable."""

    def __init__(self, *args, out_dir, run_manager=None, **kwargs):
        self._out_dir = out_dir
        self._run_manager = run_manager
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (stdlib handler method name)
        path = self.path.split("?", 1)[0]
        if path == "/status":
            if self._run_manager is None:
                self._send_json(503, {"error": "run manager unavailable"})
            else:
                self._send_json(200, self._run_manager.snapshot())
            return
        name = _SERVABLE.get(path)
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

    def do_POST(self):  # noqa: N802 (stdlib handler method name)
        if self.path.split("?", 1)[0] != "/run":
            self.send_error(404)
            return
        if self._run_manager is None:
            self._send_json(503, {"error": "run manager unavailable"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        # The only legitimate payload is a one-key {"k": <int>} object --
        # refuse an oversized body outright rather than reading an
        # arbitrary amount of attacker-controlled data into memory.
        if length > 4096:
            self._send_json(400, {"error": "request body too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) if length else b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        k, err = validate_run_k(body)
        if err:
            self._send_json(400, {"error": err})
            return
        ok, err = self._run_manager.start(k)
        if not ok:
            status = 409 if err == "a review cycle is already running" else 400
            self._send_json(status, {"error": err})
            return
        self._send_json(202, self._run_manager.snapshot())

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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


def make_run_manager(out_dir, argv_builder=None):
    """Build a _RunManager for out_dir, loaded from out_dir/report.json's
    "rerun" key if present (empty config -- run button unavailable -- if
    report.json is missing or unparseable). Split out of make_server() so
    interstellar/combined_serve.py's combined dashboard can wire the exact
    same /run + /status behavior onto its own routes instead of
    reimplementing the "load rerun config, build the manager" step."""
    out_dir = Path(out_dir).resolve()
    rerun_config = {}
    report_path = out_dir / "report.json"
    if report_path.is_file():
        try:
            rerun_config = json.loads(report_path.read_text(encoding="utf-8")).get("rerun") or {}
        except (json.JSONDecodeError, OSError):
            rerun_config = {}
    return _RunManager(out_dir, rerun_config, argv_builder=argv_builder)


def make_server(out_dir, host="127.0.0.1", port=4242, argv_builder=None):
    """Build (but do not start) a loopback-only TCP server that serves only
    report.json and index.html from out_dir, plus /run and /status (see the
    module notes above) -- backed by a fresh _RunManager loaded from
    out_dir/report.json's "rerun" key, if present. Split out from serve()
    so tests can bind an ephemeral port (port=0) and inspect server_address
    without blocking. `argv_builder` is forwarded to _RunManager -- see its
    docstring for why tests want to override it."""
    out_dir = Path(out_dir).resolve()
    if not (out_dir / "index.html").is_file():
        raise SystemExit(f"error: {out_dir} has no index.html (call write() first)")
    run_manager = make_run_manager(out_dir, argv_builder=argv_builder)
    handler = functools.partial(_ReportOnlyHandler, out_dir=out_dir, run_manager=run_manager)
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
