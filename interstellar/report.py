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
import socketserver
from datetime import datetime, timezone
from pathlib import Path

from interstellar.types import (
    ARM_CONTROL,
    ARM_TREATMENT,
    EFFECT_MCP_STARTUP_MS,
    EFFECT_SKILL_TOKENS,
    EFFECT_TOOL_CALLS,
    EFFECT_TOOL_RESULT_TOKENS,
    EFFECT_WALL_MS,
    EFFICIENCY_METRICS,
    make_report,
)

# --- cross-patch caveats -----------------------------------------------------

ISOLATION_NOTE = (
    "Control and treatment each ran k paired repeats in their own isolated "
    "sandbox (separate git worktree/copy and GROK_HOME); the user's original "
    "recorded session is not reused as the control, since it ran in a "
    "different environment."
)


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
    total = consistent = 0
    for pr in patch_results:
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


def default_caveats(k, patch_results):
    """The caveats block: k, isolation note, judge consistency rate,
    insufficient-sample flags. Always non-empty, always rendered verbatim by
    write() — this is the trust layer, not fine print."""
    caveats = [f"k={int(k)} paired repeats per arm, per patch."]
    caveats.append(ISOLATION_NOTE)

    consistent, total = _judge_consistency(patch_results)
    if total:
        rate = consistent / total
        caveats.append(
            f"Judge consistency: {consistent}/{total} position-swapped pairs "
            f"agreed on a winner ({rate:.0%}); an inconsistent pair is scored "
            f"as a tie, never resolved toward either side."
        )
    else:
        caveats.append("No judged pairs were available for this cycle.")

    flagged = _insufficient_patch_ids(patch_results)
    if flagged:
        caveats.append(
            "Insufficient samples for a confident interval on: "
            + ", ".join(str(p) for p in flagged)
            + " — treat those point estimates as directional only, not proof."
        )
    return caveats


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(*, session, baseline_version, patch_results, k=0, cost_usd=0.0,
          generated_at=None, extra_caveats=None):
    """Assemble the Report per types.make_report.

    `patch_results` must already be fully-formed make_patch_result() dicts
    (statistics/gate/verdict_line computed upstream). build() assembles the
    document and derives the cross-patch caveats; it does not recompute any
    per-patch statistic.
    """
    for pr in patch_results:
        if not pr.get("verdict_line"):
            raise ValueError(
                f"patch_result {pr.get('patch', {}).get('patch_id')!r} is "
                "missing verdict_line"
            )

    caveats = default_caveats(k, patch_results)
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

# expected_effect (types.ALL_EFFECTS) -> the efficiency metric it claims to move
EFFECT_TO_METRIC = {
    EFFECT_SKILL_TOKENS: "skill_tokens_est",
    EFFECT_TOOL_CALLS: "tool_calls",
    EFFECT_MCP_STARTUP_MS: "mcp_startup_ms",
    EFFECT_TOOL_RESULT_TOKENS: "tool_result_tokens_est",
    EFFECT_WALL_MS: "wall_ms",
}


def _esc(x) -> str:
    return html.escape(str(x if x is not None else "—"))


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
    if x is None:
        return "n/a (control ≈ 0)"
    return f"{x:+.1%}"


def _ci_text(ci):
    if not ci:
        return "insufficient samples"
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        note = ci.get("note") or ci.get("reason")
        return "insufficient samples" + (f" ({note})" if note else "")
    level = ci.get("level", 0.95)
    return f"[{_fmt_num(lo)}, {_fmt_num(hi)}] ({level:.0%} CI)"


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
    delta_abs = m.get("delta_abs")
    cls = "neutral"
    if delta_abs is not None and delta_abs != 0:
        improved = (delta_abs < 0) if lower_is_better else (delta_abs > 0)
        cls = "good" if improved else "bad"
    row_cls = f"{cls} hl" if highlight else cls
    return (
        f'<tr class="{row_cls}">'
        f"<td>{_esc(label)}{' <span class=\"tag\">target</span>' if highlight else ''}</td>"
        f"<td>{_fmt_num(m.get('control_median'))}</td>"
        f"<td>{_fmt_num(m.get('treatment_median'))}</td>"
        f"<td>{_fmt_delta(delta_abs)}</td>"
        f"<td>{_fmt_pct(m.get('delta_pct'))}</td>"
        f"<td>{_esc(_ci_text(m.get('ci')))}</td>"
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


def _wtl_bar(win_rate):
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
    rate, lo, hi = win_rate.get("rate"), win_rate.get("lo"), win_rate.get("hi")
    if rate is None:
        rate_txt = "win rate: n/a"
    else:
        rate_txt = f"win rate over decided pairs: {rate:.0%}"
        rate_txt += f" [{lo:.0%}, {hi:.0%}] 95% CI" if lo is not None and hi is not None else " (insufficient samples for a CI)"
    legend = (
        '<div class="wtl-legend">'
        f'<span class="wtl-dot wtl-win"></span>wins {wins} '
        f'<span class="wtl-dot wtl-tie"></span>ties {ties} '
        f'<span class="wtl-dot wtl-loss"></span>losses {losses}'
        f'<span class="muted"> &middot; {_esc(rate_txt)}</span>'
        "</div>"
    )
    return bar + legend


def _extra_stats(stats):
    parts = []
    sign_test = stats.get("sign_test")
    if isinstance(sign_test, dict) and sign_test.get("p") is not None:
        parts.append(
            f'<div class="stat-line">Sign test: p={sign_test["p"]:.3f}, '
            f'direction={_esc(sign_test.get("direction"))}, '
            f'n<sub>nonzero</sub>={_esc(sign_test.get("n_nonzero"))}</div>'
        )
    pass_k = stats.get("pass_k")
    if isinstance(pass_k, dict) and pass_k.get("control") is not None and pass_k.get("treatment") is not None:
        parts.append(
            f'<div class="stat-line">pass@k: control {pass_k["control"]:.0%} '
            f'&rarr; treatment {pass_k["treatment"]:.0%}</div>'
        )
    return "".join(parts)


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


def _patch_card(pr, idx):
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

    accepted = bool(gate.get("accepted"))
    gate_badge = (
        f'<span class="badge {"accept" if accepted else "reject"}">'
        f'{"ACCEPT" if accepted else "REJECT"}</span>'
    )
    reasons = gate.get("reasons") or []
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons) or '<li class="muted">No reasons recorded.</li>'

    diff_block = _diff_html(diff_text)
    if skip_reason:
        diff_block = f'<p class="muted">Not applied: {_esc(skip_reason)}</p>' + diff_block

    metric_target = EFFECT_TO_METRIC.get(expected_effect)
    table_html = _metrics_table(stats.get("efficiency") or {}, metric_target)
    wtl_html = _wtl_bar(stats.get("win_rate"))
    extra_html = _extra_stats(stats)
    matrix_line = _matrix_summary_line(matrix)

    meta_bits = [f"expects to move <strong>{_esc(expected_effect)}</strong>"]
    if source_rec:
        meta_bits.append(f"from recommendation <em>{_esc(source_rec)}</em>")
    if matrix_line:
        meta_bits.append(_esc(matrix_line))

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
        <span class="pill">session <strong>{_esc(session.get("session_id"))}</strong></span>
        <span class="pill">baseline <strong>{_esc(baseline.get("version_id"))}</strong></span>
        <span class="pill">k <strong>{_esc(k)}</strong></span>
        <span class="pill">cycle cost <strong>${cost:,.4f}</strong></span>
        <span class="pill">generated <strong>{_esc(generated_at)}</strong></span>
      </div>
    </header>
    """

    session_section = f"""
    <section class="session-card">
      <div class="section-label">Prompt replayed</div>
      <pre class="prompt-block">{_esc(session.get("prompt"))}</pre>
      <div class="kv">
        <dt>Trace file</dt><dd>{_esc(session.get("trace_file"))}</dd>
        <dt>Baseline skills</dt><dd>{_esc(baseline.get("skills"))}</dd>
        <dt>Baseline MCP servers</dt><dd>{_esc(baseline.get("mcp"))}</dd>
      </div>
    </section>
    """

    cards = "\n".join(_patch_card(pr, i) for i, pr in enumerate(patch_results, 1))
    if not cards:
        cards = '<p class="muted">No candidate patches were produced for this session.</p>'

    caveats_html = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    caveats_section = f"""
    <section class="caveats">
      <div class="section-label">Caveats &mdash; read before trusting the numbers above</div>
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
{session_section}
<section class="patches">
{cards}
</section>
{caveats_section}
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
  }
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  max-width: 100%;
  overflow-x: hidden;
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

.session-card, .patch-card, .caveats {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.95rem 1.05rem;
  margin-bottom: 1rem;
}
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
.badge.kind { color: #93c5fd; border-color: #334155; text-transform: none; }
.badge.accept { color: var(--good); border-color: var(--good); background: var(--good-dim); }
.badge.reject { color: var(--bad); border-color: var(--bad); background: var(--bad-dim); }

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

ul.reasons { margin: 0; padding-left: 1.1rem; font-size: 12.5px; }
ul.reasons li { margin-bottom: 0.2rem; }

.caveats ul { margin: 0; padding-left: 1.2rem; font-size: 13px; }
.caveats li { margin-bottom: 0.35rem; }
"""

_SCRIPT = """
document.querySelectorAll('.patch-head').forEach(function (head) {
  head.style.cursor = 'pointer';
  head.addEventListener('click', function (ev) {
    if (ev.target.closest('a, button')) return;
    var body = head.closest('.patch-card').querySelector('.patch-body');
    if (body) body.style.display = body.style.display === 'none' ? '' : 'none';
  });
});
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

def make_server(out_dir, host="127.0.0.1", port=4242):
    """Build (but do not start) a loopback-only TCP server rooted at
    out_dir. Split out from serve() so tests can bind an ephemeral port
    (port=0) and inspect server_address without blocking."""
    out_dir = Path(out_dir).resolve()
    if not (out_dir / "index.html").is_file():
        raise SystemExit(f"error: {out_dir} has no index.html (call write() first)")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.TCPServer((host, port), handler)


def serve(out_dir, port=4242):
    """Serve out_dir (report.json + index.html) over http.server, bound to
    127.0.0.1 only — never 0.0.0.0."""
    with make_server(out_dir, host="127.0.0.1", port=port) as httpd:
        host, bound_port = httpd.server_address
        print(f"serving {Path(out_dir).resolve()} at http://{host}:{bound_port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
