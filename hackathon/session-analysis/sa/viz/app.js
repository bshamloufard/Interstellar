/* Langfuse-style session trace visualizer */

const state = {
  pkg: null,
  nodes: [], // flat list with tree structure
  byId: new Map(),
  selectedId: null,
  collapsed: new Set(),
  showBars: true,
  t0: 0,
  t1: 1,
  rootIds: [],
  /** Horizontal scale: pixels per second of wall-clock time */
  pxPerSec: 1,
  timelinePx: 2400,
  /** If true, next loadLayout won't override pxPerSec (we just fitted) */
  userSetZoom: false,
};

/** Allow full-session overview (tiny px/s) up to fine tool detail */
const ZOOM_MIN = 0.05;
const ZOOM_MAX = 200;

const LANE_ICON = {
  turn: "▣",
  tool: "⚙",
  skill: "✦",
  mcp: "⬡",
  system: "◌",
  session: "◉",
  marker: "•",
};

const BAR_COLOR = {
  turn: "var(--bar-turn)",
  tool: "var(--bar-tool)",
  skill: "var(--bar-skill)",
  mcp: "var(--bar-mcp)",
  system: "var(--bar-system)",
  session: "var(--bar-system)",
  marker: "var(--bar-marker)",
};

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

async function loadJSONL(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return (await r.text())
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function loadPackage() {
  const [manifest, session, signals, turns, tools, skills, mcp, chat, timeline] =
    await Promise.all([
      loadJSON("/data/manifest.json"),
      loadJSON("/data/session.json"),
      loadJSON("/data/signals.json"),
      loadJSON("/data/turns.json"),
      loadJSON("/data/tools.json"),
      loadJSON("/data/skills.json"),
      loadJSON("/data/mcp.json"),
      loadJSON("/data/chat.json"),
      loadJSONL("/data/timeline.jsonl"),
    ]);
  return { manifest, session, signals, turns, tools, skills, mcp, chat, timeline };
}

function parseTs(ts) {
  if (!ts) return null;
  const n = Date.parse(ts);
  return Number.isNaN(n) ? null : n;
}

function fmtDur(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return "—";
  const n = Number(ms);
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60_000) return `${(n / 1000).toFixed(2)}s`;
  const m = Math.floor(n / 60_000);
  const s = ((n % 60_000) / 1000).toFixed(1);
  return `${m}m ${s}s`;
}

function fmtTokens(tok) {
  if (!tok) return null;
  const total = tok.total_tokens ?? tok.context_tokens;
  if (total == null && tok.input_tokens == null) return null;
  if (tok.input_tokens != null || tok.output_tokens != null) {
    return `${tok.input_tokens ?? "?"}→${tok.output_tokens ?? "?"}`;
  }
  return `Σ${total}`;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function isNoiseMarker(ev) {
  // TTFT etc. are metrics on the turn, not peer events like tools
  if (!ev) return false;
  if (ev.kind === "first_token") return true;
  if (ev.entity_type === "marker" && ev.kind !== "compaction") return true;
  const label = (ev.label || "").toLowerCase();
  if (label === "first token" || label === "first_token") return true;
  return false;
}

/** Build parent/child tree from timeline events. */
function buildTree(timeline) {
  const byId = new Map();
  const children = new Map();
  // parent_event_id -> TTFT ms (from first_token markers we hide)
  const ttftByParent = new Map();

  for (const ev of timeline) {
    if (isNoiseMarker(ev)) {
      const parentId = ev.parent_event_id;
      if (parentId) {
        // prefer explicit meta; else leave for turn entity lookup
        const ms = ev.meta?.ttft_ms;
        if (ms != null) ttftByParent.set(parentId, ms);
        else if (!ttftByParent.has(parentId)) ttftByParent.set(parentId, true);
      }
      continue; // do not add as a tree node
    }
    const id = ev.event_id;
    if (!id) continue;
    const node = {
      id,
      ev,
      children: [],
      depth: 0,
      ttftMs: null,
    };
    byId.set(id, node);
    children.set(id, []);
  }

  // Attach TTFT from turn package data when marker only flagged presence
  for (const node of byId.values()) {
    if (node.ev.entity_type === "turn") {
      const ent = resolveEntityEarly(node.ev);
      const ms = ent?.time_to_first_token_ms ?? node.ev.meta?.ttft_ms;
      if (ms != null) node.ttftMs = ms;
      else if (ttftByParent.has(node.id)) node.ttftMs = ms ?? null;
    }
  }

  const roots = [];
  for (const node of byId.values()) {
    const parentId = node.ev.parent_event_id;
    if (parentId && byId.has(parentId)) {
      children.get(parentId).push(node);
    } else {
      roots.push(node);
    }
  }

  // sort children + roots by start time
  const byStart = (a, b) =>
    (parseTs(a.ev.started_at) ?? 0) - (parseTs(b.ev.started_at) ?? 0) ||
    (a.ev.sequence ?? 0) - (b.ev.sequence ?? 0);

  function attach(node, depth) {
    node.depth = depth;
    node.children = (children.get(node.id) || []).slice().sort(byStart);
    for (const c of node.children) attach(c, depth + 1);
  }
  roots.sort(byStart);
  for (const r of roots) attach(r, 0);

  // Chronological roots only — do NOT dump unparented tools after all turns
  roots.sort(byStart);

  return { byId, roots };
}

/** Lightweight entity resolve used during tree build (pkg may already be set). */
function resolveEntityEarly(ev) {
  const pkg = state.pkg;
  if (!pkg) return null;
  const d = ev.detail || {};
  const id = d.entity_id || ev.entity_id;
  if (!id) return null;
  if (d.file === "turns.json" || ev.entity_type === "turn") {
    const i = pkg.turns.by_turn_id?.[id];
    return i != null ? pkg.turns.turns[i] : null;
  }
  return null;
}

function flattenVisible(roots, collapsed) {
  const out = [];
  function walk(node) {
    out.push(node);
    if (!collapsed.has(node.id)) {
      for (const c of node.children) walk(c);
    }
  }
  for (const r of roots) walk(r);
  return out;
}

function timeRange(timeline) {
  let t0 = Infinity;
  let t1 = -Infinity;
  for (const ev of timeline) {
    const a = parseTs(ev.started_at);
    const b = parseTs(ev.ended_at) ?? a;
    if (a != null) t0 = Math.min(t0, a);
    if (b != null) t1 = Math.max(t1, b);
  }
  if (!Number.isFinite(t0)) {
    t0 = Date.now();
    t1 = t0 + 1000;
  }
  if (t1 <= t0) t1 = t0 + 1000;
  return { t0, t1 };
}

function renderHeader(pkg) {
  const s = pkg.session;
  document.getElementById("title").textContent =
    s.title || s.session_summary || s.session_id || "Session";
  document.title = `Trace · ${document.getElementById("title").textContent}`;

  const turns = pkg.turns.turns?.length ?? 0;
  const tools = pkg.tools.summary?.call_count ?? 0;
  const skills = pkg.skills.summary?.activation_count ?? 0;
  const dur = (pkg.signals.activity?.session_duration_seconds ?? 0) * 1000;
  const model = s.model_id || pkg.signals.models?.primary_model_id || "—";

  document.getElementById("header-meta").innerHTML = [
    `<span class="pill">model <strong>${escapeHtml(model)}</strong></span>`,
    `<span class="pill">turns <strong>${turns}</strong></span>`,
    `<span class="pill">tools <strong>${tools}</strong></span>`,
    `<span class="pill">skills <strong>${skills}</strong></span>`,
    `<span class="pill">wall <strong>${escapeHtml(fmtDur(dur))}</strong></span>`,
  ].join("");
}

function layoutMetrics() {
  const cs = getComputedStyle(document.documentElement);
  const gutter = parseFloat(cs.getPropertyValue("--gutter")) || 340;
  const resizer = parseFloat(cs.getPropertyValue("--resizer-size")) || 5;
  const scrollEl = document.getElementById("trace-scroll");
  const viewW = scrollEl?.clientWidth || 900;
  const barsViewport = Math.max(200, viewW - gutter - resizer);
  return { gutter, resizer, scrollEl, viewW, barsViewport };
}

function sessionSpanSec() {
  return Math.max((state.t1 - state.t0) / 1000, 1);
}

/** px/s that fits the full session in the visible bars area */
function fitPxPerSec() {
  const { barsViewport } = layoutMetrics();
  const pps = barsViewport / sessionSpanSec();
  return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, pps));
}

function formatZoomLabel(pps) {
  const fit = fitPxPerSec();
  // Show human scale: how many minutes fit in view
  const { barsViewport } = layoutMetrics();
  const secVisible = barsViewport / Math.max(pps, 1e-6);
  let vis;
  if (secVisible >= 3600) vis = `~${(secVisible / 3600).toFixed(1)}h in view`;
  else if (secVisible >= 60) vis = `~${(secVisible / 60).toFixed(0)}m in view`;
  else vis = `~${secVisible.toFixed(0)}s in view`;
  const ratio = pps / fit;
  const zoomTxt =
    Math.abs(ratio - 1) < 0.08
      ? "fit"
      : ratio > 1
        ? `${ratio.toFixed(1)}×`
        : `${ratio.toFixed(2)}×`;
  return `${zoomTxt} · ${vis}`;
}

function applyTimelineScale() {
  const spanSec = sessionSpanSec();
  const { barsViewport } = layoutMetrics();
  // At least fill the viewport when zoomed out; grow when zoomed in
  state.timelinePx = Math.max(
    Math.round(barsViewport),
    Math.round(spanSec * state.pxPerSec)
  );
  document.documentElement.style.setProperty("--timeline-px", `${state.timelinePx}px`);
  const zl = document.getElementById("zoom-label");
  if (zl) zl.textContent = formatZoomLabel(state.pxPerSec);
}

function msToX(msAbs) {
  const span = Math.max(state.t1 - state.t0, 1);
  const rel = (msAbs - state.t0) / span;
  return Math.max(0, Math.min(state.timelinePx, rel * state.timelinePx));
}

/**
 * Zoom timeline keeping the wall-time under anchorClientX fixed on screen.
 * @param {number} nextPps
 * @param {number} [anchorClientX] client X; default = center of bars viewport
 */
function setZoom(nextPps, anchorClientX) {
  const { gutter, resizer, scrollEl, barsViewport } = layoutMetrics();
  if (!scrollEl) {
    state.pxPerSec = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, nextPps));
    applyTimelineScale();
    return;
  }

  const oldPps = state.pxPerSec;
  const oldPx = state.timelinePx;
  const rect = scrollEl.getBoundingClientRect();
  const anchor =
    anchorClientX != null ? anchorClientX : rect.left + gutter + resizer + barsViewport / 2;

  // Wall-time fraction under the cursor (relative to timeline content)
  const xInContent = anchor - rect.left + scrollEl.scrollLeft;
  const xInTimeline = xInContent - gutter - resizer;
  const frac = oldPx > 0 ? xInTimeline / oldPx : 0.5;

  state.pxPerSec = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, nextPps));
  state.userSetZoom = true;
  applyTimelineScale();
  renderTimeAxis();
  renderTree();

  // Keep same time under cursor
  const newXInTimeline = frac * state.timelinePx;
  const newScrollLeft = gutter + resizer + newXInTimeline - (anchor - rect.left);
  scrollEl.scrollLeft = Math.max(0, newScrollLeft);
  saveLayout();
}

function zoomByFactor(factor, anchorClientX) {
  setZoom(state.pxPerSec * factor, anchorClientX);
}

function zoomFit() {
  setZoom(fitPxPerSec());
  // After fit, scroll to start of session (or keep selection in view vertically)
  const { scrollEl } = layoutMetrics();
  if (scrollEl) scrollEl.scrollLeft = 0;
  if (state.selectedId) {
    requestAnimationFrame(() => scrollToEvent(state.selectedId));
  }
}

function renderTimeAxis() {
  applyTimelineScale();
  const axis = document.getElementById("time-axis");
  const spanMs = Math.max(state.t1 - state.t0, 1);
  // Aim for ~90–110px between ticks; cap count for huge zooms
  const approxTicks = Math.max(4, Math.min(80, Math.round(state.timelinePx / 100)));
  let html = "";
  for (let i = 0; i <= approxTicks; i++) {
    const frac = i / approxTicks;
    const left = frac * state.timelinePx;
    const ms = spanMs * frac;
    html += `<div class="tick" style="left:${left}px">${fmtDur(ms)}</div>`;
  }
  axis.innerHTML = html;
}

function barStyle(ev) {
  const a = parseTs(ev.started_at) ?? state.t0;
  const b = parseTs(ev.ended_at) ?? a;
  const left = msToX(a);
  const right = msToX(Math.max(b, a));
  let width = Math.max(right - left, 0);
  const lane = ev.lane || ev.entity_type || "system";
  const color = BAR_COLOR[lane] || BAR_COLOR[ev.entity_type] || "var(--accent)";
  const instant =
    ev.span_kind === "instant" ||
    Number(ev.duration_ms) === 0 ||
    width < 3;
  if (instant) {
    return {
      className: "span-bar instant",
      style: `left:${left}px;background:${color}`,
      leftPx: left,
      widthPx: 10,
    };
  }
  // Min width only when zoomed in enough that 1s ≥ ~2px; avoid fat bars in overview
  const minBar = state.pxPerSec >= 2 ? 4 : state.pxPerSec >= 0.5 ? 2 : 1;
  width = Math.max(width, minBar);
  return {
    className: "span-bar",
    style: `left:${left}px;width:${width}px;background:${color}`,
    leftPx: left,
    widthPx: width,
  };
}

/** Scroll the timeline so the selected bar is in view horizontally (and row vertically). */
function scrollToEvent(eventId) {
  const scrollEl = document.getElementById("trace-scroll");
  const row = scrollEl?.querySelector(`.tree-row[data-id="${CSS.escape(eventId)}"]`);
  if (!scrollEl || !row) return;

  // Vertical: keep row in view
  const rowTop = row.offsetTop;
  const rowBottom = rowTop + row.offsetHeight;
  const viewTop = scrollEl.scrollTop;
  const viewBottom = viewTop + scrollEl.clientHeight;
  if (rowTop < viewTop + 40) {
    scrollEl.scrollTop = Math.max(0, rowTop - 48);
  } else if (rowBottom > viewBottom - 20) {
    scrollEl.scrollTop = rowBottom - scrollEl.clientHeight + 24;
  }

  // Horizontal: center the bar in the bars viewport
  const bar = row.querySelector(".span-bar");
  if (!bar || !state.showBars) return;
  const cs = getComputedStyle(document.documentElement);
  const gutter = parseFloat(cs.getPropertyValue("--gutter")) || 340;
  const resizer = parseFloat(cs.getPropertyValue("--resizer-size")) || 5;
  const barLeft = parseFloat(bar.style.left) || 0;
  const barWidth = bar.classList.contains("instant")
    ? 10
    : parseFloat(bar.style.width) || 6;
  const barCenter = gutter + resizer + barLeft + barWidth / 2;
  const target = Math.max(0, barCenter - scrollEl.clientWidth / 2);
  scrollEl.scrollTo({ left: target, behavior: "smooth" });
}

function renderTree() {
  const root = document.getElementById("tree");
  const visible = flattenVisible(state.rootIds.map((id) => state.byId.get(id)).filter(Boolean), state.collapsed);

  if (!visible.length) {
    root.innerHTML = `<div class="empty">No timeline events in this package.</div>`;
    return;
  }

  root.innerHTML = visible
    .map((node) => {
      const ev = node.ev;
      const hasKids = node.children.length > 0;
      const collapsed = state.collapsed.has(node.id);
      const selected = state.selectedId === node.id;
      const icon =
        LANE_ICON[ev.entity_type] ||
        LANE_ICON[ev.lane] ||
        LANE_ICON.marker;
      const name = ev.label || ev.name || ev.kind || ev.entity_id || node.id;
      const dur = fmtDur(ev.duration_ms);
      const tok = fmtTokens(ev.tokens);
      const status = ev.status || "unknown";
      const bar = barStyle(ev);
      const ttft =
        node.ttftMs != null
          ? `<span class="tok" title="Time to first token">ttft ${escapeHtml(fmtDur(node.ttftMs))}</span>`
          : "";

      const twist = hasKids
        ? `<button type="button" class="twist" data-twist="${escapeHtml(node.id)}" aria-label="toggle">${collapsed ? "▶" : "▼"}</button>`
        : `<span class="twist placeholder">·</span>`;

      return `<div class="tree-row ${selected ? "selected" : ""}" data-id="${escapeHtml(node.id)}" style="--depth:${node.depth}">
        <div class="row-main">
          <span class="row-indent" style="--depth:${node.depth}"></span>
          ${twist}
          <span class="row-icon" title="${escapeHtml(ev.entity_type || "")}">${icon}</span>
          <span class="row-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          <div class="row-meta">
            <span class="status-dot ${escapeHtml(status)}" title="${escapeHtml(status)}"></span>
            <span class="dur">${escapeHtml(dur)}</span>
            ${ttft}
            ${tok ? `<span class="tok">${escapeHtml(tok)}</span>` : ""}
          </div>
        </div>
        <div class="row-gutter-spacer" aria-hidden="true"></div>
        <div class="row-bars">
          <div class="${bar.className}" style="${bar.style}"></div>
        </div>
      </div>`;
    })
    .join("");

  // indent via padding on name row
  root.querySelectorAll(".tree-row").forEach((row) => {
    const depth = Number(row.style.getPropertyValue("--depth") || 0);
    const indent = row.querySelector(".row-indent");
    if (indent) indent.style.width = `${depth * 14}px`;
  });

  root.querySelectorAll(".tree-row").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (e.target.closest("[data-twist]")) return;
      selectNode(row.dataset.id);
    });
  });
  root.querySelectorAll("[data-twist]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.dataset.twist;
      if (state.collapsed.has(id)) state.collapsed.delete(id);
      else state.collapsed.add(id);
      renderTree();
    });
  });
}

function resolveEntity(ev) {
  const pkg = state.pkg;
  const d = ev.detail || {};
  const id = d.entity_id || ev.entity_id;
  if (!id) return null;
  if (d.file === "turns.json" || ev.entity_type === "turn") {
    const i = pkg.turns.by_turn_id?.[id];
    return i != null ? { kind: "turn", data: pkg.turns.turns[i] } : null;
  }
  if (d.file === "tools.json" || ev.entity_type === "tool") {
    const i = pkg.tools.by_tool_call_id?.[id];
    return i != null ? { kind: "tool", data: pkg.tools.calls[i] } : null;
  }
  if (d.file === "skills.json" || ev.entity_type === "skill") {
    const i = pkg.skills.by_activation_id?.[id];
    return i != null ? { kind: "skill", data: pkg.skills.activations[i] } : null;
  }
  if (d.file === "mcp.json" || ev.entity_type === "mcp_call") {
    const i = pkg.mcp.by_mcp_call_id?.[id];
    return i != null ? { kind: "mcp", data: pkg.mcp.tool_calls[i] } : null;
  }
  return null;
}

function chatForTurn(turnId) {
  if (!turnId) return [];
  const idxs = state.pkg.chat.by_turn_id?.[turnId] || [];
  return idxs.map((i) => state.pkg.chat.messages[i]).filter(Boolean);
}

function selectNode(id, { scroll = true } = {}) {
  state.selectedId = id;
  renderTree();
  renderDetail(id);
  if (scroll) {
    // After paint
    requestAnimationFrame(() => scrollToEvent(id));
  }
}

function renderDetail(id) {
  const node = state.byId.get(id);
  const headWhen = document.getElementById("detail-when");
  const badges = document.getElementById("detail-badges");
  const preview = document.getElementById("tab-preview");
  const meta = document.getElementById("tab-metadata");

  if (!node) {
    headWhen.textContent = "Select a span";
    badges.innerHTML = "";
    preview.innerHTML = `<p class="muted">Click a row in the trace tree.</p>`;
    meta.innerHTML = `<pre class="meta-json muted">—</pre>`;
    return;
  }

  const ev = node.ev;
  const entity = resolveEntity(ev);
  const data = entity?.data;

  headWhen.textContent = ev.started_at
    ? `${ev.started_at}${ev.ended_at && ev.ended_at !== ev.started_at ? " → " + ev.ended_at : ""}`
    : "—";

  const status = ev.status || data?.status || "unknown";
  const tok = ev.tokens || data?.tokens;
  const model =
    data?.model_id ||
    (entity?.kind === "turn" ? data?.model_id : null) ||
    state.pkg.session.model_id;

  const badgeBits = [
    `<span class="badge ${status === "success" || status === "ok" ? "ok" : status === "failure" ? "err" : ""}"><em>${escapeHtml(status)}</em></span>`,
    `<span class="badge">Latency <em>${escapeHtml(fmtDur(ev.duration_ms ?? data?.span?.duration_ms))}</em></span>`,
  ];
  if (tok) {
    const tlabel = fmtTokens(tok);
    if (tlabel) badgeBits.push(`<span class="badge">Tokens <em>${escapeHtml(tlabel)}</em></span>`);
    if (tok.input_tokens != null) {
      badgeBits.push(
        `<span class="badge"><em>${tok.input_tokens}</em> prompt → <em>${tok.output_tokens ?? "—"}</em> completion</span>`
      );
    }
  }
  if (model) badgeBits.push(`<span class="badge model">${escapeHtml(model)}</span>`);
  if (ev.kind) badgeBits.push(`<span class="badge">${escapeHtml(ev.kind)}</span>`);
  badges.innerHTML = badgeBits.join("");

  // Preview content
  let html = "";
  const turnId = ev.turn_id || data?.turn_id;

  if (entity?.kind === "turn" || ev.entity_type === "turn") {
    const node = state.byId.get(id);
    const ttft = data?.time_to_first_token_ms ?? node?.ttftMs;
    if (ttft != null) {
      html += `<p class="muted" style="font-size:11px;margin:0 0 0.65rem">
        Time to first token: <strong style="color:var(--text);font-weight:600">${escapeHtml(fmtDur(ttft))}</strong>
        <span style="opacity:0.8"> — when the model started streaming (not a separate event).</span>
      </p>`;
    }
    html += turnPreviewHtml(data, turnId);
  } else if (entity?.kind === "tool") {
    html += `<div class="section-label">Tool</div>`;
    html += `<dl class="kv">
      <dt>name</dt><dd>${escapeHtml(data.tool_name)}</dd>
      <dt>call id</dt><dd>${escapeHtml(data.tool_call_id)}</dd>
      <dt>outcome</dt><dd>${escapeHtml(data.outcome || "—")}</dd>
    </dl>`;
    if (data.input_preview?.text) {
      html += msgCard("tool", data.input_preview.text, "input");
    }
    if (data.result_preview?.text) {
      html += msgCard("tool_result", data.result_preview.text, "result");
    }
    // related chat turn messages (assistant that issued call)
    if (turnId) {
      const stack = msgStackHtml(
        chatForTurn(turnId).filter(
          (m) =>
            m.role === "assistant" ||
            (m.role === "tool_result" && m.tool_call_id === data.tool_call_id)
        )
      );
      if (stack) {
        html += `<div class="section-label">Turn context</div>${stack}`;
      }
    }
  } else if (entity?.kind === "skill") {
    html += `<div class="section-label">Skill</div>`;
    html += `<dl class="kv">
      <dt>name</dt><dd>${escapeHtml(data.skill_name)}</dd>
      <dt>trigger</dt><dd>${escapeHtml(data.trigger || "—")}</dd>
    </dl>`;
    if (data.args_preview?.text) html += msgCard("user", data.args_preview.text, "args");
    if (data.agent_preview?.text) html += msgCard("assistant", data.agent_preview.text, "agent");
    if (turnId) {
      const stack = msgStackHtml(chatForTurn(turnId));
      if (stack) {
        html += `<div class="section-label">Turn messages</div>${stack}`;
      }
    }
  } else if (ev.entity_type === "compaction" || ev.kind === "compaction") {
    const tb = ev.tokens_before ?? ev.meta?.tokens_before;
    const ta = ev.tokens_after ?? ev.meta?.tokens_after;
    html += `<div class="section-label">Context compaction</div>
      <p class="muted" style="margin:0 0 0.65rem;line-height:1.45">
        The session summarized older chat to free context window space.
        Duration is a point-in-time marker (completion instant), not a multi-second span.
      </p>
      <dl class="kv">
        <dt>tokens before</dt><dd>${escapeHtml(String(tb ?? "—"))}</dd>
        <dt>tokens after</dt><dd>${escapeHtml(String(ta ?? "—"))}</dd>
        <dt>saved</dt><dd>${tb != null && ta != null ? escapeHtml(String(tb - ta)) : "—"}</dd>
        <dt>turn</dt><dd>${escapeHtml(ev.turn_id || "—")}</dd>
        <dt>when</dt><dd>${escapeHtml(ev.started_at || "—")}</dd>
      </dl>`;
  } else {
    html += previewFromEntity(data, ev);
    if (turnId) {
      const stack = msgStackHtml(chatForTurn(turnId));
      if (stack) {
        html += `<div class="section-label">Turn messages</div>${stack}`;
      }
    }
  }

  if (!html) html = `<p class="muted">No preview content for this span.</p>`;
  preview.innerHTML = html;

  const metaObj = {
    event: ev,
    entity: data || null,
  };
  meta.innerHTML = `<pre class="meta-json">${escapeHtml(JSON.stringify(metaObj, null, 2))}</pre>`;
}

function previewFromEntity(data, ev) {
  const p =
    ev.preview?.text ||
    data?.agent_preview?.text ||
    data?.user_preview?.text ||
    data?.input_preview?.text ||
    data?.result_preview?.text;
  if (!p) return "";
  return msgCard("assistant", p, "preview");
}

/**
 * Build turn preview cards.
 *
 * Chat history is preferred when complete, but after compaction it often only
 * retains a user remnant (or nothing). Always fill missing roles from
 * turns.*_preview (sourced from updates.jsonl).
 *
 * Order is always: user → reasoning → assistant (one exchange per turn).
 */
function turnPreviewHtml(data, turnId) {
  const msgs = chatForTurn(turnId).filter(
    (m) =>
      m.role !== "system" &&
      !(m.role === "user" && m.synthetic_reason === "system_reminder")
  );

  const fromChat = { user: null, reasoning: null, assistant: null };
  for (const m of msgs) {
    if (m.role === "user" && !fromChat.user) fromChat.user = m;
    else if (m.role === "reasoning" && !fromChat.reasoning) fromChat.reasoning = m;
    else if (m.role === "assistant" && !fromChat.assistant) fromChat.assistant = m;
    // skip tool_result noise in the main turn exchange view
  }

  // Prefer chat rows that actually have text; empty tool-only assistants skipped
  const chatUser = fromChat.user && hasMsgText(fromChat.user.text) ? fromChat.user : null;
  const chatReason =
    fromChat.reasoning && hasMsgText(fromChat.reasoning.text) ? fromChat.reasoning : null;
  // last non-empty assistant in turn (final reply), not first empty tool-call stub
  let chatAsst = null;
  for (const m of msgs) {
    if (m.role === "assistant" && hasMsgText(m.text)) chatAsst = m;
  }

  const userText = chatUser?.text || data?.user_preview?.text || null;
  const reasonText = chatReason?.text || data?.reasoning_preview?.text || null;
  const asstText = chatAsst?.text || data?.agent_preview?.text || null;

  const usedUpdates =
    (!chatUser && hasMsgText(data?.user_preview?.text)) ||
    (!chatAsst && hasMsgText(data?.agent_preview?.text)) ||
    (!chatReason && hasMsgText(data?.reasoning_preview?.text));

  const cards = [];
  if (hasMsgText(userText)) {
    cards.push(
      msgCard(
        "user",
        userText,
        chatUser?.message_id || data?.user_preview?.source || "user"
      )
    );
  }
  if (hasMsgText(reasonText)) {
    cards.push(
      msgCard(
        "reasoning",
        reasonText,
        chatReason?.message_id || data?.reasoning_preview?.source || "reasoning"
      )
    );
  }
  if (hasMsgText(asstText)) {
    cards.push(
      msgCard(
        "assistant",
        asstText,
        chatAsst?.message_id || data?.agent_preview?.source || "assistant"
      )
    );
  }

  if (!cards.length) {
    return `<p class="muted">No preview content for this span.
      ${turnId ? `(Nothing linked to <code>${escapeHtml(turnId)}</code>.)` : ""}
    </p>`;
  }

  let html = "";
  if (usedUpdates) {
    html += `<p class="muted" style="font-size:11px;margin:0 0 0.6rem">
      Part of this turn’s chat was compacted; filled missing sides from session update streams.
    </p>`;
  }
  html += `<div class="msg-stack">${cards.join("")}</div>`;
  return html;
}

/** True if message has visible text (tool-only assistant rows are empty). */
function hasMsgText(text) {
  return Boolean(text && String(text).trim());
}

function msgStackHtml(messages) {
  const cards = (messages || [])
    .filter((m) => hasMsgText(m?.text))
    .map((m) => msgCard(m.role, m.text, m.message_id))
    .filter(Boolean);
  if (!cards.length) return "";
  return `<div class="msg-stack">${cards.join("")}</div>`;
}

function msgCard(role, text, id) {
  if (!hasMsgText(text)) return "";
  const body = String(text).trim().slice(0, 8000);
  const label = role.replaceAll("_", " ");
  return `<div class="msg-card ${escapeHtml(role)}">
    <div class="msg-card-h"><span>${escapeHtml(label)}</span><span class="muted">${escapeHtml(id || "")}</span></div>
    <div class="msg-card-b">${escapeHtml(body)}</div>
  </div>`;
}

function collectExpandableIds(roots) {
  const ids = [];
  function walk(n) {
    if (n.children.length) {
      ids.push(n.id);
      n.children.forEach(walk);
    }
  }
  roots.forEach(walk);
  return ids;
}

function rerenderScaleKeepSelection() {
  const id = state.selectedId;
  const { scrollEl, gutter, resizer } = layoutMetrics();
  // Preserve center time when button-zooming
  let anchorClientX;
  if (scrollEl) {
    const rect = scrollEl.getBoundingClientRect();
    anchorClientX = rect.left + gutter + resizer + layoutMetrics().barsViewport / 2;
  }
  setZoom(state.pxPerSec, anchorClientX);
  if (id) {
    requestAnimationFrame(() => scrollToEvent(id));
  }
}

const LAYOUT_KEY = "sa-trace-layout-v2";

function loadLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    if (!raw) return false;
    const o = JSON.parse(raw);
    if (o.gutter && o.gutter >= 180 && o.gutter <= 700) {
      document.documentElement.style.setProperty("--gutter", `${o.gutter}px`);
    }
    if (o.detail && o.detail >= 240 && o.detail <= 900) {
      document.documentElement.style.setProperty("--detail-width", `${o.detail}px`);
    }
    if (o.pxPerSec && o.pxPerSec >= ZOOM_MIN && o.pxPerSec <= ZOOM_MAX) {
      state.pxPerSec = o.pxPerSec;
      state.userSetZoom = true;
      return true;
    }
  } catch {
    /* ignore */
  }
  return false;
}

function saveLayout() {
  const gutter = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--gutter")
  );
  const detail = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--detail-width")
  );
  try {
    localStorage.setItem(
      LAYOUT_KEY,
      JSON.stringify({
        gutter: Math.round(gutter),
        detail: Math.round(detail),
        pxPerSec: state.pxPerSec,
      })
    );
  } catch {
    /* ignore */
  }
}

/**
 * Drag a vertical resizer.
 * @param {HTMLElement} el
 * @param {"gutter" | "main"} kind
 */
function wireResizer(el, kind) {
  if (!el) return;
  let startX = 0;
  let startVal = 0;

  const onMove = (e) => {
    const dx = e.clientX - startX;
    if (kind === "gutter") {
      const next = Math.min(700, Math.max(180, startVal + dx));
      document.documentElement.style.setProperty("--gutter", `${next}px`);
    } else {
      // dragging main resizer: moving right shrinks detail, left grows detail
      const next = Math.min(900, Math.max(240, startVal - dx));
      document.documentElement.style.setProperty("--detail-width", `${next}px`);
    }
  };

  const onUp = () => {
    el.classList.remove("dragging");
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    saveLayout();
  };

  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    if (kind === "gutter") {
      startVal = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue("--gutter")
      ) || 340;
    } else {
      startVal = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue("--detail-width")
      ) || 420;
    }
    el.classList.add("dragging");
    document.body.classList.add("is-resizing");
    el.setPointerCapture?.(e.pointerId);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

function wireChrome() {
  const hadSavedZoom = loadLayout();
  wireResizer(document.getElementById("resize-gutter"), "gutter");
  wireResizer(document.getElementById("resize-main"), "main");

  document.getElementById("btn-expand-all").onclick = () => {
    state.collapsed.clear();
    renderTree();
  };
  document.getElementById("btn-collapse-all").onclick = () => {
    for (const id of collectExpandableIds(
      state.rootIds.map((i) => state.byId.get(i)).filter(Boolean)
    )) {
      state.collapsed.add(id);
    }
    renderTree();
  };
  document.getElementById("btn-zoom-in").onclick = () => {
    zoomByFactor(1.35);
  };
  document.getElementById("btn-zoom-out").onclick = () => {
    zoomByFactor(1 / 1.35);
  };
  document.getElementById("btn-zoom-fit").onclick = () => {
    zoomFit();
  };
  document.getElementById("toggle-bars").onchange = (e) => {
    state.showBars = e.target.checked;
    document.body.classList.toggle("hide-bars", !state.showBars);
  };
  document.querySelectorAll(".detail-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".detail-tabs .tab").forEach((t) => t.classList.remove("on"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("on"));
      tab.classList.add("on");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add("on");
    });
  });

  // Mouse / trackpad zoom on the timeline
  const scrollEl = document.getElementById("trace-scroll");
  if (scrollEl) {
    scrollEl.addEventListener(
      "wheel",
      (e) => {
        // Zoom: ctrl/cmd+wheel (trackpad pinch), or wheel over the time axis header,
        // or alt+wheel. Plain wheel still scrolls vertically/horizontally.
        const overAxis = e.target.closest?.(".time-header-axis, .time-header");
        const wantZoom = e.ctrlKey || e.metaKey || e.altKey || overAxis;
        if (!wantZoom || !state.showBars) return;

        e.preventDefault();
        // Normalize delta — trackpads send small values, mice large
        let dy = e.deltaY;
        if (e.deltaMode === 1) dy *= 16; // lines
        if (e.deltaMode === 2) dy *= 400; // pages
        // Smooth exponential zoom
        const factor = Math.exp(-dy * 0.0018);
        zoomByFactor(factor, e.clientX);
      },
      { passive: false }
    );

    // Double-click axis → fit overview
    document.getElementById("time-axis")?.addEventListener("dblclick", () => {
      zoomFit();
    });
  }

  // Expose whether we already have a saved zoom for main()
  state._hadSavedZoom = hadSavedZoom;
}

/** Default: expand turns, collapse deep tool noise if many children */
function defaultCollapse(roots) {
  state.collapsed.clear();
  for (const r of roots) {
    for (const c of r.children) {
      // keep tools visible under turns; collapse markers-only noise not needed
      if (c.children.length > 8) {
        // still show children of huge nodes collapsed? keep expanded for explore
      }
    }
  }
  // Collapse nothing by default so structure is visible; user can collapse
}

async function main() {
  wireChrome();
  try {
    const pkg = await loadPackage();
    state.pkg = pkg;
    renderHeader(pkg);

    const { byId, roots } = buildTree(pkg.timeline || []);
    state.byId = byId;
    state.rootIds = roots.map((r) => r.id);
    defaultCollapse(roots);

    const tr = timeRange(pkg.timeline || []);
    state.t0 = tr.t0;
    state.t1 = tr.t1;

    // Default: fit whole session so multiple turns are visible.
    // If user previously zoomed, keep that (v2 layout key).
    if (!state._hadSavedZoom) {
      state.pxPerSec = fitPxPerSec();
      state.userSetZoom = false;
    }
    renderTimeAxis();
    renderTree();

    // Auto-select first turn (or first root) — don't force-scroll horizontally on fit
    const firstTurn = roots.find((r) => r.ev.entity_type === "turn") || roots[0];
    if (firstTurn) selectNode(firstTurn.id, { scroll: !state._hadSavedZoom ? false : true });
    if (!state._hadSavedZoom) {
      const { scrollEl } = layoutMetrics();
      if (scrollEl) scrollEl.scrollLeft = 0;
    }
  } catch (err) {
    document.getElementById("title").textContent = "Failed to load package";
    document.getElementById("tree").innerHTML = `<div class="empty">${escapeHtml(
      String(err)
    )}</div>`;
  }
}

main();
