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
    `<span class="pill">session <strong>${escapeHtml(s.session_id || s.id || "—")}</strong></span>`,
    `<span class="pill">model <strong>${escapeHtml(model)}</strong></span>`,
    `<span class="pill">turns <strong>${turns}</strong></span>`,
    `<span class="pill">tools <strong>${tools}</strong></span>`,
    `<span class="pill">skills <strong>${skills}</strong></span>`,
    `<span class="pill">wall <strong>${escapeHtml(fmtDur(dur))}</strong></span>`,
  ].join("");
}

function shortSessionId(id) {
  return (id || "").split("-")[0] || id || "";
}

/** View switcher: a link to the matching interstellar review report for
 * this same session, appended (not replacing) into #header-meta next to
 * the pills above. Renders nothing when no report_url is configured (see
 * serve.py's --report-url) -- a missing link is safe, a wrong one is not:
 * this only ever LOWERS confidence (unverified) or LOUDLY flags a mismatch
 * (never silently claims a match it can't back up). Never throws into
 * main()'s caller -- a /meta.json fetch failure just means no switcher. */
async function renderViewSwitcher(pkg) {
  const container = document.getElementById("header-meta");
  if (!container) return;
  let meta;
  try {
    meta = await loadJSON("/meta.json");
  } catch (err) {
    return;
  }
  const url = meta && meta.report_url;
  if (!url) return;

  const sessionId = pkg.session.session_id || pkg.session.id || "";
  const targetId = (meta.report_session_id || "").trim();
  let cls = "view-link view-link-unverified";
  let label = "Review";
  let title = `Opens the review report at ${url} — target session could not be verified against this trace`;
  if (targetId) {
    if (targetId === sessionId) {
      cls = "view-link";
      title = `Opens the review report — verified same session (${escapeHtml(sessionId)})`;
    } else {
      cls = "view-link view-link-mismatch";
      label = `Review · ${shortSessionId(targetId)}`;
      title = `WARNING: this report is for a DIFFERENT session (${targetId}) than this trace (${sessionId})`;
    }
  }
  container.insertAdjacentHTML(
    "beforeend",
    `<span class="view-switcher">` +
      `<span class="view-seg view-seg-active">Trace</span>` +
      `<a class="${cls}" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${escapeHtml(
        title
      )}">${escapeHtml(label)}</a></span>`
  );
}

const SEV_RANK = { alert: 0, warn: 1, info: 2 };

/** Shell builtins / wrappers we skip when finding the "real" command. */
const SHELL_SKIP = new Set([
  "sudo",
  "command",
  "env",
  "nice",
  "nohup",
  "time",
  "timeout",
  "stdbuf",
  "bash",
  "sh",
  "zsh",
  "fish",
  "exec",
  // common builtins / not useful families
  "cd",
  "echo",
  "printf",
  "export",
  "set",
  "unset",
  "source",
  ".",
  "eval",
  "read",
  "test",
  "[",
  "[[",
  "true",
  "false",
  "exit",
  "return",
  "wait",
  "type",
  "alias",
  "declare",
  "local",
  "readonly",
  "pwd",
  "pushd",
  "popd",
  "let",
  "umask",
  "ulimit",
  "hash",
  "help",
  "history",
  "jobs",
  "fg",
  "bg",
  "shift",
  "getopts",
  "trap",
  "kill",
  // keywords
  "for",
  "do",
  "done",
  "if",
  "then",
  "else",
  "elif",
  "fi",
  "while",
  "until",
  "case",
  "esac",
  "in",
  "select",
  "function",
  "time",
]);

/** Prefer these when present early in the command line. */
const KNOWN_SET = new Set(
  `
  git gh hub svn hg bzr
  cargo rustc rustup rustfmt clippy
  go gofmt
  npm npx yarn pnpm bun deno node tsx tsc
  python python3 pip pip3 uv poetry pipenv conda mamba pytest ruff mypy black
  docker docker-compose podman kubectl helm kind minikube
  curl wget ssh scp rsync aria2c
  find fd grep rg ag sed awk xargs jq yq
  cat head tail less tee wc sort uniq diff patch
  ls which whereis file stat du df
  tar zip unzip gzip
  make cmake ninja bazel
  clang gcc g++ cc c++ protoc
  brew apt apt-get yum dnf pacman nix
  aws gcloud az gsutil
  terraform tofu ansible pulumi
  vim nvim code cursor claude codex grok
  open osascript pbcopy pbpaste
  ps top htop lsof ping dig nc
  ffmpeg ffprobe convert magick pandoc
  sqlite3 psql mysql redis-cli
  java javac mvn gradle dotnet
  ruby gem bundle perl php composer
  swift xcrun xcodebuild
  `.trim().split(/\s+/).filter(Boolean)
);

/**
 * Pull the shell `command` string out of tool input_preview (JSON may be truncated).
 */
function extractShellCommandText(call) {
  const prev = call.input_preview?.text || "";
  if (!prev) return "";
  try {
    const j = JSON.parse(prev);
    if (typeof j.command === "string") return j.command;
    if (typeof j.cmd === "string") return j.cmd;
  } catch {
    /* truncated JSON — fall through */
  }
  // Match "command": ".... with possible truncation (no closing quote)
  const m = prev.match(/"command"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)/);
  if (m) {
    try {
      return JSON.parse(`"${m[1]}"`);
    } catch {
      return m[1].replace(/\\n/g, "\n").replace(/\\"/g, '"').replace(/\\\\/g, "\\");
    }
  }
  // Not JSON-shaped — only use raw if it doesn't look like our tool envelope
  if (!/^\s*\{/.test(prev) && !/"variant"\s*:/.test(prev)) return prev;
  return "";
}

/**
 * Extract primary command family from a tool call (one layer under tool name).
 * e.g. run_terminal_command + "cd x && git status" → "git"
 */
function shellCommandFamily(call) {
  const name = call.tool_name || "";
  if (name !== "run_terminal_command" && name !== "bash") {
    return null;
  }
  const cmd = extractShellCommandText(call);
  if (!cmd || typeof cmd !== "string") return "shell";

  // Prefer first non-comment, non-trivial segment across the script
  const segments = cmd
    .split(/\n+/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  const normalizeBase = (tok) => {
    let base = tok.replace(/^["']|["']$/g, "");
    base = base.split("/").pop() || base;
    base = base.toLowerCase();
    if (/^python\d/.test(base)) return "python";
    if (base === "rg" || base === "ag") return "grep";
    if (base === "nodejs") return "node";
    if (base.endsWith(".py")) return "python";
    if (base.startsWith("grok") || base.startsWith("xai-grok")) return "grok";
    if (base === "docker-compose") return "docker";
    if (base === "podman-compose") return "podman";
    return base;
  };

  const isJunkTok = (tok) => {
    if (!tok || tok.startsWith("-") || tok.startsWith("#")) return true;
    if (/[{}:"]/.test(tok)) return true;
    if (/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{10,}/i.test(tok)) return true;
    if (/^[0-9a-f]{32,}$/i.test(tok)) return true;
    if (/^https?:\/\//i.test(tok)) return true;
    const base = normalizeBase(tok);
    if (base.includes("%2f") || base.startsWith("%")) return true;
    if (base.startsWith(".")) return true;
    if (base.length <= 1) return true;
    if (SHELL_SKIP.has(base)) return true;
    return false;
  };

  // 1) Prefer a known CLI anywhere in the early command text (git/cargo/curl/…)
  const scan = cmd.slice(0, 400);
  const rawToks = scan.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
  for (const raw of rawToks) {
    if (isJunkTok(raw)) continue;
    const base = normalizeBase(raw);
    if (KNOWN_SET.has(base)) return base;
  }

  // 2) Fall back: first non-junk token of first useful segment
  const trySegment = (seg) => {
    let part = seg.trim();
    part = part.replace(
      /^(?:cd\s+(?:[^\s;&|]+|"[^"]*"|'[^']*')\s*(?:&&|;)\s*)+/i,
      ""
    );
    part = part.split(/\|/)[0];
    part = part.split(/&&|\|\||;/)[0].trim();
    while (/^[A-Za-z_][A-Za-z0-9_]*=\S+\s+/.test(part)) {
      part = part.replace(/^[A-Za-z_][A-Za-z0-9_]*=\S+\s+/, "");
    }
    part = part.replace(/^[({]+\s*/, "");
    if (!part || part.startsWith("#")) return null;

    const tokens = part.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
    for (const raw of tokens) {
      if (isJunkTok(raw)) continue;
      return normalizeBase(raw).slice(0, 40);
    }
    return null;
  };

  for (const seg of segments) {
    const hit = trySegment(seg);
    if (hit) return hit;
  }
  return trySegment(cmd) || "shell";
}

/** Label for display: shell tool broken down by command family. */
function toolDisplayKey(call) {
  const fam = shellCommandFamily(call);
  if (fam) return `${call.tool_name} → ${fam}`;
  return call.tool_name || "unknown";
}

function median(nums) {
  if (!nums.length) return null;
  const a = [...nums].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function timelineEventIdForEvidence(ev) {
  if (!ev || !state.pkg) return null;
  const tl = state.pkg.timeline || [];
  if (ev.type === "tool") {
    const hit = tl.find(
      (e) => e.entity_type === "tool" && (e.entity_id === ev.id || e.event_id?.includes(ev.id))
    );
    return hit?.event_id || null;
  }
  if (ev.type === "turn") {
    const hit = tl.find((e) => e.entity_type === "turn" && (e.entity_id === ev.id || e.turn_id === ev.id));
    return hit?.event_id || null;
  }
  if (ev.type === "compaction") {
    const hit = tl.find(
      (e) =>
        (e.kind === "compaction" || e.entity_type === "compaction") &&
        (e.entity_id === ev.id || e.event_id === ev.id || String(e.tokens_before) === String(ev.id))
    );
    return hit?.event_id || tl.find((e) => e.kind === "compaction")?.event_id || null;
  }
  if (ev.type === "event") return ev.id;
  return null;
}

/**
 * Rule-based insights from the existing package schema (no LLM).
 * @returns {{ insights: object[], metrics: object }}
 */
function buildInsights(pkg) {
  const insights = [];
  const turns = pkg.turns?.turns || [];
  const calls = pkg.tools?.calls || [];
  const byName = pkg.tools?.by_name || {};
  const skills = pkg.skills || {};
  const sig = pkg.signals || {};
  const comps = pkg.compactions?.events || [];
  const mcp = pkg.mcp || {};

  const toolWall = calls.reduce((s, c) => s + (Number(c.span?.duration_ms) || 0), 0);
  const turnWall = turns.reduce((s, t) => s + (Number(t.span?.duration_ms) || 0), 0);
  const wallSec = sig.activity?.session_duration_seconds;
  const toolOk = pkg.tools?.effectiveness?.success_rate;
  const nTurns = turns.length;
  const nTools = calls.length;

  // 1) Headline — always
  insights.push({
    id: "headline",
    severity: "info",
    category: "session",
    title: "Session overview",
    detail: [
      `${nTurns} turns`,
      `${nTools} tools`,
      wallSec != null ? fmtDur(wallSec * 1000) + " wall" : null,
      toolWall > 0 ? fmtDur(toolWall) + " tool time" : null,
      toolOk != null ? `${Math.round(toolOk * 100)}% tool success` : null,
    ]
      .filter(Boolean)
      .join(" · "),
    evidence: [],
    suggestion: null,
  });

  // 2) Tool / shell-command time breakdown (one layer under tool name)
  //    Aggregate by display key: "git", "cargo", … for shell; else tool_name
  const familyStats = {};
  for (const c of calls) {
    const fam = shellCommandFamily(c);
    const key = fam ? fam : c.tool_name || "unknown";
    const bucket = (familyStats[key] ||= {
      key,
      tool_name: c.tool_name,
      fam,
      dur: 0,
      n: 0,
      fails: 0,
      calls: [],
    });
    const d = Number(c.span?.duration_ms) || 0;
    bucket.dur += d;
    bucket.n += 1;
    if (c.success === false || ["failure", "blocked", "error"].includes(c.status)) {
      bucket.fails += 1;
    }
    bucket.calls.push(c);
  }
  const famRanks = Object.values(familyStats)
    .filter((r) => r.dur > 0)
    .sort((a, b) => b.dur - a.dur);

  if (famRanks.length && toolWall > 0) {
    const top = famRanks[0];
    const share = top.dur / toolWall;
    const label = top.fam
      ? `shell → ${top.fam}`
      : top.key;
    // Always surface top time consumer at command/tool granularity
    {
      const bestCall = [...top.calls].sort(
        (a, b) => (b.span?.duration_ms || 0) - (a.span?.duration_ms || 0)
      )[0];
      insights.push({
        id: "tool_time_hog",
        severity: share >= 0.35 ? "warn" : "info",
        category: "efficiency",
        title: `${label} used ${Math.round(share * 100)}% of tool wall time`,
        detail: `${top.n} calls · ${fmtDur(top.dur)} total${
          top.fails ? ` · ${top.fails} fail` : ""
        }`,
        evidence: bestCall
          ? [{ type: "tool", id: bestCall.tool_call_id, label: label }]
          : [],
        suggestion:
          share >= 0.35
            ? top.fam
              ? `Most agent tool time is shell \`${top.fam}\` — check those commands for hangs or redundant work.`
              : `Most agent tool time is in ${top.key} — inspect those calls.`
            : null,
      });
    }

    // Shell command breakdown whenever there are multiple shell families
    const shellOnly = famRanks.filter((r) => r.fam);
    const shellWall = shellOnly.reduce((s, r) => s + r.dur, 0);
    if (shellOnly.length >= 2 && shellWall > 0) {
      const topF = shellOnly.slice(0, 6);
      insights.push({
        id: "shell_breakdown",
        severity: "info",
        category: "efficiency",
        title: "Shell time by command",
        detail: topF
          .map((r) => {
            const pct = Math.round((r.dur / shellWall) * 100);
            return `${r.fam} ${pct}% (${r.n}×, ${fmtDur(r.dur)})`;
          })
          .join(" · "),
        evidence: topF[0]?.calls?.length
          ? [
              {
                type: "tool",
                id: [...topF[0].calls].sort(
                  (a, b) => (b.span?.duration_ms || 0) - (a.span?.duration_ms || 0)
                )[0].tool_call_id,
                label: topF[0].fam,
              },
            ]
          : [],
        suggestion: null,
      });
    }
  }

  // 3) Slowest calls — show command family for shell
  let slowest = [...calls]
    .filter((c) => (c.span?.duration_ms || 0) > 0)
    .sort((a, b) => (b.span?.duration_ms || 0) - (a.span?.duration_ms || 0))
    .slice(0, 3);
  if (slowest.length) {
    const lines = slowest
      .map((c) => {
        const fam = shellCommandFamily(c);
        const label = fam ? `${fam}` : c.tool_name;
        return `${label} ${fmtDur(c.span?.duration_ms)}${c.turn_id ? ` (${c.turn_id})` : ""}`;
      })
      .join(" · ");
    insights.push({
      id: "slowest_calls",
      severity: "info",
      category: "efficiency",
      title: "Slowest tool calls",
      detail: lines,
      evidence: slowest[0]
        ? [{ type: "tool", id: slowest[0].tool_call_id, label: slowest[0].tool_name }]
        : [],
      suggestion: null,
    });
  }

  // 4) Tool failures — break shell failures down by command
  const fails = calls.filter(
    (c) => c.success === false || ["failure", "blocked", "error"].includes(c.status)
  );
  if (fails.length) {
    const by = Counter(fails.map((c) => {
      const fam = shellCommandFamily(c);
      return fam ? `shell→${fam}` : c.tool_name;
    }));
    const topFail = Object.entries(by).sort((a, b) => b[1] - a[1])[0];
    insights.push({
      id: "tool_failures",
      severity: fails.length >= 3 ? "alert" : "warn",
      category: "reliability",
      title: `${fails.length} tool failure${fails.length === 1 ? "" : "s"}`,
      detail: Object.entries(by)
        .map(([n, c]) => `${n}×${c}`)
        .join(" · "),
      evidence: [
        {
          type: "tool",
          id: fails[0].tool_call_id,
          label: toolDisplayKey(fails[0]),
        },
      ],
      suggestion: topFail
        ? `Inspect failed ${topFail[0]} on ${fails[0].turn_id || "timeline"}.`
        : null,
    });
  }

  // 5) Heavy turns (input tokens)
  const withTok = turns
    .map((t) => ({
      id: t.turn_id,
      inn: t.tokens?.input_tokens,
      out: t.tokens?.output_tokens,
      dur: t.span?.duration_ms,
      nTools: (t.tool_call_ids || []).length,
    }))
    .filter((t) => t.inn != null);
  withTok.sort((a, b) => b.inn - a.inn);
  if (withTok.length) {
    const top = withTok.slice(0, 3);
    const maxIn = top[0].inn;
    insights.push({
      id: "heavy_turns",
      severity: maxIn >= 100000 ? "warn" : "info",
      category: "efficiency",
      title: "Highest input-token turns",
      detail: top
        .map((t) => `${t.id}: ${t.inn.toLocaleString()} in`)
        .join(" · "),
      evidence: [{ type: "turn", id: top[0].id, label: top[0].id }],
      suggestion: null,
    });
  }

  // 6) Long turns
  const byDur = [...turns]
    .filter((t) => (t.span?.duration_ms || 0) > 0)
    .sort((a, b) => (b.span?.duration_ms || 0) - (a.span?.duration_ms || 0));
  if (byDur.length) {
    const top = byDur.slice(0, 3);
    const maxD = top[0].span.duration_ms;
    insights.push({
      id: "long_turns",
      severity: maxD >= 120000 ? "warn" : "info",
      category: "efficiency",
      title: "Longest turns",
      detail: top
        .map(
          (t) =>
            `${t.turn_id}: ${fmtDur(t.span.duration_ms)} (${(t.tool_call_ids || []).length} tools)`
        )
        .join(" · "),
      evidence: [{ type: "turn", id: top[0].turn_id, label: top[0].turn_id }],
      suggestion: null,
    });
  }

  // 7) TTFT spike
  const ttfts = turns
    .map((t) => ({ id: t.turn_id, ms: t.time_to_first_token_ms }))
    .filter((t) => t.ms != null && t.ms > 0);
  if (ttfts.length >= 2) {
    const med = median(ttfts.map((t) => t.ms));
    const worst = [...ttfts].sort((a, b) => b.ms - a.ms)[0];
    if (med && worst.ms >= 5000 && worst.ms >= 2 * med) {
      insights.push({
        id: "ttft_spike",
        severity: "warn",
        category: "latency",
        title: `Slow first token on ${worst.id}`,
        detail: `TTFT ${fmtDur(worst.ms)} vs median ${fmtDur(med)}`,
        evidence: [{ type: "turn", id: worst.id, label: worst.id }],
        suggestion: null,
      });
    }
  }

  // 8) Compactions
  if (comps.length >= 1) {
    const before = comps.reduce((s, c) => s + (c.tokens_before || 0), 0);
    const after = comps.reduce((s, c) => s + (c.tokens_after || 0), 0);
    const biggest = [...comps].sort(
      (a, b) => (b.tokens_before || 0) - (a.tokens_before || 0)
    )[0];
    insights.push({
      id: "compactions",
      severity: comps.length >= 3 ? "warn" : "info",
      category: "context",
      title: `${comps.length} context compaction${comps.length === 1 ? "" : "s"}`,
      detail: `~${before.toLocaleString()} → ${after.toLocaleString()} tokens (sum before/after)${
        biggest?.turn_id ? ` · largest near ${biggest.turn_id}` : ""
      }`,
      evidence: biggest
        ? biggest.turn_id
          ? [{ type: "turn", id: biggest.turn_id, label: biggest.turn_id }]
          : [{ type: "compaction", id: String(biggest.tokens_before), label: "compaction" }]
        : [],
      suggestion:
        comps.length >= 3
          ? "Context was compacted repeatedly — shorter turns or less tool dump may help."
          : null,
    });
  }

  // 9) Unused skills
  const inv = skills.inventory || [];
  const acts = skills.activations || [];
  const unused = skills.summary?.unused_skill_names || inv.filter((i) => !i.used).map((i) => i.skill_name);
  if (inv.length >= 5 && acts.length === 0) {
    const show = unused.slice(0, 5).join(", ");
    const more = unused.length > 5 ? ` +${unused.length - 5} more` : "";
    insights.push({
      id: "unused_skills",
      severity: "warn",
      category: "harness",
      title: `${inv.length} skills advertised, 0 activated`,
      detail: show ? `${show}${more}` : "No skill activations recorded",
      evidence: [],
      suggestion:
        "Many skills are injected but never activated — trimming inventory may cut prompt tokens.",
    });
  } else if (unused.length >= 5 && acts.length > 0) {
    insights.push({
      id: "unused_skills_partial",
      severity: "info",
      category: "harness",
      title: `${unused.length} skills never activated`,
      detail: unused.slice(0, 5).join(", ") + (unused.length > 5 ? ` +${unused.length - 5} more` : ""),
      evidence: [],
      suggestion: null,
    });
  } else if (acts.length > 0) {
    const used = skills.summary?.skills_used || [...new Set(acts.map((a) => a.skill_name))];
    insights.push({
      id: "skills_used",
      severity: "info",
      category: "harness",
      title: `${acts.length} skill activation${acts.length === 1 ? "" : "s"}`,
      detail: used.join(", "),
      evidence: [],
      suggestion: null,
    });
  }

  // 10) MCP down
  const failedMcp = mcp.summary?.failed_servers || [];
  if (failedMcp.length) {
    insights.push({
      id: "mcp_down",
      severity: "warn",
      category: "reliability",
      title: "MCP servers failed to connect",
      detail: failedMcp.join(", "),
      evidence: [],
      suggestion: `MCP servers failed to connect: ${failedMcp.join(", ")}.`,
    });
  }

  // 11) Cancelled turns
  const cancelled = turns.filter((t) => t.status === "cancelled" || t.outcome === "cancelled");
  if (cancelled.length) {
    insights.push({
      id: "cancelled_turns",
      severity: "info",
      category: "friction",
      title: `${cancelled.length} cancelled turn${cancelled.length === 1 ? "" : "s"}`,
      detail: cancelled.map((t) => t.turn_id).join(", "),
      evidence: [{ type: "turn", id: cancelled[0].turn_id, label: cancelled[0].turn_id }],
      suggestion: `User cancelled ${cancelled[0].turn_id} — possible friction or long wait.`,
    });
  }

  // Rank & cap: headline + prefer shell/tool efficiency cards, then severity
  const headline = insights.filter((i) => i.id === "headline");
  const PRIORITY_IDS = new Set([
    "tool_time_hog",
    "shell_breakdown",
    "tool_failures",
    "slowest_calls",
  ]);
  const rest = insights
    .filter((i) => i.id !== "headline")
    .sort((a, b) => {
      const pa = PRIORITY_IDS.has(a.id) ? 0 : 1;
      const pb = PRIORITY_IDS.has(b.id) ? 0 : 1;
      if (pa !== pb) return pa - pb;
      return (SEV_RANK[a.severity] ?? 9) - (SEV_RANK[b.severity] ?? 9);
    });
  const picked = [];
  const seenCat = new Set();
  for (const ins of rest) {
    if (picked.length >= 7) break;
    if (PRIORITY_IDS.has(ins.id) || !seenCat.has(ins.category) || picked.length < 5) {
      picked.push(ins);
      seenCat.add(ins.category);
    }
  }
  for (const ins of rest) {
    if (picked.length >= 7) break;
    if (!picked.includes(ins)) picked.push(ins);
  }
  const capped = [...headline, ...picked];

  // Dedupe suggestions — max 3 across strip
  let sugLeft = 3;
  for (const i of capped) {
    if (i.suggestion && sugLeft > 0) sugLeft -= 1;
    else if (i.suggestion && sugLeft <= 0) i.suggestion = null;
  }

  return {
    insights: capped,
    metrics: {
      tool_wall_time_ms: toolWall,
      turn_wall_time_ms: turnWall,
      tool_success_rate: toolOk,
      n_turns: nTurns,
      n_tools: nTools,
    },
  };
}

function Counter(arr) {
  const o = {};
  for (const x of arr) o[x] = (o[x] || 0) + 1;
  return o;
}

function renderInsights(pkg) {
  const strip = document.getElementById("insights-strip");
  const cardsEl = document.getElementById("insights-cards");
  const countEl = document.getElementById("insights-count");
  if (!strip || !cardsEl) return;

  const { insights } = buildInsights(pkg);
  state.insights = insights;

  if (!insights.length) {
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  const warns = insights.filter((i) => i.severity === "warn" || i.severity === "alert").length;
  countEl.textContent =
    warns > 0 ? `${insights.length} · ${warns} need attention` : `${insights.length} signals`;

  cardsEl.innerHTML = insights
    .map((ins) => {
      const clickable = (ins.evidence || []).length > 0;
      return `<button type="button" class="insight-card ${ins.severity}${
        clickable ? " clickable" : ""
      }" data-insight="${escapeHtml(ins.id)}">
        <div class="insight-sev">${escapeHtml(ins.severity)}</div>
        <h3>${escapeHtml(ins.title)}</h3>
        <p>${escapeHtml(ins.detail || "")}</p>
        ${
          ins.suggestion
            ? `<div class="insight-suggest">${escapeHtml(ins.suggestion)}</div>`
            : ""
        }
      </button>`;
    })
    .join("");

  cardsEl.querySelectorAll(".insight-card.clickable").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.insight;
      const ins = (state.insights || []).find((x) => x.id === id);
      const ev = ins?.evidence?.[0];
      if (!ev) return;
      const eventId = timelineEventIdForEvidence(ev);
      if (eventId) selectNode(eventId, { scroll: true });
    });
  });
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

  const insightsToggle = document.getElementById("insights-toggle");
  const insightsStrip = document.getElementById("insights-strip");
  if (insightsToggle && insightsStrip) {
    insightsToggle.onclick = () => {
      const collapsed = insightsStrip.classList.toggle("collapsed");
      insightsToggle.textContent = collapsed ? "Show" : "Hide";
    };
  }
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
    renderViewSwitcher(pkg); // fire-and-forget: never blocks the rest of the page on /meta.json
    renderInsights(pkg);

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
