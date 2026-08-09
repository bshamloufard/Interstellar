"""Normalize a raw Grok session directory into session_analysis/1.0 package files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "session_analysis/1.0"
MAX_PREVIEW = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _ms_to_iso(ms: float | int | None) -> str | None:
    """Convert unix epoch milliseconds (or seconds if small) to RFC3339 UTC."""
    if ms is None:
        return None
    try:
        v = float(ms)
    except (TypeError, ValueError):
        return None
    # Heuristic: values < 1e12 are seconds
    if v < 1e12:
        v *= 1000.0
    try:
        return (
            datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _update_event_ts(upd: dict[str, Any], params: dict[str, Any], meta: dict[str, Any]) -> str | None:
    """Best-effort timestamp for an updates.jsonl row."""
    if isinstance(meta, dict) and meta.get("agentTimestampMs") is not None:
        iso = _ms_to_iso(meta.get("agentTimestampMs"))
        if iso:
            return iso
    ts = upd.get("timestamp")
    if ts is not None:
        iso = _ms_to_iso(ts)
        if iso:
            return iso
    return None


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _ms_between(a: str | None, b: str | None) -> float | None:
    da, db = _parse_ts(a), _parse_ts(b)
    if not da or not db:
        return None
    return max(0.0, (db - da).total_seconds() * 1000.0)


def _trunc(text: str | None, n: int = MAX_PREVIEW) -> dict[str, Any] | None:
    if text is None:
        return None
    t = text if isinstance(text, str) else str(text)
    truncated = len(t) > n
    return {
        "text": t[:n] + ("…" if truncated else ""),
        "truncated": truncated,
        "char_count": len(t),
    }


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "summary_text" and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False)[:200])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "summary" in content:
            return _flatten_content(content["summary"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _tool_status(outcome: str | None) -> tuple[str, bool | None]:
    o = (outcome or "unknown").lower()
    if o == "success":
        return "success", True
    if o in ("error", "invalid_tool"):
        return "failure", False
    if o in ("cancelled", "permission_cancelled"):
        return "cancelled", False
    if o in ("permission_rejected", "hook_denied"):
        return "blocked", False
    if o == "followup":
        return "unknown", None
    return "unknown", None


def _turn_status(outcome: str | None) -> tuple[str, bool | None]:
    o = (outcome or "unknown").lower()
    if o == "completed":
        return "success", True
    if o == "error":
        return "failure", False
    if o == "cancelled":
        return "cancelled", False
    return "unknown", None


def _eff(
    successes: int,
    failures: int,
    cancelled: int = 0,
    blocked: int = 0,
    unknown: int = 0,
    total_duration_ms: float | None = None,
    total_tokens: int | None = None,
    outcome_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    attempt = successes + failures + cancelled + blocked + unknown
    return {
        "attempt_count": attempt,
        "success_count": successes,
        "failure_count": failures,
        "cancelled_count": cancelled,
        "blocked_count": blocked,
        "unknown_count": unknown,
        "success_rate": (successes / attempt) if attempt else None,
        "failure_rate": (failures / attempt) if attempt else None,
        "total_duration_ms": total_duration_ms,
        "total_tokens": total_tokens,
        "outcome_counts": outcome_counts or {},
    }


def _camel_to_snake_signals(raw: dict[str, Any]) -> dict[str, Any]:
    """Map product camelCase signals.json into grouped snake_case package shape."""

    def g(*keys: str, default: Any = 0) -> Any:
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
        return default

    return {
        "activity": {
            "turn_count": int(g("turnCount", default=0)),
            "user_message_count": int(g("userMessageCount", default=0)),
            "assistant_message_count": int(g("assistantMessageCount", default=0)),
            "session_duration_seconds": int(g("sessionDurationSeconds", default=0)),
            "long_pauses_count": int(g("longPausesCount", default=0)),
        },
        "friction": {
            "error_count": int(g("errorCount", default=0)),
            "tool_failure_count": int(g("toolFailureCount", default=0)),
            "cancellation_count": int(g("cancellationCount", default=0)),
            "consecutive_cancellations": int(g("consecutiveCancellations", default=0)),
            "regeneration_count": int(g("regenerationCount", default=0)),
            "edit_and_retry_count": int(g("editAndRetryCount", default=0)),
            "has_reverted": bool(g("hasReverted", default=False)),
            "positive_ratings": int(g("positiveRatings", default=0)),
            "negative_ratings": int(g("negativeRatings", default=0)),
        },
        "tools": {
            "tool_call_count": int(g("toolCallCount", default=0)),
            "tools_used": list(g("toolsUsed", default=[]) or []),
        },
        "skills": {
            "skill_call_count": int(g("skillCallCount", default=0)),
            "skills_used": list(g("skillsUsed", default=[]) or []),
        },
        "context": {
            "compaction_count": int(g("compactionCount", default=0)),
            "context_tokens_used": int(g("contextTokensUsed", default=0)),
            "context_window_tokens": int(g("contextWindowTokens", default=0)),
            "context_window_usage": int(g("contextWindowUsage", default=0)),
            "total_tokens_before_compaction": int(
                g("totalTokensBeforeCompaction", default=0)
            ),
        },
        "models": {
            "models_used": list(g("modelsUsed", default=[]) or []),
            "primary_model_id": g("primaryModelId", default=None),
        },
        "outcomes": {
            "git_commit_count": int(g("gitCommitCount", default=0)),
            "pr_created_count": int(g("prCreatedCount", default=0)),
            "pr_merged_count": int(g("prMergedCount", default=0)),
            "bash_bare_echo_count": int(g("bashBareEchoCount", default=0)),
        },
        "latency": {
            "avg_time_to_first_token_ms": float(g("avgTimeToFirstTokenMs", default=0) or 0),
            "avg_response_time_ms": float(g("avgResponseTimeMs", default=0) or 0),
            "min_time_to_first_token_ms": float(g("minTimeToFirstTokenMs", default=0) or 0),
            "max_time_to_first_token_ms": float(g("maxTimeToFirstTokenMs", default=0) or 0),
            "latency_sample_count": int(g("latencySampleCount", default=0)),
            "itl_p50_ms": g("itlP50Ms", default=None),
            "itl_p99_ms": g("itlP99Ms", default=None),
            "itl_max_ms": g("itlMaxMs", default=None),
            "itl_mean_ms": g("itlMeanMs", default=None),
            "itl_sample_count": int(g("itlSampleCount", default=0)),
            "total_chunk_count": int(g("totalChunkCount", default=0)),
        },
        "loc": {
            "agent_lines_added": int(g("agentLinesAdded", default=0)),
            "agent_lines_removed": int(g("agentLinesRemoved", default=0)),
            "agent_lines_added_reverted": int(g("agentLinesAddedReverted", default=0)),
            "agent_lines_removed_reverted": int(g("agentLinesRemovedReverted", default=0)),
            "human_lines_added": int(g("humanLinesAdded", default=0)),
            "human_lines_removed": int(g("humanLinesRemoved", default=0)),
            "human_lines_added_reverted": int(g("humanLinesAddedReverted", default=0)),
            "human_lines_removed_reverted": int(g("humanLinesRemovedReverted", default=0)),
            "agent_files_touched": int(g("agentFilesTouched", default=0)),
            "human_files_touched": int(g("humanFilesTouched", default=0)),
            "total_files_touched": int(g("totalFilesTouched", default=0)),
        },
        "infra": {
            "doom_loop_recovery_attempts": int(g("doomLoopRecoveryAttempts", default=0)),
            "doom_loop_recovery_accepted_after_budget": int(
                g("doomLoopRecoveryAcceptedAfterBudget", default=0)
            ),
            "doom_loop_recovery_top_trigger": g("doomLoopRecoveryTopTrigger", default=None),
            "doom_loop_recovery_aborted_chunks": int(
                g("doomLoopRecoveryAbortedChunks", default=0)
            ),
            "inference_idle_timeouts": int(g("inferenceIdleTimeouts", default=0)),
            "inference_idle_timeout_configured_secs": g(
                "inferenceIdleTimeoutConfiguredSecs", default=None
            ),
            "peak_rss_bytes": int(g("peakRssBytes", default=0)),
        },
    }


@dataclass
class TurnAcc:
    turn_number: int
    turn_id: str
    started_at: str | None = None
    ended_at: str | None = None
    model_id: str = "unknown"
    yolo_mode: bool | None = None
    session_relationship: str | None = None
    redirect_kind: str | None = None
    conversation_message_count: int | None = None
    outcome: str = "unknown"
    cancellation_category: str | None = None
    first_token_at: str | None = None
    loop_count: int = 0
    prompt_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    tool_call_ids: list[str] = field(default_factory=list)
    skill_activation_ids: list[str] = field(default_factory=list)
    mcp_call_ids: list[str] = field(default_factory=list)
    session_id: str | None = None


@dataclass
class ToolAcc:
    tool_call_id: str
    tool_name: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None
    outcome: str = "unknown"
    source: str | None = None
    turn_id: str | None = None
    turn_number: int | None = None
    prompt_id: str | None = None
    title: str | None = None
    namespace: str | None = None
    kind: str | None = None
    read_only: bool | None = None
    input_preview: str | None = None
    result_preview: str | None = None
    permission_decision: str | None = None
    permission_wait_ms: float | None = None
    permission_requested: bool = False
    context_tokens: int | None = None
    related_skill_ids: list[str] = field(default_factory=list)


def normalize_session(
    session_dir: Path,
    out_dir: Path,
    *,
    max_preview: int = MAX_PREVIEW,
) -> Path:
    """Build a session_analysis package at out_dir. Returns out_dir."""
    session_dir = session_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(session_dir / "summary.json") or {}
    signals_raw = _read_json(session_dir / "signals.json") or {}
    prompt_ctx = _read_json(session_dir / "prompt_context.json") or {}
    events = _read_jsonl(session_dir / "events.jsonl")
    updates = _read_jsonl(session_dir / "updates.jsonl")
    chat_rows = _read_jsonl(session_dir / "chat_history.jsonl")

    info = summary.get("info") or {}
    session_id = str(info.get("id") or session_dir.name)
    cwd = str(info.get("cwd") or prompt_ctx.get("working_directory") or "")

    # ── Parse events into turns / tools / skills / mcp ─────────────
    turns: dict[int, TurnAcc] = {}
    current_turn: int | None = None
    tools: dict[str, ToolAcc] = {}
    # pending tool starts without id — match by name FIFO within turn
    pending_tool_starts: list[tuple[str | None, str, str | None]] = []  # (turn_id, name, ts)
    open_tools_by_name: dict[str, list[str]] = defaultdict(list)  # name -> call_ids started

    skills: list[dict[str, Any]] = []
    skill_act_i = 0

    mcp_config: list[dict[str, Any]] = []
    mcp_inits: list[dict[str, Any]] = []
    mcp_server_events: list[dict[str, Any]] = []
    mcp_calls: dict[str, dict[str, Any]] = {}
    compactions: list[dict[str, Any]] = []

    # Synthetic tool ids when events lack tool_call_id on start
    synth_tool_i = 0

    for ev in events:
        et = ev.get("type")
        ts = ev.get("ts")

        if et == "turn_started":
            tn = int(ev.get("turn_number", 0))
            current_turn = tn
            t = TurnAcc(
                turn_number=tn,
                turn_id=f"turn:{tn}",
                started_at=ts,
                model_id=str(ev.get("model_id") or "unknown"),
                yolo_mode=ev.get("yolo_mode"),
                session_relationship=ev.get("session_relationship"),
                redirect_kind=ev.get("redirect_kind"),
                conversation_message_count=ev.get("conversation_message_count"),
                session_id=str(ev.get("session_id") or session_id),
            )
            turns[tn] = t

        elif et == "turn_ended":
            tn = current_turn
            if tn is not None and tn in turns:
                turns[tn].ended_at = ts
                turns[tn].outcome = str(ev.get("outcome") or "unknown")
                turns[tn].cancellation_category = ev.get("cancellation_category")

        elif et == "first_token":
            if current_turn is not None and current_turn in turns:
                turns[current_turn].first_token_at = ts

        elif et == "loop_started":
            if current_turn is not None and current_turn in turns:
                turns[current_turn].loop_count += 1

        elif et == "tool_started":
            name = str(ev.get("tool_name") or "unknown")
            tid = f"turn:{current_turn}" if current_turn is not None else None
            pending_tool_starts.append((tid, name, ts))
            # placeholder until completed gives id
            synth_tool_i += 1
            sid = f"pending:{synth_tool_i}"
            tools[sid] = ToolAcc(
                tool_call_id=sid,
                tool_name=name,
                started_at=ts,
                turn_id=tid,
                turn_number=current_turn,
            )
            open_tools_by_name[name].append(sid)

        elif et == "tool_completed":
            name = str(ev.get("tool_name") or "unknown")
            call_id = str(ev.get("tool_call_id") or "").strip()
            duration = ev.get("duration_ms")
            outcome = str(ev.get("outcome") or "unknown")
            source = ev.get("source")

            # Prefer matching open start by name
            matched_key: str | None = None
            if open_tools_by_name.get(name):
                matched_key = open_tools_by_name[name].pop(0)

            if call_id:
                if matched_key and matched_key in tools:
                    acc = tools.pop(matched_key)
                    acc.tool_call_id = call_id
                    tools[call_id] = acc
                elif call_id not in tools:
                    tools[call_id] = ToolAcc(
                        tool_call_id=call_id,
                        tool_name=name,
                        turn_id=f"turn:{current_turn}" if current_turn is not None else None,
                        turn_number=current_turn,
                    )
                acc = tools[call_id]
            elif matched_key and matched_key in tools:
                acc = tools[matched_key]
                call_id = matched_key
            else:
                synth_tool_i += 1
                call_id = f"anon:{synth_tool_i}"
                tools[call_id] = ToolAcc(
                    tool_call_id=call_id,
                    tool_name=name,
                    turn_id=f"turn:{current_turn}" if current_turn is not None else None,
                    turn_number=current_turn,
                )
                acc = tools[call_id]

            acc.tool_name = name
            acc.completed_at = ts
            if duration is not None:
                try:
                    acc.duration_ms = float(duration)
                except (TypeError, ValueError):
                    pass
            elif acc.started_at and ts:
                acc.duration_ms = _ms_between(acc.started_at, ts)
            acc.outcome = outcome
            if source:
                acc.source = str(source)
            if current_turn is not None and current_turn in turns:
                if call_id not in turns[current_turn].tool_call_ids:
                    turns[current_turn].tool_call_ids.append(call_id)
                acc.turn_id = turns[current_turn].turn_id
                acc.turn_number = current_turn

        elif et == "permission_requested":
            name = str(ev.get("tool_name") or "")
            # mark most recent open/completed of that name
            for acc in reversed(list(tools.values())):
                if acc.tool_name == name and not acc.permission_requested:
                    acc.permission_requested = True
                    break

        elif et == "permission_resolved":
            name = str(ev.get("tool_name") or "")
            for acc in reversed(list(tools.values())):
                if acc.tool_name == name:
                    acc.permission_decision = str(ev.get("decision") or "")
                    try:
                        acc.permission_wait_ms = float(ev.get("wait_ms") or 0)
                    except (TypeError, ValueError):
                        pass
                    acc.permission_requested = True
                    break

        elif et == "skill_activated":
            sid = f"skill_act:{skill_act_i}"
            skill_act_i += 1
            name = str(ev.get("skill_name") or "unknown")
            trigger = str(ev.get("trigger") or "slash_command")
            related = ev.get("related_tool_call_id")
            related_s = str(related) if related else None
            turn_id = f"turn:{current_turn}" if current_turn is not None else None
            # skill may fire just before turn_started — attach to next or current
            act = {
                "skill_activation_id": sid,
                "skill_name": name,
                "trigger": trigger,
                "ts": ts,
                "turn_id": turn_id,
                "turn_number": current_turn,
                "plugin_source": ev.get("plugin_source"),
                "related_tool_call_id": related_s,
            }
            skills.append(act)
            if current_turn is not None and current_turn in turns:
                turns[current_turn].skill_activation_ids.append(sid)
            if related_s and related_s in tools:
                tools[related_s].related_skill_ids.append(sid)

        elif et == "mcp_config_resolved":
            mcp_config.append(
                {
                    "ts": ts,
                    "servers": ev.get("servers") or [],
                    "disabled": ev.get("disabled") or [],
                }
            )

        elif et == "mcp_init_completed":
            started = ts
            # duration_ms on event
            dur = ev.get("duration_ms")
            ended = ts
            mcp_inits.append(
                {
                    "ts": ts,
                    "started_at": started,
                    "ended_at": ended,
                    "duration_ms": float(dur) if dur is not None else 0,
                    "total_servers": int(ev.get("total_servers") or 0),
                    "succeeded": int(ev.get("succeeded") or 0),
                    "failed": int(ev.get("failed") or 0),
                    "auth_required": int(ev.get("auth_required") or 0),
                    "total_tools": int(ev.get("total_tools") or 0),
                    "is_reinit": bool(ev.get("is_reinit")),
                    "failed_servers": list(ev.get("failed_servers") or []),
                }
            )

        elif et in (
            "mcp_server_starting",
            "mcp_server_connected",
            "mcp_server_failed",
            "mcp_server_toggled",
            "mcp_health_check",
            "mcp_transport_error",
            "mcp_transport_decode_error",
            "mcp_transport_reconnect",
            "mcp_auth_retry",
            "mcp_oauth_discovery_timeout",
            "mcp_tool_registration_failed",
        ):
            event_map = {
                "mcp_server_starting": "starting",
                "mcp_server_connected": "connected",
                "mcp_server_failed": "failed",
                "mcp_server_toggled": "toggled",
                "mcp_health_check": "health_check",
                "mcp_transport_error": "transport_error",
                "mcp_transport_decode_error": "transport_decode_error",
                "mcp_transport_reconnect": "transport_reconnect",
                "mcp_auth_retry": "auth_retry",
                "mcp_oauth_discovery_timeout": "oauth_discovery_timeout",
                "mcp_tool_registration_failed": "tool_registration_failed",
            }
            se = event_map.get(et, et)
            success: bool | None
            status: str
            if se == "connected":
                status, success = "success", True
            elif se == "failed":
                status, success = "failure", False
            elif se in ("transport_error", "transport_decode_error", "tool_registration_failed"):
                status, success = "failure", False
            else:
                status, success = "unknown", None
            dur = ev.get("duration_ms")
            mcp_server_events.append(
                {
                    "event": se,
                    "server_name": str(ev.get("server_name") or ""),
                    "ts": ts,
                    "span": {
                        "started_at": ts,
                        "ended_at": ts,
                        "duration_ms": float(dur) if dur is not None else 0,
                    },
                    "status": status,
                    "success": success,
                    "transport": ev.get("transport"),
                    "target": ev.get("target"),
                    "timeout_sec": ev.get("timeout_sec"),
                    "tool_count": ev.get("tool_count"),
                    "tools": ev.get("tools"),
                    "error_type": ev.get("error_type"),
                    "error_message": ev.get("error_message"),
                    "enabled": ev.get("enabled"),
                    "healthy": ev.get("healthy"),
                }
            )

        elif et == "mcp_tool_call_started":
            cid = str(ev.get("call_id") or f"mcp:{len(mcp_calls)}")
            mcp_calls[cid] = {
                "mcp_call_id": cid,
                "server_name": str(ev.get("server_name") or ""),
                "tool_name": str(ev.get("tool_name") or ""),
                "started_at": ts,
                "timeout_sec": ev.get("timeout_sec"),
                "turn_number": current_turn,
                "turn_id": f"turn:{current_turn}" if current_turn is not None else None,
            }

        elif et == "mcp_tool_call_completed":
            cid = str(ev.get("call_id") or "")
            if not cid:
                cid = f"mcp:{len(mcp_calls)}"
            row = mcp_calls.get(cid) or {
                "mcp_call_id": cid,
                "server_name": str(ev.get("server_name") or ""),
                "tool_name": str(ev.get("tool_name") or ""),
            }
            row["completed_at"] = ts
            row["duration_ms"] = ev.get("duration_ms")
            row["success"] = bool(ev.get("success"))
            row["is_timeout"] = ev.get("is_timeout")
            row["error"] = ev.get("error")
            row["reconnect_attempted"] = ev.get("reconnect_attempted")
            row["auth_retry_attempted"] = ev.get("auth_retry_attempted")
            if current_turn is not None:
                row["turn_number"] = current_turn
                row["turn_id"] = f"turn:{current_turn}"
                if cid not in turns[current_turn].mcp_call_ids:
                    turns[current_turn].mcp_call_ids.append(cid)
            mcp_calls[cid] = row

    # Attach orphan skills (before first turn) to turn 0 if exists
    if turns:
        first_tn = min(turns)
        for act in skills:
            if act["turn_id"] is None:
                act["turn_id"] = turns[first_tn].turn_id
                act["turn_number"] = first_tn
                if act["skill_activation_id"] not in turns[first_tn].skill_activation_ids:
                    turns[first_tn].skill_activation_ids.insert(0, act["skill_activation_id"])

    # ── Enrich from updates.jsonl ──────────────────────────────────
    # Text streams survive compaction; chat_history.jsonl often does not.
    tool_meta: dict[str, dict[str, Any]] = {}
    # prompt_id -> accumulated text (and ordered list for turns without prompt id yet)
    prompt_text: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"user": [], "assistant": [], "thought": []}
    )
    # Chunks that arrive before promptId is known — bucket by open turn index
    pending_user_chunks: list[str] = []
    turn_completed_order: list[str] = []  # prompt_ids in completion order
    open_prompt_id: str | None = None

    for upd in updates:
        params = upd.get("params") or {}
        update = params.get("update") or {}
        su = update.get("sessionUpdate") or update.get("session_update")
        meta = (params.get("_meta") or {}) if isinstance(params, dict) else {}
        umeta = update.get("_meta") or {}
        meta_pid = None
        meta_prompt_index: int | None = None
        if isinstance(meta, dict):
            if meta.get("promptId"):
                meta_pid = str(meta["promptId"])
                open_prompt_id = meta_pid
            # ACP user chunks often carry promptIndex (== turn_number) even without promptId
            if meta.get("promptIndex") is not None:
                try:
                    meta_prompt_index = int(meta["promptIndex"])
                except (TypeError, ValueError):
                    meta_prompt_index = None
        if isinstance(umeta, dict) and umeta.get("promptIndex") is not None and meta_prompt_index is None:
            try:
                meta_prompt_index = int(umeta["promptIndex"])
            except (TypeError, ValueError):
                pass
        # Also check update-level _meta on user_message
        upd_meta = update.get("_meta") if isinstance(update.get("_meta"), dict) else {}
        if upd_meta.get("promptIndex") is not None and meta_prompt_index is None:
            try:
                meta_prompt_index = int(upd_meta["promptIndex"])
            except (TypeError, ValueError):
                pass

        def _chunk_text() -> str:
            c = update.get("content") or {}
            if isinstance(c, dict):
                return str(c.get("text") or "")
            return str(c or "")

        def _resolve_pid(*, prefer_index: bool = False) -> str | None:
            """Bucket key for stream text.

            User chunks often lack promptId and must NOT inherit the previous
            turn's open_prompt_id (that caused off-by-one previews). Prefer
            promptIndex → turnidx:{n} for user messages.
            """
            if prefer_index and meta_prompt_index is not None:
                return f"turnidx:{meta_prompt_index}"
            if meta_pid:
                return meta_pid
            if meta_prompt_index is not None:
                return f"turnidx:{meta_prompt_index}"
            if open_prompt_id:
                return open_prompt_id
            return None

        if su == "user_message_chunk":
            text = _chunk_text()
            if not text:
                continue
            pid = _resolve_pid(prefer_index=True)
            if pid:
                if pending_user_chunks:
                    prompt_text[pid]["user"].extend(pending_user_chunks)
                    pending_user_chunks.clear()
                prompt_text[pid]["user"].append(text)
            else:
                pending_user_chunks.append(text)

        elif su == "agent_message_chunk":
            text = _chunk_text()
            if not text:
                continue
            pid = _resolve_pid(prefer_index=False)
            if pid:
                # Do not flush pending user into agent UUID buckets — users are
                # keyed by promptIndex; flushing here caused wrong-turn previews.
                prompt_text[pid]["assistant"].append(text)

        elif su == "agent_thought_chunk":
            text = _chunk_text()
            if not text:
                continue
            pid = _resolve_pid(prefer_index=False)
            if pid:
                prompt_text[pid]["thought"].append(text)

        elif su == "turn_completed":
            prompt_id = update.get("prompt_id") or meta_pid or open_prompt_id
            usage = update.get("usage")
            stop = update.get("stop_reason")
            pid_s = str(prompt_id) if prompt_id else None
            if pid_s:
                turn_completed_order.append(pid_s)
                open_prompt_id = pid_s
                if pending_user_chunks:
                    prompt_text[pid_s]["user"].extend(pending_user_chunks)
                    pending_user_chunks.clear()
            # attach to turns in wire order (first turn still missing usage)
            for tn in sorted(turns.keys()):
                t = turns[tn]
                if t.usage is None:
                    t.usage = usage if isinstance(usage, dict) else None
                    t.prompt_id = pid_s or t.prompt_id
                    t.stop_reason = str(stop) if stop else t.stop_reason
                    break
            else:
                if turns:
                    t = turns[max(turns)]
                    t.usage = usage if isinstance(usage, dict) else t.usage
                    if pid_s:
                        t.prompt_id = pid_s
                    if stop:
                        t.stop_reason = str(stop)

        elif su == "auto_compact_completed":
            cts = _update_event_ts(upd, params, meta if isinstance(meta, dict) else {})
            compactions.append(
                {
                    "ts": cts or _now_iso(),
                    "tokens_before": int(update.get("tokens_before") or 0),
                    "tokens_after": int(update.get("tokens_after") or 0),
                    "source": "auto_compact_completed",
                    "summary_preview": update.get("summary_preview"),
                    "turn_id": None,
                    "turn_number": None,
                }
            )

        elif su in ("tool_call", "tool_call_update"):
            tcid = update.get("toolCallId") or update.get("tool_call_id")
            if not tcid:
                continue
            tcid = str(tcid)
            entry = tool_meta.setdefault(tcid, {})
            ets = _update_event_ts(upd, params, meta if isinstance(meta, dict) else {})
            # ACP status on tool rows: Pending | InProgress | Completed | ...
            status = None
            if isinstance(meta, dict):
                up = meta.get("updateParams") if isinstance(meta.get("updateParams"), dict) else {}
                status = up.get("status") or update.get("status")
            if status:
                entry["acp_status"] = str(status)
            if ets:
                st = str(status or "").lower()
                # start on first sighting / in-progress; end on completed
                if st in ("", "pending", "inprogress", "in_progress") or su == "tool_call":
                    entry.setdefault("started_at", ets)
                if st in ("completed", "failed", "error", "cancelled"):
                    entry["completed_at"] = ets
                    if st == "completed":
                        entry["outcome_hint"] = "success"
                    elif st in ("failed", "error"):
                        entry["outcome_hint"] = "error"
                    elif st == "cancelled":
                        entry["outcome_hint"] = "cancelled"
                else:
                    # keep advancing end so multi-update tools get a window
                    entry.setdefault("started_at", ets)
                    entry["last_seen_at"] = ets
            if update.get("title"):
                entry["title"] = update["title"]
            raw_in = update.get("rawInput") or update.get("raw_input")
            if raw_in is not None:
                try:
                    entry["input_preview"] = json.dumps(raw_in, ensure_ascii=False)[
                        : max_preview
                    ]
                except (TypeError, ValueError):
                    entry["input_preview"] = str(raw_in)[:max_preview]
                if isinstance(raw_in, dict):
                    if raw_in.get("variant"):
                        entry["variant"] = str(raw_in["variant"])
                    if raw_in.get("backend") is True:
                        entry["backend"] = True
                    if raw_in.get("query"):
                        entry["query"] = str(raw_in["query"])
            xai = (umeta.get("x.ai/tool") if isinstance(umeta, dict) else None) or {}
            if isinstance(xai, dict):
                if xai.get("namespace"):
                    entry["namespace"] = xai["namespace"]
                if xai.get("kind"):
                    entry["kind"] = xai["kind"]
                if "read_only" in xai:
                    entry["read_only"] = xai["read_only"]
                if xai.get("name"):
                    entry["tool_name"] = str(xai["name"])
            if isinstance(meta, dict) and meta.get("totalTokens") is not None:
                try:
                    entry["context_tokens"] = int(meta["totalTokens"])
                except (TypeError, ValueError):
                    pass
            if isinstance(meta, dict) and meta.get("promptId"):
                entry["prompt_id"] = str(meta["promptId"])

            # Derive a clean tool name (avoid "Web search:" title noise)
            def _nice_tool_name() -> str:
                if entry.get("tool_name"):
                    return entry["tool_name"]
                if entry.get("variant"):
                    # WebSearch -> web_search
                    v = entry["variant"]
                    out = []
                    for i, ch in enumerate(v):
                        if ch.isupper() and i > 0:
                            out.append("_")
                        out.append(ch.lower())
                    return "".join(out)
                title = (entry.get("title") or update.get("title") or "").strip()
                if title.endswith(":"):
                    title = title[:-1].strip()
                if title:
                    return title.lower().replace(" ", "_")
                return "unknown"

            # Ensure tool exists
            if tcid not in tools:
                tools[tcid] = ToolAcc(
                    tool_call_id=tcid,
                    tool_name=_nice_tool_name(),
                    title=update.get("title"),
                )
            acc = tools[tcid]
            acc.tool_name = _nice_tool_name()
            if entry.get("title"):
                acc.title = entry["title"]
            if entry.get("input_preview"):
                acc.input_preview = entry["input_preview"]
            if entry.get("namespace"):
                acc.namespace = entry["namespace"]
            if entry.get("kind"):
                acc.kind = entry["kind"]
            if "read_only" in entry:
                acc.read_only = entry["read_only"]
            if entry.get("context_tokens") is not None:
                acc.context_tokens = entry["context_tokens"]
            if entry.get("prompt_id"):
                acc.prompt_id = entry["prompt_id"]
            # Prefer event-log times; fill gaps from updates stream timestamps
            if not acc.started_at and entry.get("started_at"):
                acc.started_at = entry["started_at"]
            end_ts = entry.get("completed_at") or entry.get("last_seen_at")
            if end_ts and not acc.completed_at:
                acc.completed_at = end_ts
            elif end_ts and acc.outcome == "unknown":
                # backend tools never get events.tool_completed — allow end to advance
                acc.completed_at = end_ts
            if entry.get("outcome_hint") and acc.outcome == "unknown":
                acc.outcome = entry["outcome_hint"]
            if (
                (acc.duration_ms is None or acc.duration_ms == 0)
                and acc.started_at
                and acc.completed_at
            ):
                d = _ms_between(acc.started_at, acc.completed_at)
                if d is not None:
                    acc.duration_ms = d

    # prompt_id → turn (for parenting backend tools that never hit events.jsonl)
    prompt_id_to_turn: dict[str, tuple[int, str]] = {}
    for tn, t in turns.items():
        if t.prompt_id:
            prompt_id_to_turn[t.prompt_id] = (tn, t.turn_id)

    def _nearest_turn(ts_iso: str | None) -> tuple[int, str] | None:
        if not ts_iso:
            return None
        cts = _parse_ts(ts_iso)
        if not cts:
            return None
        best: tuple[int, str] | None = None
        best_dist: float | None = None
        for tn, t in turns.items():
            if not t.started_at:
                continue
            s = _parse_ts(t.started_at)
            e = _parse_ts(t.ended_at or t.started_at) or s
            if not s:
                continue
            if e and s <= cts <= e:
                dist = 0.0
            elif cts < s:
                dist = (s - cts).total_seconds()
            else:
                dist = (cts - (e or s)).total_seconds()
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (tn, t.turn_id)
        if best is not None and best_dist is not None and best_dist < 3600:
            return best
        return None

    # Attach orphan tools to turns via prompt_id or nearest timestamp
    for call_id, acc in list(tools.items()):
        if acc.turn_id is not None:
            continue
        linked = None
        if acc.prompt_id and acc.prompt_id in prompt_id_to_turn:
            linked = prompt_id_to_turn[acc.prompt_id]
        if not linked:
            linked = _nearest_turn(acc.started_at or acc.completed_at)
        if linked:
            tn, tid = linked
            acc.turn_id = tid
            acc.turn_number = tn
            if call_id not in turns[tn].tool_call_ids:
                turns[tn].tool_call_ids.append(call_id)

    # Attach each compaction to the nearest turn by timestamp (for nesting/context)
    for c in compactions:
        linked = _nearest_turn(c.get("ts"))
        if linked:
            c["turn_number"] = linked[0]
            c["turn_id"] = linked[1]

    # ── Chat ───────────────────────────────────────────────────────
    messages: list[dict[str, Any]] = []
    by_role: dict[str, list[int]] = defaultdict(list)
    by_turn_msgs: dict[str, list[int]] = defaultdict(list)
    by_tool_msg: dict[str, int] = {}
    inventory: list[dict[str, Any]] = []
    inventory_seen: set[str] = set()

    # Map prompt_index on user messages to turn numbers when present
    prompt_index_to_turn: dict[int, int] = {}
    for tn, t in turns.items():
        # weak: turn_number often equals prompt order
        prompt_index_to_turn[tn] = tn

    for i, row in enumerate(chat_rows):
        raw_type = str(row.get("type") or "other")
        role_map = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "reasoning": "reasoning",
            "tool_result": "tool_result",
            "backend_tool_call": "backend_tool_call",
        }
        role = role_map.get(raw_type, "other")
        mid = f"msg:{i}"
        content = row.get("content")
        text = _flatten_content(content)
        if role == "reasoning" and not text:
            text = _flatten_content(row.get("summary"))

        turn_id = None
        turn_number = None
        if "prompt_index" in row and row["prompt_index"] is not None:
            try:
                pi = int(row["prompt_index"])
                if pi in turns:
                    turn_number = pi
                    turn_id = f"turn:{pi}"
            except (TypeError, ValueError):
                pass

        tool_calls_out = []
        for tc in row.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tcid = str(tc.get("id") or "")
            tname = str(tc.get("name") or "")
            args = tc.get("arguments")
            args_text = args if isinstance(args, str) else (
                json.dumps(args, ensure_ascii=False) if args is not None else None
            )
            tool_calls_out.append(
                {
                    "tool_call_id": tcid,
                    "tool_name": tname,
                    "arguments_text": (args_text[:max_preview] if args_text else None),
                }
            )
            if tcid:
                by_tool_msg[tcid] = i
                # enrich tool
                if tcid not in tools:
                    tools[tcid] = ToolAcc(tool_call_id=tcid, tool_name=tname or "unknown")
                if args_text and not tools[tcid].input_preview:
                    tools[tcid].input_preview = args_text[:max_preview]
                if turn_id:
                    tools[tcid].turn_id = turn_id
                    tools[tcid].turn_number = turn_number

        tr_id = row.get("tool_call_id")
        if tr_id:
            tr_id = str(tr_id)
            by_tool_msg[tr_id] = i
            if tr_id in tools and text and not tools[tr_id].result_preview:
                tools[tr_id].result_preview = text[:max_preview]

        # skill inventory from system_reminder
        if role == "user" and row.get("synthetic_reason") == "system_reminder":
            for m in re.finditer(
                r"- ([a-zA-Z0-9_-]+): (.+?)(?:\n  Absolute path: (.+))?$",
                text,
                re.M,
            ):
                sname, desc, path = m.group(1), m.group(2).strip(), m.group(3)
                if sname not in inventory_seen:
                    inventory_seen.add(sname)
                    inventory.append(
                        {
                            "skill_name": sname,
                            "description_preview": desc[:200],
                            "path": path.strip() if path else None,
                            "used": False,
                            "activation_count": 0,
                        }
                    )

        msg = {
            "message_id": mid,
            "index": i,
            "role": role,
            "raw_type": raw_type,
            "turn_id": turn_id,
            "turn_number": turn_number,
            "model_id": row.get("model_id"),
            "synthetic_reason": row.get("synthetic_reason"),
            "text": text if len(text) <= 200_000 else text[:200_000] + "…",
            "content": content if content is not None else row.get("summary"),
            "char_count": len(text),
            "tool_calls": tool_calls_out or None,
            "tool_call_id": str(tr_id) if tr_id else None,
        }
        # drop null tool_calls
        if not msg["tool_calls"]:
            del msg["tool_calls"]
        messages.append(msg)
        by_role[role].append(i)
        if turn_id:
            by_turn_msgs[turn_id].append(i)

    # Attribute unassigned user/assistant/reasoning messages to turns by order
    # Simple pass: after last system/meta, bucket by assistant boundaries
    if turns and messages:
        # Assign messages that have prompt_index already done; for others near tools
        for acc in tools.values():
            if acc.turn_id and acc.tool_call_id in by_tool_msg:
                mi = by_tool_msg[acc.tool_call_id]
                messages[mi]["turn_id"] = acc.turn_id
                messages[mi]["turn_number"] = acc.turn_number
                if mi not in by_turn_msgs[acc.turn_id]:
                    by_turn_msgs[acc.turn_id].append(mi)

    # Heuristic: assign user messages with skill_information / user_query to turns
    user_turn_msgs = [
        m
        for m in messages
        if m["role"] == "user"
        and m.get("turn_id") is None
        and m.get("synthetic_reason") is None
        and (
            "<user_query>" in (m.get("text") or "")
            or (m.get("text") or "").strip().startswith("/")
        )
    ]
    sorted_tns = sorted(turns.keys())
    for idx, m in enumerate(user_turn_msgs):
        if idx < len(sorted_tns):
            tn = sorted_tns[idx]
            m["turn_id"] = f"turn:{tn}"
            m["turn_number"] = tn
            by_turn_msgs[m["turn_id"]].append(m["index"])

    # Assign following reasoning/assistant until next user_query to same turn
    current_assign: str | None = None
    for m in messages:
        if m["role"] == "user" and m.get("turn_id"):
            current_assign = m["turn_id"]
        elif m.get("turn_id") is None and current_assign and m["role"] in (
            "assistant",
            "reasoning",
            "tool_result",
        ):
            m["turn_id"] = current_assign
            m["turn_number"] = int(current_assign.split(":")[1])
            by_turn_msgs[current_assign].append(m["index"])
        elif m["role"] == "user" and m.get("synthetic_reason"):
            pass

    # Rebuild by_turn from messages
    by_turn_msgs = defaultdict(list)
    for m in messages:
        if m.get("turn_id"):
            by_turn_msgs[m["turn_id"]].append(m["index"])

    # Previews from chat (may be incomplete after compaction)
    turn_previews: dict[str, dict[str, Any]] = defaultdict(dict)
    for m in messages:
        tid = m.get("turn_id")
        if not tid:
            continue
        prev = _trunc(m.get("text"), max_preview)
        if not prev:
            continue
        prev["source"] = m["role"] + (
            "_message" if m["role"] in ("user", "assistant") else ""
        )
        if m["role"] == "reasoning":
            prev["source"] = "reasoning_summary"
        prev["chat_message_id"] = m["message_id"]
        if m["role"] == "user" and "user_preview" not in turn_previews[tid]:
            if m.get("synthetic_reason") is None:
                turn_previews[tid]["user_preview"] = prev
        elif m["role"] == "assistant":
            turn_previews[tid]["agent_preview"] = prev
        elif m["role"] == "reasoning" and "reasoning_preview" not in turn_previews[tid]:
            turn_previews[tid]["reasoning_preview"] = prev

    # Fill / override previews from updates.jsonl streams (authoritative after compaction)
    # Map prompt_id -> turn via turn.prompt_id, else by turn_completed order.
    prompt_to_turn: dict[str, str] = {}
    for tn in sorted(turns.keys()):
        t = turns[tn]
        if t.prompt_id:
            prompt_to_turn[t.prompt_id] = t.turn_id
    # If some completions weren't linked, zip ordered prompt ids to turns missing text
    ordered_tns = sorted(turns.keys())
    for i, pid in enumerate(turn_completed_order):
        if pid not in prompt_to_turn and i < len(ordered_tns):
            # prefer matching by index when prompt_id missing on turn
            tn = ordered_tns[i]
            if turns[tn].prompt_id is None:
                turns[tn].prompt_id = pid
            prompt_to_turn.setdefault(pid, turns[tn].turn_id)

    # Also zip by completion order onto turns in order (handles mismatched ids)
    if turn_completed_order and ordered_tns:
        for i, tn in enumerate(ordered_tns):
            if i < len(turn_completed_order):
                pid = turn_completed_order[i]
                if not turns[tn].prompt_id:
                    turns[tn].prompt_id = pid
                prompt_to_turn[pid] = turns[tn].turn_id

    def _join_chunks(parts: list[str]) -> str:
        return "".join(parts).strip()

    def _apply_preview(tid: str, buckets: dict[str, list[str]]) -> None:
        user_t = _join_chunks(buckets["user"])
        asst_t = _join_chunks(buckets["assistant"])
        thought_t = _join_chunks(buckets["thought"])
        if user_t:
            prev = _trunc(user_t, max_preview)
            if prev:
                prev["source"] = "updates_user_message"
                turn_previews[tid]["user_preview"] = prev
        if asst_t:
            prev = _trunc(asst_t, max_preview)
            if prev:
                prev["source"] = "updates_agent_message"
                turn_previews[tid]["agent_preview"] = prev
        if thought_t:
            prev = _trunc(thought_t, max_preview)
            if prev:
                prev["source"] = "updates_agent_thought"
                # prefer full thought from updates over short chat remnant
                turn_previews[tid]["reasoning_preview"] = prev

    # 1) promptIndex buckets are the most reliable turn alignment (turn_number == promptIndex)
    for tn in sorted(turns.keys()):
        idx_key = f"turnidx:{tn}"
        if idx_key in prompt_text:
            _apply_preview(turns[tn].turn_id, prompt_text[idx_key])

    # 2) UUID prompt buckets fill gaps (assistant/thought often only tagged with promptId)
    for pid, buckets in prompt_text.items():
        if pid.startswith("turnidx:"):
            continue
        tid = prompt_to_turn.get(pid)
        if not tid:
            continue
        existing = turn_previews.get(tid) or {}
        # only fill sides still missing so we don't clobber promptIndex user text
        partial = {
            "user": buckets["user"] if "user_preview" not in existing else [],
            "assistant": buckets["assistant"] if "agent_preview" not in existing else [],
            "thought": buckets["thought"] if "reasoning_preview" not in existing else [],
        }
        if partial["user"] or partial["assistant"] or partial["thought"]:
            _apply_preview(tid, partial)

    # ── Build package files ────────────────────────────────────────
    related_base = {
        "manifest": "manifest.json",
        "session": "session.json",
        "signals": "signals.json",
        "turns": "turns.json",
        "tools": "tools.json",
        "skills": "skills.json",
        "mcp": "mcp.json",
        "prompts": "prompts.json",
        "chat": "chat.json",
        "context": "context.json",
        "compactions": "compactions.json",
        "timeline": "timeline.jsonl",
    }

    # session.json
    session_obj = {
        "kind": "session",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "id": session_id,
        "cwd": cwd,
        "title": summary.get("generated_title") or summary.get("session_summary"),
        "session_summary": summary.get("session_summary"),
        "generated_title": summary.get("generated_title"),
        "created_at": summary.get("created_at") or _now_iso(),
        "updated_at": summary.get("updated_at") or _now_iso(),
        "last_active_at": summary.get("last_active_at"),
        "duration_seconds": (signals_raw or {}).get("sessionDurationSeconds"),
        "model_id": summary.get("current_model_id"),
        "models_used": (signals_raw or {}).get("modelsUsed") or [],
        "agent_name": summary.get("agent_name"),
        "sandbox_profile": summary.get("sandbox_profile"),
        "reasoning_effort": summary.get("reasoning_effort"),
        "num_messages": summary.get("num_messages"),
        "num_chat_messages": summary.get("num_chat_messages"),
        "request_id": summary.get("request_id"),
        "grok_home": summary.get("grok_home"),
        "chat_format_version": summary.get("chat_format_version"),
        "next_trace_turn": summary.get("next_trace_turn"),
    }

    # signals
    signals_obj = {
        "kind": "signals",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        **_camel_to_snake_signals(signals_raw if isinstance(signals_raw, dict) else {}),
    }

    # tools package
    tool_calls_out: list[dict[str, Any]] = []
    by_tool_call_id: dict[str, int] = {}
    by_name_acc: dict[str, dict[str, Any]] = {}

    # Drop unresolved pending:* that never completed if we have real ids — keep all with data
    final_tools = {
        k: v
        for k, v in tools.items()
        if not k.startswith("pending:") or v.completed_at or v.duration_ms is not None
    }
    # If pending still there and no complete, keep for visibility
    if not final_tools:
        final_tools = tools

    for call_id, acc in final_tools.items():
        status, success = _tool_status(acc.outcome)
        if acc.duration_ms is None and acc.started_at and acc.completed_at:
            acc.duration_ms = _ms_between(acc.started_at, acc.completed_at)
        tl_id = f"tl:tool:{call_id}"
        span = {
            "started_at": acc.started_at or acc.completed_at or _now_iso(),
            "ended_at": acc.completed_at or acc.started_at,
            "duration_ms": acc.duration_ms if acc.duration_ms is not None else 0,
        }
        tokens = None
        if acc.context_tokens is not None:
            tokens = {"context_tokens": acc.context_tokens, "source": "updates_meta"}

        perm = None
        if acc.permission_requested or acc.permission_decision:
            perm = {
                "requested": acc.permission_requested,
                "decision": acc.permission_decision,
                "wait_ms": acc.permission_wait_ms,
            }

        chat_ids = []
        if call_id in by_tool_msg:
            chat_ids.append(f"msg:{by_tool_msg[call_id]}")

        row = {
            "tool_call_id": call_id,
            "timeline_event_id": tl_id,
            "uri": f"sa://tools/{call_id}",
            "tool_name": acc.tool_name,
            "namespace": acc.namespace,
            "kind": acc.kind,
            "read_only": acc.read_only,
            "title": acc.title,
            "span": span,
            "status": status,
            "success": success,
            "outcome": acc.outcome,
            "source": acc.source,
            "turn_id": acc.turn_id,
            "turn_number": acc.turn_number,
            "prompt_id": acc.prompt_id,
            "permission": perm,
            "input_preview": (
                {**_trunc(acc.input_preview, max_preview), "source": "tool_args"}  # type: ignore
                if acc.input_preview
                else None
            ),
            "result_preview": (
                {**_trunc(acc.result_preview, max_preview), "source": "tool_result"}  # type: ignore
                if acc.result_preview
                else None
            ),
            "tokens": tokens,
            "chat_message_ids": chat_ids or None,
            "related_skill_activation_ids": acc.related_skill_ids or None,
            "links": {
                "timeline": f"sa://timeline/{tl_id}",
                "turn": f"sa://turns/{acc.turn_id}" if acc.turn_id else None,
            },
        }
        # strip Nones in nested optional lists
        if not row["chat_message_ids"]:
            del row["chat_message_ids"]
        if not row["related_skill_activation_ids"]:
            del row["related_skill_activation_ids"]

        by_tool_call_id[call_id] = len(tool_calls_out)
        tool_calls_out.append(row)

        bn = by_name_acc.setdefault(
            acc.tool_name,
            {
                "tool_name": acc.tool_name,
                "ids": [],
                "ok": 0,
                "fail": 0,
                "cancel": 0,
                "block": 0,
                "unk": 0,
                "dur": 0.0,
                "outcomes": defaultdict(int),
            },
        )
        bn["ids"].append(call_id)
        bn["outcomes"][acc.outcome] += 1
        if success is True:
            bn["ok"] += 1
        elif status == "failure":
            bn["fail"] += 1
        elif status == "cancelled":
            bn["cancel"] += 1
        elif status == "blocked":
            bn["block"] += 1
        else:
            bn["unk"] += 1
        if acc.duration_ms:
            bn["dur"] += float(acc.duration_ms)

    by_name = {
        name: {
            "tool_name": name,
            "effectiveness": _eff(
                v["ok"],
                v["fail"],
                v["cancel"],
                v["block"],
                v["unk"],
                total_duration_ms=v["dur"],
                outcome_counts=dict(v["outcomes"]),
            ),
            "tool_call_ids": v["ids"],
        }
        for name, v in sorted(by_name_acc.items())
    }

    tot_ok = sum(1 for r in tool_calls_out if r["success"] is True)
    tot_fail = sum(1 for r in tool_calls_out if r["status"] == "failure")
    tot_cancel = sum(1 for r in tool_calls_out if r["status"] == "cancelled")
    tot_block = sum(1 for r in tool_calls_out if r["status"] == "blocked")
    tot_unk = sum(1 for r in tool_calls_out if r["status"] == "unknown")
    tot_dur = sum(float(r["span"]["duration_ms"] or 0) for r in tool_calls_out)
    slowest = sorted(
        tool_calls_out, key=lambda r: float(r["span"]["duration_ms"] or 0), reverse=True
    )[:10]

    tools_obj = {
        "kind": "tools",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "calls": tool_calls_out,
        "by_tool_call_id": by_tool_call_id,
        "by_name": by_name,
        "effectiveness": _eff(
            tot_ok, tot_fail, tot_cancel, tot_block, tot_unk, total_duration_ms=tot_dur
        ),
        "summary": {
            "call_count": len(tool_calls_out),
            "distinct_tool_count": len(by_name),
            "total_duration_ms": tot_dur,
            "total_tokens": None,
            "slowest_calls": [
                {
                    "tool_call_id": r["tool_call_id"],
                    "tool_name": r["tool_name"],
                    "duration_ms": r["span"]["duration_ms"],
                    "status": r["status"],
                    "turn_id": r.get("turn_id"),
                }
                for r in slowest
            ],
        },
    }

    # skills package
    used_counts: dict[str, int] = defaultdict(int)
    skill_rows: list[dict[str, Any]] = []
    by_act: dict[str, int] = {}
    sk_ok = sk_fail = sk_other = 0
    for act in skills:
        used_counts[act["skill_name"]] += 1
        rel = act.get("related_tool_call_id")
        related_tool = None
        duration_ms = 0.0
        started = act.get("ts")
        ended = act.get("ts")
        status, success = "unknown", None
        if rel and rel in by_tool_call_id:
            tr = tool_calls_out[by_tool_call_id[rel]]
            related_tool = {
                "tool_name": tr["tool_name"],
                "duration_ms": tr["span"]["duration_ms"],
                "outcome": tr["outcome"],
                "status": tr["status"],
                "success": tr["success"],
            }
            duration_ms = float(tr["span"]["duration_ms"] or 0)
            started = tr["span"]["started_at"]
            ended = tr["span"]["ended_at"]
            status, success = tr["status"], tr["success"]
        else:
            # slash: success from turn
            tid = act.get("turn_id")
            tn = act.get("turn_number")
            if tn is not None and tn in turns:
                st, su = _turn_status(turns[tn].outcome)
                status, success = st, su
            duration_ms = 0.0

        if success is True:
            sk_ok += 1
        elif status == "failure":
            sk_fail += 1
        else:
            sk_other += 1

        tl_id = f"tl:skill:{act['skill_activation_id']}"
        agent_prev = None
        user_args = None
        tid = act.get("turn_id")
        if tid and tid in turn_previews:
            agent_prev = turn_previews[tid].get("agent_preview")
            user_args = turn_previews[tid].get("user_preview")

        tokens = None
        if tn is not None and tn in turns and turns[tn].usage:
            u = turns[tn].usage or {}
            tokens = {
                "input_tokens": u.get("inputTokens") or u.get("input_tokens"),
                "output_tokens": u.get("outputTokens") or u.get("output_tokens"),
                "total_tokens": u.get("totalTokens") or u.get("total_tokens"),
                "reasoning_tokens": u.get("reasoningTokens") or u.get("reasoning_tokens"),
                "cached_read_tokens": u.get("cachedReadTokens") or u.get("cached_read_tokens"),
                "source": "turn_completed",
            }

        chat_ids = list(by_turn_msgs.get(tid or "", []))
        chat_mids = [f"msg:{i}" for i in chat_ids[:20]]

        row = {
            "skill_activation_id": act["skill_activation_id"],
            "timeline_event_id": tl_id,
            "uri": f"sa://skills/{act['skill_activation_id']}",
            "skill_name": act["skill_name"],
            "trigger": act["trigger"],
            "span": {
                "started_at": started,
                "ended_at": ended,
                "duration_ms": duration_ms,
            },
            "status": status,
            "success": success,
            "turn_id": tid,
            "turn_number": act.get("turn_number"),
            "plugin_source": act.get("plugin_source"),
            "related_tool_call_id": rel,
            "related_tool": related_tool,
            "args_preview": user_args,
            "agent_preview": agent_prev,
            "tokens": tokens,
            "chat_message_ids": chat_mids or None,
            "links": {
                "timeline": f"sa://timeline/{tl_id}",
                "turn": f"sa://turns/{tid}" if tid else None,
                "tool_call": f"sa://tools/{rel}" if rel else None,
                "chat_file": "chat.json",
            },
        }
        if not row["chat_message_ids"]:
            del row["chat_message_ids"]
        by_act[act["skill_activation_id"]] = len(skill_rows)
        skill_rows.append(row)

    for inv in inventory:
        c = used_counts.get(inv["skill_name"], 0)
        inv["used"] = c > 0
        inv["activation_count"] = c
    # ensure used skills appear in inventory
    for name, c in used_counts.items():
        if name not in inventory_seen:
            inventory.append(
                {
                    "skill_name": name,
                    "description_preview": None,
                    "path": None,
                    "used": True,
                    "activation_count": c,
                }
            )

    by_skill_name: dict[str, Any] = {}
    for row in skill_rows:
        n = row["skill_name"]
        slot = by_skill_name.setdefault(
            n,
            {
                "skill_name": n,
                "ids": [],
                "ok": 0,
                "fail": 0,
                "other": 0,
                "trig": {"slash_command": 0, "skill_md_read": 0, "skill_tool": 0},
                "first": row["span"]["started_at"],
                "last": row["span"]["started_at"],
                "turn_ok": 0,
                "turn_n": 0,
            },
        )
        slot["ids"].append(row["skill_activation_id"])
        slot["trig"][row["trigger"]] = slot["trig"].get(row["trigger"], 0) + 1
        if row["success"] is True:
            slot["ok"] += 1
        elif row["status"] == "failure":
            slot["fail"] += 1
        else:
            slot["other"] += 1
        slot["last"] = row["span"]["started_at"]
        tn = row.get("turn_number")
        if tn is not None and tn in turns:
            slot["turn_n"] += 1
            if turns[tn].outcome == "completed":
                slot["turn_ok"] += 1

    skills_by_name = {
        n: {
            "skill_name": n,
            "effectiveness": _eff(
                v["ok"],
                v["fail"],
                unknown=v["other"],
            ),
            "by_trigger": v["trig"],
            "skill_activation_ids": v["ids"],
            "first_started_at": v["first"],
            "last_started_at": v["last"],
            "in_inventory": n in inventory_seen or True,
            "turn_completed_rate": (v["turn_ok"] / v["turn_n"]) if v["turn_n"] else None,
        }
        for n, v in by_skill_name.items()
    }

    trig_tot = {"slash_command": 0, "skill_md_read": 0, "skill_tool": 0}
    for r in skill_rows:
        if r["trigger"] in trig_tot:
            trig_tot[r["trigger"]] += 1

    unused = [i["skill_name"] for i in inventory if not i["used"]]
    skills_obj = {
        "kind": "skills",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "activations": skill_rows,
        "by_activation_id": by_act,
        "by_name": skills_by_name,
        "inventory": inventory,
        "effectiveness": _eff(sk_ok, sk_fail, unknown=sk_other),
        "summary": {
            "activation_count": len(skill_rows),
            "distinct_skill_count": len(skills_by_name),
            "skills_used": sorted(used_counts.keys()),
            "inventory_count": len(inventory),
            "unused_skill_names": unused,
            "by_trigger": trig_tot,
        },
    }

    # turns package
    turn_rows: list[dict[str, Any]] = []
    by_turn_id: dict[str, int] = {}
    t_ok = t_fail = t_cancel = t_unk = 0
    t_dur = 0.0
    t_tok = 0

    for tn in sorted(turns.keys()):
        t = turns[tn]
        status, success = _turn_status(t.outcome)
        if success is True:
            t_ok += 1
        elif status == "failure":
            t_fail += 1
        elif status == "cancelled":
            t_cancel += 1
        else:
            t_unk += 1
        dur = _ms_between(t.started_at, t.ended_at)
        if dur is None:
            dur = 0.0
        t_dur += dur
        ttft = _ms_between(t.started_at, t.first_token_at)

        usage_norm = None
        tokens = None
        if t.usage:
            u = t.usage
            # handle both camelCase from updates and snake
            def ug(*keys: str) -> Any:
                for k in keys:
                    if k in u and u[k] is not None:
                        return u[k]
                return None

            usage_norm = {
                "input_tokens": ug("inputTokens", "input_tokens") or 0,
                "output_tokens": ug("outputTokens", "output_tokens") or 0,
                "total_tokens": ug("totalTokens", "total_tokens") or 0,
                "cached_read_tokens": ug("cachedReadTokens", "cached_read_tokens") or 0,
                "cache_creation_tokens": ug("cacheCreationTokens", "cache_creation_tokens")
                or 0,
                "reasoning_tokens": ug("reasoningTokens", "reasoning_tokens") or 0,
                "model_calls": ug("modelCalls", "model_calls") or 0,
                "api_duration_ms": ug("apiDurationMs", "api_duration_ms") or 0,
                "cost_usd_ticks": ug("costUsdTicks", "cost_usd_ticks"),
            }
            mu = ug("modelUsage", "model_usage")
            if isinstance(mu, dict):
                usage_norm["model_usage"] = {}
                for mk, mv in mu.items():
                    if isinstance(mv, dict):
                        usage_norm["model_usage"][mk] = {
                            "input_tokens": mv.get("inputTokens") or mv.get("input_tokens") or 0,
                            "output_tokens": mv.get("outputTokens") or mv.get("output_tokens") or 0,
                            "total_tokens": mv.get("totalTokens") or mv.get("total_tokens") or 0,
                            "cached_read_tokens": mv.get("cachedReadTokens")
                            or mv.get("cached_read_tokens")
                            or 0,
                            "cache_creation_tokens": mv.get("cacheCreationTokens")
                            or mv.get("cache_creation_tokens")
                            or 0,
                            "reasoning_tokens": mv.get("reasoningTokens")
                            or mv.get("reasoning_tokens")
                            or 0,
                            "model_calls": mv.get("modelCalls") or mv.get("model_calls") or 0,
                            "api_duration_ms": mv.get("apiDurationMs")
                            or mv.get("api_duration_ms")
                            or 0,
                            "cost_usd_ticks": mv.get("costUsdTicks") or mv.get("cost_usd_ticks"),
                        }
            tokens = {
                "input_tokens": usage_norm["input_tokens"],
                "output_tokens": usage_norm["output_tokens"],
                "total_tokens": usage_norm["total_tokens"],
                "reasoning_tokens": usage_norm["reasoning_tokens"],
                "cached_read_tokens": usage_norm["cached_read_tokens"],
                "cache_creation_tokens": usage_norm["cache_creation_tokens"],
                "source": "turn_completed",
            }
            t_tok += int(usage_norm["total_tokens"] or 0)

        # tool failures this turn
        t_fail_tools = 0
        for cid in t.tool_call_ids:
            if cid in by_tool_call_id:
                if tool_calls_out[by_tool_call_id[cid]]["status"] == "failure":
                    t_fail_tools += 1

        prevs = turn_previews.get(t.turn_id, {})
        chat_mids = [f"msg:{i}" for i in by_turn_msgs.get(t.turn_id, [])]
        tl_id = f"tl:turn:{tn}"

        row = {
            "turn_id": t.turn_id,
            "turn_number": tn,
            "timeline_event_id": tl_id,
            "uri": f"sa://turns/{t.turn_id}",
            "session_id": t.session_id or session_id,
            "span": {
                "started_at": t.started_at,
                "ended_at": t.ended_at,
                "duration_ms": dur,
            },
            "status": status,
            "success": success,
            "outcome": t.outcome,
            "cancellation_category": t.cancellation_category,
            "model_id": t.model_id,
            "yolo_mode": t.yolo_mode,
            "session_relationship": t.session_relationship,
            "redirect_kind": t.redirect_kind,
            "conversation_message_count": t.conversation_message_count,
            "prompt_id": t.prompt_id,
            "stop_reason": t.stop_reason,
            "loop_count": t.loop_count,
            "time_to_first_token_ms": ttft,
            "usage": usage_norm,
            "tokens": tokens,
            "chat_message_ids": chat_mids,
            "tool_call_ids": t.tool_call_ids,
            "skill_activation_ids": t.skill_activation_ids,
            "mcp_call_ids": t.mcp_call_ids,
            "counts": {
                "tool_calls": len(t.tool_call_ids),
                "tool_failures": t_fail_tools,
                "skill_activations": len(t.skill_activation_ids),
                "mcp_calls": len(t.mcp_call_ids),
            },
            "user_preview": prevs.get("user_preview"),
            "agent_preview": prevs.get("agent_preview"),
            "reasoning_preview": prevs.get("reasoning_preview"),
            "links": {
                "timeline": f"sa://timeline/{tl_id}",
                "prompt": f"sa://prompts/{t.turn_id}",
                "chat": f"sa://chat/{chat_mids[0]}" if chat_mids else "chat.json",
                "tools_file": "tools.json",
                "skills_file": "skills.json",
            },
        }
        by_turn_id[t.turn_id] = len(turn_rows)
        turn_rows.append(row)

    turns_obj = {
        "kind": "turns",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "turns": turn_rows,
        "by_turn_id": by_turn_id,
        "effectiveness": _eff(
            t_ok,
            t_fail,
            t_cancel,
            0,
            t_unk,
            total_duration_ms=t_dur,
            total_tokens=t_tok or None,
        ),
    }

    # mcp package
    mcp_tool_rows = []
    by_mcp = {}
    mcp_ok = mcp_fail = 0
    for cid, c in mcp_calls.items():
        ok = c.get("success")
        status = "success" if ok is True else ("failure" if ok is False else "unknown")
        if ok is True:
            mcp_ok += 1
        elif ok is False:
            mcp_fail += 1
        dur = c.get("duration_ms")
        try:
            dur_f = float(dur) if dur is not None else 0.0
        except (TypeError, ValueError):
            dur_f = 0.0
        started = c.get("started_at") or c.get("completed_at")
        ended = c.get("completed_at") or started
        tl_id = f"tl:mcp:{cid}"
        row = {
            "mcp_call_id": cid,
            "timeline_event_id": tl_id,
            "server_name": c.get("server_name"),
            "tool_name": c.get("tool_name"),
            "span": {
                "started_at": started,
                "ended_at": ended,
                "duration_ms": dur_f,
            },
            "status": status,
            "success": ok if isinstance(ok, bool) else None,
            "is_timeout": c.get("is_timeout"),
            "error": c.get("error"),
            "error_preview": _trunc(c.get("error"), max_preview) if c.get("error") else None,
            "turn_id": c.get("turn_id"),
            "turn_number": c.get("turn_number"),
            "timeout_sec": c.get("timeout_sec"),
            "reconnect_attempted": c.get("reconnect_attempted"),
            "auth_retry_attempted": c.get("auth_retry_attempted"),
        }
        by_mcp[cid] = len(mcp_tool_rows)
        mcp_tool_rows.append(row)

    mcp_inits_out = []
    for i, ini in enumerate(mcp_inits):
        ok = ini["failed"] == 0
        mcp_inits_out.append(
            {
                "timeline_event_id": f"tl:mcp_init:{i}",
                "span": {
                    "started_at": ini["ts"],
                    "ended_at": ini["ts"],
                    "duration_ms": ini["duration_ms"],
                },
                "status": "success" if ok else "failure",
                "success": ok,
                "total_servers": ini["total_servers"],
                "succeeded": ini["succeeded"],
                "failed": ini["failed"],
                "auth_required": ini["auth_required"],
                "total_tools": ini["total_tools"],
                "is_reinit": ini["is_reinit"],
                "failed_servers": ini["failed_servers"],
            }
        )

    by_server: dict[str, Any] = {}
    by_mcp_tool: dict[str, Any] = {}
    for r in mcp_tool_rows:
        sn, tn_ = r["server_name"], r["tool_name"]
        key = f"{sn}/{tn_}"
        slot = by_mcp_tool.setdefault(
            key, {"server_name": sn, "tool_name": tn_, "ok": 0, "fail": 0, "ids": [], "dur": 0.0}
        )
        slot["ids"].append(r["mcp_call_id"])
        if r["success"] is True:
            slot["ok"] += 1
        else:
            slot["fail"] += 1
        slot["dur"] += float(r["span"]["duration_ms"] or 0)
        ss = by_server.setdefault(
            sn,
            {
                "server_name": sn,
                "tool_ok": 0,
                "tool_fail": 0,
                "connect_ok": 0,
                "connect_fail": 0,
                "tools": set(),
            },
        )
        if r["success"] is True:
            ss["tool_ok"] += 1
        else:
            ss["tool_fail"] += 1
        ss["tools"].add(tn_)

    for se in mcp_server_events:
        sn = se["server_name"]
        ss = by_server.setdefault(
            sn,
            {
                "server_name": sn,
                "tool_ok": 0,
                "tool_fail": 0,
                "connect_ok": 0,
                "connect_fail": 0,
                "tools": set(),
            },
        )
        if se["event"] == "connected":
            ss["connect_ok"] += 1
        elif se["event"] == "failed":
            ss["connect_fail"] += 1

    mcp_obj = {
        "kind": "mcp",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "config_resolutions": mcp_config,
        "inits": mcp_inits_out,
        "server_events": [
            {**se, "timeline_event_id": f"tl:mcp_srv:{i}"}
            for i, se in enumerate(mcp_server_events)
        ],
        "tool_calls": mcp_tool_rows,
        "by_mcp_call_id": by_mcp,
        "by_server": {
            sn: {
                "server_name": sn,
                "connect_effectiveness": _eff(v["connect_ok"], v["connect_fail"]),
                "tool_effectiveness": _eff(v["tool_ok"], v["tool_fail"]),
                "tools_registered": sorted(v["tools"]),
            }
            for sn, v in by_server.items()
        },
        "by_tool": {
            k: {
                "server_name": v["server_name"],
                "tool_name": v["tool_name"],
                "effectiveness": _eff(
                    v["ok"], v["fail"], total_duration_ms=v["dur"]
                ),
                "mcp_call_ids": v["ids"],
            }
            for k, v in by_mcp_tool.items()
        },
        "effectiveness": _eff(mcp_ok, mcp_fail),
        "summary": {
            "server_success_count": sum(1 for s in mcp_server_events if s["event"] == "connected"),
            "server_failure_count": sum(1 for s in mcp_server_events if s["event"] == "failed"),
            "mcp_tool_call_count": len(mcp_tool_rows),
            "total_tools_registered": None,
            "total_duration_ms": sum(float(r["span"]["duration_ms"] or 0) for r in mcp_tool_rows),
            "servers_seen": sorted(by_server.keys()),
            "failed_servers": sorted(
                {s["server_name"] for s in mcp_server_events if s["event"] == "failed"}
            ),
        },
    }

    # chat
    by_message_id = {m["message_id"]: m["index"] for m in messages}
    role_counts: dict[str, int] = defaultdict(int)
    total_chars = 0
    for m in messages:
        role_counts[m["role"]] += 1
        total_chars += int(m.get("char_count") or 0)

    chat_obj = {
        "kind": "chat",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "messages": messages,
        "by_message_id": by_message_id,
        "by_turn_id": {k: sorted(set(v)) for k, v in by_turn_msgs.items()},
        "by_role": {k: v for k, v in by_role.items()},
        "by_tool_call_id": by_tool_msg,
        "summary": {
            "message_count": len(messages),
            "role_counts": dict(role_counts),
            "total_chars": total_chars,
        },
    }

    # prompts
    prompt_turns = []
    for tr in turn_rows:
        ut = tr.get("user_preview") or {}
        at = tr.get("agent_preview") or {}
        rt = tr.get("reasoning_preview") or {}
        skills_ref = []
        # parse skill from user text lightly
        utext = ut.get("text") or ""
        m = re.search(r"/([a-zA-Z0-9_-]+)\s*(.*)", utext)
        if m:
            skills_ref.append(
                {
                    "skill_name": m.group(1),
                    "path": None,
                    "args": (m.group(2) or "").strip() or None,
                }
            )
        prompt_turns.append(
            {
                "turn_id": tr["turn_id"],
                "turn_number": tr["turn_number"],
                "uri": f"sa://prompts/{tr['turn_id']}",
                "prompt_id": tr.get("prompt_id"),
                "user": {
                    "text": ut.get("text"),
                    "truncated": ut.get("truncated", False),
                    "char_count": ut.get("char_count"),
                    "has_images": False,
                    "skills_referenced": skills_ref,
                }
                if ut
                else None,
                "assistant": {
                    "text": at.get("text"),
                    "truncated": at.get("truncated", False),
                    "char_count": at.get("char_count"),
                    "model_id": tr.get("model_id"),
                }
                if at
                else None,
                "reasoning": {
                    "summary_text": rt.get("text"),
                    "truncated": rt.get("truncated", False),
                    "status": "completed",
                }
                if rt
                else None,
                "chat_message_ids": tr.get("chat_message_ids") or [],
                "links": {
                    "turn": f"sa://turns/{tr['turn_id']}",
                    "chat": "chat.json",
                },
            }
        )

    prompts_obj = {
        "kind": "prompts",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "included": {
            "user_text": True,
            "assistant_text": True,
            "reasoning_summary": True,
            "max_string_chars": max_preview,
        },
        "turns": prompt_turns,
        "by_turn_id": {p["turn_id"]: i for i, p in enumerate(prompt_turns)},
    }

    # context
    context_obj = {
        "kind": "context",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "version": prompt_ctx.get("version"),
        "prompt_mode": prompt_ctx.get("prompt_mode"),
        "audience": prompt_ctx.get("audience"),
        "system_prompt_label": prompt_ctx.get("system_prompt_label"),
        "build_timestamp_utc": prompt_ctx.get("build_timestamp_utc"),
        "current_date": prompt_ctx.get("current_date"),
        "os_name": prompt_ctx.get("os_name"),
        "shell_path": prompt_ctx.get("shell_path"),
        "working_directory": prompt_ctx.get("working_directory"),
        "is_non_interactive": prompt_ctx.get("is_non_interactive"),
        "memory_enabled": prompt_ctx.get("memory_enabled"),
        "memory_global_path": prompt_ctx.get("memory_global_path"),
        "memory_workspace_path": prompt_ctx.get("memory_workspace_path"),
        "agents_md_files": prompt_ctx.get("agents_md_files") or [],
        "persona_summaries": prompt_ctx.get("persona_summaries") or [],
    }

    # compactions
    comp_obj = {
        "kind": "compactions",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "related": related_base,
        "events": compactions,
        "summary": {
            "count": len(compactions),
            "total_tokens_before": sum(c.get("tokens_before") or 0 for c in compactions),
            "total_tokens_after": sum(c.get("tokens_after") or 0 for c in compactions),
            "max_tokens_before": max(
                (c.get("tokens_before") or 0 for c in compactions), default=None
            ),
        },
    }

    # timeline.jsonl
    timeline: list[dict[str, Any]] = []
    seq = 0

    def tl_add(row: dict[str, Any]) -> None:
        nonlocal seq
        row["sequence"] = seq
        row["session_id"] = session_id
        seq += 1
        timeline.append(row)

    for tr in turn_rows:
        tl_add(
            {
                "event_id": tr["timeline_event_id"],
                "span_kind": "span",
                "entity_type": "turn",
                "entity_id": tr["turn_id"],
                "lane": "turn",
                "kind": "turn",
                "label": f"turn {tr['turn_number']}",
                "name": tr.get("model_id"),
                "started_at": tr["span"]["started_at"],
                "ended_at": tr["span"]["ended_at"],
                "duration_ms": tr["span"]["duration_ms"],
                "status": tr["status"],
                "success": tr["success"],
                "parent_event_id": None,
                "turn_id": tr["turn_id"],
                "turn_number": tr["turn_number"],
                "depth": 1,
                "outcome": tr["outcome"],
                "tokens": tr.get("tokens"),
                "preview": tr.get("agent_preview") or tr.get("user_preview"),
                "detail": {
                    "file": "turns.json",
                    "entity_type": "turn",
                    "entity_id": tr["turn_id"],
                    "uri": tr["uri"],
                    "index_key": "by_turn_id",
                },
                "meta": {
                    "model_id": tr.get("model_id"),
                    "ttft_ms": tr.get("time_to_first_token_ms"),
                },
                "color_hint": "turn",
            }
        )
        # first_token is kept only as turns[].time_to_first_token_ms / meta.ttft_ms
        # (not a peer timeline row like tools)

    for r in tool_calls_out:
        parent = None
        if r.get("turn_id") and r["turn_id"] in by_turn_id:
            parent = turn_rows[by_turn_id[r["turn_id"]]]["timeline_event_id"]
        tl_add(
            {
                "event_id": r["timeline_event_id"],
                "span_kind": "span",
                "entity_type": "tool",
                "entity_id": r["tool_call_id"],
                "lane": "tool",
                "kind": "tool_call",
                "label": r["tool_name"],
                "name": r["tool_name"],
                "started_at": r["span"]["started_at"],
                "ended_at": r["span"]["ended_at"],
                "duration_ms": r["span"]["duration_ms"],
                "status": r["status"],
                "success": r["success"],
                "parent_event_id": parent,
                "parent_entity_type": "turn" if parent else None,
                "parent_entity_id": r.get("turn_id"),
                "turn_id": r.get("turn_id"),
                "turn_number": r.get("turn_number"),
                "depth": 2 if parent else 1,
                "outcome": r["outcome"],
                "tokens": r.get("tokens"),
                "preview": r.get("input_preview") or r.get("result_preview"),
                "detail": {
                    "file": "tools.json",
                    "entity_type": "tool",
                    "entity_id": r["tool_call_id"],
                    "uri": r["uri"],
                    "index_key": "by_tool_call_id",
                },
                "meta": {"title": r.get("title")},
                "color_hint": r["tool_name"],
            }
        )

    for r in skill_rows:
        parent = None
        if r.get("turn_id") and r["turn_id"] in by_turn_id:
            parent = turn_rows[by_turn_id[r["turn_id"]]]["timeline_event_id"]
        tl_add(
            {
                "event_id": r["timeline_event_id"],
                "span_kind": "span",
                "entity_type": "skill",
                "entity_id": r["skill_activation_id"],
                "lane": "skill",
                "kind": "skill_activation",
                "label": r["skill_name"],
                "name": r["skill_name"],
                "started_at": r["span"]["started_at"],
                "ended_at": r["span"]["ended_at"],
                "duration_ms": r["span"]["duration_ms"],
                "status": r["status"],
                "success": r["success"],
                "parent_event_id": parent,
                "parent_entity_type": "turn" if parent else None,
                "parent_entity_id": r.get("turn_id"),
                "turn_id": r.get("turn_id"),
                "turn_number": r.get("turn_number"),
                "depth": 2 if parent else 1,
                "preview": r.get("agent_preview") or r.get("args_preview"),
                "detail": {
                    "file": "skills.json",
                    "entity_type": "skill",
                    "entity_id": r["skill_activation_id"],
                    "uri": r["uri"],
                    "index_key": "by_activation_id",
                },
                "meta": {"trigger": r["trigger"]},
                "color_hint": r["skill_name"],
            }
        )

    for r in mcp_tool_rows:
        parent = None
        if r.get("turn_id") and r["turn_id"] in by_turn_id:
            parent = turn_rows[by_turn_id[r["turn_id"]]]["timeline_event_id"]
        tl_add(
            {
                "event_id": r["timeline_event_id"],
                "span_kind": "span",
                "entity_type": "mcp_call",
                "entity_id": r["mcp_call_id"],
                "lane": "mcp",
                "kind": "mcp_tool_call",
                "label": f"{r['server_name']}/{r['tool_name']}",
                "name": r["tool_name"],
                "started_at": r["span"]["started_at"],
                "ended_at": r["span"]["ended_at"],
                "duration_ms": r["span"]["duration_ms"],
                "status": r["status"],
                "success": r["success"],
                "parent_event_id": parent,
                "turn_id": r.get("turn_id"),
                "turn_number": r.get("turn_number"),
                "depth": 2 if parent else 1,
                "detail": {
                    "file": "mcp.json",
                    "entity_type": "mcp_call",
                    "entity_id": r["mcp_call_id"],
                    "uri": f"sa://mcp/{r['mcp_call_id']}",
                    "index_key": "by_mcp_call_id",
                },
                "meta": {"server_name": r["server_name"]},
                "color_hint": "mcp",
            }
        )

    for i, c in enumerate(compactions):
        parent = None
        if c.get("turn_id") and c["turn_id"] in by_turn_id:
            parent = turn_rows[by_turn_id[c["turn_id"]]]["timeline_event_id"]
        tb, ta = c.get("tokens_before") or 0, c.get("tokens_after") or 0
        tl_add(
            {
                "event_id": f"tl:compaction:{i}",
                "span_kind": "instant",
                "entity_type": "compaction",
                "entity_id": f"compaction:{i}",
                "lane": "system",
                "kind": "compaction",
                "label": f"compaction {tb}→{ta}",
                "name": "compaction",
                "started_at": c.get("ts"),
                "ended_at": c.get("ts"),
                "duration_ms": 0,
                "status": "success",
                "success": True,
                "parent_event_id": parent,
                "parent_entity_type": "turn" if parent else None,
                "parent_entity_id": c.get("turn_id"),
                "turn_id": c.get("turn_id"),
                "turn_number": c.get("turn_number"),
                "depth": 2 if parent else 1,
                "tokens_before": tb,
                "tokens_after": ta,
                "detail": {
                    "file": "compactions.json",
                    "entity_type": "compaction",
                    "entity_id": str(i),
                    "uri": "sa://compactions",
                    "index_key": "events",
                },
                "meta": {
                    "tokens_before": tb,
                    "tokens_after": ta,
                    "source": c.get("source"),
                },
                "color_hint": "system",
            }
        )

    # sort timeline by started_at
    def sort_key(row: dict[str, Any]) -> tuple:
        ts = row.get("started_at") or ""
        return (ts, row.get("sequence", 0))

    timeline.sort(key=sort_key)
    for i, row in enumerate(timeline):
        row["sequence"] = i

    # manifest
    def fentry(
        path: str,
        role: str,
        schema: str,
        fmt: str,
        required: bool,
        pks: list[str],
        fks: list[str],
        count: int | None = None,
    ) -> dict[str, Any]:
        e: dict[str, Any] = {
            "path": path,
            "role": role,
            "schema": schema,
            "format": fmt,
            "required": required,
            "primary_keys": pks,
            "foreign_keys": fks,
            "present": True,
        }
        if count is not None:
            e["record_count"] = count
        return e

    manifest = {
        "kind": "manifest",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "generated_at": _now_iso(),
        "generator": {"name": "sa.normalize", "version": "0.1.0"},
        "source": {
            "session_dir": str(session_dir),
            "event_schema_version": "1.0",
        },
        "options": {
            "include_user_text": True,
            "include_assistant_text": True,
            "include_chat": True,
            "include_tool_args": True,
            "include_tool_output_preview": True,
            "include_timeline": True,
            "max_string_chars": max_preview,
        },
        "files": {
            "session": fentry("session.json", "identity", "session.schema.json", "json", True, ["session_id"], []),
            "signals": fentry("signals.json", "counters", "signals.schema.json", "json", True, [], ["session_id"]),
            "turns": fentry(
                "turns.json", "turns", "turns.schema.json", "json", True,
                ["turn_id"], ["session_id", "tool_call_id", "skill_activation_id", "message_id"],
                len(turn_rows),
            ),
            "tools": fentry(
                "tools.json", "tools", "tools.schema.json", "json", True,
                ["tool_call_id"], ["session_id", "turn_id", "message_id"],
                len(tool_calls_out),
            ),
            "skills": fentry(
                "skills.json", "skills", "skills.schema.json", "json", True,
                ["skill_activation_id"], ["session_id", "turn_id", "tool_call_id"],
                len(skill_rows),
            ),
            "mcp": fentry(
                "mcp.json", "mcp", "mcp.schema.json", "json", True,
                ["mcp_call_id"], ["session_id", "turn_id"],
                len(mcp_tool_rows),
            ),
            "prompts": fentry(
                "prompts.json", "prompt digests", "prompts.schema.json", "json", False,
                [], ["turn_id", "message_id"], len(prompt_turns),
            ),
            "chat": fentry(
                "chat.json", "full chat", "chat.schema.json", "json", True,
                ["message_id"], ["session_id", "turn_id", "tool_call_id"],
                len(messages),
            ),
            "context": fentry("context.json", "env", "context.schema.json", "json", True, [], ["session_id"]),
            "compactions": fentry(
                "compactions.json", "compactions", "compactions.schema.json", "json", True,
                [], ["session_id"], len(compactions),
            ),
            "timeline": fentry(
                "timeline.jsonl", "viz timeline", "timeline_event.schema.json", "jsonl", False,
                ["timeline_event_id"], ["turn_id", "tool_call_id", "skill_activation_id"],
                len(timeline),
            ),
        },
        "join_keys": {
            "session_id": {
                "description": "Session PK",
                "defined_in": ["session"],
                "referenced_by": ["turns", "tools", "skills", "chat"],
            },
            "turn_id": {
                "description": "turn:{n}",
                "format": "turn:{turn_number}",
                "defined_in": ["turns"],
                "referenced_by": ["tools", "skills", "chat", "timeline"],
            },
            "turn_number": {
                "description": "wire turn number",
                "defined_in": ["turns"],
                "referenced_by": ["tools", "skills", "timeline"],
            },
            "tool_call_id": {
                "description": "tool call id",
                "defined_in": ["tools"],
                "referenced_by": ["turns", "skills", "chat", "timeline"],
            },
            "skill_activation_id": {
                "description": "skill_act:{n}",
                "format": "skill_act:{n}",
                "defined_in": ["skills"],
                "referenced_by": ["turns", "timeline"],
            },
            "prompt_id": {
                "description": "prompt request id",
                "defined_in": ["turns"],
                "referenced_by": [],
            },
            "mcp_call_id": {
                "description": "mcp call id",
                "defined_in": ["mcp"],
                "referenced_by": ["turns", "timeline"],
            },
            "message_id": {
                "description": "msg:{n}",
                "format": "msg:{n}",
                "defined_in": ["chat"],
                "referenced_by": ["turns", "tools", "skills", "prompts"],
            },
            "timeline_event_id": {
                "description": "tl:…",
                "defined_in": ["timeline"],
                "referenced_by": ["turns", "tools", "skills", "mcp"],
            },
        },
        "graph": [
            {"from": "turns", "to": "tools", "via": "tool_call_ids", "cardinality": "1:n"},
            {"from": "turns", "to": "skills", "via": "skill_activation_ids", "cardinality": "1:n"},
            {"from": "turns", "to": "chat", "via": "chat_message_ids", "cardinality": "1:n"},
            {"from": "timeline", "to": "turns", "via": "detail", "cardinality": "n:1"},
            {"from": "timeline", "to": "tools", "via": "detail", "cardinality": "n:1"},
            {"from": "timeline", "to": "skills", "via": "detail", "cardinality": "n:1"},
        ],
        "uri_scheme": {
            "prefix": "sa://",
            "templates": {
                "session": "sa://session",
                "turn": "sa://turns/{turn_id}",
                "tool_call": "sa://tools/{tool_call_id}",
                "skill_activation": "sa://skills/{skill_activation_id}",
                "prompt": "sa://prompts/{turn_id}",
                "chat_message": "sa://chat/{message_id}",
                "mcp_call": "sa://mcp/{mcp_call_id}",
            },
        },
    }

    # write all
    def dump(name: str, obj: Any) -> None:
        (out_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    dump("manifest.json", manifest)
    dump("session.json", session_obj)
    dump("signals.json", signals_obj)
    dump("turns.json", turns_obj)
    dump("tools.json", tools_obj)
    dump("skills.json", skills_obj)
    dump("mcp.json", mcp_obj)
    dump("prompts.json", prompts_obj)
    dump("chat.json", chat_obj)
    dump("context.json", context_obj)
    dump("compactions.json", comp_obj)
    with (out_dir / "timeline.jsonl").open("w", encoding="utf-8") as f:
        for row in timeline:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Normalize a Grok session into session_analysis/1.0")
    p.add_argument("session_dir", type=Path, help="Raw session directory")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output package directory (default: ./packages/<session_id>)",
    )
    p.add_argument("--max-preview", type=int, default=MAX_PREVIEW)
    args = p.parse_args(argv)

    session_dir = args.session_dir
    if not session_dir.is_dir():
        print(f"error: not a directory: {session_dir}", flush=True)
        return 1

    summary = _read_json(session_dir / "summary.json") or {}
    sid = (summary.get("info") or {}).get("id") or session_dir.name
    out = args.out or (Path("packages") / str(sid))
    path = normalize_session(session_dir, out, max_preview=args.max_preview)
    print(f"wrote package → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
