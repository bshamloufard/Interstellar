"""Frozen interface contract for the review loop.

Every stage in the loop hands the next one of these shapes. They are plain
dicts, built by the constructor functions below, for the same reason the
normalizer's schema is: the whole pipeline stays stdlib-JSON-serializable and
diffable, and a stage's output can be dropped into a file and read by a human
without a deserializer.

The rule that makes parallel implementation possible: **these shapes do not
change**. A stage that needs a key not declared here is a stage whose contract
is wrong, and that is a conversation, not an edit.

Flow:

    HarnessVersion  --patches-->  HarnessVersion'
          |                             |
          +---------- replay -----------+
                        |
                   ReplayMatrix  -->  [Grade]  -->  stats  -->  Report
"""

from __future__ import annotations

CONTRACT_VERSION = "1"

# --- patch kinds ------------------------------------------------------------
# Closed set. A closed action space is what makes the acceptance gate and the
# per-patch statistics tractable (SkillOpt's bounded add/delete/replace rule).
PATCH_SKILL_TRUNCATE = "skill.truncate"   # delete cited line ranges from a SKILL.md
PATCH_SKILL_REMOVE = "skill.remove"       # drop a skill from the harness entirely
PATCH_SKILL_INSERT = "skill.insert"       # insert generated lines at an anchor
PATCH_MCP_REMOVE = "mcp.remove"           # drop one [mcp_servers.X] table
PATCH_RULES_APPEND = "rules.append"       # append one imperative line to the rules

ALL_PATCH_KINDS = (
    PATCH_SKILL_TRUNCATE,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_INSERT,
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
)

# The measured metric a patch claims it will move. Named here so the report can
# check the claim against what actually moved instead of grading on vibes.
EFFECT_SKILL_TOKENS = "skill_tokens"
EFFECT_TOOL_CALLS = "tool_calls"
EFFECT_MCP_STARTUP_MS = "mcp_startup_ms"
EFFECT_TOOL_RESULT_TOKENS = "tool_result_tokens"
EFFECT_WALL_MS = "wall_ms"

ALL_EFFECTS = (
    EFFECT_SKILL_TOKENS,
    EFFECT_TOOL_CALLS,
    EFFECT_MCP_STARTUP_MS,
    EFFECT_TOOL_RESULT_TOKENS,
    EFFECT_WALL_MS,
)

# Efficiency metrics read off a normalized trace. Report order is this order.
# `lower_is_better` is False for nothing here today, but the flag is what keeps
# a future metric (success rate) from being read upside down.
EFFICIENCY_METRICS = (
    ("total_tokens", True),
    ("cost_usd", True),
    ("wall_ms", True),
    ("tool_calls", True),
    ("skill_tokens_est", True),
    ("skill_wasted_tokens_est", True),
    ("tool_result_tokens_est", True),
    ("mcp_startup_ms", True),
    ("duplicate_calls", True),
    ("turns", True),
)

ARM_CONTROL = "control"
ARM_TREATMENT = "treatment"

SEVERITIES = ("high", "medium", "low")


def make_skill(name, *, path, content, scope="user", description=""):
    """One skill as the harness sees it.

    `content` is carried in full, not by path, because a candidate harness is a
    derived object: patching must not mutate the file the user is still using.
    """
    text = content
    return {
        "name": name,
        "path": str(path),
        "scope": scope,              # "user" | "project"
        "description": description,  # frontmatter description, "" when absent
        "content": text,
        "bytes": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "tokens_est": len(text.encode("utf-8")) // 4,
        "sha256": None,              # filled by harness.snapshot
    }


def make_mcp_server(name, *, command="", args=None, enabled=True):
    return {
        "name": name,
        "command": command,
        "args": list(args or []),
        "enabled": bool(enabled),
    }


def make_harness_version(*, version_id, skills, mcp_servers, config_text,
                         extra_rules=None, source=None, warnings=None):
    """A content-addressed snapshot of the thing being optimized.

    `extra_rules` are lines appended to the system prompt at run time via
    `--rules`; they are part of the harness identity, so they feed version_id.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "version_id": version_id,
        "skills": list(skills),
        "mcp_servers": list(mcp_servers),
        "config_text": config_text,
        "extra_rules": list(extra_rules or []),
        "source": source,            # where it was snapshotted from
        "warnings": list(warnings or []),
    }


def make_patch(*, patch_id, kind, target, rationale, expected_effect,
               source_recommendation=None, line_ranges=None, insert_text=None,
               insert_after_line=None, rule_text=None, diff="",
               skipped_reason=None):
    """One bounded, typed edit against a HarnessVersion.

    Exactly one of the payload fields is meaningful per kind:
      skill.truncate -> line_ranges     (1-based inclusive [start, end] pairs)
      skill.remove   -> (none)
      skill.insert   -> insert_text + insert_after_line
      mcp.remove     -> (none; `target` is the server name)
      rules.append   -> rule_text

    `diff` is the unified diff the user will read and accept. It is filled in by
    harness.apply, which is the only thing that knows the resulting file text.
    `skipped_reason` non-None means the patch was rejected before application
    (lint violation, missing line ranges, out-of-range) and must still appear in
    the report — a silently dropped recommendation is indistinguishable from one
    that was never made.
    """
    return {
        "patch_id": patch_id,
        "kind": kind,
        "target": target,              # skill name, mcp server name, or "rules"
        "rationale": rationale,
        "expected_effect": expected_effect,
        "source_recommendation": source_recommendation,
        "line_ranges": [list(r) for r in (line_ranges or [])],
        "insert_text": insert_text,
        "insert_after_line": insert_after_line,
        "rule_text": rule_text,
        "diff": diff,
        "skipped_reason": skipped_reason,
    }


def make_patch_application(*, patch_id, applied, reason="", diff="",
                           lines_changed=0):
    return {
        "patch_id": patch_id,
        "applied": bool(applied),
        "reason": reason,
        "diff": diff,
        "lines_changed": int(lines_changed),
    }


def make_run_result(*, arm, repeat, ok, prompt, home, workdir,
                    session_id=None, trace=None, response_text="",
                    cost_usd=None, wall_s=None, turns=None, usage=None,
                    error=None, argv=None):
    """One execution of one prompt under one harness.

    `trace` is the normalized trace dict (normalizer/schema.py shape) or None
    when the run failed. Nothing downstream may assume it is present.

    `usage` is grok's own token accounting, taken verbatim from the headless
    stdout JSON: `{input_tokens, cache_read_input_tokens,
    cache_creation_input_tokens, output_tokens, reasoning_tokens,
    total_tokens}`. It lives here rather than on the trace because the
    normalizer works from the session store, which records no token counts —
    so token and cost figures are only ever available on the RunResult. Any
    grader wanting them must read the run, not the trace.
    """
    return {
        "arm": arm,                  # ARM_CONTROL | ARM_TREATMENT
        "repeat": int(repeat),
        "ok": bool(ok),
        "prompt": prompt,
        "home": str(home),
        "workdir": str(workdir),
        "session_id": session_id,
        "trace": trace,
        "response_text": response_text,
        "cost_usd": cost_usd,
        "wall_s": wall_s,
        "turns": turns,
        "usage": dict(usage or {}),
        "error": error,
        "argv": list(argv or []),
    }


def make_replay_matrix(*, prompt, k, arms, control_version_id,
                       treatment_version_id, patch_id=None):
    """k paired runs of the same prompt under two harnesses.

    Both arms run under the same isolation. The user's original recorded session
    is NOT an arm — it ran in a different environment, so using it as the
    control would confound the patch with the environment change.
    """
    return {
        "prompt": prompt,
        "k": int(k),
        "patch_id": patch_id,
        "control_version_id": control_version_id,
        "treatment_version_id": treatment_version_id,
        "arms": arms,                # {ARM_CONTROL: [RunResult], ARM_TREATMENT: [...]}
    }


def make_grade(*, repeat, control_efficiency, treatment_efficiency,
               judge=None, regressions=None, control_ok=True,
               treatment_ok=True):
    return {
        "repeat": int(repeat),
        "control_ok": bool(control_ok),
        "treatment_ok": bool(treatment_ok),
        "control_efficiency": dict(control_efficiency or {}),
        "treatment_efficiency": dict(treatment_efficiency or {}),
        "judge": judge,              # make_judge_verdict() or None
        "regressions": list(regressions or []),
    }


def make_judge_verdict(*, verdict, consistent, raw=None, reason=""):
    """A position-swapped pairwise verdict.

    `verdict` is ARM_CONTROL, ARM_TREATMENT, or "tie". When the two orderings
    disagree, `consistent` is False and `verdict` MUST be "tie" — an
    order-dependent preference is not evidence, and resolving it toward either
    side is exactly the position bias the swap exists to detect.
    """
    return {
        "verdict": verdict,
        "consistent": bool(consistent),
        "reason": reason,
        "raw": list(raw or []),
    }


def make_regression(*, kind, severity, detail, evidence=""):
    return {
        "kind": kind,
        "severity": severity,        # one of SEVERITIES
        "detail": detail,
        "evidence": evidence,
    }


def make_patch_result(*, patch, application, matrix_summary, grades,
                      statistics, gate, verdict_line):
    """Everything measured about one patch. This is what the report renders.

    `matrix_summary` is the ReplayMatrix minus the full traces — traces are
    large and belong in their own files, addressed by path.
    """
    return {
        "patch": patch,
        "application": application,
        "matrix": matrix_summary,
        "grades": list(grades),
        "statistics": statistics,
        "gate": gate,                # {accepted: bool, reasons: [str]}
        "verdict_line": verdict_line,  # one plain-language sentence
    }


def make_report(*, session, baseline_version, patch_results, caveats,
                cost_usd=0.0, k=0, generated_at=None):
    return {
        "contract_version": CONTRACT_VERSION,
        "generated_at": generated_at,
        "session": session,          # {session_id, title, prompt, trace_file}
        "baseline_version": baseline_version,  # {version_id, skills: n, mcp: n}
        "k": int(k),
        "cost_usd": float(cost_usd),
        "patch_results": list(patch_results),
        "caveats": list(caveats),    # plain-language strings, always rendered
    }
