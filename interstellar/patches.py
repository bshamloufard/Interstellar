"""Turn analyzer recommendations into typed, bounded, lint-clean Patch objects.

Two jobs, kept apart on purpose (same split as analyzer/analyze.py):

  from_recommendations()  - deterministic. Maps each analyzer recommendation
                             onto zero or more Patch dicts (interstellar.types
                             shapes) using the recommendation's own fields plus
                             `digest` for per-target evidence numbers.
  synthesize_insert()     - the only model call. One grok-dev --json-schema
                             call for `skill_add_lines`, which needs generated
                             text the recommendation itself does not carry.

This module never touches disk and never applies a patch — it only builds the
typed request. `interstellar.harness.apply` is the only thing that reads a
Patch and produces a diff.
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
from pathlib import Path

from interstellar.types import (
    EFFECT_MCP_STARTUP_MS,
    EFFECT_SKILL_TOKENS,
    EFFECT_TOOL_CALLS,
    EFFECT_TOOL_RESULT_TOKENS,
    EFFECT_WALL_MS,
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
    PATCH_SKILL_INSERT,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_TRUNCATE,
    make_patch,
)

GROK = Path.home() / ".local" / "bin" / "grok-dev"

# Bounded patches: a patch may change at most this many lines. Enforced here
# for the two kinds where the line count is knowable before application
# (skill.truncate from its cited ranges, skill.insert from the synthesized
# text). skill.remove / mcp.remove drop a whole named entity as one bounded
# action regardless of its size, so the cap does not apply to them.
MAX_PATCH_LINES = 60

# Recommendation types that become one rules.append line each, and which
# measured metric each one claims to move.
_RULES_APPEND_TYPES = {
    "tool_redundant": EFFECT_TOOL_CALLS,
    "tool_output_truncate": EFFECT_TOOL_RESULT_TOKENS,
    "subagent_inline": EFFECT_WALL_MS,
    "permission_friction": EFFECT_WALL_MS,
}

# mcp_latency / mcp_fix_broken only become a patch when the analyzer's own
# action text asks for removal or disabling; otherwise there is nothing safe
# to patch (a broken/slow server the analyzer wants *fixed* is out of scope
# for a bounded harness edit).
_REMOVAL_ACTION_RE = re.compile(
    r"\b(disable|remove|disconnect|uninstall|drop)\b", re.IGNORECASE
)

_INSERT_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {"type": "string"},
            "description": "at most 4 lines of markdown to add to the skill",
        },
        "anchor_line": {
            "type": "integer",
            "description": "1-based line number in the skill text to insert "
                            "after; 0 means insert at the very top",
        },
    },
    "required": ["lines", "anchor_line"],
}

# --- security lint -----------------------------------------------------------
# Applied to any text we are about to inject into a harness (a synthesized
# skill insert, or a rules.append line lifted from the analyzer's own output).
# Each entry is (family name, pattern). A hit records "family: matched text".
_LINT_PATTERNS = (
    ("url", re.compile(r"https?://\S+", re.IGNORECASE)),
    ("shell_fetch", re.compile(r"\b(curl|wget|nc|netcat)\b", re.IGNORECASE)),
    ("base64_blob", re.compile(r"(?:[A-Za-z0-9+/]{4}){10,}={0,2}")),
    ("prompt_injection", re.compile(
        r"ignore (all )?previous instructions|disregard the above|^\s*system:",
        re.IGNORECASE | re.MULTILINE)),
    ("disable_safety", re.compile(
        r"disable\s+(all\s+)?(safety|permission)|bypass(es|ing)?\s+permission"
        r"|skip\s+permission", re.IGNORECASE)),
    ("credential_exfil", re.compile(
        r"auth\.json|XAI_API_KEY|~/\.ssh", re.IGNORECASE)),
)


def security_lint(text: str) -> list[str]:
    """Return one violation string per hit family, or [] for clean text."""
    violations = []
    for name, pattern in _LINT_PATTERNS:
        m = pattern.search(text or "")
        if m:
            violations.append(f"{name}: matched {m.group(0)!r}")
    return violations


# --- the model call ----------------------------------------------------------

class GrokCallError(RuntimeError):
    """Raised by `_default_grok` when grok-dev did not return a usable
    structured result. `synthesize_insert` catches this (along with any
    other exception) and returns None -- this class exists so the specific
    reason is legible to anything that inspects it directly (tests, logs),
    instead of collapsing into an opaque JSONDecodeError that looks
    indistinguishable from "the model declined"."""


def _extract_structured_output(out: dict):
    """Pull the schema-validated result out of grok-dev's parsed JSON
    envelope `out`.

    Reads the validated `structuredOutput` field the CLI attaches at top
    level when `--json-schema` is passed, not the `text` message buffer:
    `text` happens to carry the same JSON on today's backend, but that is
    not the contract -- a different backend can deliver the schema result
    only via a synthetic tool call, leaving `text` empty or prose.
    `structuredOutputError` non-null means the call failed even if `text`
    looks parseable. `text` is used only when `structuredOutput` is absent
    from the response entirely (an older CLI, or a schema-less call).

    Split out from `_default_grok` so this precedence logic is testable
    without shelling out to grok-dev.
    """
    error = out.get("structuredOutputError")
    if error:
        raise GrokCallError(f"structuredOutputError: {error}")

    if "structuredOutput" in out:
        structured = out["structuredOutput"]
        if structured is not None:
            return structured
        raise GrokCallError("structuredOutput is null with no structuredOutputError")

    # structuredOutput key absent entirely -- fall back to the text buffer.
    text = out.get("text") or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise GrokCallError(f"no structuredOutput and text was not valid JSON: {e}") from e


def _default_grok(prompt, schema, *, timeout=600):
    """Shell out to grok-dev, same invocation pattern as analyzer.analyze."""
    cmd = [str(GROK), "-p", prompt,
           "--json-schema", json.dumps(schema),
           "--permission-mode", "bypassPermissions",
           "--disallowed-tools", "run_terminal_command,read_file,write,search_replace"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise GrokCallError(f"grok-dev stdout was not JSON: {e}") from e
    return _extract_structured_output(out)


def synthesize_insert(rec, skill_text, *, grok=_default_grok):
    """One grok-dev call producing <=4 lines of skill text plus an anchor.

    Returns `(text, insert_after_line)` on success, or None when the model
    call fails, the response is malformed, or the generated text fails
    `security_lint`. `grok` takes `(prompt, schema)` and returns the already
    -parsed JSON dict — the default implementation shells out and parses the
    grok-dev CLI's own JSON envelope; tests inject a fake that skips the
    subprocess entirely.
    """
    total_lines = len(skill_text.splitlines())
    prompt = (
        "You write short additions to an agent harness skill file. A session "
        "audit found this skill was loaded and then followed by discovery "
        "calls (list_dir/grep/glob), which usually means the skill left out "
        "something the agent needed to proceed.\n\n"
        f"Recommendation action: {rec.get('action', '')}\n"
        f"Evidence: {rec.get('evidence', '')}\n\n"
        "Write at most 4 lines of markdown that state the missing guidance "
        "directly and concretely - no more. Also choose the 1-based line "
        f"number in the skill text (0-{total_lines}) after which to insert "
        "it; 0 means insert at the very top, before line 1.\n\n"
        f"--- SKILL TEXT ({total_lines} lines) ---\n{skill_text}\n--- END ---"
    )
    try:
        result = grok(prompt, _INSERT_SCHEMA)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    lines = result.get("lines")
    anchor = result.get("anchor_line")
    if not isinstance(lines, list) or not lines or len(lines) > 4:
        return None
    if not all(isinstance(l, str) for l in lines):
        return None
    if not isinstance(anchor, int) or isinstance(anchor, bool):
        return None
    if anchor < 0 or anchor > total_lines:
        return None
    text = "\n".join(l.rstrip() for l in lines)
    if not text.strip():
        return None
    if security_lint(text):
        return None
    return text, anchor


# --- helpers shared by the mapping rows --------------------------------------

def _next_patch_id(counter) -> str:
    return f"patch-{next(counter):03d}"


def _split_targets(target: str) -> list[str]:
    """analyzer/prompt.md's collapsing rule names several targets in one
    comma-joined string (spacing is inconsistent in practice); split them."""
    return [t.strip() for t in (target or "").split(",") if t.strip()]


def _rationale(rec, *, extra=None):
    parts = [(rec.get("action") or "").strip()]
    evidence = (rec.get("evidence") or "").strip()
    if evidence:
        parts.append(f"Evidence: {evidence}")
    if extra:
        parts.append(extra)
    confidence = rec.get("confidence")
    if confidence:
        parts.append(f"confidence={confidence}")
    return " ".join(p for p in parts if p)


def _lines_in_ranges(ranges) -> int:
    return sum(max(0, end - start + 1) for start, end in ranges)


def _validate_and_coalesce_ranges(ranges, skill_lines=None):
    """Validate each cited range individually (1-based, start<=end, and
    within the skill's length when known), then sort and coalesce
    overlapping and adjacent (prev_end + 1 == next_start) ranges into one.

    This is normalization, not reinterpretation: coalescing never changes
    which lines are deleted, only how the same deletion is spelled, so the
    total touched-line count is identical before and after. An individually
    invalid range is dropped on its own, with a note; it does not take the
    whole patch down with it.

    Returns (coalesced_ranges, drop_notes).
    """
    valid = []
    notes = []
    try:
        items = list(ranges)
    except TypeError:
        return [], [f"line_ranges was not a list: {ranges!r}"]

    for r in items:
        # A malformed shape here (a flat [start, end] pair instead of
        # [[start, end]], a bare int, a string...) must degrade to one
        # dropped range with a note, never raise -- one bad shape from the
        # analyzer must not take down every other recommendation in the
        # same call.
        try:
            pair = list(r)
        except TypeError:
            notes.append(f"dropped malformed range {r!r}: not a [start, end] pair")
            continue
        if len(pair) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
            notes.append(f"dropped malformed range {r!r}")
            continue
        start, end = pair
        if start < 1 or end < start:
            notes.append(f"dropped invalid range [{start}, {end}]: not 1-based/ordered")
            continue
        if skill_lines is not None and end > skill_lines:
            notes.append(f"dropped range [{start}, {end}]: exceeds skill length "
                         f"({skill_lines} lines)")
            continue
        valid.append((start, end))

    valid.sort()
    coalesced = []
    for start, end in valid:
        if coalesced and start <= coalesced[-1][1] + 1:
            prev_start, prev_end = coalesced[-1]
            coalesced[-1] = (prev_start, max(prev_end, end))
        else:
            coalesced.append((start, end))
    return [list(r) for r in coalesced], notes


def _count_orphans(text, ranges):
    """Count dangling fragments a verbatim range-deletion would leave in
    `text`: (a) '## ' headings immediately followed by another heading or
    EOF (a heading whose body was deleted), and (b) non-blank lines that
    immediately follow a deleted region which itself contained a heading (a
    body paragraph whose heading was deleted).

    This is a measurement, not a fixer: it neither repairs nor rejects the
    patch, only reports what partial-section deletion would leave behind.
    """
    original = text.splitlines()
    deleted = set()
    for start, end in ranges:
        deleted.update(range(start, end + 1))

    survivors = [(i, line) for i, line in enumerate(original, start=1)
                 if i not in deleted]
    n = len(survivors)

    count = 0
    for k, (lineno, line) in enumerate(survivors):
        if line.startswith("## "):
            if k + 1 == n or survivors[k + 1][1].startswith("## "):
                count += 1
            continue
        if not line.strip():
            continue
        # Walk back over the contiguous deleted block immediately preceding
        # this survivor (if any); if it contained a heading, this line's
        # owning heading is gone and it is now a stray paragraph.
        j = lineno - 1
        block_had_heading = False
        while j >= 1 and j in deleted:
            if original[j - 1].startswith("## "):
                block_had_heading = True
            j -= 1
        if block_had_heading:
            count += 1
    return count


# --- per-type patch builders ---------------------------------------------------

def _skill_truncate_patch(rec, skill_index, patch_id):
    target = rec.get("target", "")
    # Deliberately not coerced/iterated here: a malformed shape (e.g. a flat
    # [start, end] pair where [[start, end]] was meant) must be diagnosed
    # per-range inside _validate_and_coalesce_ranges, not raise here and take
    # every other recommendation in the same from_recommendations call down
    # with it.
    raw_ranges = rec.get("line_ranges") or []
    if not raw_ranges:
        return make_patch(
            patch_id=patch_id, kind=PATCH_SKILL_TRUNCATE, target=target,
            rationale=_rationale(rec), expected_effect=EFFECT_SKILL_TOKENS,
            source_recommendation=rec, line_ranges=[],
            skipped_reason="skill_truncate recommendation cited no line_ranges")

    skill = skill_index.get(target)
    if skill is None:
        # Without the skill's recorded content there is nothing to validate
        # ranges against and no text to run the orphan check on -- emitting
        # a "not skipped" patch here would look bounded and measured while
        # actually being neither, and harness.apply will fail on it anyway.
        return make_patch(
            patch_id=patch_id, kind=PATCH_SKILL_TRUNCATE, target=target,
            rationale=_rationale(rec), expected_effect=EFFECT_SKILL_TOKENS,
            source_recommendation=rec, line_ranges=[],
            skipped_reason=f"skill {target!r} not found in harness version "
                           "(cannot validate line ranges or measure orphans)")

    skill_lines = skill.get("lines")
    ranges, drop_notes = _validate_and_coalesce_ranges(raw_ranges, skill_lines)

    if not ranges:
        rationale = _rationale(rec, extra="; ".join(drop_notes) if drop_notes else None)
        return make_patch(
            patch_id=patch_id, kind=PATCH_SKILL_TRUNCATE, target=target,
            rationale=rationale, expected_effect=EFFECT_SKILL_TOKENS,
            source_recommendation=rec, line_ranges=[],
            skipped_reason="all cited line_ranges were invalid: " + "; ".join(drop_notes))

    skipped = None
    touched = _lines_in_ranges(ranges)
    if touched > MAX_PATCH_LINES:
        skipped = (f"truncate touches {touched} lines, exceeds "
                   f"MAX_PATCH_LINES={MAX_PATCH_LINES}")

    orphans = _count_orphans(skill["content"], ranges)
    plural = "" if orphans == 1 else "s"
    extra_parts = list(drop_notes)
    extra_parts.append(
        f"leaves {orphans} orphaned fragment{plural} from partial section deletion")

    return make_patch(
        patch_id=patch_id, kind=PATCH_SKILL_TRUNCATE, target=target,
        rationale=_rationale(rec, extra="; ".join(extra_parts) if extra_parts else None),
        expected_effect=EFFECT_SKILL_TOKENS, source_recommendation=rec,
        line_ranges=ranges, skipped_reason=skipped)


def _skill_remove_patches(rec, digest_skills, counter):
    out = []
    for name in _split_targets(rec.get("target", "")):
        info = digest_skills.get(name)
        extra = f"tokens_est={info['tokens_est']}" if info else None
        out.append(make_patch(
            patch_id=_next_patch_id(counter), kind=PATCH_SKILL_REMOVE, target=name,
            rationale=_rationale(rec, extra=extra),
            expected_effect=EFFECT_SKILL_TOKENS, source_recommendation=rec))
    return out


def _skill_insert_patch(rec, skill_index, patch_id, *, grok=_default_grok):
    target = rec.get("target", "")
    rationale = _rationale(rec)
    skill = skill_index.get(target)
    if skill is None:
        return make_patch(
            patch_id=patch_id, kind=PATCH_SKILL_INSERT, target=target,
            rationale=rationale, expected_effect=EFFECT_SKILL_TOKENS,
            source_recommendation=rec,
            skipped_reason=f"skill {target!r} not found in harness version")

    result = synthesize_insert(rec, skill["content"], grok=grok)
    if result is None:
        return make_patch(
            patch_id=patch_id, kind=PATCH_SKILL_INSERT, target=target,
            rationale=rationale, expected_effect=EFFECT_SKILL_TOKENS,
            source_recommendation=rec,
            skipped_reason="synthesize_insert produced no usable text "
                           "(model error, malformed response, or a "
                           "security_lint violation)")

    text, anchor = result
    touched = len(text.splitlines())
    skipped = None
    if touched > MAX_PATCH_LINES:
        skipped = (f"insert touches {touched} lines, exceeds "
                   f"MAX_PATCH_LINES={MAX_PATCH_LINES}")
    return make_patch(
        patch_id=patch_id, kind=PATCH_SKILL_INSERT, target=target,
        rationale=rationale, expected_effect=EFFECT_SKILL_TOKENS,
        source_recommendation=rec, insert_text=text, insert_after_line=anchor,
        skipped_reason=skipped)


def _mcp_remove_patches(rec, digest_mcp, counter):
    out = []
    for name in _split_targets(rec.get("target", "")):
        info = digest_mcp.get(name)
        extra = (f"tool_count={info.get('tool_count')} "
                 f"connect_ms={info.get('connect_ms')}") if info else None
        out.append(make_patch(
            patch_id=_next_patch_id(counter), kind=PATCH_MCP_REMOVE, target=name,
            rationale=_rationale(rec, extra=extra),
            expected_effect=EFFECT_MCP_STARTUP_MS, source_recommendation=rec))
    return out


def _mcp_action_patches(rec, digest_mcp, digest_failed, counter):
    """mcp_latency / mcp_fix_broken -> mcp.remove, only when the
    recommendation's own action text asks for removal/disabling."""
    action = rec.get("action") or ""
    if not _REMOVAL_ACTION_RE.search(action):
        return [make_patch(
            patch_id=_next_patch_id(counter), kind=PATCH_MCP_REMOVE,
            target=rec.get("target", ""), rationale=_rationale(rec),
            expected_effect=EFFECT_MCP_STARTUP_MS, source_recommendation=rec,
            skipped_reason=f"action does not request removal/disable: {action!r}")]

    out = []
    for name in _split_targets(rec.get("target", "")):
        failed = digest_failed.get(name)
        info = digest_mcp.get(name)
        if failed:
            extra = (f"wasted_ms={failed.get('wasted_ms')} "
                     f"error_type={failed.get('error_type')}")
        elif info:
            extra = (f"tool_count={info.get('tool_count')} "
                     f"connect_ms={info.get('connect_ms')}")
        else:
            extra = None
        out.append(make_patch(
            patch_id=_next_patch_id(counter), kind=PATCH_MCP_REMOVE, target=name,
            rationale=_rationale(rec, extra=extra),
            expected_effect=EFFECT_MCP_STARTUP_MS, source_recommendation=rec))
    return out


def _rules_append_patch(rec, effect, patch_id):
    rule_text = (rec.get("action") or "").strip()
    rationale = _rationale(rec)
    reasons = security_lint(rule_text)
    touched = len(rule_text.splitlines()) or 1  # an empty string is still "one" empty line
    if touched > MAX_PATCH_LINES:
        reasons.append(f"rules.append touches {touched} lines, exceeds "
                       f"MAX_PATCH_LINES={MAX_PATCH_LINES}")
    return make_patch(
        patch_id=patch_id, kind=PATCH_RULES_APPEND, target="rules",
        rationale=rationale, expected_effect=effect, source_recommendation=rec,
        rule_text=rule_text,
        skipped_reason="; ".join(reasons) if reasons else None)


# --- dedup ---------------------------------------------------------------------

def _diff_key(patch):
    """A key over the fields that determine the eventual diff. `diff` itself
    is always "" here — patches.py never applies a patch — so this is
    reconstructed from the payload fields instead."""
    kind = patch["kind"]
    if kind == PATCH_SKILL_TRUNCATE:
        return tuple(tuple(r) for r in patch["line_ranges"])
    if kind == PATCH_SKILL_INSERT:
        return (patch["insert_text"], patch["insert_after_line"])
    if kind == PATCH_RULES_APPEND:
        return patch["rule_text"]
    return None  # skill.remove / mcp.remove: (kind, target) alone is unique


def dedup(patches):
    """Collapse patches with identical (kind, target, diff_key). Keeps the
    first occurrence and notes the merged count in its rationale."""
    kept_by_key = {}
    counts = {}
    out = []
    for p in patches:
        key = (p["kind"], p["target"], _diff_key(p))
        if key not in kept_by_key:
            kept_by_key[key] = p
            counts[key] = 1
            out.append(p)
        else:
            counts[key] += 1
    for key, p in kept_by_key.items():
        if counts[key] > 1:
            p["rationale"] = (p["rationale"] +
                               f" [dedup: merged {counts[key]} identical recommendations]")
    return out


# --- entry point ----------------------------------------------------------------

def from_recommendations(recs, version, *, digest, grok=_default_grok):
    """Map analyzer recommendations onto Patch dicts.

    `grok` is forwarded to `synthesize_insert` for `skill_add_lines`
    recommendations; it defaults to the real grok-dev-shelling implementation
    so production callers need not pass it. It exists as a parameter (beyond
    what the analyzer-type mapping table names) purely so tests can inject a
    fake and never spawn grok-dev, per the global no-grok-in-tests rule.
    """
    skill_index = {s["name"]: s for s in version.get("skills", [])}
    digest_skills = {s["skill"]: s for s in digest.get("installed_skills_never_loaded", [])}
    digest_mcp = digest.get("mcp_servers", {})
    digest_failed = {f["server"]: f for f in digest.get("failed_mcp_servers", [])}

    counter = itertools.count(1)
    patches = []
    for rec in recs:
        rtype = rec.get("type")
        if rtype == "skill_keep":
            continue
        elif rtype == "skill_truncate":
            patches.append(_skill_truncate_patch(rec, skill_index, _next_patch_id(counter)))
        elif rtype == "skill_remove":
            patches.extend(_skill_remove_patches(rec, digest_skills, counter))
        elif rtype == "skill_add_lines":
            patches.append(_skill_insert_patch(rec, skill_index, _next_patch_id(counter), grok=grok))
        elif rtype == "mcp_remove":
            patches.extend(_mcp_remove_patches(rec, digest_mcp, counter))
        elif rtype in ("mcp_latency", "mcp_fix_broken"):
            patches.extend(_mcp_action_patches(rec, digest_mcp, digest_failed, counter))
        elif rtype in _RULES_APPEND_TYPES:
            patches.append(_rules_append_patch(rec, _RULES_APPEND_TYPES[rtype], _next_patch_id(counter)))
        else:
            # Recorded, not silently discarded: the analyzer's schema is an
            # enum today, but a recommendation from an older/newer analyzer
            # build could name a type this mapping doesn't handle. Not
            # raising is still correct (one unrecognized type must not crash
            # the loop) -- but a dropped recommendation with no trace
            # anywhere is indistinguishable from one the analyzer never
            # made (types.make_patch's own docstring; final-review.md I5),
            # so this becomes a skipped patch like any other
            # rejected-before-ranking recommendation instead of a silent
            # gap in the patch list.
            patches.append(make_patch(
                patch_id=_next_patch_id(counter), kind=rtype or "unknown",
                target=rec.get("target", ""), rationale=_rationale(rec),
                expected_effect="", source_recommendation=rec,
                skipped_reason=f"unrecognized recommendation type: {rtype!r}"))
    return patches
