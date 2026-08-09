"""The canonical trace.json schema.

One shape, two producers: the HuggingFace OTel dataset today, real grok
sessions later. The analyzer prompt only ever sees this — so it does not
need rewriting when the input source changes.

Everything here is plain dicts so the output is stdlib-JSON-serializable
and diffable. No pydantic, no dataclass-to-dict dance.
"""

SCHEMA_VERSION = "1"

# --- span kinds -------------------------------------------------------------
# Kept deliberately small. A viewer colours by these; the analyzer reasons
# about them. New kinds are additive.
KIND_SESSION = "session"
KIND_LLM = "llm"        # one model call
KIND_TOOL = "tool"      # a native tool invocation
KIND_MCP = "mcp"        # a tool served by an MCP server
KIND_SKILL = "skill"    # a SKILL.md load
KIND_SUBAGENT = "subagent"

ALL_KINDS = (KIND_SESSION, KIND_LLM, KIND_TOOL, KIND_MCP, KIND_SKILL, KIND_SUBAGENT)

# --- duration provenance ----------------------------------------------------
# The single most important honesty flag in this file. The HF dataset has no
# tool spans, so any tool duration derived from the gap between LLM calls is a
# guess contaminated by harness overhead and rate-limit backoff. The analyzer
# MUST refuse to raise latency findings on anything but MEASURED.
DUR_MEASURED = "measured"           # real start/end on the span itself
DUR_DERIVED = "derived"             # inferred from a gap; single call, so unambiguous
DUR_DERIVED_SHARED = "derived_shared"  # gap split across N concurrent calls
DUR_UNKNOWN = "unknown"             # could not be established at all

TRUSTWORTHY_DURATIONS = (DUR_MEASURED,)


def make_span(
    span_id,
    kind,
    name,
    *,
    parent_id=None,
    t_start_ms=0,
    duration_ms=0,
    duration_source=DUR_UNKNOWN,
    status="ok",
    tokens=None,
    attrs=None,
):
    """Build one span. t_start_ms is relative to session start, so a trace is
    positionable without any date parsing in the viewer."""
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown span kind: {kind}")
    return {
        "id": span_id,
        "parent_id": parent_id,
        "kind": kind,
        "name": name,
        "t_start_ms": int(t_start_ms),
        "duration_ms": int(duration_ms),
        "duration_source": duration_source,
        "status": status,
        "tokens": tokens or {},
        "attrs": attrs or {},
    }


def make_trace(session, spans, inventory, metrics, warnings):
    return {
        "schema_version": SCHEMA_VERSION,
        "session": session,
        "spans": spans,
        "inventory": inventory,
        "metrics": metrics,
        # Anything the normalizer could not do faithfully. Surfaced to the
        # analyzer so it can scope its own confidence instead of inventing it.
        "warnings": warnings,
    }
