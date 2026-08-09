# Synthetic trace corpus for the grok-build session analyzer

Generates **real grok-build sessions** with **planted, documented defects**, normalizes
them to a canonical `trace.json`, and scores an analyzer agent against the plant.

The point of the plant is that "good synthetic data" is otherwise unfalsifiable. If
you don't know what the analyzer is supposed to find, you're grading its output by
vibes. Here every scenario ships with the answer key, held in a field the model
never sees.

## Why not the HuggingFace dataset

`Exgentic/agent-llm-traces` (1781 sessions, 38k spans) was evaluated first and
rejected for this purpose. Measured against what the analyzer needs:

| need | HF dataset | grok sessions |
|---|---|---|
| tool durations | none — every span is `llm_call`; only derivable from inter-span gaps, 8% of which are negative | **measured** `tool_completed.duration_ms` |
| skill loads | `Skill` tool declared, `Available skills:` list **empty in 300/300 rows**, 0 invocations | derived from `read_file` of `SKILL.md`, with byte/section detail |
| subagents | 1 row in 400, no `parent_span_id` anywhere | `spawn_subagent` spans |
| MCP | one synthetic `environment` server, no connect/health events | real servers with connect ms, tool counts, handshake failures |
| permission waits | absent | `permission_resolved.wait_ms` |

It remains useful for tool-inventory work (955 tools defined vs 258 called is a
genuinely strong signal) — see `normalizer/normalize.py`. It cannot support skill,
subagent, or latency findings at all.

## Pipeline

```
make_skills.py --install      fixture skills with planted defects
real_skills.py --install      13 real public skills (Anthropic / superpowers)
        |
run_corpus.py --all           drives grok-dev headlessly, one session per scenario
        |                     -> manifest.json  (scenario -> session_id + answer key)
        |
build_corpus.py               normalize + label + COVERAGE GATE
        |                     -> ../traces/corpus/*.json + index.json
        |
analyzer/analyze.py --all     digest (deterministic) -> grok --json-schema
        |                     -> analyzer/results/*.result.json
        |
analyzer/score.py             recall against the plant
```

## Upstream: native skill events (a959d1a, + our 7af84ba)

Alex's `a959d1a` added `skill_activated` to `events.jsonl` — `skill_name`,
`trigger` (`slash_command` | `skill_md_read` | `skill_tool`), `plugin_source`,
and `related_tool_call_id` — plus `skillCallCount` / `skillsUsed` in
`signals.json`. `skill_md_read` is exactly the derivation this repo had been
doing, now first-class. The normalizer prefers it and marks each span
`source: native_event`; the old heuristic remains as fallback so pre-rebase
traces still read (`source: derived`).

One gap remained, and we patched main for it (`7af84ba`). `skill_activated`
recorded THAT a skill activated but not what it cost. On the `slash_command`
trigger there is no backing tool call, so no `tool_completed` row exists to
join for a size — the body entered context unpriced. Since every
truncate/remove recommendation is driven by token cost, main could not produce
the data this corpus needs. We added optional `skill_bytes` / `skill_lines`,
measured from the skill path each call site already holds. Optional, not
required, so an unreadable file records "not measured" rather than a
misleading 0.

Verified end to end: `/strict-audit app.py` now emits
`skill_bytes: 5832, skill_lines: 125` with no tool call in the session.

## The mechanism that did not previously exist

Skills now have a native event (above). Two span kinds are still **derived**,
because main has no dedicated event for either — each is tagged
`attrs.derived_from` so nothing masquerades as native instrumentation:

- **subagent** — a `spawn_subagent` tool call
- **mcp** — a `use_tool` call, server parsed from the qualified tool name

(There is still no `skill` *tool* in the registered toolset — verified by asking
a live build to enumerate its own tools, 31 registered, none is `skill`. Skills
reach the model as a `read_file` of `SKILL.md` or via slash. Both are now
reported by `skill_activated`.)

These two are the obvious next candidates for the same upstream treatment
`skill_activated` just received.

On top of that it runs a **section-utilization** pass: each `## ` section of a
loaded skill is checked for whether its *distinctive* vocabulary (its words minus
words shared with other sections) appears anywhere downstream. That is what makes
line-level recommendations possible — a truncation finding cites real
`[start, end]` ranges rather than a guess.

Two deliberate honesty constraints:

- A section with no distinctive vocabulary is reported `referenced: null`
  (unjudgeable), never "unused".
- Verbatim-duplicate sections are detected and excluded from cancelling each
  other's vocabulary, and reported separately as their own finding.

## Ground truth

`ground_truth.json` / `real_ground_truth.json` map each skill to an expected
verdict — `keep`, `truncate`, `remove`, `add_lines` — plus why. `manifest.json`
carries the per-scenario `expects`. `build_corpus.py` attaches it to each trace
under a top-level `ground_truth` key, and `analyzer/analyze.py:digest()` strips
it before the model sees anything.

## Coverage gate

`build_corpus.py` exits non-zero unless every finding type has raw material in the
corpus. A prompt path that never fires during development is a prompt path that
ships untested, so this is a build gate rather than a report.

## Scoring

`score.py` reports recall against the plant. Precision is handled deliberately
loosely: the idle/failed MCP servers are flagged in every session and those
findings are *true*, just environmental rather than scenario-specific. Counting
them as false positives would punish correct work. The one precision rule enforced
is the one the prompt actually mandates — every recommendation must cite a number.

## Costs

Roughly $0.05 per session to generate, $0.04 per session to analyze.

## Reset

```sh
python3 make_skills.py --clean     # only deletes dirs carrying the fixture sentinel
python3 real_skills.py --clean
cp ~/.grok/config.toml.bak ~/.grok/config.toml   # restores pre-MCP config
```
