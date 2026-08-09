#!/usr/bin/env python3
"""Install a corpus of REALISTIC skills modelled on genuinely popular public ones.

`make_skills.py` plants obvious, cartoonish defects — repeated filler lines,
bodies that are 90% glossary. That is useful for a smoke test but it flatters
the analyzer: nothing that blunt survives contact with a real ~/.grok/skills
directory. This file is the harder fixture. Every skill here is written to look
like something a working developer actually installed, with the names and
descriptions taken from real public skills where we could find them.

Sources (see the report in git history for per-skill attribution):
  - github.com/anthropics/skills                  (official Anthropic skills)
  - github.com/obra/superpowers                   (the `superpowers` skill pack)
  - ~/.grok/bundled/skills/                       (skills shipped with the CLI)
  - github.com/karanb192/awesome-claude-skills    (community catalog)

The defects are correspondingly subtler. A `truncate` skill here is not 80%
filler; it is a genuinely good skill that happens to carry two or three sections
of historical rationale, restated philosophy, or a duplicated worked example.
That is what real skill rot looks like, and it is what the analyzer has to catch.

Verdicts, same vocabulary as make_skills.py:
  keep      - lean, every section actionable
  truncate  - useful core, plus specific low-value sections (named in ground truth)
  remove    - would never be invoked in a Python/git workspace; pure context tax

Usage:  python3 real_skills.py --install    (writes to ~/.grok/skills)
        python3 real_skills.py --clean      (removes them again)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SKILLS_DIR = Path.home() / ".grok" / "skills"
HERE = Path(__file__).parent

# Marks fixture-owned dirs so --clean can never touch a real user skill.
SENTINEL = "x-synthetic-fixture: true"


def frontmatter(name, description, extra=""):
    return f"""---
name: {name}
description: {description}
{SENTINEL}
{extra}---
"""


# ==========================================================================
# KEEP - lean, every section earns its tokens
# ==========================================================================

TDD = """
# Test-Driven Development (TDD)

## The Iron Law

No production code without a failing test that demands it. Not "write the test
after". Not "the test is obvious, skip it". The test comes first, and you watch
it fail before you make it pass.

## Red-Green-Refactor

**RED** — Write the smallest test that expresses the next unmet requirement.
Run it. Read the failure message. If it fails for the wrong reason (import
error, typo) the test is not yet red — fix it until the failure is the one you
predicted.

**GREEN** — Write the least code that makes the test pass. Hardcoding a return
value is legitimate here; the next test will force generality. Run the full
suite, not just the new test.

**REFACTOR** — Now that the behavior is pinned, clean up. Extract, rename,
delete duplication. Re-run the suite after each move. If the suite goes red
during refactor, revert rather than debug forward.

## Good Tests

- One behavior per test. If the name needs "and", split it.
- Assert on observable behavior, not on internal calls. `assert result == 7`,
  not `assert mock_adder.call_count == 1`.
- Name the test after the requirement: `test_rejects_negative_quantity`.
- No conditionals in tests. A branch in a test means two tests.

## Bug Fixes

A bug means a test was missing. Write the test that reproduces it, watch it
fail, then fix. The reproduction test is the deliverable as much as the fix is.

## Verification Checklist

Before claiming a feature is done:

1. Every new behavior has a test that failed before the implementation existed.
2. The full suite passes, run just now, in this shell.
3. No test was weakened or skipped to get green.
"""

VERIFICATION = """
# Verification Before Completion

## The Iron Law

Evidence before assertions. You may not say "done", "fixed", "passing", or
"working" until you have run the command that proves it and read the output.

## The Gate

Before any completion claim, answer all three in the affirmative:

1. Did I run the verification command in this session, just now?
2. Did I read the actual output, not the exit code alone?
3. Does the output show the specific thing I am about to claim?

Any "no" means you are not done — you are guessing.

## Common Failures

- Claiming tests pass after editing the test file but before re-running.
- Reporting a build succeeded when only the typecheck ran.
- Saying "the fix works" when only one of the three reported symptoms was
  reproduced and checked.
- Marking a task complete because the diff looks right.

## What Counts as Evidence

The command, and its output, pasted or summarized faithfully. "Tests pass" is
not evidence. `47 passed, 0 failed in 3.2s` is evidence. If a step was skipped,
say which and why.
"""

WORKTREES = """
# Using Git Worktrees

## When to Use

Before starting feature work that should not disturb the current checkout, and
always before executing a written implementation plan. Isolation means the user
can keep working in the main checkout while the agent runs.

## Step 0 - Detect Existing Isolation

You may already be isolated. Check before creating anything:

```bash
git rev-parse --show-toplevel
git rev-parse --git-common-dir
```

If `--git-dir` and `--git-common-dir` differ, you are already in a worktree.
Stop here and report the path. Note that a submodule also looks unusual to this
check — confirm with `git rev-parse --show-superproject-working-tree`; if that
returns a path you are in a submodule, not a worktree, and should treat the
repo as normal.

## Step 1 - Create the Workspace

Prefer a native worktree tool if the harness offers one. Otherwise:

```bash
git worktree add ../<repo>-<feature> -b <feature>
```

Place it as a sibling of the repo, never inside it — a nested worktree ends up
in the parent's index and confuses every subsequent `git status`.

## Step 2 - Project Setup

A fresh worktree has no ignored files, so dependencies are missing. Install for
the project's ecosystem before running anything:

- Python: `uv sync` or `pip install -e .[dev]` inside a fresh venv
- Node: `npm ci`
- Rust: nothing needed; `cargo` shares the target dir only if configured
- Go: nothing needed

Copy across any untracked-but-required local config (`.env`, local settings)
that the main checkout has.

## Step 3 - Verify a Clean Baseline

Run the project's test command before writing any code. A worktree that starts
red hides which failures you caused. If the baseline is red, report that to the
user and ask whether to proceed.

## Cleanup

When the branch is merged or abandoned:

```bash
git worktree remove ../<repo>-<feature>
git worktree prune
```

Never `rm -rf` a worktree directory; that leaves a stale administrative entry
behind that blocks re-creating the same path.
"""

REQUEST_REVIEW = """
# Requesting Code Review

## When to Use

After completing a task or a major feature, and always before merging. The
point is to have someone check the work against the requirements, not to have
them admire the diff.

## Prepare the Diff

1. Rebase onto the current base branch so the diff shows only your change.
2. Run the full test suite and record the result.
3. Read your own diff top to bottom. Remove debug prints, commented-out code,
   and stray formatting churn — these consume reviewer attention that should go
   to the logic.

## Write the Request

Give the reviewer what they need to judge correctness, in this order:

- **What changed and why** — one paragraph, in terms of behavior, not files.
- **The requirement it satisfies** — link the issue, or quote the ask.
- **How it was verified** — the exact commands and their output.
- **What you are unsure about** — name the specific decisions you want
  challenged. A review with no stated uncertainty tends to come back with only
  style comments.

## Scope the Ask

State the review depth you want: a correctness pass, a design pass, or a
full audit. Reviewers calibrate effort to the ask, and an unscoped request
usually gets the cheapest reading.

## Do Not

- Do not request review on a branch with failing tests unless you say so
  explicitly and explain why.
- Do not bundle an unrelated refactor into a feature branch. Split it.
- Do not mark feedback resolved without either making the change or replying
  with the reason you did not.
"""


# ==========================================================================
# TRUNCATE - a good skill carrying real dead weight
# ==========================================================================

SYSTEMATIC_DEBUGGING = """
# Systematic Debugging

## The Iron Law

No fix without a root cause. A change that makes the symptom disappear is not a
fix until you can explain the mechanism by which the bug produced the symptom.

## When to Use

Any bug, any failing test, any unexpected behavior — before proposing a fix.
This includes "it works on my machine", flaky tests, and anything a previous
attempt already tried and failed to fix.

## Phase 1 - Root Cause Investigation

Read the error. All of it, including the stack frames you think are boilerplate.
Then:

1. Reproduce it deterministically. If you cannot reproduce it, you cannot
   confirm a fix; get to a reproduction first, even a slow or ugly one.
2. Find the last known-good state. `git log`, `git bisect`, or simply the last
   version where the test passed.
3. Read the code on the actual failing path. Not the code you assume runs —
   confirm it with a print, a breakpoint, or a log line.

## Phase 2 - Pattern Analysis

Ask whether this is one bug or an instance of a class:

- Does the same mistake appear elsewhere in the file, module, or codebase?
- Is there a working analogue nearby? Diff your broken path against it.
- Did the failure appear at a boundary — serialization, encoding, timezone,
  None-vs-missing, off-by-one at a slice edge? Those are where bugs cluster.

## Phase 3 - Hypothesis and Testing

State the hypothesis in one sentence, in cause-and-effect form: "the cache key
omits the tenant id, so tenant B reads tenant A's row." Then design the cheapest
experiment that would falsify it. Run it. If the experiment does not falsify the
hypothesis, you have a root cause; if it does, you have a new hypothesis and a
narrowed search space, which is progress.

Do not skip the falsification step because the hypothesis feels obvious. The
obvious hypothesis is wrong often enough that the check pays for itself.

## Phase 4 - Implementation

Write a failing test that reproduces the bug at the level of the root cause,
not at the level of the symptom. Fix. Watch the test go green. Then re-run the
whole suite — root-cause fixes touch shared code and break neighbours more often
than symptom patches do.

## Red Flags - Stop and Restart the Process

If you catch yourself doing any of these, you have left the process:

- Adding a `try/except` around the failing line without knowing what raises.
- Changing a value from 5 to 10 to "see if that helps".
- Reverting to an older library version without reading its changelog.
- Fixing a test by loosening the assertion.
- Writing "this should fix it" instead of "this fixes it because".

## Common Rationalizations

Every one of these has been said immediately before a wasted afternoon, and
each is a restatement of the same underlying error — substituting confidence for
evidence:

*"I don't need to reproduce it, the stack trace is clear."* The stack trace tells
you where the program noticed the problem, not where the problem originated.

*"This is a one-line fix, the process is overkill."* The process is cheap on a
one-line fix. It is exactly on the fixes that feel small that it gets skipped
and the real cause survives.

*"I've seen this exact error before."* You have seen an error with the same
text. Same text, different cause, is the single most common way debugging goes
wrong.

*"The tests pass now, so it's fixed."* The tests passing is consistent with a
fix and also consistent with having disabled the condition that triggered the
detection.

*"I'll add logging and come back to it."* Logging without a hypothesis produces
volume, not information. Form the hypothesis first, then log the one thing that
distinguishes it.

*"It's probably a race condition."* Sometimes true, and almost always reached
too early, because it is the explanation that requires no further investigation.

## Signals From Your Human Partner That You Are Doing It Wrong

Certain phrases from the person you are working with are reliable indicators
that you have drifted out of the process and into guessing. Learn to hear them
as a full stop rather than as ordinary feedback:

- "Are you sure that's the actual cause?" — you have presented a correlation.
- "Didn't we already try that?" — you are cycling.
- "Why would that make a difference?" — you cannot explain the mechanism.
- "Just revert it." — your changes have accumulated past the point where anyone
  can reason about which one mattered.
- "Let me take a look." — you have burned their patience; hand over cleanly with
  a summary of what has been ruled out, and what has not.

When you hear one of these, do not defend the current attempt. Return to Phase
1, state plainly what is known versus assumed, and re-enter the loop.

## Quick Reference

A condensed restatement of the four phases above, for when you have internalized
the process and want the checklist without the explanation:

1. **Investigate** — reproduce deterministically, find last known-good, read the
   real failing path.
2. **Pattern** — is this one bug or a class of bug? Is there a working analogue?
3. **Hypothesize** — one sentence, cause and effect, then design the cheapest
   falsifying experiment and run it.
4. **Implement** — failing test at the root-cause level, fix, green, full suite.

And the red-flag list, condensed: bare except, magic-number tweaking, blind
version pinning, loosened assertions, "should fix it".

## Origins and Influences

This procedure is a synthesis of several older sources. The four-phase shape
comes from the scientific-method framings common in the debugging literature of
the 1990s and 2000s, most directly the "make it fail, make it repeatable" line
of argument. The insistence on falsification rather than confirmation is
borrowed straight from Popper by way of that literature. The "root cause, not
symptom" framing traces to manufacturing quality practice — the five-whys
technique — which entered software through the lean movement.

None of this history is required in order to apply the procedure, and readers
who find it interesting are better served by the original works than by this
summary. It is retained here mainly because the section headings below once
referenced the original numbering, and removing it would leave several internal
cross-references dangling.

## When the Process Reveals No Root Cause

Occasionally the investigation genuinely bottoms out: the behavior is caused by
a third-party service, a hardware fault, or an upstream bug you cannot fix.
Say so explicitly, document what was ruled out, and choose a mitigation
deliberately — a retry, a guard, a pin — with a comment explaining that it is a
mitigation and not a fix, and a link to the upstream issue.
"""

CODE_REVIEW = """
# Strict Code Quality Review

## Core Prompt

Perform a deep code quality audit of the current branch's changes. Rethink how
to structure the changes to meaningfully improve quality without changing
behavior. Improve abstractions and modularity, reduce spaghetti, improve
succinctness and legibility. Be ambitious: if there is a clear path to improving
the implementation that involves restructuring part of the codebase, take it.

## Be Ambitious About Structural Simplification

Do not stop at "this could be a bit cleaner." Look for reframings that make
whole branches, helpers, modes, conditionals, or layers disappear. Prefer the
solution that makes the code feel inevitable in hindsight. Assume there is often
a move available that uses the existing architecture more effectively and makes
the change dramatically simpler. If you see a path to delete complexity rather
than rearrange it, push hard for that path.

## Giant Files and Giant Functions

Flag any file that has grown past roughly 800 lines, and any function past
roughly 60. Size alone is not a defect, but it is a reliable smell: report what
distinct responsibilities the file has accumulated, and propose the split along
those seams. A 1200-line file with one coherent responsibility is fine; say so
and move on.

## Spaghetti Conditions

Flag boolean expressions that have grown by accretion — more than three clauses,
mixed `and`/`or` without parentheses, negations of compound conditions. For each,
propose either a named predicate function or an early return that flattens the
nesting. Show the rewritten form, not just the complaint.

## Abstraction Quality

Ask of each new abstraction:

- Does it have exactly one reason to change?
- Does its name describe what it guarantees, or how it is implemented? The
  latter is a leak.
- Are there exactly two callers, one of which is a test? A premature
  abstraction is worse than duplication; say so.
- Does it take a boolean flag that selects between two behaviors? That is two
  functions wearing a trenchcoat.

## Naming

Names are the cheapest place to fix a comprehension problem and the one most
often skipped in review because it feels like bikeshedding. It is not, when the
name is actively wrong.

- A name that describes the implementation (`dict_of_lists`, `helper2`,
  `process_data_v2`) rather than the meaning is a finding.
- A boolean named without a predicate (`flag`, `status`, `check`) is a finding;
  `is_expired` and `has_quota` read correctly at the call site.
- Abbreviations that are not universal in the domain: expand them. `cfg` and
  `ctx` are fine; `usr_prf_mgr` is not.
- A name that lies is the most serious naming defect. `get_user` that also
  creates the user on miss must be renamed or split, and this outranks every
  structural finding in the same file.

## Comments and Docstrings

- Comments that restate the code are noise; flag them for deletion, not rewrite.
- Comments that explain *why* — a workaround, a non-obvious ordering
  constraint, a link to the upstream bug — are the ones worth demanding when
  they are absent. Any surprising line without one is a finding.
- A docstring that documents parameters the function no longer takes is worse
  than no docstring. Check every docstring in the diff against the signature
  immediately above it.
- Commented-out code has no place in a merged diff. Version control is the
  archive.

## Error Handling

- Bare `except:` or `except Exception:` without a re-raise is a defect. Name it.
- Errors swallowed into a log line and then execution continues — flag every
  instance and ask what the caller is supposed to do with the corrupt state.
- Error messages that do not include the offending value are half a message.
- A retry loop with no cap, no backoff, and no distinction between retryable and
  fatal errors is a production incident waiting for a slow dependency.
- Exceptions used for ordinary control flow across a module boundary: ask
  whether an explicit result type reads better.

## Concurrency and Resource Lifetime

- Any file, socket, lock, or connection acquired outside a `with` block: flag
  it, and check every early return and raise path between acquisition and
  release.
- Shared mutable state touched from more than one thread or task without a lock
  or a queue. Ask specifically about read-modify-write sequences, which look
  atomic and are not.
- `async def` functions that call blocking I/O. One `requests.get` inside an
  event loop stalls every other coroutine, and the symptom appears far from the
  cause.
- Background tasks created and not awaited or tracked: the reference is dropped,
  the task may be garbage collected mid-flight, and the exception is swallowed.

## Tests in the Diff

A review of the implementation without a review of the tests is half a review.

- Does each new behavior have a test that would fail if the behavior were
  removed? Mentally delete the new code and ask which test goes red. If none
  does, the test is decorative.
- Tests that assert on mock call counts rather than on outcomes will pass
  through any refactor and catch nothing.
- Tests sharing mutable module-level fixtures, or depending on execution order:
  flag now, because the failure will surface much later as a flake.
- A test with a `sleep` in it is a test that will be flaky on loaded CI.

## Performance Sanity Check

Do not micro-optimize in review, but do flag the two mistakes that are cheap to
catch and expensive to discover in production:

- A query inside a loop over rows returned by another query. Name it as N+1 and
  point at the batched form.
- An O(n) membership test — `x in some_list` — inside a loop over a collection
  of comparable size. A set costs one line.

## Reporting Format

Report findings most severe first. For each: `file:line`, one sentence on what
is wrong, one sentence on why it matters, and the concrete replacement. Do not
pad the report with findings you do not believe in — a review with three real
findings is more useful than one with twenty, of which three are real.

## Historical Notes on This Document

This skill began life as a manual review checklist maintained in a spreadsheet
by the platform team, some time around the second year of the codebase. It moved
to the internal wiki when the spreadsheet grew past what a single screen could
show, and was encoded here when the wiki page stopped being read.

The original checklist had 47 numbered items. Roughly two-thirds of those are now
enforced automatically by the formatter and the linter, and have been dropped
from the active procedure above — line length, import ordering, trailing
whitespace, quote style, the naming conventions for test functions, the rule
about module-level mutable defaults, and so on. Several more items concerned a
templating system that has since been removed from the codebase entirely.

The numbering in the old wiki page does not correspond to anything in this
document. Review comments written before the migration sometimes cite "checklist
item 12" or similar; those references cannot be resolved and should be read as
generic requests for cleanup.

## Philosophy of Code Quality

Code quality is not an absolute, and reviewers who treat it as one produce
reviews that are technically correct and practically ignored. What counts as
acceptable depends on the age of the codebase, the size of the team, the
expected lifetime of the module, and the cost of getting it wrong. A prototype
scheduled for deletion in a month warrants different scrutiny from a payments
path that will outlive everyone currently employed.

There is a further tension between local and global quality. A change can be a
clear local improvement and a global regression, if it introduces a second way
of doing something the codebase already does one way. Consistency has value
independent of the merits of the convention being followed, and a reviewer who
optimizes each diff in isolation will, over enough diffs, produce a codebase
with no conventions at all.

Reviewers should also be honest about the limits of their own judgement. Much of
what feels like a quality instinct is familiarity: code written in the idiom you
learned first reads as clean, and code written in another idiom reads as
convoluted, and neither reaction is evidence. When you find yourself unable to
articulate why something is worse — only that you would have written it
differently — that is the moment to let it go.

Finally, the purpose of review is to catch defects and transfer knowledge, in
that order. A review that catches no defects and teaches nothing has cost two
people an hour. Calibrate accordingly, and resist the temptation to apply
uniform standards across dissimilar code.

## Glossary

**Abstraction leak** — an implementation detail visible through an interface.

**Spaghetti condition** — a boolean expression grown by accretion past the point
of readability.

**God object** — a type that has accumulated responsibilities properly belonging
to several other types.

**Shotgun surgery** — a change that requires small edits in many unrelated
places, indicating a missing abstraction.

**Primitive obsession** — modelling a domain concept as a bare string or int,
so that validity has to be re-checked at every use site.

**Feature envy** — a method that reads more of another object's state than its
own, and therefore belongs on that other object.

**Temporal coupling** — two calls that must happen in a particular order, with
nothing in the type signature saying so.

## Further Reading

Numerous books and articles cover these topics in more depth than a review skill
can. Reviewers are encouraged to develop their own judgement rather than
deferring to any single source, and no specific reading is required in order to
apply the procedure above effectively.
"""

SECURITY_REVIEW = """
# Security Review

## Scope

Review the pending changes on the current branch for security defects. Focus on
the diff — pre-existing issues in untouched code are worth noting once, briefly,
but the deliverable is a judgement on whether this change is safe to merge.

## Procedure

1. Get the diff: `git diff $(git merge-base HEAD main)...HEAD`.
2. Identify every point where the change touches a trust boundary: user input,
   network response, file path, environment variable, deserialization, subprocess
   invocation, template render, or SQL construction.
3. For each, trace the value from source to sink. Ask what an attacker who
   controls the source can make happen at the sink.
4. Report findings with severity, `file:line`, the attack, and the fix.

## Injection

- **SQL** — any query built with f-strings, `%`, `.format()`, or concatenation
  is a finding, even when the interpolated value "cannot" be attacker-controlled
  today. Require parameterized queries.
- **Command** — `subprocess` with `shell=True` and any non-literal component is a
  finding. Require the list form.
- **Path** — any path built from user input must be resolved and checked to be
  inside the intended root: `Path(root, user).resolve().is_relative_to(root)`.
- **Template** — autoescaping disabled, or `| safe` applied to a value that is
  not provably static.

## Authentication and Authorization

- Every new endpoint or handler: what authenticates the caller, and what
  authorizes this specific object? Missing object-level authorization is the
  most common real vulnerability in application code and the easiest to miss in
  review, because the code looks complete.
- Comparisons of secrets, tokens, or signatures must be constant-time.
- Session or token lifetime changes, and anything that widens a scope, get
  flagged regardless of intent.

## Secrets and Sensitive Data

- No credentials, keys, or tokens in the diff. Check test fixtures too.
- Log statements that include a request body, a header dict, an exception with
  embedded arguments, or a user record.
- New third-party calls: what is being sent, and was that reviewed?

## Cryptography

Nothing in a diff should be implementing a primitive. What you are checking is
correct use of a library:

- Random values used for tokens, session ids, password reset links, or nonces
  must come from `secrets`, never `random`. `random` is seeded predictably and
  its output is recoverable from a handful of samples.
- Passwords hashed with anything other than a memory-hard KDF — argon2, scrypt,
  or bcrypt — is a finding. A single round of SHA-256, salted or not, is not
  password hashing.
- Encryption without authentication (AES-CBC, AES-ECB, raw XOR) is a finding.
  Require an AEAD mode.
- A hardcoded IV, nonce, or salt is a finding even when the key is loaded from
  the environment.
- Certificate verification disabled (`verify=False`, a custom trust-all
  callback) is a finding regardless of the comment next to it.

## Deserialization and File Handling

- `pickle.loads`, `yaml.load` without `SafeLoader`, `eval`, `exec`, and
  `marshal` applied to anything that crossed a process boundary: each is a
  direct path to code execution. There is no safe way to accept a pickle from an
  untrusted source.
- Archive extraction (`zipfile`, `tarfile`) without validating entry paths.
  `tarfile.extractall` on an untrusted archive writes wherever the archive says.
- File uploads: the stored filename must be generated, not taken from the
  client, and the content type must be determined by inspection rather than by
  the declared header.
- Temporary files created with a predictable name in a shared directory.

## Web-Specific Checks

If the diff touches an HTTP layer:

- New state-changing endpoints need CSRF protection unless they are token
  authenticated and the token is not sent ambiguously.
- Redirects built from a request parameter must be validated against an
  allowlist; an open redirect is a phishing primitive and a common OAuth
  vulnerability.
- CORS changes: `Access-Control-Allow-Origin` reflecting the request Origin,
  combined with credentials, is equivalent to no same-origin policy at all.
- Cookies set without `HttpOnly`, `Secure`, and an explicit `SameSite`.
- Error responses that differ observably between "no such user" and "wrong
  password" leak account existence.

## Multi-Tenancy

In any system with tenants, the single highest-value check is that every query
added by the diff is scoped by tenant. Read each new query and find the tenant
predicate. If it is missing, the finding is critical regardless of whether the
calling code happens to filter afterwards — the next caller will not.

Also check caches: a cache key that omits the tenant id serves one tenant's data
to another, and does so intermittently, which makes it very hard to diagnose
from reports.

## Dependencies

Any new or bumped dependency: check that the package name is the one intended
(typosquats are a live threat), that the version is not yanked, and that the
change does not pull in a transitive package with a known advisory.

Also check how it is pinned. A range specifier on a build-time dependency means
the next CI run may install code nobody reviewed. Lockfile changes that touch
packages unrelated to the stated dependency change deserve a question.

## Reporting

Order by exploitability, not by CVSS score. For each finding give the concrete
attack — "a user who can set `filename` can read `/etc/passwd`" — not the
category name. If you find nothing, say that plainly and list what you checked;
an empty report with no stated coverage is indistinguishable from not looking.

## Background: Why OWASP, and How the Categories Have Moved

The category names used above descend from the OWASP Top Ten, which has been
revised roughly every three to four years since its first publication. The
revisions matter mainly because the names have shifted underneath long-lived
internal documents, and reviewers occasionally encounter older tickets using
retired terminology.

Injection was for many years the single top category, covering SQL, command, and
LDAP injection together. A later revision merged cross-site scripting into
Injection, which surprised people who had been treating XSS as a separate
discipline. "Broken Access Control" rose to the top position in the 2021
revision, reflecting what practitioners had been saying for a decade: that
missing object-level authorization is more common in real applications than any
injection class. "Sensitive Data Exposure" was renamed to "Cryptographic
Failures" to shift the emphasis from the symptom to the cause. "Insecure
Deserialization" was folded into "Software and Data Integrity Failures", which
also absorbed supply-chain concerns after several high-profile incidents.

None of this affects the procedure above. It is recorded here because internal
tickets from before the renames still use the old category labels, and a
reviewer reading them should know that "A3: Sensitive Data Exposure" and
"Cryptographic Failures" refer to the same body of concerns.

## Compliance Framework Cross-Reference

The following table was assembled when the organization was pursuing its first
audit, and maps review findings onto control identifiers in several frameworks.
It is not used during review — the auditors work from their own evidence
collection — but it is kept here so the mapping is not lost.

- Access control findings map to SOC 2 CC6.1, CC6.2, and CC6.3; to ISO 27001
  Annex A 9.2 and A 9.4; and to PCI DSS requirements 7 and 8.
- Cryptographic findings map to SOC 2 CC6.7; ISO 27001 A 10.1; PCI DSS 3 and 4.
- Logging findings map to SOC 2 CC7.2; ISO 27001 A 12.4; PCI DSS 10.
- Dependency and supply-chain findings map to SOC 2 CC8.1; ISO 27001 A 14.2;
  PCI DSS 6.3.
- Input validation findings map to PCI DSS 6.5 and to the corresponding secure
  development controls in ISO 27001 A 14.2.5.

The identifiers above reflect the framework versions current at the time the
table was written and have not been re-verified since. Anyone actually preparing
audit evidence should work from the current control catalogs rather than this
list.

## Glossary of Vulnerability Classes

**IDOR (Insecure Direct Object Reference)** — an endpoint that accepts an object
identifier and returns the object without checking that the caller may see it.

**SSRF (Server-Side Request Forgery)** — the server can be induced to make a
request to an attacker-chosen URL, typically reaching internal services.

**XXE (XML External Entity)** — an XML parser configured to resolve external
entities, allowing file read or SSRF via a crafted document.

**Prototype pollution** — a JavaScript-specific class where attacker-controlled
keys reach `Object.prototype` and change behavior globally.

**Deserialization gadget** — a class whose deserialization has side effects,
chained by an attacker to reach code execution.

**TOCTOU (Time of Check to Time of Use)** — a check and the action it guards are
separated in time, and the state changes in between.

**Zip slip** — an archive entry whose path escapes the extraction directory via
`../` components.
"""

REFACTORING = """
# Refactoring Patterns

## What This Skill Is For

Detecting code smells and applying a named, behavior-preserving transformation
to remove them. Refactoring means the tests do not change. If you are changing
tests, you are changing behavior, and this is not a refactor.

## Preconditions

Never start without all three:

1. A test suite that covers the code you are about to move, running green now.
2. A clean working tree, so you can revert cheaply.
3. A named target smell. "Tidying up" is not a target and produces diffs nobody
   can review.

## Smell: Long Function

**Detect** — more than one level of abstraction in the body; comments that act
as section headers; a variable used only in one contiguous block.

**Transform** — Extract Function on each commented block, naming the new
function after the comment. Then delete the comment; if the name says it, the
comment is redundant. Repeat until the outer function reads as a sequence of
named steps.

## Smell: Long Parameter List

**Detect** — more than four parameters, or several parameters that are always
passed together.

**Transform** — Introduce Parameter Object for the co-travelling group. If the
parameters are flags, prefer Replace Parameter with Explicit Methods — one
function per flag combination that is actually used.

## Smell: Conditional Complexity

**Detect** — nesting past two levels; a chain of `elif` dispatching on a type or
a string tag.

**Transform** — Replace Nested Conditional with Guard Clauses first, because it
is nearly free and often reveals that the remaining logic is simple. If a type
tag remains, Replace Conditional with Polymorphism, or in Python a dispatch
dict, which is usually the lighter-weight version of the same idea.

## Smell: Duplicated Code

**Detect** — the same three-or-more-line sequence in two places. Two occurrences
is a data point; three is a pattern.

**Transform** — Extract Function if the duplicates are identical; Form Template
Method if they differ only in the middle. Resist extracting when the two copies
happen to be identical today but are governed by different requirements — they
will diverge, and the shared function will grow a flag.

## Smell: Primitive Obsession

**Detect** — a bare `str` that is validated in more than one place; an `int`
carrying units; a tuple whose fields are documented in a comment.

**Transform** — Replace Data Value with Object, or in Python a `NewType` or a
frozen dataclass. The win is that validation happens once, at construction.

## Working Safely

Make one transformation at a time. Run the suite after each. Commit after each
green run — a refactoring branch with one enormous commit cannot be bisected,
and bisectability is most of the value of doing this in small steps at all.

## The Catalog and Its Numbering

The transformation names above are drawn from the standard refactoring catalog,
which has existed in several editions with substantially different contents. The
first edition used Java examples throughout and included a number of
transformations that only make sense in a language with checked exceptions and
no first-class functions. The second edition re-did the examples in JavaScript
and dropped or renamed several entries.

This matters for one practical reason: the numbering. Older internal documents
and code comments occasionally cite refactorings by their catalog page or by an
informal number, and those citations do not resolve against the current edition.
"Refactoring 6.1" is not a stable reference. When citing a transformation, use
its name.

Several names have also drifted. "Replace Temp with Query" was formerly split
across two entries; "Extract Method" became "Extract Function" once the catalog
stopped assuming everything was a method on a class; "Move Method" and "Move
Field" were unified into "Move". Where this document and an older one disagree
on a name, prefer this one, and understand that the underlying transformation is
almost certainly the same.

## Extended Discussion: When Not to Refactor

There is a persistent belief that refactoring is always net positive because it
is behavior-preserving, and therefore risk-free. This is not true, and the ways
it is untrue are worth setting out at length even though none of them change the
procedure above.

Behavior preservation is only as good as the test suite. On code with thin
coverage, a refactor is an unverified rewrite with a reassuring name. The honest
sequence there is to add characterization tests first — tests that assert what
the code currently does, correct or not — and only then transform.

Refactoring also has a real cost in review attention and in merge conflicts. A
large restructuring that lands on a Friday will conflict with every branch in
flight, and the cost of resolving those conflicts is borne by people who did not
choose to do the refactor. Restructurings should be small, frequent, and
announced, or large, rare, and coordinated. The failure mode is large,
frequent, and unannounced.

There is a further category of code that should simply be left alone: code that
is stable, well-understood by nobody, and about to be deleted. Untouched code
that works has an empirical track record. Refactored code has an argument. On a
module scheduled for removal in a quarter, the track record is worth more.

Finally, the aesthetic pull of refactoring is strong enough that it substitutes
for progress. If you cannot state which future change the refactor makes easier,
you are tidying, and tidying is a legitimate but low-priority activity that
should not be presented as risk-reduction work.

## Related Reading

The catalog itself is the reference; secondary summaries, including this one,
lose the preconditions and the mechanics that make each transformation safe.
No specific reading is required to apply the procedure above.
"""

WRITING_PLANS = """
# Writing Plans

## When to Use

You have a spec or a set of requirements for a multi-step task. Write the plan
before touching code. A plan that is written after the code is a summary, and
summaries do not catch design problems.

## Scope Check

First decide whether a plan is warranted. A task that is one file and under an
hour does not need one. A task that touches three subsystems, or that another
session will execute, does. If in doubt, write the plan — an unnecessary plan
costs ten minutes; a missing one costs a rewrite.

## Plan Structure

```
  # <Feature> Implementation Plan

  ## Goal
  One paragraph. What is true after this is done that is not true now.

  ## Constraints
  Global rules applying to every task below: the language version, the test
  command, files that must not change, backward-compatibility requirements.

  ## Task N: <Component>
  - Files: exact paths, created or modified
  - Change: what the code does after this task
  - Verify: the exact command that proves the task is done
```

## Task Right-Sizing

Each task should be independently verifiable and take a single focused session.
If a task's Verify line is "run the whole feature", the task is too big — split
it until each one has a specific check. If two tasks always have to be done
together to keep the suite green, merge them; a task that cannot land on its own
is not a task.

## No Placeholders

Every task names real file paths and real function names. "Update the relevant
handlers" is not a task. If you do not know which handlers, that ignorance is
itself the first task: "Task 1: enumerate the handlers that call `parse_row`."

## Self-Review Before Handing Off

Read the plan as though you were the executor with no context:

- Can I do Task 1 without asking a question?
- Does each task say how to verify it?
- Is the ordering forced by real dependencies, or did I just list them in the
  order I thought of them?
- What is the riskiest assumption, and which task tests it? Move that task
  earlier.

## Rationale: Why Written Plans Beat Working It Out As You Go

The argument for writing plans down is not that planning is inherently virtuous.
It is narrower than that, and rests on three observations about how multi-step
work actually fails.

The first is that design errors are cheapest to fix before code exists. This is
the oldest observation in software engineering and remains the least
internalized. A wrong interface caught in a plan costs a paragraph rewrite; the
same error caught after four files depend on it costs a day.

The second is that a plan externalizes the dependency ordering. Held in the
head, the ordering is re-derived on every interruption, and the re-derivation is
lossy. Written down, it survives a weekend.

The third is about handoff. Work that is executed in a different session, by a
different person or a different agent, has to be transmitted somehow. A plan is
the transmission format. The alternative — reconstructing intent from a diff —
is strictly worse and is the reason so much half-finished work is abandoned
rather than picked up.

Against these, the standard objection is that plans go stale. They do. A plan is
a hypothesis about the order of work, and hypotheses get revised. The response is
to revise the plan when reality diverges, not to skip writing it. A stale plan
that is corrected in five minutes is still cheaper than no plan.

## Alternative Formats Considered and Rejected

Several other plan formats were tried before settling on the structure above,
and are recorded here so they are not re-proposed.

**A single prose document.** Readable, but there is no natural place to record
per-task verification, so verification silently disappears from the process.

**A ticket per task in the issue tracker.** Good for coordination across people,
bad for a single execution session: the context is spread across N pages and the
global constraints have nowhere to live.

**A checklist with no file paths.** Fast to write, and consistently produces the
"update the relevant handlers" failure mode described above.

**A full design document with alternatives and trade-offs.** Appropriate for
architectural decisions, disproportionate for implementation work. If the
decision needs that treatment, write a design doc and let the plan reference it.

## Appendix: Worked Example

The following is a complete plan for a small feature, reproduced in full to show
the structure applied end to end. It restates the structure section above, in
filled-in form.

```
  # Rate-Limit Response Headers Implementation Plan

  ## Goal
  Every API response carries X-RateLimit-Limit, X-RateLimit-Remaining, and
  X-RateLimit-Reset, so clients can back off before being throttled.

  ## Constraints
  - Python 3.11, no new dependencies
  - Test command: pytest tests/ -x
  - Do not change the existing limiter storage schema

  ## Task 1: Expose remaining-quota from the limiter
  - Files: src/limits/bucket.py
  - Change: TokenBucket gains a `snapshot()` returning (limit, remaining, reset_at)
  - Verify: pytest tests/limits/test_bucket.py -k snapshot

  ## Task 2: Attach headers in middleware
  - Files: src/http/middleware.py
  - Change: RateLimitMiddleware calls snapshot() and sets the three headers on
    every response, including 429s
  - Verify: pytest tests/http/test_middleware.py -k headers

  ## Task 3: Document the headers
  - Files: docs/api/rate-limits.md
  - Change: new section listing the three headers and their semantics
  - Verify: grep -c X-RateLimit docs/api/rate-limits.md returns 3
```
"""


# ==========================================================================
# REMOVE - would never be invoked in a Python/git workspace
# ==========================================================================

SLACK_GIF = """
# Slack GIF Creator

## Slack's Constraints

Slack rejects or degrades animations that exceed its limits. Design to these
numbers from the start rather than compressing at the end:

- Emoji: 128x128 px, under 128 KB, square
- Inline image: under 2 MB for reliable autoplay
- Frame rate: 15-20 fps reads as smooth; below 10 fps reads as broken

## Workflow

1. Decide the loop length. Slack emoji read best at 1-2 seconds.
2. Render frames with PIL at 2x the target size, then downsample once at the
   end — downsampling per frame produces shimmer.
3. Assemble with `imageio` or `PIL.Image.save(save_all=True)`.
4. Validate with the size checker before uploading; an oversized emoji fails
   silently at upload time.

## Animation Concepts

- **Ease, don't lerp.** Linear motion reads as mechanical. Use an ease-in-out
  curve on position and scale.
- **Loop seamlessly.** The last frame must lead back into the first. Test by
  playing the GIF twice through.
- **Hold the readable frame.** A 30-frame animation where the subject is legible
  in 3 frames is a flicker, not an animation.

## Palette

GIF is limited to 256 colors. Quantize deliberately with a fixed palette rather
than letting the encoder choose per frame, which causes color flicker across the
loop. Transparency must be a single fully-transparent index — GIF has no alpha
channel, so soft edges against an unknown Slack background will fringe.

## Validation

Run the bundled checker before delivering: it verifies dimensions, byte size,
frame count, and that the first and last frames differ (a common bug produces a
one-frame "animation" that Slack displays as a static image).
"""

ALGORITHMIC_ART = """
# Algorithmic Art

## Setup

Sketches are p5.js, run in a single HTML file with the library inlined. Always
seed the RNG explicitly so a result can be reproduced:

```js
let SEED = 12345;
function setup() { randomSeed(SEED); noiseSeed(SEED); }
```

Never use unseeded `random()`. An unreproducible good result is a lost result.

## Parameter Exploration

Expose every magic number as a named constant at the top of the sketch, then
generate a contact sheet: render the same seed across a grid of parameter
values so the user can see the space rather than one point in it. Three to five
parameters is the practical limit before the grid stops being readable.

## Flow Fields

The canonical construction: sample Perlin noise over the canvas, map the noise
value to an angle, and advect particles along the resulting vector field.

- Noise scale controls feature size. 0.001-0.01 is the useful band; above that
  the field becomes uncorrelated noise.
- Draw particle trails with low alpha and many steps rather than thick strokes;
  density does the shading.
- Reseed particles that leave the canvas, or the field empties out.

## Particle Systems

Keep the simulation and the rendering separate. A particle holds position,
velocity, age, and nothing about how it is drawn. This makes it cheap to
re-render the same simulation with different marks.

Cap the particle count by measured frame time, not by a guess. A sketch that
drops to 5 fps is not exploratory.

## Color

Work in HSB. Pick a narrow hue band and vary saturation and brightness — wide
hue variation reads as noise. Sample the palette from a fixed list rather than
randomizing hue per particle.

## Composition

- Leave negative space. A canvas filled edge to edge has no subject.
- Break symmetry deliberately; perfect symmetry reads as a test pattern.
- Render at 4x the display size and downsample for the final export.

## Originality

Create original work. Do not reproduce or closely imitate an identifiable
artist's series, and do not use an artist's name as a style prompt. If the user
references a specific artist, take the formal property they are after — the
palette, the density, the mark type — and build it independently.

## Export

Export a still with `saveCanvas()` at high resolution, and record the seed and
every parameter value in the filename or an adjacent JSON sidecar. A rendered
image without its parameters cannot be iterated on.
"""

BRAND_GUIDELINES = """
# Brand Guidelines

## When to Apply

Any artifact that a person outside the team will see: slides, one-pagers,
landing pages, diagrams, report covers. Internal scratch work does not need it.

## Color

- Primary text on light backgrounds: near-black, never pure `#000`.
- Backgrounds: the warm off-white from the palette, not `#fff`.
- Accent: one accent per artifact. Two accents means neither is an accent.
- Charts get the sequential palette in order; do not reassign colors per chart.

## Typography

- Headings and body use the brand serif and sans respectively, with the system
  fallback stack specified in the reference file.
- Set body copy at 16-18 px with a 1.5-1.6 line height.
- Never set body copy in the display weight, and never letterspace lowercase
  text.

## Layout

Use the 8 px spacing scale. Keep line length between 60 and 75 characters. Left
align body text; centered paragraphs are for pull quotes only.

## Logo Use

Maintain clear space equal to the height of the logotype on all sides. Do not
recolor, outline, rotate, or place the logo on a busy photograph. On dark
backgrounds use the supplied reversed asset rather than inverting it yourself.

## Checking Your Work

Before delivering, view the artifact at 50% zoom. If the hierarchy is not
readable at that size, the type scale is too flat.
"""

IOS_RELEASE = """
# iOS App Release

## Pre-flight

Before starting a release build, confirm all of these on the release branch:

- Version and build number bumped in the target's Info settings
- `CFBundleShortVersionString` matches the marketing version in App Store Connect
- No `DEBUG` conditionals guarding shipping behavior
- Third-party SDK keys point at production, not staging

## Archive

```bash
xcodebuild -workspace App.xcworkspace \\
  -scheme App -configuration Release \\
  -archivePath build/App.xcarchive archive
```

Archive from a clean build directory. An incremental archive can silently embed
a stale framework binary and fail App Store validation with an error that names
the wrong target.

## Signing

Use manual signing for release builds. Automatic signing regenerates profiles
and has produced builds signed with a development certificate that passed local
validation and were rejected on upload.

Verify before export:

```bash
codesign -dv --verbose=4 build/App.xcarchive/Products/Applications/App.app
```

Confirm the authority line names the Distribution certificate and the entitlements
match `App.entitlements` exactly. A mismatch in the push-notification environment
is the most common cause of silent APNs failure in production.

## Export and Upload

Export with an `ExportOptions.plist` specifying `method: app-store`, then upload
with `xcrun altool` or Transporter. Wait for the processing email before
attempting to attach the build to a version — attaching a still-processing build
leaves the version in a state that has to be cleared by support.

## TestFlight

Push to internal testers first. Internal builds skip review and surface crashes
within an hour of distribution. Only promote to an external group after the
internal group has generated at least one session per device family you support.

## Symbolication

Upload the dSYM bundle to the crash reporter immediately after archiving. The
dSYM lives inside the archive and is discarded if the archive is cleaned;
unsymbolicated production crash reports cannot be recovered later.

## App Review

- Provide a demo account with working credentials for any login wall. Review
  rejects for a non-functional demo account more often than for anything else.
- If the build adds a permission prompt, the usage-description string must
  explain the user-facing benefit, not the technical reason.
- Note any feature behind a server flag in the review notes; reviewers who see a
  flag-disabled feature described in the changelog will reject for missing
  functionality.

## Phased Release and Rollback

Enable phased release for anything touching networking or persistence. There is
no rollback on the App Store: the only remedy for a bad build is to halt the
phased rollout and ship a fix, which takes at minimum a full review cycle.
Expedited review exists but is granted sparingly and should be reserved for data
loss or a total launch failure.
"""


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
# (description, body, verdict, why, low_value_sections, source)
SKILLS = {
    # ---- keep -------------------------------------------------------------
    "test-driven-development": (
        "Use when implementing any feature or bugfix, before writing implementation code",
        TDD, "keep",
        "Lean. Five sections, all procedural, all reachable from an ordinary "
        "Python feature request.",
        [],
        "github.com/obra/superpowers/blob/main/skills/test-driven-development (real name + verbatim description)",
    ),
    "verification-before-completion": (
        "Use when about to claim work is complete, fixed, or passing, before committing or creating PRs - requires running verification commands and confirming output before making any success claims; evidence before assertions always",
        VERIFICATION, "keep",
        "Shortest skill in the set and entirely actionable; the gate applies to "
        "every task the agent finishes.",
        [],
        "github.com/obra/superpowers/blob/main/skills/verification-before-completion (real name + verbatim description)",
    ),
    "using-git-worktrees": (
        "Use when starting feature work that needs isolation from current workspace or before executing implementation plans - ensures an isolated workspace exists via native tools or git worktree fallback",
        WORKTREES, "keep",
        "Every section is a step with a command. Directly applicable in a git "
        "workspace.",
        [],
        "github.com/obra/superpowers/blob/main/skills/using-git-worktrees (real name + verbatim description)",
    ),
    "requesting-code-review": (
        "Use when completing tasks, implementing major features, or before merging to verify work meets requirements",
        REQUEST_REVIEW, "keep",
        "Compact checklist, no historical or philosophical padding.",
        [],
        "github.com/obra/superpowers/blob/main/skills/requesting-code-review (real name + verbatim description)",
    ),

    # ---- truncate ---------------------------------------------------------
    "systematic-debugging": (
        "Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes",
        SYSTEMATIC_DEBUGGING, "truncate",
        "The four phases and the red flags are the working core. Quick Reference "
        "restates the phases verbatim, Origins is pure lineage, and the two "
        "rationalization/signals sections are long prose that never changes an action.",
        [
            "Common Rationalizations",
            "Signals From Your Human Partner That You Are Doing It Wrong",
            "Quick Reference",
            "Origins and Influences",
        ],
        "github.com/obra/superpowers/blob/main/skills/systematic-debugging (real name, verbatim description, real section titles)",
    ),
    "code-review": (
        "Run an extremely strict maintainability review for abstraction quality, giant files, and spaghetti-condition growth. Use for a deep code quality audit or an especially harsh maintainability review.",
        CODE_REVIEW, "truncate",
        "The review procedure is strong; the last four sections are a spreadsheet "
        "migration story, a page of calibration philosophy, definitions the model "
        "already knows, and a Further Reading section that recommends nothing.",
        [
            "Historical Notes on This Document",
            "Philosophy of Code Quality",
            "Glossary",
            "Further Reading",
        ],
        "~/.grok/bundled/skills/code-review/SKILL.md (real name + verbatim description, body modelled on it)",
    ),
    "security-review": (
        "Complete a security review of the pending changes on the current branch",
        SECURITY_REVIEW, "truncate",
        "Procedure and the per-class checks are actionable. The OWASP renaming "
        "history, the stale compliance control mapping, and the glossary are "
        "reference material that never affects a finding.",
        [
            "Background: Why OWASP, and How the Categories Have Moved",
            "Compliance Framework Cross-Reference",
            "Glossary of Vulnerability Classes",
        ],
        "Claude Code built-in /security-review (real name + verbatim description); body written from domain knowledge",
    ),
    "refactoring-patterns": (
        "Code smell detection and systematic refactoring techniques for behavior-preserving cleanup",
        REFACTORING, "truncate",
        "The smell/transform pairs are the useful part. Catalog numbering trivia, "
        "a long essay on when not to refactor, and a Related Reading section that "
        "names no reading are dead weight.",
        [
            "The Catalog and Its Numbering",
            "Extended Discussion: When Not to Refactor",
            "Related Reading",
        ],
        "github.com/karanb192/awesome-claude-skills (real catalog entry name + paraphrased description); body written from domain knowledge",
    ),
    "writing-plans": (
        "Use when you have a spec or requirements for a multi-step task, before touching code",
        WRITING_PLANS, "truncate",
        "Structure, right-sizing and self-review are the skill. The rationale essay, "
        "the rejected-alternatives log, and the appendix — which re-renders the "
        "structure section as a filled-in example — are duplicated or inert.",
        [
            "Rationale: Why Written Plans Beat Working It Out As You Go",
            "Alternative Formats Considered and Rejected",
            "Appendix: Worked Example",
        ],
        "github.com/obra/superpowers/blob/main/skills/writing-plans (real name, verbatim description, real section titles)",
    ),

    # ---- remove -----------------------------------------------------------
    "slack-gif-creator": (
        "Knowledge and utilities for creating animated GIFs optimized for Slack. Provides constraints, validation tools, and animation concepts. Use when users request animated GIFs for Slack.",
        SLACK_GIF, "remove",
        "Nothing in a Python analyzer repo will ever trigger this. Pure context tax.",
        [],
        "github.com/anthropics/skills/tree/main/skills/slack-gif-creator (real name + description from the repo frontmatter)",
    ),
    "algorithmic-art": (
        "Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use this when users request creating art using code, generative art, flow fields, or particle systems.",
        ALGORITHMIC_ART, "remove",
        "Creative-coding skill with no overlap with the workspace; never invoked.",
        [],
        "github.com/anthropics/skills/tree/main/skills/algorithmic-art (real name + verbatim description)",
    ),
    "brand-guidelines": (
        "Applies official brand colors and typography to any artifact that may benefit from the company look-and-feel. Use it when brand colors or style guidelines, visual formatting, or company design standards apply.",
        BRAND_GUIDELINES, "remove",
        "Design-system skill. No artifact in this workspace is user-facing; never invoked.",
        [],
        "github.com/anthropics/skills/tree/main/skills/brand-guidelines (real name, description lightly de-Anthropic-ised)",
    ),
    "ios-app-release": (
        "Archive, sign, and ship an iOS build through TestFlight and App Store review, including dSYM upload and phased rollout.",
        IOS_RELEASE, "remove",
        "Mobile-platform skill in a Python/git repo with no Xcode project. Never invoked.",
        [],
        "NO REAL SOURCE FOUND - written from domain knowledge as the mobile-specific 'remove' case",
    ),
}


def build():
    """Returns {name: (description, body, verdict, why, low_value_sections, source)}."""
    return SKILLS


def install():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    truth = {}
    for name, (desc, body, verdict, why, low_value, source) in build().items():
        d = SKILLS_DIR / name
        d.mkdir(exist_ok=True)
        text = frontmatter(name, desc) + body
        (d / "SKILL.md").write_text(text)
        lines = text.count("\n") + 1
        sections = [l[3:].strip() for l in body.splitlines() if l.startswith("## ")]
        # A named low-value section that isn't in the body is a fixture bug, and
        # would silently make the scoring unfalsifiable. Fail loudly instead.
        missing = [s for s in low_value if s not in sections]
        assert not missing, f"{name}: low_value_sections not found in body: {missing}"
        entry = {
            "expected_verdict": verdict,
            "why": why,
            "source": source,
            "sections": sections,
            "bytes": len(text),
            "lines": lines,
            "tokens_est": len(text) // 4,
        }
        if low_value:
            entry["low_value_sections"] = low_value
        truth[name] = entry
        print(f"  {verdict:9} {name:32} {lines:4} lines  {len(text):6,}B  "
              f"{len(sections):2} sections  {len(low_value)} low-value")

    out = HERE / "real_ground_truth.json"
    out.write_text(json.dumps({"skills": truth}, indent=2))

    total = sum(v["bytes"] for v in truth.values())
    dead = sum(v["bytes"] for v in truth.values() if v["expected_verdict"] == "remove")
    by_verdict = {}
    for v in truth.values():
        by_verdict[v["expected_verdict"]] = by_verdict.get(v["expected_verdict"], 0) + 1
    print(f"\ninstalled {len(truth)} skills into {SKILLS_DIR}")
    print("verdicts: " + "  ".join(f"{k}={n}" for k, n in sorted(by_verdict.items())))
    print(f"total {total:,}B (~{total // 4:,} tok)  |  never-invoked {dead:,}B (~{dead // 4:,} tok)")
    print(f"ground truth -> {out}")


def clean():
    if not SKILLS_DIR.exists():
        print("nothing to clean")
        return
    removed = 0
    for name in build():
        d = SKILLS_DIR / name
        f = d / "SKILL.md"
        # Two locks: the dir must be one of ours by name, AND carry the sentinel.
        # The name check also keeps us off make_skills.py's fixtures, which share
        # the sentinel and live in the same directory.
        if f.is_file() and SENTINEL in f.read_text():
            shutil.rmtree(d)
            removed += 1
    print(f"removed {removed} fixture skills")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        clean()
    elif a.install:
        install()
    else:
        for n, (d, b, v, w, lv, src) in build().items():
            n_sec = sum(1 for l in b.splitlines() if l.startswith("## "))
            print(f"{v:9} {n:32} {len(b):6,}B  {n_sec:2} sections")
