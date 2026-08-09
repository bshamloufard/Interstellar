#!/usr/bin/env python3
"""Generate a skill fixture set with PLANTED, DOCUMENTED defects.

The point is ground truth. Synthetic data is only worth anything if you know
in advance what the analyzer is supposed to find — otherwise you are grading
its output by vibes. Every skill here is created with an intended verdict, and
`ground_truth.json` records it. Later we score the agent against that file.

Defect classes, one per intended recommendation type:
  keep       - lean, invoked, earns its context
  truncate   - invoked, but most of the body is never relevant
  remove     - never invoked at all; pure context tax
  add_lines  - invoked, but missing an instruction, which causes observable
               thrash in the trace (the agent retries / flails)

Usage:  python3 make_skills.py --install    (writes to ~/.grok/skills)
        python3 make_skills.py --clean      (removes them again)
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


# --------------------------------------------------------------------------
# 1. KEEP - lean and genuinely used
# --------------------------------------------------------------------------
LEAN_BODY = """
# Python Import Hygiene

When asked to clean up imports in a Python file:

1. Read the file.
2. Remove imports that no name in the file references.
3. Order the remaining imports: stdlib, third-party, local — blank line between groups.
4. Do not reorder anything inside a `try:`/`except ImportError:` block; those are load-bearing.
5. Never delete an import named in `__all__`.

Report which imports you removed and why.
"""

# --------------------------------------------------------------------------
# 2. TRUNCATE - a small useful core buried in a large irrelevant body
# --------------------------------------------------------------------------
FILLER_SECTIONS = [
    ("Historical Context", """
This skill originated from an internal audit process developed over several years.
The original process was a manual checklist maintained in a spreadsheet, later
migrated to a wiki, and eventually encoded here. Understanding this lineage is
not required in order to apply the skill, but it explains some of the naming.
The original checklist had 47 items, many of which are now handled automatically
by linters and are therefore omitted from the active procedure below."""),
    ("Rationale and Philosophy", """
Code quality is not an absolute. What counts as acceptable depends on the age of
the codebase, the size of the team, and the expected lifetime of the module under
review. A prototype scheduled for deletion in a month warrants different scrutiny
than a payments path. Reviewers should calibrate accordingly, and should resist
the temptation to apply uniform standards across dissimilar code."""),
    ("Team Conventions (deprecated)", """
Earlier versions of this document specified tab-based indentation and an 80
column limit. Those conventions have been superseded by the automatic formatter
and should no longer be enforced manually during review. They are retained here
only so that older review comments referencing them remain intelligible."""),
    ("Glossary", """
Abstraction leak: an implementation detail visible through an interface.
Spaghetti condition: a boolean expression grown by accretion past readability.
God object: a type that accumulated responsibilities belonging elsewhere.
Shotgun surgery: a change that requires edits in many unrelated places."""),
    ("Further Reading", """
Numerous books and articles cover these topics in depth. Reviewers are encouraged
to develop their own judgement rather than relying on any single source. No
specific reading is required in order to use this skill effectively."""),
]

def bloated_body():
    core = """
# Strict Code Audit

## Procedure

1. List the changed files.
2. For each file, flag any function longer than 60 lines.
3. Flag any conditional with more than 3 clauses.
4. Report findings most severe first. Cite `file:line`.
"""
    out = [core]
    # Repeat the filler so the useful core is a small fraction of the body.
    for _ in range(3):
        for title, text in FILLER_SECTIONS:
            out.append(f"\n## {title}\n{text}\n")
    return "".join(out)


# --------------------------------------------------------------------------
# 3. REMOVE - never invoked
# --------------------------------------------------------------------------
UNUSED = {
    "release-notes": (
        "Draft release notes from merged pull requests for a tagged version.",
        "# Release Notes\n\nCollect merged PRs since the last tag, group by label, "
        "and write a changelog entry per group.\n" + ("\nDetail line for the release process.\n" * 40),
    ),
    "db-migrate": (
        "Plan and apply a database schema migration with rollback safety.",
        "# DB Migration\n\nWrite the forward migration, then the rollback, then verify "
        "both against a scratch database.\n" + ("\nDetail line about migration safety.\n" * 40),
    ),
    "k8s-deploy": (
        "Deploy a service to Kubernetes and verify rollout health.",
        "# Kubernetes Deploy\n\nApply the manifest, watch the rollout, roll back on "
        "failed readiness probes.\n" + ("\nDetail line about deployment verification.\n" * 40),
    ),
}

# --------------------------------------------------------------------------
# 4. ADD_LINES - invoked but under-specified, causing observable thrash
# --------------------------------------------------------------------------
# Deliberately omits: which file to write to, and that it must not create the
# file if it exists. The agent has to guess, which shows up in the trace as
# extra exploratory tool calls.
UNDERSPECIFIED = """
# Changelog Entry

Add an entry describing the current change.

Keep it to one line. Use the past tense.
"""


def build():
    """Returns {name: (description, body, verdict, why)}."""
    skills = {
        "py-import-hygiene": (
            "Clean up unused and misordered Python imports in a file.",
            LEAN_BODY, "keep",
            "Lean body, every line actionable, invoked during the run.",
        ),
        "strict-audit": (
            "Run a strict code quality audit flagging long functions and complex conditionals.",
            bloated_body(), "truncate",
            "Only the 6-line Procedure section is used; the rest is history, "
            "philosophy, deprecated conventions, glossary and further reading.",
        ),
        "changelog-entry": (
            "Add a one-line changelog entry for the current change.",
            UNDERSPECIFIED, "add_lines",
            "Never says WHICH file to edit or how to handle an existing entry, "
            "so the agent burns tool calls discovering it.",
        ),
    }
    for name, (desc, body) in UNUSED.items():
        skills[name] = (desc, body, "remove", "Never invoked in any session; pure context tax.")
    return skills


def install():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    truth = {}
    for name, (desc, body, verdict, why) in build().items():
        d = SKILLS_DIR / name
        d.mkdir(exist_ok=True)
        text = frontmatter(name, desc) + body
        (d / "SKILL.md").write_text(text)
        lines = text.count("\n") + 1
        truth[name] = {
            "expected_verdict": verdict,
            "why": why,
            "bytes": len(text),
            "lines": lines,
            "tokens_est": len(text) // 4,
        }
        print(f"  {verdict:9} {name:22} {lines:4} lines  {len(text):6,}B")

    out = HERE / "ground_truth.json"
    out.write_text(json.dumps({"skills": truth}, indent=2))
    total = sum(v["bytes"] for v in truth.values())
    waste = sum(v["bytes"] for v in truth.values() if v["expected_verdict"] == "remove")
    print(f"\ninstalled {len(truth)} skills into {SKILLS_DIR}")
    print(f"total {total:,}B (~{total // 4:,} tok)  |  never-invoked {waste:,}B (~{waste // 4:,} tok)")
    print(f"ground truth -> {out}")


def clean():
    if not SKILLS_DIR.exists():
        print("nothing to clean")
        return
    removed = 0
    for d in SKILLS_DIR.iterdir():
        f = d / "SKILL.md"
        # Only ever delete dirs we stamped. Never touch a real user skill.
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
        for n, (d, b, v, w) in build().items():
            print(f"{v:9} {n:22} {len(b):6,}B")
