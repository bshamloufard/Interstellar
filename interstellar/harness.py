"""Harness snapshot, materialize, and patch-apply.

Turns a real grok home (+ optional project dir) into a content-addressed
`HarnessVersion` (interstellar/types.py), materializes any version into a
fresh, isolated `GROK_HOME`, and applies a list of `Patch` objects to a
version to derive a new one -- purely, without touching disk.

Isolation is GROK_HOME: setting that env var makes grok resolve skills/,
config.toml, and sessions/ from the given dir. This module never writes to
the user's real ~/.grok; callers own choosing `dest` for materialize().

GROK_HOME does not fully isolate the skill/MCP catalog: grok also discovers
skills from `<grok_home>/bundled/skills/` (ships with the binary), from
`~/.agents/skills/` (not disableable), from the equivalent project-scoped
roots (`<project>/.grok/skills/`, `<project>/.agents/skills/`), and, unless
disabled, from Claude- and Cursor-sourced skills/MCP servers under
`~/.claude*` and `~/.cursor`. `materialize()` seals the last kind
(compat.claude/compat.cursor) by forcing all six cells of both tables to
`false` in config.toml -- appending fresh tables when absent, rewriting in
place when the version's own config already has one, so a user's existing
`[compat.claude]` table can never leave a surface un-sealed while the
recorded warning claims otherwise. The other roots are not sealable, so
`snapshot()` only surfaces a same-name collision as a warning via
`uncontrolled_skills()` rather than trying to hide them.

Project-scope skills (`scope == "project"`, read from `<project>/.grok/
skills/`) need to be materialized into `<project_dest>/.grok/skills/`, not
`<dest>/skills/`: grok's own precedence is `./.grok/skills/` >
`<repo>/.grok/skills/` > `$GROK_HOME/skills/`, so a project skill written
into the GROK_HOME tier is silently shadowed by the original, unpatched copy
still sitting in the replay workdir. See `materialize(..., project_dest=)`.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import tomllib
from pathlib import Path

from interstellar.types import (
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
    PATCH_SKILL_INSERT,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_TRUNCATE,
    make_harness_version,
    make_mcp_server,
    make_patch_application,
    make_skill,
)

# One TOML table header per line, e.g. "[mcp_servers.foo]" or
# "[[marketplace.sources]]". Anchored so we don't match text mid-line.
_TOML_HEADER_RE = re.compile(r"^\s*\[\[?([^\[\]]+)\]\]?\s*$")

# Skill roots grok discovers that GROK_HOME does not control, checked against
# the grok home. "bundled/skills" is relative to the grok home -- verified
# empirically (not just from docs): grok auto-creates and populates
# `<home>/bundled/skills/` with 23 skills on first run against a *fresh*,
# otherwise-empty home, so the root ships with the binary but is resolved
# per-GROK_HOME, not globally. "~/.agents/skills" is always the real user
# home, independent of GROK_HOME. Neither is disableable, so a `skill.remove`
# patch targeting a name that also lives here is a no-op we must disclose,
# not a bug to fix -- see module docstring.
UNCONTROLLED_SKILL_ROOTS = ("bundled/skills", "~/.agents/skills")

# Project-relative counterparts, checked separately against `project_dir`
# since they aren't expressible as a single grok-home-relative label.
UNCONTROLLED_PROJECT_SKILL_ROOTS = ("project/.grok/skills", "project/.agents/skills")

# The six on/off surfaces every [compat.<vendor>] table exposes, each
# defaulting to true (enabled) when omitted -- confirmed against
# ~/.grok/docs/user-guide/05-configuration.md.
_COMPAT_CELLS = ("skills", "rules", "agents", "mcps", "hooks", "sessions")
_COMPAT_VENDORS = ("claude", "cursor")


# --- hashing -----------------------------------------------------------------

def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_version_id(skills, config_text, extra_rules=()):
    """sha256 over sorted (name, sha256) skill pairs + config sha + extra_rules,
    first 12 hex.

    Sorting the skill pairs makes this independent of iteration/directory
    order, so two identical harnesses always produce the same id. extra_rules
    are hashed in list order (not sorted) since they're appended at run time
    via `--rules` and their order can matter to the model. types.py's
    `make_harness_version` docstring states these are part of the harness
    identity ("so they feed version_id") -- without this, a `rules.append`
    patch produces a treatment version_id identical to its control's, per
    review-harness-replay.md Important 5.
    """
    h = hashlib.sha256()
    for name, sha in sorted((s["name"], s["sha256"] or "") for s in skills):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("utf-8"))
        h.update(b"\x00")
    h.update(_sha256_text(config_text).encode("utf-8"))
    for rule in extra_rules:
        h.update(b"\x00")
        h.update(rule.encode("utf-8"))
    return h.hexdigest()[:12]


# --- frontmatter ---------------------------------------------------------

def _frontmatter_description(content):
    """Pull `description:` out of a SKILL.md's leading `---` frontmatter."""
    if not content.startswith("---"):
        return ""
    end = content.find("\n---", 3)
    if end == -1:
        return ""
    for line in content[3:end].splitlines():
        line = line.strip()
        if line.startswith("description:"):
            value = line[len("description:"):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return ""


# --- uncontrolled roots ---------------------------------------------------

def _uncontrolled_root(home, label):
    if label == "~/.agents/skills":
        # Always the real user home -- deliberately not GROK_HOME-relative,
        # since grok reads this root regardless of which home is active.
        return Path.home() / ".agents" / "skills"
    return Path(home) / label


def _uncontrolled_project_root(project_dir, label):
    if label == "project/.grok/skills":
        return Path(project_dir) / ".grok" / "skills"
    return Path(project_dir) / ".agents" / "skills"


def _skills_under(root):
    if not root.is_dir():
        return []
    return [entry.name for entry in sorted(root.iterdir())
            if (entry / "SKILL.md").is_file()]


def uncontrolled_skills(home, project_dir=None) -> list[dict]:
    """Skills grok's catalog includes that this harness does not govern.

    Returns `{name, root}` for every skill found under
    `UNCONTROLLED_SKILL_ROOTS`, plus -- when `project_dir` is given --
    `UNCONTROLLED_PROJECT_SKILL_ROOTS`. Consumed by the report to render a
    caveat; does not affect `apply()`.
    """
    out = []
    for label in UNCONTROLLED_SKILL_ROOTS:
        root = _uncontrolled_root(home, label)
        out.extend({"name": n, "root": label} for n in _skills_under(root))
    if project_dir is not None:
        for label in UNCONTROLLED_PROJECT_SKILL_ROOTS:
            root = _uncontrolled_project_root(project_dir, label)
            out.extend({"name": n, "root": label} for n in _skills_under(root))
    return out


def _uncontrolled_warnings(skills, home, project_dir=None):
    names = {s["name"] for s in skills}
    # A controlled scope="project" skill is *sourced from* exactly
    # project/.grok/skills -- that's its origin, not a shadow, and
    # materialize(..., project_dest=) routes it back to that same tier. Only
    # flag project/.grok/skills entries this version did NOT already capture,
    # e.g. a skill added to the project after this snapshot was taken.
    project_names = {s["name"] for s in skills if s["scope"] == "project"}
    seen = set()
    out = []
    for u in uncontrolled_skills(home, project_dir=project_dir):
        if u["root"] == "project/.grok/skills" and u["name"] in project_names:
            continue
        key = (u["name"], u["root"])
        if u["name"] in names and key not in seen:
            seen.add(key)
            out.append(
                f"skill '{u['name']}' also exists in {u['root']}/ — removing "
                "it from the harness will not remove it from the agent's "
                "catalog"
            )
    return out


# --- snapshot ------------------------------------------------------------

def _read_skills(skills_dir, scope, warnings):
    out = []
    if not skills_dir.is_dir():
        return out
    for entry in sorted(skills_dir.iterdir()):
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # Recorded, not swallowed: a dropped skill changes version_id and
            # the materialized home as though it were never installed, so its
            # absence must be visible to whoever reads this version.
            warnings.append(f"could not read skill '{entry.name}' at {skill_md}: {exc}")
            continue
        skill = make_skill(
            entry.name,
            path=skill_md,
            content=content,
            scope=scope,
            description=_frontmatter_description(content),
        )
        skill["sha256"] = _sha256_text(content)
        out.append(skill)
    return out


def _read_mcp_servers(config_path, warnings):
    """Parse `[mcp_servers.*]` tables from config.toml. Never raises."""
    if not config_path.is_file():
        warnings.append(f"no config.toml at {config_path}")
        return [], ""
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"could not read config.toml: {exc}")
        return [], ""
    try:
        parsed = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        warnings.append(f"could not parse config.toml: {exc}")
        return [], config_text
    servers = []
    for name, table in (parsed.get("mcp_servers") or {}).items():
        if not isinstance(table, dict):
            continue
        servers.append(make_mcp_server(
            name,
            command=table.get("command", ""),
            args=table.get("args", []),
            enabled=table.get("enabled", True),
        ))
    return servers, config_text


def snapshot(grok_home: Path, project_dir: Path | None = None):
    """Build a content-addressed `HarnessVersion` from a real grok home."""
    grok_home = Path(grok_home)
    warnings = []

    skills = _read_skills(grok_home / "skills", scope="user", warnings=warnings)
    if project_dir is not None:
        skills += _read_skills(
            Path(project_dir) / ".grok" / "skills", scope="project", warnings=warnings,
        )

    mcp_servers, config_text = _read_mcp_servers(grok_home / "config.toml", warnings)

    warnings.extend(_uncontrolled_warnings(skills, grok_home, project_dir=project_dir))

    return make_harness_version(
        version_id=_compute_version_id(skills, config_text),
        skills=skills,
        mcp_servers=mcp_servers,
        config_text=config_text,
        source=str(grok_home),
        warnings=warnings,
    )


# --- materialize -----------------------------------------------------------

def _find_table_block(lines, table_name):
    """Locate a top-level `[table_name]` table's line span: (start, end),
    `end` exclusive, spanning from its header through to (not including) the
    next header line. Returns None if no such table exists. Only matches the
    table by exact name -- a subtable like `[table_name.sub]` ends the block,
    it is not swept into it (compat tables are flat, no subtables per docs).
    """
    start = None
    end = None
    for i, line in enumerate(lines):
        m = _TOML_HEADER_RE.match(line)
        if not m:
            continue
        header = m.group(1).strip()
        if start is not None:
            end = i
            break
        if header == table_name:
            start = i
    if start is None:
        return None
    return start, (end if end is not None else len(lines))


_COMPAT_CELL_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=\s*(true|false)\s*(#.*)?$")


def _existing_compat_cells(lines, start, end):
    """Cell -> bool for every recognized `cell = true|false` line in a
    [compat.<vendor>] block's body (lines[start+1:end])."""
    cells = {}
    for line in lines[start + 1:end]:
        m = _COMPAT_CELL_RE.match(line)
        if m and m.group(1) in _COMPAT_CELLS:
            cells[m.group(1)] = m.group(2) == "true"
    return cells


def _compat_table_text(vendor):
    body = "".join(f"{cell} = false\n" for cell in _COMPAT_CELLS)
    return f"[compat.{vendor}]\n{body}\n"


def _seal_config_text(config_text):
    """Force every cell of [compat.claude] and [compat.cursor] to `false`.

    Rewrites an existing table's cells in place (never appends a second
    header for a table that already exists -- that's a TOML duplicate-table
    parse error) and appends a fresh table when one is absent. Returns
    `(new_config_text, notes)`; `notes` describes exactly what happened to
    each vendor's table so the caller can record an accurate warning instead
    of a blanket "sealed" claim that may not be true for every cell.
    """
    lines = config_text.splitlines(keepends=True)
    notes = []
    for vendor in _COMPAT_VENDORS:
        table_name = f"compat.{vendor}"
        block = _find_table_block(lines, table_name)
        if block is None:
            if lines and lines[-1].strip():
                lines.append("\n")
            lines.extend(_compat_table_text(vendor).splitlines(keepends=True))
            notes.append(f"[compat.{vendor}] added (was absent, defaults to enabled)")
            continue
        start, end = block
        existing = _existing_compat_cells(lines, start, end)
        # Cells default to enabled (true) when the table exists but omits
        # them -- see _COMPAT_CELLS comment -- so an absent cell still counts
        # as an override once we force it to false.
        overridden = [c for c in _COMPAT_CELLS if existing.get(c, True) is not False]
        lines[start:end] = _compat_table_text(vendor).splitlines(keepends=True)
        if overridden:
            notes.append(
                f"[compat.{vendor}] existing table rewritten, cells forced "
                f"false that were not already false: {', '.join(overridden)}"
            )
        else:
            notes.append(f"[compat.{vendor}] already fully sealed")
    return "".join(lines), notes


def materialize(version, dest: Path, *, auth_from: Path, seal: bool = True,
                 project_dest: Path | None = None) -> Path:
    """Write `version` out as a runnable GROK_HOME at `dest`.

    Idempotent: skills/ and config.toml are replaced wholesale so a stale
    skill from a previous materialize never survives.

    `seal=True` (default) forces `[compat.claude]`/`[compat.cursor]` to fully
    disabled in the written config.toml, since GROK_HOME alone does not stop
    grok from also discovering Claude/Cursor-sourced skills and MCP servers
    on the same machine (see module docstring). This always actually seals --
    an existing table is rewritten, not skipped -- and the warning recorded
    on `version["warnings"]` names exactly what was changed. Pass
    `seal=False` to opt out.

    `scope == "project"` skills are written to `<project_dest>/.grok/skills/`
    when `project_dest` is given (grok's own precedence puts that root above
    `$GROK_HOME/skills/`, so a project skill written only into `dest` would
    be shadowed by the original file still in the replay workdir). Without
    `project_dest`, project-scope skills are not written anywhere and a
    warning names them -- never written into the GROK_HOME tier where the
    caller would believe, wrongly, that they took effect.
    """
    # Resolved to absolute here, once, for every caller: grok resolves a
    # relative GROK_HOME against its OWN cwd (the run's workdir in a replay,
    # not wherever this tool was invoked from), so a relative dest silently
    # points grok at an empty home instead of this one and it reports
    # "Not signed in" instead of finding auth.json. This is the backstop --
    # every home path in the system originates here.
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    user_skills = [s for s in version["skills"] if s["scope"] != "project"]
    project_skills = [s for s in version["skills"] if s["scope"] == "project"]

    skills_dir = dest / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True)
    for skill in user_skills:
        skill_dir = skills_dir / skill["name"]
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill["content"], encoding="utf-8")

    if project_dest is not None:
        # Wiped unconditionally, even when this version has zero project
        # skills, so a stale project skill from a previous materialize()
        # call never survives -- the same idempotency guarantee as skills_dir.
        project_skills_dir = Path(project_dest) / ".grok" / "skills"
        if project_skills_dir.exists():
            shutil.rmtree(project_skills_dir)
        if project_skills:
            project_skills_dir.mkdir(parents=True)
            for skill in project_skills:
                skill_dir = project_skills_dir / skill["name"]
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(skill["content"], encoding="utf-8")
    elif project_skills:
        names = ", ".join(sorted(s["name"] for s in project_skills))
        warning = (
            f"project-scope skill(s) not materialized (no project_dest "
            f"given): {names} -- patches against them will not take "
            "effect, since grok resolves ./.grok/skills/ above "
            "$GROK_HOME/skills/ and the original copy still wins"
        )
        if warning not in version["warnings"]:
            version["warnings"].append(warning)

    config_text = version["config_text"]
    if seal:
        config_text, notes = _seal_config_text(config_text)
        warning = "sealed materialized home: " + "; ".join(notes)
        if warning not in version["warnings"]:
            version["warnings"].append(warning)
    (dest / "config.toml").write_text(config_text, encoding="utf-8")

    auth_src = Path(auth_from) / "auth.json"
    if not auth_src.is_file():
        raise FileNotFoundError(f"no auth.json at {auth_src}")
    # Copied, not symlinked: grok rewrites auth.json on token refresh, and a
    # symlink would let that refresh leak back into the source home.
    dest_auth = dest / "auth.json"
    shutil.copyfile(auth_src, dest_auth)
    # copyfile copies bytes only, not mode -- explicitly restore 0600 so a
    # 0600 source credential doesn't land world/group-readable under the
    # default umask.
    os.chmod(dest_auth, 0o600)

    (dest / "sessions").mkdir(parents=True, exist_ok=True)
    return dest


# --- apply -------------------------------------------------------------------

def _skip(patch_id, reason):
    return make_patch_application(patch_id=patch_id, applied=False, reason=reason)


def _find_skill_index(skills, name):
    for i, s in enumerate(skills):
        if s["name"] == name:
            return i
    return None


def _rebuild_skill(old_skill, new_content):
    skill = make_skill(
        old_skill["name"],
        path=old_skill["path"],
        content=new_content,
        scope=old_skill["scope"],
        description=old_skill["description"],
    )
    skill["sha256"] = _sha256_text(new_content)
    return skill


def _unified_diff(old_text, new_text, label):
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    return "\n".join(difflib.unified_diff(
        old_lines, new_lines, fromfile=label, tofile=label, lineterm="",
    ))


def _merge_ranges(ranges):
    """Sort ascending and fold overlapping [start, end] pairs into one."""
    ordered = sorted((int(s), int(e)) for s, e in ranges)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _apply_skill_truncate(patch, skills):
    patch_id, name = patch["patch_id"], patch["target"]
    idx = _find_skill_index(skills, name)
    if idx is None:
        return _skip(patch_id, f"skill not found: {name}"), skills

    ranges = patch.get("line_ranges") or []
    if not ranges:
        return _skip(patch_id, "no line_ranges given"), skills

    skill = skills[idx]
    lines = skill["content"].splitlines(keepends=True)
    num_lines = len(lines)
    for start, end in ranges:
        if not (1 <= start <= end <= num_lines):
            return _skip(
                patch_id,
                f"line range [{start}, {end}] out of bounds for {name} "
                f"({num_lines} lines)",
            ), skills

    # Apply merged, non-overlapping ranges high-to-low so an earlier
    # deletion never shifts a later range's indices.
    new_lines = list(lines)
    deleted = 0
    for start, end in sorted(_merge_ranges(ranges), reverse=True):
        deleted += end - start + 1
        del new_lines[start - 1:end]
    new_content = "".join(new_lines)

    new_skills = list(skills)
    new_skills[idx] = _rebuild_skill(skill, new_content)
    diff = _unified_diff(skill["content"], new_content, f"{name}/SKILL.md")
    return make_patch_application(
        patch_id=patch_id, applied=True, diff=diff, lines_changed=deleted,
    ), new_skills


def _apply_skill_remove(patch, skills):
    patch_id, name = patch["patch_id"], patch["target"]
    idx = _find_skill_index(skills, name)
    if idx is None:
        return _skip(patch_id, f"skill not found: {name}"), skills

    old = skills[idx]
    diff = _unified_diff(old["content"], "", f"{name}/SKILL.md")
    new_skills = skills[:idx] + skills[idx + 1:]
    return make_patch_application(
        patch_id=patch_id, applied=True, diff=diff, lines_changed=old["lines"],
    ), new_skills


def _apply_skill_insert(patch, skills):
    patch_id, name = patch["patch_id"], patch["target"]
    idx = _find_skill_index(skills, name)
    if idx is None:
        return _skip(patch_id, f"skill not found: {name}"), skills

    insert_text = patch.get("insert_text")
    if not insert_text:
        return _skip(patch_id, "no insert_text given"), skills

    after = patch.get("insert_after_line")
    if after is None:
        after = 0

    skill = skills[idx]
    lines = skill["content"].splitlines(keepends=True)
    num_lines = len(lines)
    if not (0 <= after <= num_lines):
        return _skip(
            patch_id,
            f"insert_after_line {after} out of bounds for {name} ({num_lines} lines)",
        ), skills

    block = insert_text if insert_text.endswith("\n") else insert_text + "\n"
    inserted_lines = block.splitlines(keepends=True)
    new_lines = lines[:after] + inserted_lines + lines[after:]
    new_content = "".join(new_lines)

    new_skills = list(skills)
    new_skills[idx] = _rebuild_skill(skill, new_content)
    diff = _unified_diff(skill["content"], new_content, f"{name}/SKILL.md")
    return make_patch_application(
        patch_id=patch_id, applied=True, diff=diff, lines_changed=len(inserted_lines),
    ), new_skills


def _apply_mcp_remove(patch, mcp_servers, config_text):
    patch_id, name = patch["patch_id"], patch["target"]
    target = f"mcp_servers.{name}"

    lines = config_text.splitlines(keepends=True)
    remove = [False] * len(lines)
    in_target = False
    found = False
    for i, line in enumerate(lines):
        m = _TOML_HEADER_RE.match(line)
        if m:
            header = m.group(1).strip()
            # Matches the server's own table and any nested subtable, e.g.
            # "mcp_servers.foo.env" -- but not "mcp_servers.foobar".
            in_target = header == target or header.startswith(target + ".")
            found = found or in_target
        remove[i] = in_target

    if not found:
        return _skip(patch_id, f"mcp server not found: {name}"), mcp_servers, config_text

    new_config_text = "".join(l for i, l in enumerate(lines) if not remove[i])
    new_servers = [m for m in mcp_servers if m["name"] != name]
    diff = _unified_diff(config_text, new_config_text, "config.toml")
    return make_patch_application(
        patch_id=patch_id, applied=True, diff=diff, lines_changed=sum(remove),
    ), new_servers, new_config_text


def _apply_rules_append(patch, extra_rules):
    patch_id = patch["patch_id"]
    rule_text = patch.get("rule_text")
    if not rule_text:
        return _skip(patch_id, "no rule_text given"), extra_rules

    new_rules = list(extra_rules) + [rule_text]
    diff = _unified_diff("", rule_text, "extra_rules")
    return make_patch_application(
        patch_id=patch_id, applied=True, diff=diff, lines_changed=1,
    ), new_rules


def apply(version, patches):
    """Derive a new `HarnessVersion` with `patches` applied. Never touches disk."""
    skills = list(version["skills"])
    mcp_servers = list(version["mcp_servers"])
    config_text = version["config_text"]
    extra_rules = list(version["extra_rules"])
    applications = []

    for patch in patches:
        kind = patch["kind"]
        if kind == PATCH_SKILL_TRUNCATE:
            app, skills = _apply_skill_truncate(patch, skills)
        elif kind == PATCH_SKILL_REMOVE:
            app, skills = _apply_skill_remove(patch, skills)
        elif kind == PATCH_SKILL_INSERT:
            app, skills = _apply_skill_insert(patch, skills)
        elif kind == PATCH_MCP_REMOVE:
            app, mcp_servers, config_text = _apply_mcp_remove(patch, mcp_servers, config_text)
        elif kind == PATCH_RULES_APPEND:
            app, extra_rules = _apply_rules_append(patch, extra_rules)
        else:
            app = _skip(patch["patch_id"], f"unknown patch kind: {kind}")
        applications.append(app)

    new_version = make_harness_version(
        version_id=_compute_version_id(skills, config_text, extra_rules),
        skills=skills,
        mcp_servers=mcp_servers,
        config_text=config_text,
        extra_rules=extra_rules,
        source=version.get("source"),
        warnings=list(version.get("warnings") or []),
    )
    return new_version, applications
