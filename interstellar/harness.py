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
`~/.agents/skills/` (not disableable), and, unless disabled, from Claude- and
Cursor-sourced skills/MCP servers under `~/.claude*` and `~/.cursor`.
`materialize()` seals the second kind (compat.claude/compat.cursor) by
appending disable tables to config.toml. `bundled/skills/` and
`~/.agents/skills/` are not sealable -- they are a constant across both arms
of a paired comparison, so `snapshot()` only surfaces them as a warning via
`uncontrolled_skills()` rather than trying to hide them.
"""

from __future__ import annotations

import difflib
import hashlib
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

# Skill roots grok discovers that GROK_HOME does not control. "bundled/skills"
# is relative to the grok home (it ships with the binary); "~/.agents/skills"
# is always the real user home, independent of GROK_HOME. Neither is
# disableable, so a `skill.remove` patch targeting a name that also lives here
# is a no-op we must disclose, not a bug to fix -- see module docstring.
UNCONTROLLED_SKILL_ROOTS = ("bundled/skills", "~/.agents/skills")

# Appended (never prepended) to a materialized config.toml so grok stops
# discovering skills/rules/agents/mcp servers/hooks/sessions sourced from a
# Claude or Cursor install on the same machine. Appending, not prepending,
# means a `mcp.remove` patch's table-header text surgery -- which runs on the
# version's config_text before this is added -- can never land inside these
# tables.
_SEAL_CONFIG_TEXT = """
[compat.claude]
skills = false
rules = false
agents = false
mcps = false
hooks = false
sessions = false

[compat.cursor]
skills = false
rules = false
agents = false
mcps = false
hooks = false
sessions = false
"""


# --- hashing -----------------------------------------------------------------

def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_version_id(skills, config_text):
    """sha256 over sorted (name, sha256) skill pairs + config sha, first 12 hex.

    Sorting makes this independent of iteration/directory order, so two
    identical harnesses always produce the same id.
    """
    h = hashlib.sha256()
    for name, sha in sorted((s["name"], s["sha256"] or "") for s in skills):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("utf-8"))
        h.update(b"\x00")
    h.update(_sha256_text(config_text).encode("utf-8"))
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


def uncontrolled_skills(home) -> list[dict]:
    """Skills grok's catalog includes that GROK_HOME does not govern.

    Returns `{name, root}` for every skill found under `UNCONTROLLED_SKILL_ROOTS`.
    Consumed by the report to render a caveat; does not affect `apply()`.
    """
    out = []
    for label in UNCONTROLLED_SKILL_ROOTS:
        root = _uncontrolled_root(home, label)
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if (entry / "SKILL.md").is_file():
                out.append({"name": entry.name, "root": label})
    return out


def _uncontrolled_warnings(skills, home):
    names = {s["name"] for s in skills}
    seen = set()
    out = []
    for u in uncontrolled_skills(home):
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

def _read_skills(skills_dir, scope):
    out = []
    if not skills_dir.is_dir():
        return out
    for entry in sorted(skills_dir.iterdir()):
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
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
    except OSError as exc:
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

    skills = _read_skills(grok_home / "skills", scope="user")
    if project_dir is not None:
        skills += _read_skills(Path(project_dir) / ".grok" / "skills", scope="project")

    mcp_servers, config_text = _read_mcp_servers(grok_home / "config.toml", warnings)

    warnings.extend(_uncontrolled_warnings(skills, grok_home))

    return make_harness_version(
        version_id=_compute_version_id(skills, config_text),
        skills=skills,
        mcp_servers=mcp_servers,
        config_text=config_text,
        source=str(grok_home),
        warnings=warnings,
    )


# --- materialize -----------------------------------------------------------

_SEAL_WARNING = (
    "sealed materialized home: appended [compat.claude]/[compat.cursor] "
    "disable tables so Claude/Cursor-sourced skills and MCP servers can't "
    "leak into the isolated run"
)


def _is_sealed(config_text):
    for line in config_text.splitlines():
        m = _TOML_HEADER_RE.match(line)
        if m and m.group(1).strip() in ("compat.claude", "compat.cursor"):
            return True
    return False


def _seal_config_text(config_text):
    if _is_sealed(config_text):
        return config_text
    sep = "" if (not config_text or config_text.endswith("\n")) else "\n"
    return config_text + sep + _SEAL_CONFIG_TEXT.lstrip("\n")


def materialize(version, dest: Path, *, auth_from: Path, seal: bool = True) -> Path:
    """Write `version` out as a runnable GROK_HOME at `dest`.

    Idempotent: skills/ and config.toml are replaced wholesale so a stale
    skill from a previous materialize never survives.

    `seal=True` (default) appends `[compat.claude]`/`[compat.cursor]` disable
    tables to the written config.toml, since GROK_HOME alone does not stop
    grok from also discovering Claude/Cursor-sourced skills and MCP servers
    on the same machine (see module docstring). Pass `seal=False` to opt out.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    skills_dir = dest / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True)
    for skill in version["skills"]:
        skill_dir = skills_dir / skill["name"]
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill["content"], encoding="utf-8")

    config_text = version["config_text"]
    if seal:
        config_text = _seal_config_text(config_text)
        if _SEAL_WARNING not in version["warnings"]:
            version["warnings"].append(_SEAL_WARNING)
    (dest / "config.toml").write_text(config_text, encoding="utf-8")

    auth_src = Path(auth_from) / "auth.json"
    if not auth_src.is_file():
        raise FileNotFoundError(f"no auth.json at {auth_src}")
    # Copied, not symlinked: grok rewrites auth.json on token refresh, and a
    # symlink would let that refresh leak back into the source home.
    shutil.copyfile(auth_src, dest / "auth.json")

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
        version_id=_compute_version_id(skills, config_text),
        skills=skills,
        mcp_servers=mcp_servers,
        config_text=config_text,
        extra_rules=extra_rules,
        source=version.get("source"),
        warnings=list(version.get("warnings") or []),
    )
    return new_version, applications
