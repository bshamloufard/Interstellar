"""Tests for interstellar.harness: snapshot / materialize / apply.

Never calls grok-dev -- everything here is built with tempfile dirs and the
types.py constructors, per the review-loop test constraints.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from interstellar import harness
from interstellar.types import (
    EFFECT_SKILL_TOKENS,
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
    PATCH_SKILL_INSERT,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_TRUNCATE,
    make_harness_version,
    make_mcp_server,
    make_patch,
    make_skill,
)

SKILL_A = """---
name: alpha
description: Alpha skill for tests.
---

line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10
"""

SKILL_B = "---\nname: beta\ndescription: Beta skill.\n---\n\nbeta body\n"

CONFIG_TEXT = """[marketplace]
default_skills_installs_purged = true

[mcp_servers.filesystem]
command = "npx"
args = [
    "-y",
    "@modelcontextprotocol/server-filesystem",
]
startup_timeout_sec = 120

[mcp_servers.memory]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-memory"]

[mcp_servers.memory.env]
MEMORY_FILE_PATH = "/tmp/memory.jsonl"

[mcp_servers.sequential-thinking]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-sequential-thinking"]

[ui]
yolo = false
"""


def _write_grok_home(root: Path, *, skills=(("alpha", SKILL_A), ("beta", SKILL_B)),
                      config_text=CONFIG_TEXT):
    (root / "skills").mkdir(parents=True, exist_ok=True)
    for name, content in skills:
        d = root / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    if config_text is not None:
        (root / "config.toml").write_text(config_text, encoding="utf-8")
    return root


class SnapshotTests(unittest.TestCase):
    def test_deterministic_version_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            v1 = harness.snapshot(home)
            v2 = harness.snapshot(home)
            self.assertEqual(v1["version_id"], v2["version_id"])
            self.assertEqual({s["name"] for s in v1["skills"]}, {"alpha", "beta"})

    def test_skill_byte_change_moves_version_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            before = harness.snapshot(home)
            (home / "skills" / "alpha" / "SKILL.md").write_text(
                SKILL_A + "\nextra line\n", encoding="utf-8"
            )
            after = harness.snapshot(home)
            self.assertNotEqual(before["version_id"], after["version_id"])

    def test_frontmatter_description_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            v = harness.snapshot(home)
            alpha = next(s for s in v["skills"] if s["name"] == "alpha")
            self.assertEqual(alpha["description"], "Alpha skill for tests.")
            self.assertEqual(alpha["scope"], "user")
            self.assertIsNotNone(alpha["sha256"])

    def test_project_skills_marked_project_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _write_grok_home(root / "home")
            project = root / "proj"
            (project / ".grok" / "skills" / "gamma").mkdir(parents=True)
            (project / ".grok" / "skills" / "gamma" / "SKILL.md").write_text(
                "---\nname: gamma\ndescription: Project skill.\n---\nbody\n",
                encoding="utf-8",
            )
            v = harness.snapshot(home, project_dir=project)
            names_scopes = {s["name"]: s["scope"] for s in v["skills"]}
            self.assertEqual(names_scopes["gamma"], "project")
            self.assertEqual(names_scopes["alpha"], "user")

    def test_mcp_servers_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            v = harness.snapshot(home)
            names = {m["name"] for m in v["mcp_servers"]}
            self.assertEqual(names, {"filesystem", "memory", "sequential-thinking"})
            fs = next(m for m in v["mcp_servers"] if m["name"] == "filesystem")
            self.assertEqual(fs["command"], "npx")
            self.assertTrue(fs["enabled"])

    def test_missing_config_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home", config_text=None)
            v = harness.snapshot(home)
            self.assertEqual(v["mcp_servers"], [])
            self.assertTrue(any("config.toml" in w for w in v["warnings"]))

    def test_unparseable_config_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home", config_text="not [ valid toml =")
            v = harness.snapshot(home)
            self.assertEqual(v["mcp_servers"], [])
            self.assertTrue(any("config.toml" in w for w in v["warnings"]))


def _version(skills, config_text=CONFIG_TEXT, mcp_servers=None):
    return make_harness_version(
        version_id="deadbeef0000",
        skills=skills,
        mcp_servers=mcp_servers or [],
        config_text=config_text,
    )


def _skill(name, content, scope="user", description=""):
    s = make_skill(name, path=f"skills/{name}/SKILL.md", content=content,
                    scope=scope, description=description)
    s["sha256"] = harness._sha256_text(content)
    return s


TEN_LINES = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"


class ApplyTruncateTests(unittest.TestCase):
    def test_deletes_exactly_cited_lines(self):
        version = _version([_skill("alpha", TEN_LINES)])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="alpha",
            rationale="trim", expected_effect=EFFECT_SKILL_TOKENS,
            line_ranges=[[3, 5]],
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        self.assertEqual(apps[0]["lines_changed"], 3)
        new_content = new_version["skills"][0]["content"]
        self.assertEqual(
            new_content.splitlines(),
            ["line 1", "line 2", "line 6", "line 7", "line 8", "line 9", "line 10"],
        )
        # original version untouched (apply is pure)
        self.assertEqual(version["skills"][0]["content"], TEN_LINES)

    def test_overlapping_descending_ranges(self):
        version = _version([_skill("alpha", TEN_LINES)])
        # Given out of order and overlapping (5-9 then 3-7): union is 3-9.
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="alpha",
            rationale="trim", expected_effect=EFFECT_SKILL_TOKENS,
            line_ranges=[[5, 9], [3, 7]],
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        new_content = new_version["skills"][0]["content"]
        self.assertEqual(new_content.splitlines(), ["line 1", "line 2", "line 10"])

    def test_out_of_range_is_skipped_not_raised(self):
        version = _version([_skill("alpha", TEN_LINES)])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="alpha",
            rationale="trim", expected_effect=EFFECT_SKILL_TOKENS,
            line_ranges=[[8, 12]],
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertFalse(apps[0]["applied"])
        self.assertTrue(apps[0]["reason"])
        # unchanged
        self.assertEqual(new_version["skills"][0]["content"], TEN_LINES)

    def test_missing_skill_is_skipped(self):
        version = _version([_skill("alpha", TEN_LINES)])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="nope",
            rationale="trim", expected_effect=EFFECT_SKILL_TOKENS,
            line_ranges=[[1, 2]],
        )
        _, apps = harness.apply(version, [patch])
        self.assertFalse(apps[0]["applied"])
        self.assertIn("not found", apps[0]["reason"])

    def test_version_id_changes_after_apply(self):
        version = _version([_skill("alpha", TEN_LINES)])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_TRUNCATE, target="alpha",
            rationale="trim", expected_effect=EFFECT_SKILL_TOKENS,
            line_ranges=[[3, 5]],
        )
        new_version, _ = harness.apply(version, [patch])
        self.assertNotEqual(version["version_id"], new_version["version_id"])


class ApplyOtherKindsTests(unittest.TestCase):
    def test_skill_remove_drops_skill(self):
        version = _version([_skill("alpha", TEN_LINES), _skill("beta", "x\n")])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_REMOVE, target="alpha",
            rationale="drop", expected_effect=EFFECT_SKILL_TOKENS,
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        self.assertEqual([s["name"] for s in new_version["skills"]], ["beta"])

    def test_skill_insert_at_anchor(self):
        version = _version([_skill("alpha", TEN_LINES)])
        patch = make_patch(
            patch_id="p1", kind=PATCH_SKILL_INSERT, target="alpha",
            rationale="add", expected_effect=EFFECT_SKILL_TOKENS,
            insert_text="new line a\nnew line b", insert_after_line=2,
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        lines = new_version["skills"][0]["content"].splitlines()
        self.assertEqual(lines[:4], ["line 1", "line 2", "new line a", "new line b"])

    def test_rules_append(self):
        version = _version([])
        patch = make_patch(
            patch_id="p1", kind=PATCH_RULES_APPEND, target="rules",
            rationale="add rule", expected_effect=EFFECT_SKILL_TOKENS,
            rule_text="Never run destructive commands without confirmation.",
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        self.assertEqual(
            new_version["extra_rules"],
            ["Never run destructive commands without confirmation."],
        )


class ApplyMcpRemoveTests(unittest.TestCase):
    def test_removes_only_named_table(self):
        mcp_servers = [
            make_mcp_server("filesystem", command="npx"),
            make_mcp_server("memory", command="npx"),
            make_mcp_server("sequential-thinking", command="npx"),
        ]
        version = _version([], mcp_servers=mcp_servers)
        patch = make_patch(
            patch_id="p1", kind=PATCH_MCP_REMOVE, target="memory",
            rationale="unused", expected_effect=EFFECT_SKILL_TOKENS,
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        text = new_version["config_text"]
        self.assertNotIn("[mcp_servers.memory]", text)
        self.assertNotIn("[mcp_servers.memory.env]", text)
        self.assertNotIn("MEMORY_FILE_PATH", text)
        # other tables survive intact
        self.assertIn("[mcp_servers.filesystem]", text)
        self.assertIn("[mcp_servers.sequential-thinking]", text)
        self.assertIn("[ui]", text)
        self.assertEqual(
            {m["name"] for m in new_version["mcp_servers"]},
            {"filesystem", "sequential-thinking"},
        )

    def test_unknown_server_is_skipped(self):
        version = _version([], mcp_servers=[make_mcp_server("filesystem")])
        patch = make_patch(
            patch_id="p1", kind=PATCH_MCP_REMOVE, target="ghost",
            rationale="unused", expected_effect=EFFECT_SKILL_TOKENS,
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertFalse(apps[0]["applied"])
        self.assertEqual(new_version["config_text"], CONFIG_TEXT)

    def test_does_not_prefix_match_similar_name(self):
        # "filesystem" must not also delete a hypothetical "filesystem2" table.
        config_text = CONFIG_TEXT + "\n[mcp_servers.filesystem2]\ncommand = \"x\"\n"
        version = _version([], config_text=config_text,
                            mcp_servers=[make_mcp_server("filesystem"),
                                         make_mcp_server("filesystem2")])
        patch = make_patch(
            patch_id="p1", kind=PATCH_MCP_REMOVE, target="filesystem",
            rationale="unused", expected_effect=EFFECT_SKILL_TOKENS,
        )
        new_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        self.assertIn("[mcp_servers.filesystem2]", new_version["config_text"])


class MaterializeTests(unittest.TestCase):
    def _auth_dir(self, tmp):
        auth_dir = Path(tmp) / "auth_src"
        auth_dir.mkdir()
        (auth_dir / "auth.json").write_text('{"token": "fake"}', encoding="utf-8")
        return auth_dir

    def test_materialize_matches_version_and_copies_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([_skill("alpha", TEN_LINES), _skill("beta", "x\n")])
            dest = Path(tmp) / "grok_home"
            # seal=False here: sealing behavior has its own dedicated tests
            # below, this test is about the general copy semantics.
            result = harness.materialize(version, dest, auth_from=auth_dir, seal=False)

            self.assertEqual(result, dest)
            self.assertEqual(
                {p.name for p in (dest / "skills").iterdir()}, {"alpha", "beta"}
            )
            self.assertEqual(
                (dest / "skills" / "alpha" / "SKILL.md").read_text(), TEN_LINES
            )
            self.assertEqual((dest / "config.toml").read_text(), CONFIG_TEXT)
            self.assertEqual((dest / "auth.json").read_text(), '{"token": "fake"}')
            self.assertTrue((dest / "sessions").is_dir())

    def test_rematerialize_drops_stale_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            dest = Path(tmp) / "grok_home"
            v1 = _version([_skill("alpha", TEN_LINES), _skill("beta", "x\n")])
            harness.materialize(v1, dest, auth_from=auth_dir)
            self.assertTrue((dest / "skills" / "beta").exists())

            v2 = _version([_skill("alpha", TEN_LINES)])
            harness.materialize(v2, dest, auth_from=auth_dir)
            self.assertEqual({p.name for p in (dest / "skills").iterdir()}, {"alpha"})
            self.assertFalse((dest / "skills" / "beta").exists())

    def test_materialize_seals_fresh_config_with_no_compat_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            text = (dest / "config.toml").read_text()
            self.assertIn("[compat.claude]", text)
            self.assertIn("[compat.cursor]", text)
            for cell in ("skills", "rules", "agents", "mcps", "hooks", "sessions"):
                self.assertIn(f"{cell} = false", text)
            # original recorded config still present, seal only appended
            self.assertIn("[mcp_servers.filesystem]", text)
            self.assertTrue(text.index("[mcp_servers.filesystem]") <
                             text.index("[compat.claude]"))
            tomllib.loads(text)  # must still be valid, parseable TOML

            self.assertEqual(len(version["warnings"]), 1)
            self.assertIn("sealed materialized home", version["warnings"][0])
            self.assertIn("added (was absent", version["warnings"][0])

    def test_materialize_seal_false_opts_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir, seal=False)
            text = (dest / "config.toml").read_text()
            self.assertNotIn("[compat.claude]", text)
            self.assertEqual(text, CONFIG_TEXT)
            self.assertEqual(version["warnings"], [])

    def test_materialize_rewrites_existing_partial_compat_table_never_fabricates(self):
        # Regression test for Critical 1: a user who left [compat.claude]
        # present but only turned off hooks must not get an unsealed harness
        # plus an unqualified claim that it was sealed -- every cell must end
        # up false, and the warning must name exactly what was overridden.
        config_text = CONFIG_TEXT + "\n[compat.claude]\nhooks = false\n"
        version = _version([_skill("alpha", TEN_LINES)], config_text=config_text)
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            text = (dest / "config.toml").read_text()

            self.assertEqual(text.count("[compat.claude]"), 1)  # no dup header
            parsed = tomllib.loads(text)  # must not raise
            all_false = {c: False for c in
                         ("skills", "rules", "agents", "mcps", "hooks", "sessions")}
            self.assertEqual(parsed["compat"]["claude"], all_false)
            self.assertEqual(parsed["compat"]["cursor"], all_false)

            expected = (
                "sealed materialized home: [compat.claude] existing table "
                "rewritten, cells forced false that were not already false: "
                "skills, rules, agents, mcps, sessions; [compat.cursor] added "
                "(was absent, defaults to enabled)"
            )
            self.assertIn(expected, version["warnings"])

    def test_materialize_records_already_sealed_without_fabricating_override(self):
        already_sealed = "".join(
            f"\n[compat.{vendor}]\n" +
            "".join(f"{c} = false\n" for c in
                    ("skills", "rules", "agents", "mcps", "hooks", "sessions"))
            for vendor in ("claude", "cursor")
        )
        config_text = CONFIG_TEXT + already_sealed
        version = _version([_skill("alpha", TEN_LINES)], config_text=config_text)
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            warning = next(w for w in version["warnings"] if w.startswith("sealed"))
            self.assertIn("[compat.claude] already fully sealed", warning)
            self.assertIn("[compat.cursor] already fully sealed", warning)
            self.assertNotIn("overridden", warning)
            text = (dest / "config.toml").read_text()
            self.assertEqual(text.count("[compat.claude]"), 1)
            self.assertEqual(text.count("[compat.cursor]"), 1)

    def test_materialize_does_not_duplicate_seal_warning_on_rematerialize(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            harness.materialize(version, dest, auth_from=auth_dir)
            seal_warnings = [w for w in version["warnings"] if w.startswith("sealed")]
            self.assertEqual(len(seal_warnings), 1)
            text = (dest / "config.toml").read_text()
            self.assertEqual(text.count("[compat.claude]"), 1)

    def test_auth_json_mode_forced_to_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            (auth_dir / "auth.json").chmod(0o600)
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            mode = (dest / "auth.json").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_auth_json_mode_forced_even_when_source_is_looser(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            (auth_dir / "auth.json").chmod(0o644)
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)
            mode = (dest / "auth.json").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_missing_auth_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_auth = Path(tmp) / "no_auth"
            empty_auth.mkdir()
            version = _version([_skill("alpha", TEN_LINES)])
            dest = Path(tmp) / "grok_home"
            with self.assertRaises(FileNotFoundError):
                harness.materialize(version, dest, auth_from=empty_auth)


class ProjectSkillMaterializeTests(unittest.TestCase):
    """Regression coverage for Critical 2: a scope="project" skill must be
    materialized at the tier that actually wins under grok's own precedence
    (./.grok/skills/ > $GROK_HOME/skills/), never silently into the tier
    that gets shadowed."""

    def _auth_dir(self, tmp):
        auth_dir = Path(tmp) / "auth_src"
        auth_dir.mkdir()
        (auth_dir / "auth.json").write_text('{"token": "fake"}', encoding="utf-8")
        return auth_dir

    def test_project_skill_routed_to_project_dest_not_user_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([
                _skill("alpha", TEN_LINES, scope="user"),
                _skill("proj-skill", "project body\n", scope="project"),
            ])
            dest = Path(tmp) / "grok_home"
            project_dest = Path(tmp) / "workdir"
            harness.materialize(
                version, dest, auth_from=auth_dir, project_dest=project_dest,
            )

            self.assertTrue((dest / "skills" / "alpha" / "SKILL.md").is_file())
            self.assertFalse((dest / "skills" / "proj-skill").exists())
            self.assertEqual(
                (project_dest / ".grok" / "skills" / "proj-skill" / "SKILL.md")
                .read_text(),
                "project body\n",
            )
            self.assertFalse(any("not materialized" in w for w in version["warnings"]))

    def test_project_skill_without_project_dest_warns_and_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            version = _version([
                _skill("alpha", TEN_LINES, scope="user"),
                _skill("proj-skill", "project body\n", scope="project"),
            ])
            dest = Path(tmp) / "grok_home"
            harness.materialize(version, dest, auth_from=auth_dir)

            self.assertFalse((dest / "skills" / "proj-skill").exists())
            self.assertTrue((dest / "skills" / "alpha").exists())
            warning = next(w for w in version["warnings"] if "not materialized" in w)
            self.assertIn("proj-skill", warning)
            self.assertIn("patches against them will not take effect", warning)

    def test_rematerialize_drops_stale_project_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth_dir = self._auth_dir(tmp)
            dest = Path(tmp) / "grok_home"
            project_dest = Path(tmp) / "workdir"
            v1 = _version([_skill("proj-skill", "v1\n", scope="project")])
            harness.materialize(v1, dest, auth_from=auth_dir, project_dest=project_dest)
            self.assertTrue(
                (project_dest / ".grok" / "skills" / "proj-skill").exists()
            )

            v2 = _version([])
            harness.materialize(v2, dest, auth_from=auth_dir, project_dest=project_dest)
            self.assertFalse(
                (project_dest / ".grok" / "skills" / "proj-skill").exists()
            )


class NonUtf8AndUnreadableTests(unittest.TestCase):
    """Regression coverage for Important 3/4: a bad skill or config must be
    recorded as a warning, never raised past snapshot() and never silently
    dropped with no trace."""

    def test_non_utf8_skill_does_not_raise_and_is_warned(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home", skills=(("alpha", SKILL_A),))
            bad = home / "skills" / "bad"
            bad.mkdir()
            (bad / "SKILL.md").write_bytes(b"---\nname: bad\n---\ncaf\xe9\n")
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                v = harness.snapshot(home)  # must not raise
            self.assertEqual({s["name"] for s in v["skills"]}, {"alpha"})
            self.assertTrue(any("bad" in w for w in v["warnings"]))

    def test_non_utf8_config_does_not_raise_and_is_warned(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home", config_text=None)
            (home / "config.toml").write_bytes(b'[ui]\ntitle = "caf\xe9"\n')
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                v = harness.snapshot(home)  # must not raise
            self.assertEqual(v["mcp_servers"], [])
            self.assertTrue(any("config.toml" in w for w in v["warnings"]))

    def test_unreadable_skill_is_warned_not_silently_dropped(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("running as root: permission bits don't block reads")
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home", skills=(("alpha", SKILL_A),))
            locked = home / "skills" / "locked"
            locked.mkdir()
            skill_md = locked / "SKILL.md"
            skill_md.write_text("secret\n", encoding="utf-8")
            skill_md.chmod(0o000)
            try:
                with mock.patch.object(
                    harness.Path, "home", return_value=Path(tmp) / "no_such_home"
                ):
                    v = harness.snapshot(home)
            finally:
                skill_md.chmod(0o644)
            self.assertEqual({s["name"] for s in v["skills"]}, {"alpha"})
            self.assertTrue(any("locked" in w for w in v["warnings"]))


class VersionIdRulesTests(unittest.TestCase):
    """Regression coverage for Important 5: extra_rules must feed version_id
    per types.py's make_harness_version docstring, or a rules.append patch
    produces a treatment indistinguishable by id from its control."""

    def test_rules_append_produces_different_version_id_than_control(self):
        version = _version([_skill("alpha", TEN_LINES)])
        control_version, _ = harness.apply(version, [])
        patch = make_patch(
            patch_id="p1", kind=PATCH_RULES_APPEND, target="rules",
            rationale="add rule", expected_effect=EFFECT_SKILL_TOKENS,
            rule_text="Be brief.",
        )
        treatment_version, apps = harness.apply(version, [patch])
        self.assertTrue(apps[0]["applied"])
        self.assertNotEqual(
            control_version["version_id"], treatment_version["version_id"],
        )


class UncontrolledSkillsTests(unittest.TestCase):
    def test_bundled_skills_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            bundled = home / "bundled" / "skills" / "code-review"
            bundled.mkdir(parents=True)
            (bundled / "SKILL.md").write_text("bundled body\n", encoding="utf-8")

            # Isolate from whatever the real dev machine's ~/.agents/skills
            # happens to contain.
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                found = harness.uncontrolled_skills(home)
            self.assertEqual(found, [{"name": "code-review", "root": "bundled/skills"}])

    def test_agents_home_skills_detected_independent_of_grok_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            fake_user_home = Path(tmp) / "fake_user_home"
            (fake_user_home / ".agents" / "skills" / "gizmo").mkdir(parents=True)
            (fake_user_home / ".agents" / "skills" / "gizmo" / "SKILL.md").write_text(
                "x\n", encoding="utf-8"
            )
            with mock.patch.object(harness.Path, "home", return_value=fake_user_home):
                found = harness.uncontrolled_skills(home)
            self.assertEqual(found, [{"name": "gizmo", "root": "~/.agents/skills"}])

    def test_no_uncontrolled_roots_present_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                self.assertEqual(harness.uncontrolled_skills(home), [])

    def test_snapshot_warns_on_name_collision_with_bundled(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")  # has "alpha", "beta"
            bundled = home / "bundled" / "skills" / "alpha"
            bundled.mkdir(parents=True)
            (bundled / "SKILL.md").write_text("bundled alpha\n", encoding="utf-8")

            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                v = harness.snapshot(home)

            matches = [w for w in v["warnings"] if "alpha" in w and "bundled" in w]
            self.assertEqual(len(matches), 1)
            self.assertIn("also exists in bundled/skills/", matches[0])
            # beta has no uncontrolled duplicate -- no warning for it
            self.assertFalse(any("beta" in w for w in v["warnings"]))
            # apply()/patching is unaffected: 'alpha' is still a controlled skill
            self.assertIn("alpha", {s["name"] for s in v["skills"]})

    def test_snapshot_no_false_positive_when_no_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                v = harness.snapshot(home)
            self.assertEqual(v["warnings"], [])

    # -- project-relative roots (Critical 2 follow-up) --------------------

    def test_project_grok_skills_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            project = Path(tmp) / "proj"
            (project / ".grok" / "skills" / "shadow").mkdir(parents=True)
            (project / ".grok" / "skills" / "shadow" / "SKILL.md").write_text(
                "x\n", encoding="utf-8"
            )
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                found = harness.uncontrolled_skills(home, project_dir=project)
            self.assertIn({"name": "shadow", "root": "project/.grok/skills"}, found)

    def test_project_agents_skills_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            project = Path(tmp) / "proj"
            (project / ".agents" / "skills" / "ghost").mkdir(parents=True)
            (project / ".agents" / "skills" / "ghost" / "SKILL.md").write_text(
                "x\n", encoding="utf-8"
            )
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                found = harness.uncontrolled_skills(home, project_dir=project)
            self.assertIn({"name": "ghost", "root": "project/.agents/skills"}, found)

    def test_project_roots_not_checked_when_project_dir_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                found = harness.uncontrolled_skills(home)  # no project_dir
            self.assertEqual(found, [])

    def test_uncontrolled_warnings_excludes_captured_project_skill_self_match(self):
        # A scope="project" skill IS sourced from project/.grok/skills --
        # that's its origin, not a shadow, and materialize(project_dest=)
        # routes it back to that same tier. Must not self-flag.
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            project = Path(tmp) / "proj"
            (project / ".grok" / "skills" / "gamma").mkdir(parents=True)
            (project / ".grok" / "skills" / "gamma" / "SKILL.md").write_text(
                "x\n", encoding="utf-8"
            )
            skills = [_skill("gamma", "x\n", scope="project")]
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                warnings = harness._uncontrolled_warnings(skills, home, project_dir=project)
            self.assertEqual(warnings, [])

    def test_uncontrolled_warnings_flags_project_skill_this_version_did_not_capture(self):
        # "alpha" is controlled as a user-scope skill but this version was
        # never told about the project directory (e.g. an older snapshot) --
        # a same-named project skill there would still shadow it at replay.
        with tempfile.TemporaryDirectory() as tmp:
            home = _write_grok_home(Path(tmp) / "home")
            project = Path(tmp) / "proj"
            (project / ".grok" / "skills" / "alpha").mkdir(parents=True)
            (project / ".grok" / "skills" / "alpha" / "SKILL.md").write_text(
                "shadow copy\n", encoding="utf-8"
            )
            skills = [_skill("alpha", TEN_LINES, scope="user")]
            with mock.patch.object(
                harness.Path, "home", return_value=Path(tmp) / "no_such_home"
            ):
                warnings = harness._uncontrolled_warnings(skills, home, project_dir=project)
            matches = [w for w in warnings
                       if "alpha" in w and "project/.grok/skills" in w]
            self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
