"""Tests for interstellar/replay.py.

Nothing here spawns grok-dev or calls the real harness.materialize: the grok
invocation goes through the injectable `runner` callable and run_matrix's
call into materialize goes through the injectable `materialize` callable, so
these tests assert argv/env/call-order directly and fabricate results
instead.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from interstellar.replay import GROK, isolated_workdir, run_matrix, run_once
from interstellar.types import ARM_CONTROL, ARM_TREATMENT, make_harness_version


def _ok_result(argv, sessionId="sess-1", **fields):
    payload = {"sessionId": sessionId, "num_turns": 2, "total_cost_usd": 0.01,
               "stopReason": "end_turn", "text": "done",
               "usage": {"input_tokens": 100, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0, "output_tokens": 20,
                         "reasoning_tokens": 5, "total_tokens": 125}}
    payload.update(fields)
    return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")


def _write_session(root: Path, session_id: str, *, with_events=True):
    """Fake `<home>/sessions/<cwd>/<session_id>/` dir, normalizable when
    `with_events` (a bare events.jsonl is enough for normalize_session to
    return a trace keyed by the dir name)."""
    d = root / "sessions" / "fake-cwd" / session_id
    d.mkdir(parents=True, exist_ok=True)
    if with_events:
        lines = [
            json.dumps({"type": "turn_started", "ts": "2024-01-01T00:00:00Z",
                       "turn_number": 0}),
            json.dumps({"type": "turn_ended", "ts": "2024-01-01T00:00:01Z",
                       "outcome": "completed"}),
        ]
        (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    return d


def _ok_run(argv, kwargs, sessionId="sess-1", **fields):
    """Like _ok_result, but also lands a normalizable session dir under the
    GROK_HOME the runner was actually given -- needed for any test that
    expects ok=True, now that a run with no findable session dir is
    correctly ok=False (Important C)."""
    home = Path(kwargs["env"]["GROK_HOME"])
    _write_session(home, sessionId)
    return _ok_result(argv, sessionId=sessionId, **fields)


def _write_broken_session(root: Path, session_id: str):
    """A session dir whose first event has no `ts`. grok_normalize computes
    t0 from events[0] and crashes on the first later event that does have
    one -- this is the exact failure Critical A exists to contain."""
    d = root / "sessions" / "fake-cwd" / session_id
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"type": "turn_started", "turn_number": 0}),  # no ts
        json.dumps({"type": "turn_ended", "ts": "2024-01-01T00:00:01Z",
                   "outcome": "completed"}),
    ]
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    return d


class RunOnceArgvEnvTests(unittest.TestCase):
    """The grok invocation itself: argv shape and GROK_HOME isolation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.workdir = Path(self.tmp.name) / "work"
        self.home.mkdir()
        self.workdir.mkdir()
        # run_once resolves both before use (fix round 3); resolve here too
        # so string comparisons below don't trip on e.g. macOS's /var ->
        # /private/var symlink, which .resolve() collapses.
        self.home = self.home.resolve()
        self.workdir = self.workdir.resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_argv_and_env(self):
        captured = {}

        def fake_runner(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _ok_run(argv, kwargs)

        result = run_once("do the thing", self.home, workdir=self.workdir,
                          runner=fake_runner)

        argv = captured["argv"]
        self.assertEqual(argv[0], str(GROK))
        self.assertIn("-p", argv)
        self.assertIn("do the thing", argv)
        self.assertIn("--output-format", argv)
        self.assertIn("json", argv)
        self.assertIn("--permission-mode", argv)
        self.assertIn("bypassPermissions", argv)
        self.assertNotIn("--model", argv)  # not requested
        self.assertNotIn("--rules", argv)  # none supplied

        self.assertEqual(captured["kwargs"]["cwd"], str(self.workdir))
        self.assertEqual(captured["kwargs"]["env"]["GROK_HOME"], str(self.home))
        self.assertTrue(result["ok"])
        self.assertEqual(result["session_id"], "sess-1")
        self.assertEqual(result["home"], str(self.home))
        self.assertEqual(result["workdir"], str(self.workdir))

    def test_home_and_workdir_resolved_to_absolute_even_when_given_relative(self):
        # Fix round 3: grok resolves a relative GROK_HOME against its own
        # cwd (the workdir passed via cwd=), not the caller's -- a relative
        # `home` sent grok looking in the wrong place entirely, finding
        # nothing, and reporting itself signed out. Every run in a live
        # 18-run cycle failed exactly this way.
        rel_home = Path(os.path.relpath(self.home, Path.cwd()))
        rel_workdir = Path(os.path.relpath(self.workdir, Path.cwd()))
        self.assertFalse(rel_home.is_absolute())
        self.assertFalse(rel_workdir.is_absolute())

        captured = {}

        def fake_runner(argv, **kwargs):
            captured["kwargs"] = kwargs
            return _ok_run(argv, kwargs)

        result = run_once("p", rel_home, workdir=rel_workdir, runner=fake_runner)

        grok_home = captured["kwargs"]["env"]["GROK_HOME"]
        cwd = captured["kwargs"]["cwd"]
        self.assertTrue(Path(grok_home).is_absolute(), grok_home)
        self.assertTrue(Path(cwd).is_absolute(), cwd)
        self.assertEqual(grok_home, str(self.home))
        self.assertEqual(cwd, str(self.workdir))
        # Also true of what lands on the RunResult -- a relative path there
        # is useless to a reader whose cwd is not the runner's.
        self.assertTrue(Path(result["home"]).is_absolute())
        self.assertTrue(Path(result["workdir"]).is_absolute())

    def test_model_and_extra_rules_forwarded(self):
        captured = {}

        def fake_runner(argv, **kwargs):
            captured["argv"] = argv
            return _ok_result(argv)

        run_once("p", self.home, workdir=self.workdir, model="grok-build",
                 extra_rules=["Never delete files.", "Prefer stdlib."],
                 runner=fake_runner)

        argv = captured["argv"]
        self.assertIn("--model", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "grok-build")
        self.assertIn("--rules", argv)
        rules_value = argv[argv.index("--rules") + 1]
        self.assertIn("Never delete files.", rules_value)
        self.assertIn("Prefer stdlib.", rules_value)

    def test_arm_and_repeat_stamped_on_result(self):
        result = run_once("p", self.home, workdir=self.workdir,
                          runner=lambda argv, **kw: _ok_result(argv),
                          arm=ARM_TREATMENT, repeat=3)
        self.assertEqual(result["arm"], ARM_TREATMENT)
        self.assertEqual(result["repeat"], 3)

    def test_usage_and_response_text_passed_through_verbatim(self):
        result = run_once("p", self.home, workdir=self.workdir,
                          runner=lambda argv, **kw: _ok_result(argv))
        self.assertEqual(result["response_text"], "done")
        self.assertEqual(result["usage"], {
            "input_tokens": 100, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0, "output_tokens": 20,
            "reasoning_tokens": 5, "total_tokens": 125,
        })

    def test_usage_defaults_to_empty_dict_when_absent(self):
        # Missing usage must stay {} -- never a fabricated zeroed dict.
        def fake_runner(argv, **kwargs):
            payload = {"sessionId": "sess-1", "text": "done"}
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)
        self.assertEqual(result["usage"], {})


class SessionDiscoveryTests(unittest.TestCase):
    """Session dir discovery diffs before/after; it must not just grab
    whatever is newest, since repeats run concurrently against shared
    templates in practice (though not the same home in run_matrix)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.workdir = Path(self.tmp.name) / "work"
        self.home.mkdir()
        self.workdir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_ignores_pre_existing_session_dir(self):
        # A session dir that was already there before this run starts.
        _write_session(self.home, "pre-existing", with_events=True)

        def fake_runner(argv, **kwargs):
            # Simulate grok landing a brand new session dir as a side effect
            # of "running", the way the real subprocess would.
            _write_session(self.home, "new-session", with_events=True)
            return _ok_result(argv, sessionId="new-session")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_id"], "new-session")
        self.assertIsNotNone(result["trace"])
        self.assertEqual(result["trace"]["session"]["session_id"], "new-session")

    def test_picks_sole_new_dir_when_stdout_has_no_session_id(self):
        _write_session(self.home, "pre-existing", with_events=True)

        def fake_runner(argv, **kwargs):
            _write_session(self.home, "new-session-2", with_events=True)
            return _ok_result(argv, sessionId=None)

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["trace"])
        self.assertEqual(result["trace"]["session"]["session_id"], "new-session-2")

    def test_no_new_session_dir_marks_run_not_ok(self):
        # Important C: a run with no findable session dir has no trace, and
        # nine of ten efficiency metrics are read off the trace -- it must
        # not be reported as an unremarked success.
        def fake_runner(argv, **kwargs):
            return _ok_result(argv, sessionId="never-landed")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIsNone(result["trace"])
        self.assertIsNotNone(result["error"])
        self.assertIn("no session directory", result["error"])


class FailureModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.workdir = Path(self.tmp.name) / "work"
        self.home.mkdir()
        self.workdir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_timeout_produces_ok_false(self):
        def fake_runner(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 1))

        result = run_once("p", self.home, workdir=self.workdir,
                          timeout=1, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])
        self.assertIn("timeout", result["error"])
        self.assertIsNone(result["trace"])

    def test_unparseable_stdout_produces_ok_false(self):
        def fake_runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="not json", stderr="")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])

    def test_nonzero_exit_produces_ok_false(self):
        def fake_runner(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 1, stdout=json.dumps({"sessionId": None}), stderr="boom")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])

    def test_never_raises_on_oserror(self):
        def fake_runner(argv, **kwargs):
            raise FileNotFoundError("no such binary")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIn("no such binary", result["error"])

    def test_never_raises_on_arbitrary_runner_exception(self):
        # Critical B, at the run_once level: not just {TimeoutExpired,
        # OSError} -- anything.
        def fake_runner(argv, **kwargs):
            raise subprocess.SubprocessError("child process died weirdly")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIn("child process died weirdly", result["error"])

    def test_normalizer_crash_does_not_raise_and_sets_ok_false(self):
        # Critical A: a malformed session (first event missing `ts`) makes
        # grok_normalize raise TypeError. run_once must swallow it, not
        # propagate it, and must not report the run as ok.
        def fake_runner(argv, **kwargs):
            _write_broken_session(self.home, "broken")
            return _ok_result(argv, sessionId="broken")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertFalse(result["ok"])
        self.assertIsNone(result["trace"])
        self.assertIsNotNone(result["error"])
        self.assertIn("normalize", result["error"].lower())


class IsolatedWorkdirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # isolated_workdir resolves both src and dest; resolve here too so
        # string/path comparisons below don't trip on e.g. macOS's /var ->
        # /private/var symlink, which .resolve() collapses.
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_copy(self):
        src = self.root / "src"
        src.mkdir()
        (src / "app.py").write_text("print('hi')\n")
        dest = self.root / "dest"

        out = isolated_workdir(src, dest)

        self.assertEqual(out, dest)
        self.assertEqual((dest / "app.py").read_text(), "print('hi')\n")
        # Mutating the copy must not touch the source.
        (dest / "app.py").write_text("print('mutated')\n")
        self.assertEqual((src / "app.py").read_text(), "print('hi')\n")

    def test_preserves_uncommitted_and_untracked_files_in_a_git_repo(self):
        # Important D: this is the regression the copy-over-worktree ruling
        # exists to prevent. `git worktree add --detach` reproduces HEAD
        # only, which would silently drop both of these -- exactly the
        # state the prompt being replayed actually ran against.
        if shutil.which("git") is None:
            self.skipTest("git not available")
        src = self.root / "repo"
        src.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=src, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=src, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=src, check=True)
        (src / "committed.py").write_text("committed\n")
        subprocess.run(["git", "add", "committed.py"], cwd=src, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True)
        (src / "committed.py").write_text("dirty\n")       # uncommitted change
        (src / "untracked.py").write_text("untracked\n")   # untracked file
        dest = self.root / "dest"

        isolated_workdir(src, dest)

        self.assertEqual((dest / "committed.py").read_text(), "dirty\n")
        self.assertEqual((dest / "untracked.py").read_text(), "untracked\n")
        self.assertFalse((dest / ".git").exists())

    def test_ignores_git_and_build_dirs(self):
        src = self.root / "src"
        (src / ".git").mkdir(parents=True)
        (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (src / "target").mkdir()
        (src / "target" / "big.bin").write_text("junk\n")
        (src / "keep.py").write_text("keep\n")
        dest = self.root / "dest"

        isolated_workdir(src, dest)

        self.assertTrue((dest / "keep.py").exists())
        self.assertFalse((dest / ".git").exists())
        self.assertFalse((dest / "target").exists())

    def test_dest_inside_src_terminates_and_excludes_itself(self):
        # Final review C1: the documented default puts the cycle's own
        # output (--out) inside `workspace`, so `dest` is routinely a
        # descendant of `src`. An unconditional copytree recurses into its
        # own destination and copies it into itself until it dies on path
        # length -- reproduced here with the same shape as the real bug
        # (dest several directories deep under src, alongside a sibling
        # run's directory and ordinary source content).
        src = self.root / "workspace"
        src.mkdir()
        (src / "keep.py").write_text("keep\n")
        (src / "untracked.py").write_text("untracked\n")
        # A sibling run's own output, already sitting under the same
        # ancestor as `dest` -- must not be swept into this run's copy.
        sibling = src / "runs" / "ts1" / "scratch" / "p1" / "runs" / "treatment-0-workdir"
        sibling.mkdir(parents=True)
        (sibling / "leftover.txt").write_text("stale\n")
        dest = src / "runs" / "ts1" / "scratch" / "p1" / "runs" / "control-0-workdir"

        out = isolated_workdir(src, dest)

        self.assertEqual(out, dest)
        # Terminated (a broken implementation raises OSError from a path
        # that grows without bound long before reaching here) and excluded
        # itself: the copy must not contain a `runs/` subtree at all, since
        # `runs/` is entirely the ancestor branch that led to `dest`.
        self.assertFalse((dest / "runs").exists())
        # Preserved: an untracked file and ordinary source content, exactly
        # the property Important D's fix already established for git state.
        self.assertEqual((dest / "keep.py").read_text(), "keep\n")
        self.assertEqual((dest / "untracked.py").read_text(), "untracked\n")

    def test_dest_equal_to_src_raises(self):
        src = self.root / "workspace"
        src.mkdir()

        with self.assertRaises(ValueError):
            isolated_workdir(src, src)

    def test_src_inside_dest_raises_instead_of_deleting_src(self):
        # dest being an ancestor of src is unsafe for a different reason:
        # the pre-copy `rmtree(dest)` would delete `src` along with it.
        dest = self.root / "outer"
        src = dest / "workspace"
        src.mkdir(parents=True)
        (src / "keep.py").write_text("keep\n")

        with self.assertRaises(ValueError):
            isolated_workdir(src, dest)
        # And, critically, src must still be there -- the guard has to fire
        # before any destructive step, not after.
        self.assertTrue((src / "keep.py").exists())


class RunMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.scratch = self.root / "scratch"
        self.auth_from = self.root / "auth-src"
        self.workspace.mkdir(parents=True)
        self.auth_from.mkdir(parents=True)
        (self.auth_from / "auth.json").write_text("{}")
        (self.workspace / "app.py").write_text("x = 1\n")

        self.control = make_harness_version(
            version_id="c1", skills=[], mcp_servers=[], config_text="")
        self.treatment = make_harness_version(
            version_id="t1", skills=[], mcp_servers=[], config_text="",
            extra_rules=["Be brief."])

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _fake_materialize(version, dest, *, auth_from, project_dest=None):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "sessions").mkdir(parents=True, exist_ok=True)
        return dest

    def test_scratch_inside_workspace_does_not_break_every_run(self):
        # Final review C1, exercised at the level the bug actually fired:
        # the documented default puts --out (and therefore scratch) inside
        # workspace itself. Before the isolated_workdir fix, this made every
        # single run's copytree recurse into its own destination -- every
        # run in the matrix failed with a path error that looked, to the
        # user, like an auth failure.
        workspace = self.root / "repo"
        workspace.mkdir()
        (workspace / "app.py").write_text("x = 1\n")
        scratch = workspace / "runs" / "ts1" / "scratch" / "p1" / "runs"

        def fake_runner(argv, **kwargs):
            return _ok_run(argv, kwargs)

        matrix = run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=workspace, scratch=scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        for arm in (ARM_CONTROL, ARM_TREATMENT):
            for r in matrix["arms"][arm]:
                self.assertTrue(r["ok"], r.get("error"))

    def test_produces_2k_results(self):
        def fake_runner(argv, **kwargs):
            return _ok_run(argv, kwargs)

        matrix = run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=2, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=2,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertEqual(matrix["k"], 2)
        self.assertEqual(matrix["control_version_id"], "c1")
        self.assertEqual(matrix["treatment_version_id"], "t1")
        self.assertEqual(len(matrix["arms"][ARM_CONTROL]), 2)
        self.assertEqual(len(matrix["arms"][ARM_TREATMENT]), 2)
        total = len(matrix["arms"][ARM_CONTROL]) + len(matrix["arms"][ARM_TREATMENT])
        self.assertEqual(total, 4)
        for r in matrix["arms"][ARM_CONTROL]:
            self.assertTrue(r["ok"])
            self.assertEqual(r["arm"], ARM_CONTROL)
        for r in matrix["arms"][ARM_TREATMENT]:
            self.assertTrue(r["ok"])
            self.assertEqual(r["arm"], ARM_TREATMENT)

    def test_treatment_extra_rules_come_from_the_version_dict(self):
        captured = []

        def fake_runner(argv, **kwargs):
            captured.append(argv)
            return _ok_result(argv)

        run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        control_argv, treatment_argv = captured
        self.assertNotIn("--rules", control_argv)
        self.assertIn("--rules", treatment_argv)
        self.assertIn("Be brief.", treatment_argv[treatment_argv.index("--rules") + 1])

    def test_tolerates_one_arm_failing(self):
        # Fail every treatment run, succeed every control run. The fake
        # runner distinguishes them by looking at cwd, which run_matrix
        # names per-(arm, repeat).
        def fake_runner(argv, **kwargs):
            if "treatment" in kwargs["cwd"]:
                raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
            return _ok_run(argv, kwargs)

        matrix = run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=2, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=2,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertEqual(len(matrix["arms"][ARM_CONTROL]), 2)
        self.assertEqual(len(matrix["arms"][ARM_TREATMENT]), 2)
        self.assertTrue(all(r["ok"] for r in matrix["arms"][ARM_CONTROL]))
        self.assertTrue(all(not r["ok"] for r in matrix["arms"][ARM_TREATMENT]))
        # The matrix itself is always returned, never raised, regardless of
        # how many individual runs failed.
        self.assertIn("arms", matrix)

    def test_tolerates_arbitrary_runner_exception(self):
        # Critical B: not just {TimeoutExpired, OSError} -- run_matrix must
        # survive any exception a runner (or materialize, or isolated_workdir)
        # can throw, or one bad run destroys every already-succeeded run and
        # everything already spent on them.
        def fake_runner(argv, **kwargs):
            if "treatment" in kwargs["cwd"]:
                raise subprocess.SubprocessError("child process died weirdly")
            return _ok_run(argv, kwargs)

        matrix = run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertTrue(matrix["arms"][ARM_CONTROL][0]["ok"])
        self.assertFalse(matrix["arms"][ARM_TREATMENT][0]["ok"])
        self.assertIn("child process died weirdly",
                      matrix["arms"][ARM_TREATMENT][0]["error"])

    def test_tolerates_materialize_exception(self):
        def flaky_materialize(version, dest, *, auth_from, project_dest=None):
            if version["version_id"] == "t1":
                raise RuntimeError("disk full")
            return self._fake_materialize(version, dest, auth_from=auth_from,
                                          project_dest=project_dest)

        matrix = run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=lambda argv, **kw: _ok_run(argv, kw),
            materialize=flaky_materialize,
        )

        self.assertTrue(matrix["arms"][ARM_CONTROL][0]["ok"])
        self.assertFalse(matrix["arms"][ARM_TREATMENT][0]["ok"])
        self.assertIn("disk full", matrix["arms"][ARM_TREATMENT][0]["error"])

    def test_project_skills_removed_before_materialize_with_project_dest(self):
        # The new requirement: grok resolves ./.grok/skills above GROK_HOME,
        # so a patched project-scope skill written only to GROK_HOME would
        # be silently shadowed by the unpatched copy isolated_workdir just
        # reproduced. Order must be: (1) copy workdir, (2) delete its
        # .grok/skills, (3) materialize with project_dest=that workdir.
        (self.workspace / ".grok" / "skills" / "demo").mkdir(parents=True)
        (self.workspace / ".grok" / "skills" / "demo" / "SKILL.md").write_text("ORIGINAL\n")
        (self.workspace / ".grok" / "rules").mkdir(parents=True)  # untouched sibling

        calls = []

        def spy_materialize(version, dest, *, auth_from, project_dest=None):
            calls.append({
                "project_dest": project_dest,
                "grok_skills_present": (project_dest / ".grok" / "skills").exists()
                                       if project_dest else None,
                "grok_rules_present": (project_dest / ".grok" / "rules").exists()
                                      if project_dest else None,
            })
            return self._fake_materialize(version, dest, auth_from=auth_from,
                                          project_dest=project_dest)

        run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=lambda argv, **kw: _ok_result(argv),
            materialize=spy_materialize,
        )

        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIsNotNone(call["project_dest"])
            # .grok/skills was deleted from the workdir copy by the time
            # materialize ran ...
            self.assertFalse(call["grok_skills_present"])
            # ... but only skills/, not the rest of .grok/.
            self.assertTrue(call["grok_rules_present"])
            # project_dest really is the (rest of the) workspace copy.
            self.assertTrue((call["project_dest"] / "app.py").exists())

    def test_grok_home_and_cwd_absolute_even_with_relative_scratch_and_workspace(self):
        # Fix round 3, reproduced at the run_matrix level: this is the exact
        # case that broke a live cycle -- scratch/workspace given as
        # relative paths, which used to flow straight through into a
        # relative GROK_HOME.
        rel_scratch = Path(os.path.relpath(self.scratch, Path.cwd()))
        rel_workspace = Path(os.path.relpath(self.workspace, Path.cwd()))
        self.assertFalse(rel_scratch.is_absolute())
        self.assertFalse(rel_workspace.is_absolute())

        captured = []

        def fake_runner(argv, **kwargs):
            captured.append(kwargs)
            return _ok_run(argv, kwargs)

        run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=1, workspace=rel_workspace, scratch=rel_scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertEqual(len(captured), 2)
        for kwargs in captured:
            self.assertTrue(Path(kwargs["env"]["GROK_HOME"]).is_absolute())
            self.assertTrue(Path(kwargs["cwd"]).is_absolute())

    def test_dispatch_order_is_interleaved(self):
        # A single worker dequeues submitted tasks strictly FIFO, so with
        # max_parallel=1 the observed call order is exactly the submission
        # order -- this is what must be interleaved, not arm-major.
        order = []
        lock = threading.Lock()

        def fake_runner(argv, **kwargs):
            home = kwargs["env"]["GROK_HOME"]
            tag = Path(home).name.removesuffix("-home")  # "control-0" etc.
            with lock:
                order.append(tag)
            return _ok_result(argv)

        run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=3, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=1,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertEqual(order, [
            "control-0", "treatment-0",
            "control-1", "treatment-1",
            "control-2", "treatment-2",
        ])

    def test_each_repeat_gets_its_own_home_and_workdir(self):
        seen_homes, seen_cwds = set(), set()

        def fake_runner(argv, **kwargs):
            seen_homes.add(kwargs["env"]["GROK_HOME"])
            seen_cwds.add(kwargs["cwd"])
            return _ok_result(argv)

        run_matrix(
            "prompt", control=self.control, treatment=self.treatment,
            k=2, workspace=self.workspace, scratch=self.scratch,
            auth_from=self.auth_from, max_parallel=3,
            runner=fake_runner, materialize=self._fake_materialize,
        )

        self.assertEqual(len(seen_homes), 4)
        self.assertEqual(len(seen_cwds), 4)


if __name__ == "__main__":
    unittest.main()
