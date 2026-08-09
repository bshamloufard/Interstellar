"""Tests for interstellar/replay.py.

Nothing here spawns grok-dev: the invocation is built through the injectable
`runner` callable, so these tests assert argv/env directly and fabricate
CompletedProcess-like results instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from interstellar.replay import GROK, isolated_workdir, run_matrix, run_once
from interstellar.types import ARM_CONTROL, ARM_TREATMENT


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


class RunOnceArgvEnvTests(unittest.TestCase):
    """The grok invocation itself: argv shape and GROK_HOME isolation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.workdir = Path(self.tmp.name) / "work"
        self.home.mkdir()
        self.workdir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_argv_and_env(self):
        captured = {}

        def fake_runner(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _ok_result(argv)

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

    def test_no_new_session_dir_leaves_trace_none(self):
        def fake_runner(argv, **kwargs):
            return _ok_result(argv, sessionId="never-landed")

        result = run_once("p", self.home, workdir=self.workdir, runner=fake_runner)

        self.assertTrue(result["ok"])  # stdout parsed fine
        self.assertIsNone(result["trace"])


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


class IsolatedWorkdirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_copy_for_non_repo(self):
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

    def test_git_worktree_for_repo(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        src = self.root / "repo"
        src.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=src, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=src, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=src, check=True)
        (src / "app.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=src, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True)
        dest = self.root / "dest"

        out = isolated_workdir(src, dest)

        self.assertEqual(out, dest)
        self.assertEqual((dest / "app.py").read_text(), "print('hi')\n")
        # It's a worktree, not a plain copy: it has its own .git pointer file.
        self.assertTrue((dest / ".git").exists())

        subprocess.run(["git", "worktree", "remove", "--force", str(dest)],
                       cwd=src, check=True)


class RunMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.control_home = self.root / "control-home"
        self.treatment_home = self.root / "treatment-home"
        self.workspace = self.root / "workspace"
        self.scratch = self.root / "scratch"
        for d in (self.control_home, self.treatment_home, self.workspace):
            d.mkdir(parents=True)
        (self.workspace / "app.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_produces_2k_results(self):
        def fake_runner(argv, **kwargs):
            return _ok_result(argv)

        matrix = run_matrix(
            "prompt", control=self.control_home, treatment=self.treatment_home,
            k=2, workspace=self.workspace, scratch=self.scratch,
            max_parallel=2, runner=fake_runner,
            control_version_id="c1", treatment_version_id="t1",
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

    def test_tolerates_one_arm_failing(self):
        # Fail every treatment run, succeed every control run. The fake
        # runner distinguishes them by looking at cwd, which run_matrix
        # names per-(arm, repeat).
        def fake_runner(argv, **kwargs):
            if "treatment" in kwargs["cwd"]:
                raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
            return _ok_result(argv)

        matrix = run_matrix(
            "prompt", control=self.control_home, treatment=self.treatment_home,
            k=2, workspace=self.workspace, scratch=self.scratch,
            max_parallel=2, runner=fake_runner,
        )

        self.assertEqual(len(matrix["arms"][ARM_CONTROL]), 2)
        self.assertEqual(len(matrix["arms"][ARM_TREATMENT]), 2)
        self.assertTrue(all(r["ok"] for r in matrix["arms"][ARM_CONTROL]))
        self.assertTrue(all(not r["ok"] for r in matrix["arms"][ARM_TREATMENT]))
        # The matrix itself is always returned, never raised, regardless of
        # how many individual runs failed.
        self.assertIn("arms", matrix)

    def test_each_repeat_gets_its_own_home_and_workdir(self):
        seen_homes, seen_cwds = set(), set()

        def fake_runner(argv, **kwargs):
            seen_homes.add(kwargs["env"]["GROK_HOME"])
            seen_cwds.add(kwargs["cwd"])
            return _ok_result(argv)

        run_matrix(
            "prompt", control=self.control_home, treatment=self.treatment_home,
            k=2, workspace=self.workspace, scratch=self.scratch,
            max_parallel=3, runner=fake_runner,
        )

        self.assertEqual(len(seen_homes), 4)
        self.assertEqual(len(seen_cwds), 4)


if __name__ == "__main__":
    unittest.main()
