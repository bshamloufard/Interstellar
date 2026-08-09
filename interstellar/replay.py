#!/usr/bin/env python3
"""Run a prompt under an isolated harness, capture and normalize the session.

Every model call is a child `grok-dev` process, isolated by `GROK_HOME`:
setting that env var makes grok resolve skills/, config.toml, and sessions/
from the given directory instead of the user's real `~/.grok`. Nothing here
ever points GROK_HOME at the user's real home.

Two building blocks:

  run_once      one grok-dev invocation -> one RunResult
  run_matrix    k repeats of control and k repeats of treatment, each in its
                own workdir and its own copy of the home directory, so
                concurrent runs never share a sessions/ dir or a config.toml

The grok invocation itself goes through an injectable `runner` callable
(default `subprocess.run`) so tests can assert the argv and env without
spawning a process.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .types import ARM_CONTROL, ARM_TREATMENT, make_replay_matrix, make_run_result

GROK = Path.home() / ".local" / "bin" / "grok-dev"
REPO_ROOT = Path(__file__).resolve().parent.parent

# `git worktree add` mutates the source repo's .git dir (index, worktree
# list); concurrent adds against the same repo can race on that lock. One
# process-wide lock serializes just the git call, not the (slow) copy.
_GIT_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# session discovery + normalization
# --------------------------------------------------------------------------

def _session_dirs(home: Path) -> set[Path]:
    """Every `<home>/sessions/<cwd-encoded>/<session-uuid>/` dir that exists now."""
    root = Path(home) / "sessions"
    if not root.is_dir():
        return set()
    return {p for p in root.glob("*/*") if p.is_dir()}


def _normalize(session_dir: Path):
    """Import normalizer/grok_normalize.py and normalize one session dir.

    Same import dance as synth/build_corpus.py: the normalizer is a sibling
    top-level package, not installed, so it is reached via sys.path.
    """
    normalizer_dir = str(REPO_ROOT / "normalizer")
    if normalizer_dir not in sys.path:
        sys.path.insert(0, normalizer_dir)
    from grok_normalize import normalize_session  # noqa: E402

    return normalize_session(session_dir)


# --------------------------------------------------------------------------
# run_once
# --------------------------------------------------------------------------

def run_once(prompt, home: Path, *, workdir: Path, timeout=600, model=None,
             extra_rules: list[str] = None, runner=subprocess.run,
             arm=ARM_CONTROL, repeat=0):
    """Run one prompt under one isolated GROK_HOME. Never raises.

    `arm`/`repeat` are stamped onto the returned RunResult; run_matrix is the
    only caller that needs to set them to anything other than the default.
    """
    home = Path(home)
    workdir = Path(workdir)
    extra_rules = list(extra_rules or [])

    argv = [str(GROK), "-p", prompt, "--output-format", "json",
            "--permission-mode", "bypassPermissions"]
    if model:
        argv += ["--model", model]
    if extra_rules:
        # --rules <TEXT> takes a single value; multiple rule lines are joined
        # into one block rather than repeating the flag (undocumented as
        # repeatable, unlike --allow/--deny).
        argv += ["--rules", "\n".join(extra_rules)]

    env = {**os.environ, "GROK_HOME": str(home)}

    before = _session_dirs(home)
    t0 = time.time()
    try:
        proc = runner(argv, cwd=str(workdir), env=env,
                      capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return make_run_result(
            arm=arm, repeat=repeat, ok=False, prompt=prompt, home=home,
            workdir=workdir, wall_s=round(time.time() - t0, 3),
            error=f"timeout after {timeout}s", argv=argv,
        )
    except OSError as exc:
        return make_run_result(
            arm=arm, repeat=repeat, ok=False, prompt=prompt, home=home,
            workdir=workdir, wall_s=round(time.time() - t0, 3),
            error=f"{type(exc).__name__}: {exc}", argv=argv,
        )
    wall_s = round(time.time() - t0, 3)

    ok, error, out = True, None, {}
    try:
        out = json.loads(proc.stdout)
        if not isinstance(out, dict):
            raise TypeError("stdout json is not an object")
    except (json.JSONDecodeError, TypeError) as exc:
        ok = False
        error = (f"unparseable stdout ({exc}); exit={proc.returncode} "
                 f"stderr={(proc.stderr or '')[-400:]!r}")
    if ok and proc.returncode != 0:
        ok = False
        error = f"grok exited {proc.returncode}: {(proc.stderr or '')[-400:]}"

    session_id = out.get("sessionId")

    # Session dir discovery: diff before/after rather than trust newest
    # mtime — repeats run concurrently, so mtime ordering across threads is
    # not reliable, but the before/after set difference around this one
    # subprocess call is.
    new_dirs = sorted(_session_dirs(home) - before, key=lambda p: p.name)
    session_dir = next((p for p in new_dirs if p.name == session_id), None)
    if session_dir is None and len(new_dirs) == 1:
        session_dir = new_dirs[0]

    trace = _normalize(session_dir) if session_dir is not None else None

    return make_run_result(
        arm=arm, repeat=repeat, ok=ok, prompt=prompt, home=home, workdir=workdir,
        session_id=session_id, trace=trace,
        response_text=out.get("text") or "",
        cost_usd=out.get("total_cost_usd"), wall_s=wall_s,
        turns=out.get("num_turns"), usage=out.get("usage") or {},
        error=error, argv=argv,
    )


# --------------------------------------------------------------------------
# isolated_workdir
# --------------------------------------------------------------------------

def _is_git_repo(path: Path) -> bool:
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           cwd=str(path), capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


def isolated_workdir(src: Path, dest: Path) -> Path:
    """Copy a project workspace into a throwaway dir.

    `git worktree add --detach` when `src` is a repo (cheap, and keeps the
    working tree at the same commit without touching `src`); a plain
    recursive copy otherwise. Either way, mutations under `dest` cannot leak
    back into `src` or into another run's `dest`.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    if _is_git_repo(src):
        with _GIT_LOCK:
            r = subprocess.run(
                ["git", "worktree", "add", "--detach", str(dest)],
                cwd=str(src), capture_output=True, text=True, timeout=60,
            )
        if r.returncode == 0:
            return dest
        # Fall through to a plain copy if worktree creation failed for any
        # reason (e.g. dirty index git refuses to snapshot cleanly).

    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git"))
    return dest


# --------------------------------------------------------------------------
# run_matrix
# --------------------------------------------------------------------------

def _copy_home(template: Path, dest: Path) -> Path:
    """Own copy of a materialized GROK_HOME: same skills/config/auth, fresh
    (empty) sessions/ dir so no run's sessions/ is ever shared."""
    template, dest = Path(template), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, dest, ignore=shutil.ignore_patterns("sessions"))
    (dest / "sessions").mkdir(parents=True, exist_ok=True)
    return dest


def run_matrix(prompt, *, control: Path, treatment: Path, k: int,
               workspace: Path, scratch: Path, max_parallel: int = 3,
               timeout=600, model=None,
               control_version_id="", treatment_version_id="",
               control_extra_rules: list[str] = None,
               treatment_extra_rules: list[str] = None,
               patch_id=None, runner=subprocess.run):
    """k paired, isolated repeats of `prompt` under `control` and `treatment`.

    `control`/`treatment` are already-materialized GROK_HOME dirs (see
    harness.materialize) shared as read-only templates; each repeat gets its
    own copy of the template and its own copy of `workspace`, so one run's
    file or session-dir mutations can never leak into another's, even when
    both arms run concurrently in the same ThreadPoolExecutor.

    Never raises: a run that fails (bad home copy, bad workdir, grok error)
    becomes an ok=False RunResult like any other failure inside run_once, so
    one arm's failure never prevents the rest of the matrix from completing.

    Dispatch order is interleaved -- (control, r0), (treatment, r0),
    (control, r1), (treatment, r1), ... -- rather than arm-major, and
    submitted to the executor in that order. Every paired statistic
    downstream (permutation test, bootstrap, win rate) assumes the k pairs
    are exchangeable draws; running all of one arm and then all of the
    other perfectly confounds temporal drift (model-side changes, endpoint
    load, machine load) with the arm. With max_parallel > 1 the actual
    execution overlap is approximate -- what matters is both arms are drawn
    from the same stretch of wall-clock time rather than two disjoint
    blocks.

    Recording `dispatch_order` and each run's start time was also asked for
    (docs/plans/review-loop.md discussion), but there is nowhere on the
    frozen ReplayMatrix/RunResult shapes to put them yet -- that needs a
    contract change this function cannot make on its own, so for now the
    interleaving itself is real (submission order below is the interleaved
    order) but is not yet independently recorded on the return value.
    """
    scratch = Path(scratch)
    plan = []
    for i in range(k):
        plan.append((ARM_CONTROL, i, Path(control), control_extra_rules))
        plan.append((ARM_TREATMENT, i, Path(treatment), treatment_extra_rules))

    def _task(arm, repeat, home_template, rules):
        tag = f"{arm}-{repeat}"
        try:
            home = _copy_home(home_template, scratch / f"{tag}-home")
            workdir = isolated_workdir(Path(workspace), scratch / f"{tag}-workdir")
        except OSError as exc:
            return make_run_result(
                arm=arm, repeat=repeat, ok=False, prompt=prompt,
                home=home_template, workdir=workspace,
                error=f"setup failed: {type(exc).__name__}: {exc}",
            )
        return run_once(prompt, home, workdir=workdir, timeout=timeout,
                        model=model, extra_rules=rules, runner=runner,
                        arm=arm, repeat=repeat)

    arms = {ARM_CONTROL: [None] * k, ARM_TREATMENT: [None] * k}
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        # Submitted in `plan` order (dict comprehensions preserve insertion
        # order), so submission order == the interleaved dispatch order.
        futures = {pool.submit(_task, arm, repeat, home, rules): (arm, repeat)
                  for arm, repeat, home, rules in plan}
        for fut in futures:
            arm, repeat = futures[fut]
            arms[arm][repeat] = fut.result()

    return make_replay_matrix(
        prompt=prompt, k=k, arms=arms,
        control_version_id=control_version_id,
        treatment_version_id=treatment_version_id,
        patch_id=patch_id,
    )
