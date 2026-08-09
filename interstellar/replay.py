#!/usr/bin/env python3
"""Run a prompt under an isolated harness, capture and normalize the session.

Every model call is a child `grok-dev` process, isolated by `GROK_HOME`:
setting that env var makes grok resolve skills/, config.toml, and sessions/
from the given directory instead of the user's real `~/.grok`. Nothing here
ever points GROK_HOME at the user's real home.

Two building blocks:

  run_once      one grok-dev invocation -> one RunResult
  run_matrix    k repeats of control and k repeats of treatment, each in its
                own workdir and its own materialized home, so concurrent
                runs never share a sessions/ dir, a config.toml, or a
                .grok/skills/ dir

The grok invocation itself goes through an injectable `runner` callable
(default `subprocess.run`) so tests can assert the argv and env without
spawning a process. `run_matrix`'s call into `harness.materialize` is
likewise injectable (default `harness.materialize`) for the same reason.

Nothing in this module ever raises out to its caller: every failure mode
(timeout, bad stdout, a crashing normalizer, an unexpected exception from
anywhere in the call chain) becomes an `ok=False` RunResult instead. That
matters more here than it would elsewhere -- these runs cost real money, and
one malformed session must never discard every run that already succeeded.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .harness import materialize as _default_materialize
from .types import ARM_CONTROL, ARM_TREATMENT, make_replay_matrix, make_run_result

GROK = Path.home() / ".local" / "bin" / "grok-dev"
REPO_ROOT = Path(__file__).resolve().parent.parent


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

    A run only counts as `ok` when there is a trace to grade: nine of the ten
    EFFICIENCY_METRICS are read off the normalized trace, so a run with no
    trace is not a data point. `ok=False` covers timeout, unparseable stdout,
    a non-zero exit, no findable session dir, a normalizer crash, and -- as a
    last resort -- any other exception raised anywhere in this function.

    `home` and `workdir` are resolved to absolute paths before anything else:
    grok resolves a relative `GROK_HOME` against its own cwd (`workdir`, via
    the `cwd=` kwarg below), not the caller's cwd, so a relative `home` sends
    grok looking for the home in the wrong place entirely -- it finds
    nothing, silently materializes a fresh empty one, and reports itself
    signed out. A relative `workdir` is also useless once recorded on the
    RunResult, since the report's reader has a different cwd than the runner
    did.
    """
    home = Path(home).resolve()
    workdir = Path(workdir).resolve()
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
    t0 = time.time()

    try:
        before = _session_dirs(home)
        proc = runner(argv, cwd=str(workdir), env=env,
                      capture_output=True, text=True, timeout=timeout)
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
        # mtime — repeats run concurrently, so mtime ordering across threads
        # is not reliable, but the before/after set difference around this
        # one subprocess call is.
        new_dirs = sorted(_session_dirs(home) - before, key=lambda p: p.name)
        session_dir = next((p for p in new_dirs if p.name == session_id), None)
        if session_dir is None and len(new_dirs) == 1:
            session_dir = new_dirs[0]

        trace = None
        if session_dir is not None:
            try:
                trace = _normalize(session_dir)
            except Exception as exc:
                # A malformed session (e.g. an events.jsonl whose first event
                # has no `ts`, which grok_normalize's t0 computation cannot
                # survive) must not take the whole run down with it, or the
                # whole matrix along with it.
                if ok:
                    ok = False
                    error = f"normalize failed: {type(exc).__name__}: {exc}"
        elif ok:
            # grok exited cleanly but nothing new landed under sessions/
            # (e.g. it resumed/appended to an existing session instead of
            # creating one) -- there is no trace to grade, so this is not a
            # countable success.
            ok = False
            error = "no session directory found for run"

        return make_run_result(
            arm=arm, repeat=repeat, ok=ok, prompt=prompt, home=home, workdir=workdir,
            session_id=session_id, trace=trace,
            response_text=out.get("text") or "",
            cost_usd=out.get("total_cost_usd"), wall_s=wall_s,
            turns=out.get("num_turns"), usage=out.get("usage") or {},
            error=error, argv=argv,
        )
    except subprocess.TimeoutExpired:
        return make_run_result(
            arm=arm, repeat=repeat, ok=False, prompt=prompt, home=home,
            workdir=workdir, wall_s=round(time.time() - t0, 3),
            error=f"timeout after {timeout}s", argv=argv,
        )
    except Exception as exc:
        # Last resort: subprocess.SubprocessError/ValueError from a bad argv,
        # OSError from a missing binary, or anything else -- never escape.
        return make_run_result(
            arm=arm, repeat=repeat, ok=False, prompt=prompt, home=home,
            workdir=workdir, wall_s=round(time.time() - t0, 3),
            error=f"{type(exc).__name__}: {exc}", argv=argv,
        )


# --------------------------------------------------------------------------
# isolated_workdir
# --------------------------------------------------------------------------

# `.git` is never useful inside a replay workdir; the rest are large,
# regenerable directories that would make every run pay a real copy cost
# (and, for `target/`, a very large one in this repo) for no benefit.
# Excluded only at the top level of `src` (see `_make_ignore`) -- matching
# by basename anywhere in the tree would also skip an unrelated,
# legitimately-named directory deeper down.
_TOP_LEVEL_IGNORE_NAMES = frozenset({".git", "target", "node_modules",
                                     "__pycache__", ".venv"})


def _make_ignore(src: Path, dest: Path):
    """Build the `shutil.copytree` ignore callable for one `isolated_workdir`
    call. Path-based, not name-based: it excludes `dest` itself, and any
    directory that *contains* `dest`, wherever encountered while walking
    `src`. `dest` is commonly inside `src` -- the cycle's own output
    directory defaults to `<workspace>/runs/<ts>/...`, so `scratch` (and
    every sibling run's home/workdir under it) typically lives inside the
    very workspace being copied. Skipping the topmost containing directory
    (not just `dest` itself) both prevents copytree from recursing into its
    own destination -- which would otherwise copy it into itself until it
    dies on path length -- and, as a side effect, keeps every other run's
    already-materialized output out of this run's copy instead of paying a
    real, unbounded copy cost for it on every one of the 2*k runs.
    """
    dest_may_be_inside_src = dest.is_relative_to(src)

    def _ignore(dirpath, names):
        dirpath = Path(dirpath).resolve()
        skip = set()
        if dirpath == src:
            skip |= _TOP_LEVEL_IGNORE_NAMES & set(names)
        if dest_may_be_inside_src:
            for name in names:
                if dest.is_relative_to((dirpath / name).resolve()):
                    skip.add(name)
        return skip

    return _ignore


def isolated_workdir(src: Path, dest: Path) -> Path:
    """Copy the project workspace into a throwaway dir.

    Always a plain recursive copy -- never `git worktree`. `git worktree add
    --detach` checks out HEAD, which drops every uncommitted modification
    and every untracked file. The prompt being replayed ran against the
    user's actual working state, uncommitted and untracked files included
    (in this repo, that is most of it: `analyzer/`, `synth/`,
    `trace-viewer/`, `traces/` are all untracked); a HEAD checkout silently
    replays a different world. A plain copy also means nothing is ever
    written into the user's real repo (no `.git/worktrees/` registration to
    forget to clean up) and two invocations of the same command behave
    identically regardless of git state.

    `src` and `dest` are resolved to absolute paths up front, and the copy's
    ignore predicate is path-aware (see `_make_ignore`) so that `dest` being
    a descendant of `src` -- the documented default, since `--out` lives
    inside `workspace` unless told otherwise -- does not make copytree walk
    into its own output. `dest == src`, or `src` nested inside `dest` (which
    would make the pre-copy `rmtree(dest)` below delete `src`), cannot be
    made safe by excluding anything and are refused outright rather than
    silently producing an empty or destroyed workdir.
    """
    src = Path(src).resolve()
    dest = Path(dest).resolve()
    if dest == src or src.is_relative_to(dest):
        raise ValueError(
            f"isolated_workdir: dest ({dest}) is src itself, or an ancestor "
            f"of src ({src}) -- cannot be safely excluded from the copy")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=_make_ignore(src, dest))
    return dest


# --------------------------------------------------------------------------
# run_matrix
# --------------------------------------------------------------------------

def run_matrix(prompt, *, control, treatment, k: int, workspace: Path,
               scratch: Path, auth_from: Path, max_parallel: int = 3,
               timeout=600, model=None, patch_id=None,
               runner=subprocess.run, materialize=_default_materialize):
    """k paired, isolated repeats of `prompt` under `control` and `treatment`.

    `control`/`treatment` are HarnessVersion dicts (harness.snapshot /
    harness.apply output) -- run_matrix materializes each one itself, once
    per run, via the injectable `materialize` (default harness.materialize),
    rather than reusing one pre-built home. Per run, in this order:

      1. copy `workspace` into a fresh workdir (isolated_workdir);
      2. delete any `.grok/skills/` that copy carries;
      3. materialize the arm's home with `project_dest` pointing at that
         same workdir.

    Order matters. grok's skill precedence is `./.grok/skills/` (highest) >
    `<repo>/.grok/skills/` > `$GROK_HOME/skills/` (lowest). Materializing
    into GROK_HOME alone is not enough: the workdir copy still carries the
    project's *original, unpatched* `.grok/skills/`, which outranks
    anything written to GROK_HOME and would silently shadow a patched
    project-scope skill -- the treatment arm would equal control with no
    error anywhere. Step 2 removes that shadow; step 3's `project_dest`
    gives the version's project-scope skills (if any) the tier that
    actually wins.

    Each repeat gets its own workdir and its own home -- concurrent runs
    never share a sessions/ dir, a config.toml, or a .grok/skills/ dir, even
    both arms running at once in the same ThreadPoolExecutor.

    Never raises: a run that fails for any reason -- a bad workdir copy, a
    bad materialize, a runner exception of any kind, a crash anywhere inside
    run_once -- becomes an ok=False RunResult, exactly like any other
    failure. These runs cost real money; one malformed run must never
    discard every run around it that already succeeded.

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

    Recording `dispatch_order` and each run's start time on the returned
    matrix was also asked for, but there is nowhere on the frozen
    ReplayMatrix/RunResult shapes to put them yet -- that needs a contract
    change this function cannot make on its own, so for now the interleaving
    itself is real (submission order below is the interleaved order) but is
    not yet independently recorded on the return value.
    """
    # Resolved once, up front: everything derived from `scratch` below --
    # every per-run home_dest/workdir_dest -- must be absolute, the same
    # reason run_once resolves `home`/`workdir` itself (see its docstring).
    # A relative `scratch` would otherwise reproduce the exact bug even
    # though run_once also resolves, because the setup-failure branch below
    # records home_dest/workdir_dest on a RunResult without ever calling
    # run_once.
    scratch = Path(scratch).resolve()
    workspace = Path(workspace).resolve()
    auth_from = Path(auth_from).resolve()
    plan = []
    for i in range(k):
        plan.append((ARM_CONTROL, i, control))
        plan.append((ARM_TREATMENT, i, treatment))

    def _task(arm, repeat, version):
        # except Exception, not except OSError: setup (isolated_workdir,
        # materialize) can fail with shutil.Error, a bad HarnessVersion, or
        # anything else, and run_once is a second, independent safety net,
        # not a substitute for this one -- a future collected with
        # fut.result() re-raises in the main thread, which would abort every
        # other run in the matrix, succeeded or not.
        tag = f"{arm}-{repeat}"
        home_dest = scratch / f"{tag}-home"
        workdir_dest = scratch / f"{tag}-workdir"
        try:
            workdir = isolated_workdir(Path(workspace), workdir_dest)
            project_skills = workdir / ".grok" / "skills"
            if project_skills.exists():
                shutil.rmtree(project_skills)
            home = materialize(version, home_dest, auth_from=auth_from,
                               project_dest=workdir)
            return run_once(prompt, home, workdir=workdir, timeout=timeout,
                            model=model, extra_rules=version.get("extra_rules") or [],
                            runner=runner, arm=arm, repeat=repeat)
        except Exception as exc:
            # Record the paths this run actually attempted to use, not a
            # shared template it never touched, so the RunResult is honest
            # about where to look.
            return make_run_result(
                arm=arm, repeat=repeat, ok=False, prompt=prompt,
                home=home_dest, workdir=workdir_dest,
                error=f"setup failed: {type(exc).__name__}: {exc}",
            )

    arms = {ARM_CONTROL: [None] * k, ARM_TREATMENT: [None] * k}
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        # Submitted in `plan` order (dict comprehensions preserve insertion
        # order), so submission order == the interleaved dispatch order.
        futures = {pool.submit(_task, arm, repeat, version): (arm, repeat)
                  for arm, repeat, version in plan}
        for fut in futures:
            arm, repeat = futures[fut]
            arms[arm][repeat] = fut.result()

    return make_replay_matrix(
        prompt=prompt, k=k, arms=arms,
        control_version_id=control.get("version_id"),
        treatment_version_id=treatment.get("version_id"),
        patch_id=patch_id,
    )
