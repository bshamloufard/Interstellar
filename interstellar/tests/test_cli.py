"""Tests for interstellar.cli: the review-loop orchestrator.

Never calls grok-dev or any other subprocess -- every path that would spawn
one is either skipped (--dry-run) or given an injected fake, per the
project's global test constraint.

    python3 -m unittest discover interstellar/tests
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from interstellar import cli
from interstellar.types import (
    EFFICIENCY_METRICS,
    PATCH_MCP_REMOVE,
    PATCH_RULES_APPEND,
    PATCH_SKILL_INSERT,
    PATCH_SKILL_REMOVE,
    PATCH_SKILL_TRUNCATE,
    make_harness_version,
    make_skill,
)

CORPUS_TRACE = cli.REPO_ROOT / "traces" / "corpus" / "skill_bloat_019fe41b.json"
CACHED_RESULTS_DIR = cli.REPO_ROOT / "analyzer" / "results"


def _refuse_to_spawn(*args, **kwargs):
    raise AssertionError("a --dry-run cycle must never spawn a subprocess")


class _FixedNow:
    def __init__(self, start):
        self._t = start

    def __call__(self):
        self._t += 1
        return datetime.fromtimestamp(self._t, tz=timezone.utc)


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

class ArgParsingTests(unittest.TestCase):
    def test_defaults(self):
        args = cli.build_arg_parser().parse_args(["review", "trace.json"])
        self.assertEqual(args.command, "review")
        self.assertEqual(args.trace, ["trace.json"])
        self.assertEqual(args.k, cli.DEFAULT_K)
        self.assertEqual(args.max_patches, cli.DEFAULT_MAX_PATCHES)
        self.assertIsNone(args.out)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.serve)
        self.assertEqual(args.port, cli.DEFAULT_PORT)
        self.assertIsNone(args.budget_usd)
        self.assertFalse(args.use_cached_analysis)
        self.assertIsNone(args.model)
        self.assertEqual(args.max_parallel, cli.DEFAULT_MAX_PARALLEL)
        self.assertEqual(args.timeout, cli.DEFAULT_TIMEOUT)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.max_token_regression, cli.DEFAULT_MAX_TOKEN_REGRESSION)
        self.assertIsNone(args.grok_home)
        self.assertFalse(args.no_preflight)

    def test_flags_and_overrides(self):
        args = cli.build_arg_parser().parse_args([
            "review", "a.json", "b.json",
            "--k", "5", "--max-patches", "2", "--out", "/tmp/out",
            "--dry-run", "--serve", "--port", "9000", "--budget-usd", "1.5",
            "--use-cached-analysis", "--model", "grok-4.5", "--max-parallel", "1",
            "--timeout", "30", "--seed", "7", "--max-token-regression", "0.2",
            "--grok-home", "/tmp/home", "--no-preflight",
        ])
        self.assertEqual(args.trace, ["a.json", "b.json"])
        self.assertEqual(args.k, 5)
        self.assertEqual(args.max_patches, 2)
        self.assertEqual(args.out, Path("/tmp/out"))
        self.assertTrue(args.dry_run)
        self.assertTrue(args.serve)
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.budget_usd, 1.5)
        self.assertTrue(args.use_cached_analysis)
        self.assertEqual(args.model, "grok-4.5")
        self.assertEqual(args.max_parallel, 1)
        self.assertEqual(args.timeout, 30)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.max_token_regression, 0.2)
        self.assertEqual(args.grok_home, Path("/tmp/home"))
        self.assertTrue(args.no_preflight)

    def test_missing_trace_is_an_error(self):
        with self.assertRaises(SystemExit):
            cli.build_arg_parser().parse_args(["review"])

    def test_missing_command_is_an_error(self):
        with self.assertRaises(SystemExit):
            cli.build_arg_parser().parse_args([])

    def test_main_dispatches_to_cmd_review(self):
        with tempfile.TemporaryDirectory() as td:
            trace_file = Path(td) / "trace.json"
            trace_file.write_text("{}")
            with mock.patch.object(cli, "run_review") as fake_run:
                fake_run.return_value = {"ok": True}
                rc = cli.main(["review", str(trace_file), "--dry-run"])
        self.assertEqual(rc, 0)
        fake_run.assert_called_once()
        _, kwargs = fake_run.call_args
        self.assertTrue(kwargs["dry_run"])

    def test_relative_out_and_grok_home_are_resolved_to_absolute(self):
        # Regression: a relative --out (the default shape, runs/<ts>, IS
        # relative) used to reach _run_preflight as a relative GROK_HOME;
        # grok resolves that against ITS OWN cwd and reports a plain path
        # bug as "not signed in". Every user path must be absolute by the
        # time it leaves argument parsing.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td).resolve()
            trace_file = td_path / "trace.json"
            trace_file.write_text("{}")
            grok_home_dir = td_path / "some_home"
            grok_home_dir.mkdir()

            cwd = os.getcwd()
            os.chdir(td_path)
            try:
                with mock.patch.object(cli, "run_review") as fake_run:
                    fake_run.return_value = {"ok": True}
                    cli.main(["review", "trace.json", "--dry-run",
                             "--out", "relative_out", "--grok-home", "some_home"])
            finally:
                os.chdir(cwd)

        args, kwargs = fake_run.call_args
        self.assertTrue(Path(args[0]).is_absolute())
        self.assertEqual(args[0], trace_file)
        self.assertTrue(kwargs["out_dir"].is_absolute())
        self.assertEqual(kwargs["out_dir"], td_path / "relative_out")
        self.assertTrue(kwargs["grok_home_override"].is_absolute())
        self.assertEqual(kwargs["grok_home_override"], grok_home_dir)


class ResolveTracePathsTests(unittest.TestCase):
    def test_literal_path_must_exist(self):
        with self.assertRaises(SystemExit):
            cli._resolve_trace_paths(["/no/such/file.json"])

    def test_glob_with_no_matches_errors(self):
        with self.assertRaises(SystemExit):
            cli._resolve_trace_paths(["/no/such/dir/*.json"])

    def test_glob_expands(self):
        paths = cli._resolve_trace_paths([str(cli.REPO_ROOT / "traces" / "corpus" / "skill_bloat_*.json")])
        self.assertEqual(paths, [CORPUS_TRACE])


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

class ResolvePromptTests(unittest.TestCase):
    def test_prefers_ground_truth_prompt(self):
        trace = {"ground_truth": {"prompt": "  do the thing  "}, "session": {}}
        prompt, source = cli._resolve_prompt(trace)
        self.assertEqual(prompt, "do the thing")
        self.assertEqual(source, "ground_truth.prompt")

    def test_falls_back_to_user_query_in_session_dir(self):
        with tempfile.TemporaryDirectory() as td:
            session_dir = Path(td)
            history = [
                {"type": "system", "content": "sys"},
                {"type": "user", "content": [{"type": "text", "text": "<user_info>\nfoo\n</user_info>"}]},
                {"type": "user", "content": [{"type": "text",
                 "text": "<user_query>\nRead app.py\n</user_query>"}]},
            ]
            (session_dir / "chat_history.jsonl").write_text(
                "\n".join(json.dumps(h) for h in history), encoding="utf-8")
            trace = {"session": {"session_dir": str(session_dir)}}
            prompt, source = cli._resolve_prompt(trace)
        self.assertEqual(prompt, "Read app.py")
        self.assertIn("user_query", source)

    def test_fails_loudly_when_nothing_found(self):
        trace = {"session": {"session_dir": "/no/such/dir"}}
        with self.assertRaises(SystemExit):
            cli._resolve_prompt(trace)


class ResolveBaselineHomeTests(unittest.TestCase):
    def test_override_wins(self):
        home, source = cli._resolve_baseline_home({}, "/some/override")
        self.assertEqual(home, Path("/some/override"))
        self.assertIn("override", source)

    def test_falls_back_to_live_home_when_session_dir_missing(self):
        home, source = cli._resolve_baseline_home({"session": {}}, None)
        self.assertEqual(home, Path.home() / ".grok")
        self.assertIn("live", source)

    def test_uses_recorded_home_when_it_looks_real(self):
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "home"
            (fake_home / "skills").mkdir(parents=True)
            (fake_home / "config.toml").write_text("")
            session_dir = fake_home / "sessions" / "cwd-enc" / "sess-1"
            session_dir.mkdir(parents=True)
            home, source = cli._resolve_baseline_home(
                {"session": {"session_dir": str(session_dir)}}, None)
        # .resolve()d (e.g. macOS's /tmp -> /private/tmp), not just
        # string-equal to the unresolved fixture path -- see the module
        # docstring note on why every path here must end up absolute.
        self.assertEqual(home, fake_home.resolve())
        self.assertTrue(home.is_absolute())
        self.assertIn("recorded session_dir", source)


class PatchImpactScoreTests(unittest.TestCase):
    def _patch(self, kind, target):
        return {"kind": kind, "target": target, "patch_id": "patch-001"}

    def test_truncate_scores_measured_wasted_tokens(self):
        dig = {"skills_loaded": [{"skill": "strict-audit", "unreferenced_tokens_est": 728}]}
        score = cli._patch_impact_score(self._patch(PATCH_SKILL_TRUNCATE, "strict-audit"), dig)
        self.assertEqual(score, 728.0)

    def test_remove_of_never_loaded_skill_scores_zero(self):
        dig = {"skills_loaded": [{"skill": "other", "tokens_est": 999}]}
        score = cli._patch_impact_score(self._patch(PATCH_SKILL_REMOVE, "unused-skill"), dig)
        self.assertEqual(score, 0.0)

    def test_truncate_outranks_zero_cost_remove(self):
        dig = {"skills_loaded": [{"skill": "strict-audit", "unreferenced_tokens_est": 728}]}
        truncate = self._patch(PATCH_SKILL_TRUNCATE, "strict-audit")
        remove = self._patch(PATCH_SKILL_REMOVE, "never-loaded")
        self.assertGreater(cli._patch_impact_score(truncate, dig),
                           cli._patch_impact_score(remove, dig))

    def test_rules_append_scores_zero(self):
        score = cli._patch_impact_score(self._patch(PATCH_RULES_APPEND, "rules"), {})
        self.assertEqual(score, 0.0)


class PatchImpactUnitTests(unittest.TestCase):
    def test_skill_truncate_and_remove_share_tokens_unit(self):
        self.assertEqual(cli._patch_impact_unit(
            {"kind": PATCH_SKILL_TRUNCATE}), "tokens_est")
        self.assertEqual(cli._patch_impact_unit(
            {"kind": PATCH_SKILL_REMOVE}), "tokens_est")

    def test_skill_insert_and_mcp_remove_share_ms_unit(self):
        # skill.insert's only measured evidence (discovery-call ms after the
        # load) is a time quantity, not a token count -- it must NOT be
        # grouped with skill.truncate/remove's token-denominated class, even
        # though all three share expected_effect=EFFECT_SKILL_TOKENS.
        self.assertEqual(cli._patch_impact_unit({"kind": PATCH_SKILL_INSERT}), "ms")
        self.assertEqual(cli._patch_impact_unit({"kind": PATCH_MCP_REMOVE}), "ms")

    def test_rules_append_is_unmeasured(self):
        self.assertEqual(cli._patch_impact_unit({"kind": PATCH_RULES_APPEND}), "unmeasured")


class RankAndSelectTests(unittest.TestCase):
    def test_truncates_to_max_patches_and_logs_drops(self):
        dig = {"skills_loaded": [
            {"skill": "high", "unreferenced_tokens_est": 900},
            {"skill": "low", "unreferenced_tokens_est": 10},
        ]}
        patches_list = [
            {"patch_id": "patch-001", "kind": PATCH_SKILL_TRUNCATE, "target": "high",
             "skipped_reason": None},
            {"patch_id": "patch-002", "kind": PATCH_SKILL_TRUNCATE, "target": "low",
             "skipped_reason": None},
            {"patch_id": "patch-003", "kind": PATCH_SKILL_REMOVE, "target": "bad",
             "skipped_reason": "skill not found: bad"},
        ]
        selected, dropped = cli._rank_and_select(patches_list, dig, max_patches=1)
        self.assertEqual([p["patch_id"] for p in selected], ["patch-001"])
        dropped_ids = {d["patch_id"] for d in dropped}
        self.assertEqual(dropped_ids, {"patch-002", "patch-003"})
        by_id = {d["patch_id"]: d for d in dropped}
        # dropped entries are full Patch dicts (report.build(dropped_patches=)
        # needs kind/target/skipped_reason/rationale/source_recommendation),
        # plus impact_score/impact_unit for the CLI's own ranking transparency.
        self.assertEqual(by_id["patch-002"]["kind"], PATCH_SKILL_TRUNCATE)
        self.assertEqual(by_id["patch-002"]["target"], "low")
        self.assertIn("budget", by_id["patch-002"]["skipped_reason"])
        self.assertEqual(by_id["patch-002"]["impact_unit"], "tokens_est")
        self.assertEqual(by_id["patch-002"]["impact_score"], 10.0)
        # already-skipped patches keep their own original skipped_reason
        # verbatim, untouched by ranking.
        self.assertEqual(by_id["patch-003"]["skipped_reason"], "skill not found: bad")
        self.assertIsNone(by_id["patch-003"]["impact_score"])

    def test_keeps_everything_when_under_budget(self):
        patches_list = [
            {"patch_id": "patch-001", "kind": PATCH_RULES_APPEND, "target": "rules",
             "skipped_reason": None},
        ]
        selected, dropped = cli._rank_and_select(patches_list, {}, max_patches=3)
        self.assertEqual(len(selected), 1)
        self.assertEqual(dropped, [])

    def test_never_ranks_ms_against_tokens_across_classes(self):
        # The exact shape of the reported bug: two mcp.remove patches
        # (measured in ms) score just above a skill.truncate (measured in
        # tokens_est) on raw magnitude alone. A flat top-N would pick both
        # mcp patches and drop the skill patch; round-robin across the two
        # unit classes must give the skill patch a slot instead.
        dig = {
            "skills_loaded": [{"skill": "strict-audit", "unreferenced_tokens_est": 728}],
            "failed_mcp_servers": [
                {"server": "fetch", "wasted_ms": 748},
                {"server": "git", "wasted_ms": 741},
            ],
        }
        patches_list = [
            {"patch_id": "patch-001", "kind": PATCH_SKILL_TRUNCATE,
             "target": "strict-audit", "skipped_reason": None},
            {"patch_id": "patch-002", "kind": PATCH_MCP_REMOVE,
             "target": "fetch", "skipped_reason": None},
            {"patch_id": "patch-003", "kind": PATCH_MCP_REMOVE,
             "target": "git", "skipped_reason": None},
        ]
        selected, dropped = cli._rank_and_select(patches_list, dig, max_patches=2)
        selected_ids = {p["patch_id"] for p in selected}
        self.assertIn("patch-001", selected_ids)  # the skill patch got a slot
        self.assertEqual(len(selected_ids), 2)
        self.assertEqual(len(dropped), 1)  # only the weaker of the two mcp patches drops
        self.assertEqual(dropped[0]["patch_id"], "patch-003")
        self.assertIn("'ms' impact class", dropped[0]["skipped_reason"])

    def test_round_robin_gives_each_class_a_slot_before_a_second_pick(self):
        dig = {
            "skills_loaded": [
                {"skill": "big", "unreferenced_tokens_est": 5000},
                {"skill": "small", "unreferenced_tokens_est": 5},
            ],
            "mcp_servers": {"only-mcp": {"connect_ms": 100}},
        }
        patches_list = [
            {"patch_id": "patch-001", "kind": PATCH_SKILL_TRUNCATE, "target": "big",
             "skipped_reason": None},
            {"patch_id": "patch-002", "kind": PATCH_SKILL_TRUNCATE, "target": "small",
             "skipped_reason": None},
            {"patch_id": "patch-003", "kind": PATCH_MCP_REMOVE, "target": "only-mcp",
             "skipped_reason": None},
        ]
        # max_patches=2: round-robin must take the best of each class first
        # (big, only-mcp) rather than both tokens_est patches (big, small).
        selected, _ = cli._rank_and_select(patches_list, dig, max_patches=2)
        self.assertEqual({p["patch_id"] for p in selected}, {"patch-001", "patch-003"})


class RunHealthTests(unittest.TestCase):
    def _run_result(self, ok):
        return {"ok": ok}

    def test_counts_ok_against_requested_k_not_attempted_count(self):
        # Only 1 repeat actually ran (e.g. fail-fast stopped early), but the
        # denominator must stay the ORIGINALLY requested k=3 so the shortfall
        # itself is visible, not hidden behind a 1/1-looks-fine denominator.
        matrix_summary = {"arms": {
            "control": [self._run_result(True)],
            "treatment": [self._run_result(False)],
        }}
        health = cli._run_health(matrix_summary, k=3)
        self.assertEqual(health["control"], (1, 3))
        self.assertEqual(health["treatment"], (0, 3))

    def test_all_ok(self):
        matrix_summary = {"arms": {
            "control": [self._run_result(True)] * 3,
            "treatment": [self._run_result(True)] * 3,
        }}
        health = cli._run_health(matrix_summary, k=3)
        self.assertEqual(health["control"], (3, 3))
        self.assertEqual(health["treatment"], (3, 3))

    def test_empty_arms_is_zero_of_k(self):
        health = cli._run_health({"arms": {"control": [], "treatment": []}}, k=3)
        self.assertEqual(health["control"], (0, 3))
        self.assertEqual(health["treatment"], (0, 3))

    def test_missing_matrix_summary_is_zero_of_k(self):
        health = cli._run_health(None, k=3)
        self.assertEqual(health["control"], (0, 3))
        self.assertEqual(health["treatment"], (0, 3))


class CountRunsTests(unittest.TestCase):
    def test_totals_across_patches(self):
        patch_results = [
            {"matrix": {"arms": {"control": [{"ok": True}, {"ok": False}],
                                 "treatment": [{"ok": True}, {"ok": True}]}}},
            {"matrix": {"arms": {"control": [{"ok": False}],
                                 "treatment": [{"ok": False}]}}},
        ]
        total, ok = cli._count_runs(patch_results)
        self.assertEqual(total, 6)
        self.assertEqual(ok, 3)

    def test_no_runs_is_zero_zero(self):
        patch_results = [{"matrix": {"arms": {"control": [], "treatment": []}}}]
        self.assertEqual(cli._count_runs(patch_results), (0, 0))


class CacheReadRangeTests(unittest.TestCase):
    def _run(self, cache_read):
        return {"usage": {"cache_read_input_tokens": cache_read}} if cache_read is not None \
            else {"usage": {}}

    def test_min_max_across_patches_and_arms(self):
        patch_results = [
            {"matrix": {"arms": {
                "control": [self._run(27648), self._run(41600)],
                "treatment": [self._run(16768)],
            }}},
            {"matrix": {"arms": {
                "control": [self._run(41728)],
                "treatment": [self._run(31360), self._run(42752)],
            }}},
        ]
        self.assertEqual(cli._cache_read_range(patch_results), (16768, 42752))

    def test_none_when_no_usage_data(self):
        patch_results = [{"matrix": {"arms": {
            "control": [self._run(None)], "treatment": [self._run(None)],
        }}}]
        self.assertIsNone(cli._cache_read_range(patch_results))

    def test_none_for_empty_patch_results(self):
        self.assertIsNone(cli._cache_read_range([]))


class CacheWarmthCaveatTests(unittest.TestCase):
    def test_none_when_no_usage_data(self):
        self.assertIsNone(cli._cache_warmth_caveat([]))

    def test_states_this_cycles_real_observed_range(self):
        patch_results = [{"matrix": {"arms": {
            "control": [{"usage": {"cache_read_input_tokens": 16768}}],
            "treatment": [{"usage": {"cache_read_input_tokens": 42752}}],
        }}}]
        caveat = cli._cache_warmth_caveat(patch_results)
        self.assertIn("16,768", caveat)
        self.assertIn("42,752", caveat)
        self.assertIn("prompt-cache warmth", caveat)
        self.assertIn("cost_usd", caveat)

    def test_never_states_a_canned_figure(self):
        # The 16k-43k figure from the reported cycle must never appear
        # unless THIS cycle's own data actually produced it.
        patch_results = [{"matrix": {"arms": {
            "control": [{"usage": {"cache_read_input_tokens": 5000}}],
            "treatment": [{"usage": {"cache_read_input_tokens": 5200}}],
        }}}]
        caveat = cli._cache_warmth_caveat(patch_results)
        self.assertIn("5,000", caveat)
        self.assertIn("5,200", caveat)
        self.assertNotIn("16,", caveat)
        self.assertNotIn("43,", caveat)


class CacheSensitiveMetricsTests(unittest.TestCase):
    def test_classifies_every_efficiency_metric(self):
        # Full coverage now (team ruling, backed by real correlation data
        # for cost_usd/wall_ms): no metric should read as "unclassified"
        # by simple absence anymore.
        for name, _lower_is_better in EFFICIENCY_METRICS:
            self.assertIn(name, cli.CACHE_SENSITIVE_METRICS, f"{name} has no entry")

    def test_empirically_verified_entries(self):
        # corr(cache_read_input_tokens, cost_usd) = -0.744, n=50: strong,
        # correctly signed -> True.
        self.assertTrue(cli.CACHE_SENSITIVE_METRICS["cost_usd"])
        # corr(cache_read_input_tokens, wall_s) = +0.201, n=50: weak and
        # the wrong sign -> False, despite the plausible-sounding intuition.
        self.assertFalse(cli.CACHE_SENSITIVE_METRICS["wall_ms"])

    def test_classified_by_construction_entries(self):
        self.assertTrue(cli.CACHE_SENSITIVE_METRICS["total_tokens"])
        for name in ("skill_tokens_est", "skill_wasted_tokens_est",
                    "tool_result_tokens_est", "mcp_startup_ms",
                    "tool_calls", "turns", "duplicate_calls"):
            self.assertFalse(cli.CACHE_SENSITIVE_METRICS[name])


def _patch_for_verdict(kind="skill.truncate", target="strict-audit"):
    return {"kind": kind, "target": target}


class VerdictLineTests(unittest.TestCase):
    """The exact bug: a k=3 patch that measured a real improvement (47%
    token reduction, judged tie, zero regressions) printed as "REJECTED
    (trends favorable)" because REJECTED was the default for anything
    neither accepted nor provisional, regardless of why."""

    def test_accepted(self):
        gate_result = {"accepted": True, "provisional": False,
                       "directional": "favorable", "reasons": ["all clear"]}
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("ACCEPTED:"))

    def test_provisional(self):
        gate_result = {"accepted": False, "provisional": True,
                       "directional": "favorable", "reasons": ["provisional pass"]}
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("PROVISIONAL:"))

    def test_insufficient_samples_favorable_is_directional_not_rejected(self):
        # The reported bug, verbatim shape: k=3 -> n=3 judged pairs -> n <
        # MIN_SAMPLES_FOR_ACCEPT (5) -> stats.gate() cannot call it either
        # way, but the evidence trends favorable.
        gate_result = {
            "accepted": False, "provisional": False, "directional": "favorable",
            "reasons": ["insufficient_samples_for_acceptance: only 3 judged "
                       "pair(s) -- k<5 cannot support an accept decision"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("DIRECTIONAL: FAVORABLE:"))
        self.assertNotIn("REJECTED", line)

    def test_insufficient_samples_unfavorable_is_directional_not_rejected(self):
        gate_result = {
            "accepted": False, "provisional": False, "directional": "unfavorable",
            "reasons": ["insufficient_samples_for_acceptance: only 3 judged pair(s)"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("DIRECTIONAL: UNFAVORABLE:"))
        self.assertNotIn("REJECTED", line)

    def test_real_rejection_with_enough_samples_is_rejected(self):
        # n >= MIN_SAMPLES_FOR_ACCEPT, an actual quality regression -- no
        # "insufficient_samples_for_acceptance" reason at all, so this IS
        # the state REJECTED should be reserved for.
        gate_result = {
            "accepted": False, "provisional": False, "directional": "unfavorable",
            "reasons": ["quality regression: 4 loss(es), win-rate lower CI "
                       "0.1 does not clear 0.5"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("REJECTED:"))

    def test_high_severity_regression_with_enough_samples_is_rejected(self):
        gate_result = {
            "accepted": False, "provisional": False, "directional": "unfavorable",
            "reasons": ["1 high-severity regression(s): mcp server X crashed"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("REJECTED:"))

    def test_unknown_directional_never_renders_as_neutral_or_rejected(self):
        # Zero usable pairs -- every run failed on at least one arm.
        gate_result = {
            "accepted": False, "provisional": False, "directional": "unknown",
            "reasons": ["no successful paired runs: 0 of 3 runs produced usable data"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("DIRECTIONAL: UNKNOWN:"))
        self.assertNotIn("NEUTRAL", line)
        self.assertNotIn("REJECTED", line)

    def test_not_applied_patch_renders_as_directional_unknown(self):
        # The exact shape run_review hand-builds when harness.apply skips a
        # patch (lint failure, out-of-range lines, ...): nothing was ever
        # measured, which must read the same as the zero-usable-pairs case
        # above, not as a real "neutral" (measured-and-even) reading.
        gate_result = {
            "accepted": False, "provisional": False, "directional": "unknown",
            "reasons": ["patch not applied: skill not found: strict-audit"],
        }
        line = cli._verdict_line(_patch_for_verdict(), gate_result)
        self.assertTrue(line.startswith("DIRECTIONAL: UNKNOWN:"))


class AllDirectionalForSampleSizeTests(unittest.TestCase):
    def _pr(self, accepted=False, provisional=False, insufficient=True):
        reasons = (["insufficient_samples_for_acceptance: only 3 judged pair(s)"]
                   if insufficient else ["quality regression: 4 loss(es)"])
        return {"gate": {"accepted": accepted, "provisional": provisional,
                        "reasons": reasons}}

    def test_true_when_every_patch_is_sample_size_blocked(self):
        prs = [self._pr(), self._pr()]
        self.assertTrue(cli._all_directional_for_sample_size(prs))

    def test_false_when_any_patch_accepted(self):
        prs = [self._pr(), self._pr(accepted=True, insufficient=False)]
        self.assertFalse(cli._all_directional_for_sample_size(prs))

    def test_false_when_any_patch_provisional(self):
        prs = [self._pr(), self._pr(provisional=True, insufficient=False)]
        self.assertFalse(cli._all_directional_for_sample_size(prs))

    def test_false_when_a_patch_is_a_real_rejection(self):
        prs = [self._pr(), self._pr(insufficient=False)]
        self.assertFalse(cli._all_directional_for_sample_size(prs))

    def test_false_when_empty(self):
        self.assertFalse(cli._all_directional_for_sample_size([]))


class CostEstimateTests(unittest.TestCase):
    def test_uses_real_manifest_numbers(self):
        estimate = cli._cost_estimate(
            [{"patch_id": "patch-001"}], k=3, use_cached_analysis=True,
            repo_root=cli.REPO_ROOT)
        self.assertEqual(estimate["planned_replay_runs"], 6)
        self.assertIsNotNone(estimate["mean_session_cost_usd"])
        self.assertGreater(estimate["mean_session_cost_sample_n"], 0)
        self.assertEqual(estimate["analyzer_cost_estimate_usd"], 0.0)
        self.assertIsNotNone(estimate["estimated_total_usd"])
        self.assertIn("mean cost_usd observed across", estimate["note"])

    def test_no_patches_means_zero_runs(self):
        estimate = cli._cost_estimate([], k=3, use_cached_analysis=True,
                                      repo_root=cli.REPO_ROOT)
        self.assertEqual(estimate["planned_replay_runs"], 0)
        self.assertEqual(estimate["estimated_total_usd"], 0.0)


class CheckBudgetTests(unittest.TestCase):
    def test_none_budget_never_refuses(self):
        cli._check_budget({"estimated_total_usd": 1000.0}, None)  # must not raise

    def test_over_budget_refuses(self):
        with self.assertRaises(SystemExit):
            cli._check_budget({"estimated_total_usd": 1.0}, 0.0)

    def test_under_budget_passes(self):
        cli._check_budget({"estimated_total_usd": 0.5}, 1.0)  # must not raise

    def test_missing_estimate_refuses(self):
        with self.assertRaises(SystemExit):
            cli._check_budget({"estimated_total_usd": None}, 5.0)


class CheckWorkspaceOutCollisionTests(unittest.TestCase):
    """final-review.md C1: --out defaulting inside the replayed workspace
    (the documented default, runs/<ts>, IS inside the repo when the repo
    itself is the workspace -- the headline use case) makes
    isolated_workdir copy the workspace into itself. Must refuse before
    the preflight call, not discover it as a misdiagnosed auth failure."""

    def test_none_workspace_never_refuses(self):
        cli._check_workspace_out_collision(None, Path("/tmp/out"))  # must not raise

    def test_out_inside_workspace_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            out_dir = workspace / "runs" / "20260101T000000Z"
            with self.assertRaises(SystemExit) as ctx:
                cli._check_workspace_out_collision(workspace, out_dir)
            message = str(ctx.exception)
            self.assertIn(str(workspace.resolve()), message)
            self.assertIn(str(out_dir.resolve()), message)

    def test_workspace_inside_out_refuses(self):
        # The reverse nesting -- out_dir is an ancestor of workspace --
        # would have the exact same self-copy problem the other direction.
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            workspace = out_dir / "checkout" / "repo"
            with self.assertRaises(SystemExit):
                cli._check_workspace_out_collision(workspace, out_dir)

    def test_identical_paths_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            with self.assertRaises(SystemExit):
                cli._check_workspace_out_collision(path, path)

    def test_sibling_directories_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "repo"
            out_dir = root / "runs"
            workspace.mkdir()
            cli._check_workspace_out_collision(workspace, out_dir)  # must not raise

    def test_unrelated_paths_pass(self):
        cli._check_workspace_out_collision(Path("/tmp/a/workspace"), Path("/tmp/b/out"))


class PreflightTests(unittest.TestCase):
    """`_run_preflight` in isolation -- no `run_review` involved."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.out_dir = root / "out"
        self.auth_home = root / "auth_home"
        self.auth_home.mkdir()
        (self.auth_home / "auth.json").write_text('{"token": "fake"}')

    def _version(self, *, skills=None, **kwargs):
        return make_harness_version(
            version_id="v1", skills=skills or [], mcp_servers=[],
            config_text="", **kwargs)

    def test_success_does_not_raise(self):
        cli._run_preflight(self._version(), out_dir=self.out_dir,
                           auth_from=self.auth_home, model=None, timeout=30,
                           runner=_fake_runner)  # must not raise

    def test_failure_raises_with_verbatim_error(self):
        def failing_runner(argv, **kwargs):
            return _FakeCompletedProcess("not json", returncode=1,
                                         stderr="Error: Not signed in.")

        with self.assertRaises(SystemExit) as ctx:
            cli._run_preflight(self._version(), out_dir=self.out_dir,
                               auth_from=self.auth_home, model=None, timeout=30,
                               runner=failing_runner)
        message = str(ctx.exception)
        self.assertIn("preflight check failed", message)
        self.assertIn("Not signed in.", message)
        self.assertIn("--no-preflight", message)

    def test_runner_exception_raises_systemexit_too(self):
        def exploding_runner(argv, **kwargs):
            raise OSError("no such file: grok-dev")

        with self.assertRaises(SystemExit) as ctx:
            cli._run_preflight(self._version(), out_dir=self.out_dir,
                               auth_from=self.auth_home, model=None, timeout=30,
                               runner=exploding_runner)
        self.assertIn("no such file: grok-dev", str(ctx.exception))

    def test_does_not_leak_a_warning_into_the_real_version(self):
        # A project-scope skill with no project_dest triggers a
        # "not materialized" warning inside harness.materialize -- that
        # warning is real for the actual replay (which DOES pass
        # project_dest) but meaningless for this throwaway connectivity
        # probe, and must not contaminate the version everything else reads.
        version = self._version(
            skills=[make_skill("proj-skill", path="p", content="---\n---\nx",
                               scope="project")])
        original_warnings = list(version["warnings"])
        cli._run_preflight(version, out_dir=self.out_dir, auth_from=self.auth_home,
                           model=None, timeout=30, runner=_fake_runner)
        self.assertEqual(version["warnings"], original_warnings)


# --------------------------------------------------------------------------
# --dry-run: real corpus trace, cached analysis, no spawning
# --------------------------------------------------------------------------

class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CORPUS_TRACE.is_file(), "fixture trace missing from corpus")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name) / "out"
        # A minimal, hermetic harness source -- snapshot() tolerates a
        # missing skills/ dir and an empty config.toml just fine.
        self.grok_home = Path(self._tmp.name) / "grok_home"
        self.grok_home.mkdir()
        (self.grok_home / "config.toml").write_text("")

    def test_dry_run_produces_patches_and_cost_estimate_without_spawning(self):
        stdout = io.StringIO()
        with mock.patch("subprocess.run", side_effect=_refuse_to_spawn):
            with contextlib.redirect_stdout(stdout):
                plan = cli.run_review(
                    CORPUS_TRACE, out_dir=self.out_dir, k=3, max_patches=2,
                    dry_run=True, use_cached_analysis=True,
                    grok_home_override=self.grok_home,
                )

        self.assertIn("selected_patches", plan)
        self.assertGreater(len(plan["selected_patches"]), 0)
        self.assertLessEqual(len(plan["selected_patches"]), 2)
        for p in plan["selected_patches"]:
            self.assertEqual(p["planned_runs"], 6)  # 2 * k
            self.assertIn("impact_score", p)
            self.assertIn("impact_unit", p)
        self.assertIsNotNone(plan["cost_estimate"]["estimated_total_usd"])
        self.assertGreater(plan["cost_estimate"]["estimated_total_usd"], 0)

        # side effects: plan.json + progress.json written under --out, and
        # nothing else (no report, since dry-run never runs the matrix).
        self.assertTrue((self.out_dir / "plan.json").is_file())
        self.assertTrue((self.out_dir / "progress.json").is_file())
        self.assertFalse((self.out_dir / "report.json").exists())

        progress = json.loads((self.out_dir / "progress.json").read_text())
        self.assertEqual(progress["status"], "dry_run_complete")

        # --dry-run must not run silently: the plan has to be readable on
        # stdout before/without ever spending anything.
        printed = stdout.getvalue()
        self.assertNotEqual(printed, "")
        self.assertIn(plan["prompt_source"], printed)
        self.assertIn(plan["baseline"]["version_id"], printed)
        for p in plan["selected_patches"]:
            self.assertIn(p["patch_id"], printed)
        self.assertIn("estimated cost", printed)
        self.assertIn(str(self.out_dir / "plan.json"), printed)

    def test_dry_run_skips_skill_insert_synthesis_call(self):
        # Confirms patches.from_recommendations gets a no-spawn grok stub in
        # dry-run: a fabricated skill_add_lines rec must not raise or spawn.
        recs = [{"type": "skill_add_lines", "target": "nonexistent-skill",
                "action": "add missing guidance", "evidence": "e", "confidence": "low"}]

        def fake_digest(trace):
            return {}

        def fake_analyze(dig, model=None):
            return {"session_verdict": "minor_waste", "recommendations": recs}, {"cost_usd": 0.01}

        with mock.patch("subprocess.run", side_effect=_refuse_to_spawn):
            plan = cli.run_review(
                CORPUS_TRACE, out_dir=self.out_dir, k=1, max_patches=3,
                dry_run=True, grok_home_override=self.grok_home,
                digest_fn=fake_digest, analyze_fn=fake_analyze,
            )
        # skill not found in the (skill-less) fixture harness -> patch is
        # correctly reported as not appliable, never silently dropped.
        self.assertEqual(plan["selected_patches"], [])
        reasons = [d["skipped_reason"] for d in plan["dropped_patches"]]
        self.assertTrue(any("not found" in r for r in reasons))

    def test_budget_usd_zero_refuses(self):
        with mock.patch("subprocess.run", side_effect=_refuse_to_spawn):
            with self.assertRaises(SystemExit) as ctx:
                cli.run_review(
                    CORPUS_TRACE, out_dir=self.out_dir, k=3, max_patches=2,
                    dry_run=True, use_cached_analysis=True, budget_usd=0.0,
                    grok_home_override=self.grok_home,
                )
        self.assertIn("exceeds --budget-usd", str(ctx.exception))
        # the plan must have been computed (and journaled) before refusal
        progress = json.loads((self.out_dir / "progress.json").read_text())
        self.assertEqual(progress["status"], "planned")
        self.assertIn("cost_estimate", progress)


# --------------------------------------------------------------------------
# progress journaling across a full (fully-faked) cycle
# --------------------------------------------------------------------------

class _FakeCompletedProcess:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _fake_runner(argv, **kwargs):
    # replay.run_once only counts a run as `ok` when a NEW
    # <home>/sessions/*/<session-id>/ directory shows up after the call (it
    # diffs the directory set before/after) -- a real grok-dev process
    # creates that as a side effect of running; this fake must too, or
    # every fake run reads as a failure once anything (like the all-failed
    # exit check) actually looks at `ok`. An empty/missing events.jsonl
    # inside it is fine: grok_normalize.normalize_session returns None for
    # that cleanly, without raising, so `trace` stays None but `ok` stays
    # True -- exactly the "ok run, nothing to grade off the trace" case
    # grade.efficiency() already treats as all-None metrics rather than a
    # crash.
    env = kwargs.get("env") or {}
    home = env.get("GROK_HOME")
    session_id = "fake-session"
    if home:
        (Path(home) / "sessions" / "fake-cwd" / session_id).mkdir(
            parents=True, exist_ok=True)
    return _FakeCompletedProcess(json.dumps({
        "sessionId": session_id,
        "text": "fake response text",
        "total_cost_usd": 0.002,
        "num_turns": 1,
        "usage": {"total_tokens": 10},
    }))


def _fake_judge_grok(prompt_text, schema, model=None):
    return {"winner": "tie", "reason": "fake judge"}


def _fake_stats_summarize(grades, *, primary_effect, seed=0):
    return {"win_rate": {"wins": 0, "ties": len(grades), "losses": 0, "n": len(grades),
                        "rate": None, "lo": None, "hi": None, "note": None},
            "regressions": [], "efficiency": {}, "primary_metric": primary_effect}


def _fake_stats_gate(summary, *, primary_effect=None, max_token_regression=0.10):
    return {"accepted": False, "provisional": True, "directional": "neutral",
            "reasons": ["fake gate: not enough signal"]}


def _sample_size_blocked_stats_gate(summary, *, primary_effect=None, max_token_regression=0.10):
    """The real reported bug's exact gate shape: k=3 -> n=3 judged pairs,
    below stats.MIN_SAMPLES_FOR_ACCEPT (5), so accepted/provisional are
    both False purely on sample size -- with a favorable trend, per the
    real cycle that surfaced this (1458 -> 774 skill tokens, judged tie,
    zero regressions)."""
    return {"accepted": False, "provisional": False, "directional": "favorable",
            "reasons": ["insufficient_samples_for_acceptance: only 3 judged "
                       "pair(s) -- k<5 cannot support an accept decision at "
                       "any confidence"]}


class ProgressJournalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self.grok_home = root / "grok_home"
        self.grok_home.mkdir()
        (self.grok_home / "config.toml").write_text("")

        self.auth_home = root / "auth_home"
        self.auth_home.mkdir()
        (self.auth_home / "auth.json").write_text('{"token": "fake"}')

        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "app.py").write_text("print('hi')\n")

        self.trace_path = root / "trace.json"
        self.trace_path.write_text(json.dumps({
            "ground_truth": {"prompt": "Do the thing.", "cost_usd": 0.05, "turns": 3},
            "session": {"session_id": "sess-1", "title": "test session",
                       "cwd": str(self.workspace)},
        }))

        self.out_dir = root / "out"

        self.recs = [
            {"type": "tool_redundant", "target": "grep",
             "action": "Reuse the previous grep result instead of re-running it.",
             "evidence": "duplicate_calls: 2", "confidence": "medium"},
            {"type": "permission_friction", "target": "n/a",
             "action": "Batch related file edits into one turn to reduce prompts.",
             "evidence": "permission_wait_ms: 4000", "confidence": "low"},
        ]

    def _run(self, **overrides):
        kwargs = dict(
            out_dir=self.out_dir, k=1, max_patches=5, dry_run=False,
            grok_home_override=self.grok_home, auth_from=self.auth_home,
            digest_fn=lambda trace: {}, analyze_fn=lambda dig, model=None: (
                {"session_verdict": "minor_waste", "recommendations": self.recs}, {}),
            runner=_fake_runner, judge_grok=_fake_judge_grok,
            stats_summarize=_fake_stats_summarize, stats_gate=_fake_stats_gate,
            now=_FixedNow(1_700_000_000),
        )
        kwargs.update(overrides)
        return cli.run_review(self.trace_path, **kwargs)

    def test_prints_a_summary_line_per_patch_and_the_report_path(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rpt = self._run()
        printed = stdout.getvalue()
        self.assertNotEqual(printed, "")
        for pr in rpt["patch_results"]:
            self.assertIn(pr["patch"]["patch_id"], printed)
            self.assertIn(pr["verdict_line"], printed)
        self.assertIn(str(self.out_dir / "index.html"), printed)
        # provisional patches (the default fake gate) are not the
        # sample-size-blocked case -- no --k suggestion should print.
        self.assertNotIn("directional-only", printed)

    def test_sample_size_blocked_favorable_patch_prints_directional_not_rejected(self):
        # End-to-end reproduction of the real report: a favorable,
        # sample-size-blocked gate result must never print as REJECTED,
        # and the cycle summary must point the user at --k 10.
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rpt = self._run(stats_gate=_sample_size_blocked_stats_gate)
        printed = stdout.getvalue()

        for pr in rpt["patch_results"]:
            self.assertTrue(pr["verdict_line"].startswith("DIRECTIONAL: FAVORABLE:"))
        self.assertNotIn("REJECTED", printed)
        self.assertIn("DIRECTIONAL: FAVORABLE", printed)
        self.assertIn(f"--k {cli.stats.MIN_SAMPLES_FOR_CI}", printed)
        self.assertIn("directional-only", printed)

    def test_report_carries_cache_warmth_caveat_and_metric_flags(self):
        # Real observed cache_read_input_tokens swing, varying per call
        # (like the reported live cycle) rather than a fixed number.
        # Exactly 4 values for exactly the 4 replay calls this fixture's
        # k=1, 2-selected-patch cycle makes (preflight disabled so its own
        # probe call -- not part of any patch's matrix -- can't shift the
        # range this test asserts on).
        cache_reads = [16768, 41600, 27648, 41728]
        call = {"n": 0}

        def varying_cache_runner(argv, **kwargs):
            env = kwargs.get("env") or {}
            home = env.get("GROK_HOME")
            if home:
                (Path(home) / "sessions" / "fake-cwd" / "fake-session").mkdir(
                    parents=True, exist_ok=True)
            reads = cache_reads[call["n"] % len(cache_reads)]
            call["n"] += 1
            return _FakeCompletedProcess(json.dumps({
                "sessionId": "fake-session",
                "text": "fake response text",
                "total_cost_usd": 0.05,
                "num_turns": 1,
                "usage": {"total_tokens": 47000, "cache_read_input_tokens": reads},
            }))

        rpt = self._run(runner=varying_cache_runner, no_preflight=True)

        self.assertEqual(rpt["cache_sensitive_metrics"], cli.CACHE_SENSITIVE_METRICS)
        caveats_text = " ".join(rpt["caveats"])
        self.assertIn("prompt-cache warmth", caveats_text)
        self.assertIn(f"{min(cache_reads):,}", caveats_text)
        self.assertIn(f"{max(cache_reads):,}", caveats_text)

    def test_no_patches_report_carries_metric_flags_but_no_cache_caveat(self):
        # Nothing was measured -- the classification metadata is still a
        # constant fact worth attaching, but there is no real range to
        # disclose, so the caveat itself must not appear (never a
        # fabricated generic warning).
        rpt = self._run(analyze_fn=lambda dig, model=None: (
            {"session_verdict": "clean", "recommendations": []}, {}))
        self.assertEqual(rpt["patch_results"], [])
        self.assertEqual(rpt["cache_sensitive_metrics"], cli.CACHE_SENSITIVE_METRICS)
        caveats_text = " ".join(rpt["caveats"])
        self.assertNotIn("prompt-cache warmth", caveats_text)

    def test_journals_progress_after_each_patch(self):
        writes = []
        real_write_json = cli._write_json

        def spy(path, data):
            if Path(path).name == "progress.json":
                writes.append(json.loads(json.dumps(data, default=str)))
            real_write_json(path, data)

        with mock.patch.object(cli, "_write_json", side_effect=spy):
            rpt = self._run()

        self.assertEqual(len(rpt["patch_results"]), 2)

        completed_lengths = [len(w.get("completed_patches", [])) for w in writes
                             if "completed_patches" in w]
        # journaled at least once per patch, growing monotonically to 2
        self.assertIn(1, completed_lengths)
        self.assertIn(2, completed_lengths)
        self.assertEqual(completed_lengths, sorted(completed_lengths))

        final = writes[-1]
        self.assertEqual(final["status"], "done")
        patch_ids_seen = {c["patch_id"] for c in final["completed_patches"]}
        selected_ids = {pr["patch"]["patch_id"] for pr in rpt["patch_results"]}
        self.assertEqual(patch_ids_seen, selected_ids)

    def test_progress_file_survives_and_matches_final_state(self):
        self._run()
        progress = json.loads((self.out_dir / "progress.json").read_text())
        self.assertEqual(progress["status"], "done")
        self.assertEqual(len(progress["completed_patches"]), 2)
        self.assertTrue((self.out_dir / "report.json").is_file())
        self.assertTrue((self.out_dir / "index.html").is_file())

    def test_report_never_fabricates_baseline_from_original_session(self):
        rpt = self._run()
        caveats_text = " ".join(rpt["caveats"])
        self.assertIn("NOT the statistical control", caveats_text)
        self.assertIn("$0.05", caveats_text)

    def test_writes_only_inside_out_dir(self):
        # scratch/materialized homes (one per run, per replay.run_matrix)
        # and per-patch matrices must all live under --out.
        self._run()
        scratch = self.out_dir / "scratch"
        self.assertTrue(scratch.is_dir())
        control_homes = list(scratch.glob("*/runs/control-0-home/config.toml"))
        treatment_homes = list(scratch.glob("*/runs/treatment-0-home/config.toml"))
        self.assertEqual(len(control_homes), 2)  # one per selected patch
        self.assertEqual(len(treatment_homes), 2)
        patches_dir = self.out_dir / "patches"
        self.assertTrue(patches_dir.is_dir())
        matrix_files = list(patches_dir.glob("*/matrix.json"))
        self.assertEqual(len(matrix_files), 2)

    # -- preflight, fail-fast, and the all-failed exit code --------------

    def test_out_inside_workspace_refuses_before_preflight_spawns_anything(self):
        # End-to-end reproduction of C1: out_dir defaulting/landing inside
        # the trace's recorded workspace must refuse at plan time, before
        # the preflight call -- not spend anything on a cycle that would
        # copy the workspace into itself.
        with self.assertRaises(SystemExit) as ctx:
            with mock.patch.object(cli, "_run_preflight", side_effect=AssertionError(
                    "preflight must never be reached when out_dir is nested "
                    "inside the workspace")):
                self._run(out_dir=self.workspace / "runs" / "cycle")
        message = str(ctx.exception)
        self.assertIn(str(self.workspace.resolve()), message)
        self.assertFalse((self.out_dir / "progress.json").exists())

    def test_preflight_cost_is_folded_into_reported_cycle_cost(self):
        def costed_runner(argv, **kwargs):
            env = kwargs.get("env") or {}
            home = env.get("GROK_HOME", "")
            if "preflight-home" in home:
                return _FakeCompletedProcess(json.dumps({
                    "sessionId": "preflight", "text": "OK",
                    "total_cost_usd": 0.0111,
                }))
            return _fake_runner(argv, **kwargs)

        rpt = self._run(runner=costed_runner)
        # 2 patches x 2 arms x k=1 x $0.002 (per _fake_runner) + one
        # preflight call at $0.0111 -- the preflight's own real cost must
        # show up in the total, not vanish (final-review.md I1).
        self.assertAlmostEqual(rpt["cost_usd"], 4 * 0.002 + 0.0111, places=6)

    def test_report_names_what_the_cost_figure_excludes(self):
        rpt = self._run()
        caveats_text = " ".join(rpt["caveats"])
        self.assertIn("does NOT include judge", caveats_text)

    def test_dropped_patches_appear_in_the_report_not_just_progress_json(self):
        # final-review.md I4: report.build(dropped_patches=) is the
        # purpose-built, non-truncating surface -- must actually be wired
        # up, not left dead while cli.py rolls its own truncated summary.
        recs = list(self.recs) + [
            {"type": "skill_add_lines", "target": "nonexistent-skill",
             "action": "add missing guidance", "evidence": "e", "confidence": "low"},
        ]
        rpt = self._run(
            max_patches=1,  # forces a real budget cut too, not just the lint one
            analyze_fn=lambda dig, model=None: (
                {"session_verdict": "minor_waste", "recommendations": recs}, {}))
        caveats_text = " ".join(rpt["caveats"])
        self.assertIn("Considered but never run", caveats_text)
        # the lint/not-found rejection (skill_add_lines on a skill that
        # doesn't exist in this fixture's skill-less harness) -- rejected
        # before ranking, by patches.py itself, never spawning grok-dev.
        self.assertIn("not found in harness version", caveats_text)
        # a genuine budget cut (3 recs in, max_patches=1)
        self.assertIn("budget: ranked", caveats_text)

    def test_preflight_runs_before_the_loop_and_aborts_on_failure(self):
        def always_fail(argv, **kwargs):
            return _FakeCompletedProcess("not json", returncode=1,
                                         stderr="Error: Not signed in.")

        with self.assertRaises(SystemExit) as ctx:
            self._run(runner=always_fail)  # no_preflight defaults to False
        self.assertIn("preflight check failed", str(ctx.exception))

        # aborted before the loop: progress never left the planning stage
        progress = json.loads((self.out_dir / "progress.json").read_text())
        self.assertEqual(progress["status"], "planned")
        self.assertNotIn("completed_patches", progress)
        self.assertFalse((self.out_dir / "report.json").exists())

    def test_relative_out_dir_still_produces_absolute_grok_home_for_the_runner(self):
        # The exact regression: a relative out_dir used to reach the runner
        # as a relative GROK_HOME (grok resolves that against ITS OWN cwd,
        # not this process's, and reports "not signed in" for what is
        # actually a path bug). Drives run_review directly with a relative
        # out_dir string, from a chdir'd cwd, and checks every GROK_HOME
        # the runner actually received -- preflight's call included.
        seen_homes = []

        def recording_runner(argv, **kwargs):
            seen_homes.append((kwargs.get("env") or {}).get("GROK_HOME"))
            return _fake_runner(argv, **kwargs)

        cwd = os.getcwd()
        os.chdir(self.out_dir.parent)  # self.out_dir doesn't exist yet
        try:
            self._run(out_dir=Path(self.out_dir.name), runner=recording_runner)
        finally:
            os.chdir(cwd)

        self.assertGreater(len(seen_homes), 0)
        for home in seen_homes:
            self.assertIsNotNone(home)
            self.assertTrue(Path(home).is_absolute(),
                            f"GROK_HOME {home!r} handed to the runner was relative")

    def test_no_preflight_flag_skips_the_check(self):
        # Fails only when routed through the preflight-only materialized
        # home; a real per-patch replay call is unaffected.
        def fail_only_preflight(argv, **kwargs):
            home = (kwargs.get("env") or {}).get("GROK_HOME", "")
            if "preflight-home" in home:
                return _FakeCompletedProcess("not json", returncode=1, stderr="boom")
            return _fake_runner(argv, **kwargs)

        with self.assertRaises(SystemExit):
            self._run(runner=fail_only_preflight)  # preflight on: aborts
        rpt = self._run(runner=fail_only_preflight, no_preflight=True)  # skipped: succeeds
        self.assertEqual(len(rpt["patch_results"]), 2)

    def test_fail_fast_stops_after_first_pair_and_exits_nonzero_when_all_fail(self):
        calls = []

        def always_fail(argv, **kwargs):
            calls.append(argv)
            return _FakeCompletedProcess("not json", returncode=1,
                                         stderr="Error: Not signed in.")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                self._run(k=3, max_parallel=1, runner=always_fail, no_preflight=True)
        message = str(ctx.exception)
        self.assertIn("all", message)
        self.assertIn("failed this cycle", message)

        # fail-fast: 2 calls (one pair) per patch, not 2*k=6 -- 2 patches total
        self.assertEqual(len(calls), 4)

        progress = json.loads((self.out_dir / "progress.json").read_text())
        self.assertEqual(progress["status"], "failed_no_data")
        # the report is still written for debugging, despite the raise
        self.assertTrue((self.out_dir / "index.html").is_file())

        printed = stdout.getvalue()
        self.assertIn("stopped after the first paired run failed on both arms", printed)

    def test_partial_failure_is_reported_as_run_health(self):
        # Only the very first grok-dev call (patch-001's control repeat 0,
        # deterministic under max_parallel=1) fails; everything else
        # succeeds. Not a both-arms failure, so no fail-fast: all k=3
        # repeats run, and the shortfall must show up as 2/3, not silently
        # round down to a clean 1/1 or get skipped.
        state = {"n": 0}

        def fail_first_call_only(argv, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                return _FakeCompletedProcess("not json", returncode=1, stderr="boom")
            return _fake_runner(argv, **kwargs)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rpt = self._run(k=3, max_parallel=1, runner=fail_first_call_only,
                            no_preflight=True)

        first = rpt["patch_results"][0]
        control_runs = first["matrix"]["arms"]["control"]
        treatment_runs = first["matrix"]["arms"]["treatment"]
        self.assertEqual(len(control_runs), 3)
        self.assertEqual(len(treatment_runs), 3)
        self.assertEqual(sum(1 for r in control_runs if r["ok"]), 2)
        self.assertEqual(sum(1 for r in treatment_runs if r["ok"]), 3)

        printed = stdout.getvalue()
        self.assertIn("control 2/3 ok, treatment 3/3 ok", printed)

        progress = json.loads((self.out_dir / "progress.json").read_text())
        first_completed = progress["completed_patches"][0]
        self.assertEqual(first_completed["control_ok"], 2)
        self.assertEqual(first_completed["treatment_ok"], 3)
        self.assertIsNone(first_completed["fail_fast_reason"])


if __name__ == "__main__":
    unittest.main()
