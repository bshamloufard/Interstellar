"""Tests for analyze()'s structuredOutput-first precedence (see
interstellar/patches.py's identical fix for the rationale). No test calls
grok-dev: subprocess.run is patched to return a canned CLI envelope.

    python3 -m unittest analyzer.test_analyze
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import analyze


def _proc(stdout_obj):
    proc = MagicMock()
    proc.stdout = json.dumps(stdout_obj)
    proc.stderr = ""
    return proc


class AnalyzePrecedenceTests(unittest.TestCase):
    def test_reads_structured_output_over_text(self):
        out = {
            "text": json.dumps({"session_verdict": "clean", "recommendations": []}),
            "structuredOutput": {"session_verdict": "significant_waste",
                                  "recommendations": [{"type": "skill_keep",
                                                        "target": "x", "action": "a",
                                                        "evidence": "e", "confidence": "high"}]},
            "structuredOutputError": None,
            "total_cost_usd": 0.02,
            "usage": {"input_tokens": 100},
        }
        with patch("analyze.subprocess.run", return_value=_proc(out)):
            recs, meta = analyze.analyze({"session": {}})
        self.assertEqual(recs["session_verdict"], "significant_waste")
        self.assertEqual(meta["source"], "structuredOutput")
        self.assertEqual(meta["cost_usd"], 0.02)

    def test_structured_output_error_returns_none_even_if_text_parses(self):
        out = {
            "text": json.dumps({"session_verdict": "clean", "recommendations": []}),
            "structuredOutput": None,
            "structuredOutputError": "schema validation failed",
        }
        with patch("analyze.subprocess.run", return_value=_proc(out)):
            recs, meta = analyze.analyze({"session": {}})
        self.assertIsNone(recs)
        self.assertEqual(meta["source"], "structuredOutput")
        self.assertIn("structuredOutputError", meta)

    def test_falls_back_to_text_when_structured_output_absent(self):
        out = {"text": json.dumps({"session_verdict": "clean", "recommendations": []})}
        with patch("analyze.subprocess.run", return_value=_proc(out)):
            recs, meta = analyze.analyze({"session": {}})
        self.assertEqual(recs["session_verdict"], "clean")
        self.assertEqual(meta["source"], "text_fallback")

    def test_unparseable_text_fallback_returns_none_with_source(self):
        out = {"text": "not json"}
        with patch("analyze.subprocess.run", return_value=_proc(out)):
            recs, meta = analyze.analyze({"session": {}})
        self.assertIsNone(recs)
        self.assertEqual(meta["source"], "text_fallback")
        self.assertEqual(meta["error"], "model output not JSON")

    def test_unparseable_stdout_returns_none_with_source(self):
        proc = MagicMock()
        proc.stdout = "not json at all"
        proc.stderr = "stderr text"
        with patch("analyze.subprocess.run", return_value=proc):
            recs, meta = analyze.analyze({"session": {}})
        self.assertIsNone(recs)
        self.assertEqual(meta["source"], "stdout")
        self.assertEqual(meta["error"], "unparseable")


if __name__ == "__main__":
    unittest.main()
