#!/usr/bin/env python3
"""Lightweight tests for PGSM STK parameter export (no audio, no FEM/ROM, no STK runtime)."""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_stk_parameter_export import (  # noqa: E402
    NOTE_SET,
    PYTHON_ROLE,
    RENDERER_TARGET,
    REQUIRED_RENDER_GROUPS,
    SAMPLE_SET,
    build_parameter_export,
    expected_wav_filename,
    expected_wav_paths,
    write_parameter_export,
)


class TestPgsmStkParameterExport(unittest.TestCase):
    def test_build_export_schema(self) -> None:
        doc = build_parameter_export(repo_root=REPO)
        self.assertEqual(doc["renderer"], RENDERER_TARGET)
        self.assertEqual(doc["python_role"], PYTHON_ROLE)
        self.assertEqual(len(doc["renders"]), 9)
        self.assertEqual(doc.get("expected_render_count"), 9)
        for row in doc["renders"]:
            self.assertIn(row["sample_id"], SAMPLE_SET)
            self.assertIn(row["note_name"], NOTE_SET)
            for group in REQUIRED_RENDER_GROUPS:
                self.assertIn(group, row, msg=f"missing {group} in {row['sample_id']}/{row['note_name']}")
            modes = row["body_model"]["modes"]
            self.assertGreaterEqual(len(modes), 5)
            for mode in modes:
                self.assertIn("frequency_hz", mode)
                self.assertIn("gain", mode)
                self.assertIn("tau_or_q", mode)
                self.assertIn("component", mode)

    def test_v2_physical_difference_audit(self) -> None:
        doc = build_parameter_export(repo_root=REPO, demo_version="v2")
        self.assertEqual(doc["demo_version"], "pgsm_stk_guitar_demo_v2")
        audit = doc.get("physical_difference_audit") or {}
        self.assertIn("per_sample", audit)
        self.assertIn("factor_spread", audit)
        self.assertEqual(audit.get("anchor_note"), "A2")
        for sid in SAMPLE_SET:
            row = audit["per_sample"][sid]
            self.assertIn("bridge_mobility_factor", row)
            self.assertIn("modal_frequency_hz", row)
            self.assertIn("string_body_mix", row)
        paths = expected_wav_paths(REPO, demo_version="v2")
        self.assertTrue(all("pgsm_stk_guitar_demo_v2" in str(p) for p in paths))
        for row in doc["renders"]:
            self.assertIn("pgsm_stk_guitar_demo_v2", row["output_model"]["output_wav_path"])
            self.assertFalse(row["output_model"].get("normalize_rms", True))

    def test_samples_differ_physically(self) -> None:
        doc = build_parameter_export(repo_root=REPO)
        a2 = {r["sample_id"]: r for r in doc["renders"] if r["note_name"] == "A2"}
        self.assertNotEqual(
            a2["sample_001"]["bridge_model"]["bridge_mobility"],
            a2["sample_002"]["bridge_model"]["bridge_mobility"],
        )
        self.assertNotEqual(
            a2["sample_001"]["body_model"]["soundhole_radiation_factor"],
            a2["sample_002"]["body_model"]["soundhole_radiation_factor"],
        )
        self.assertGreater(
            a2["sample_001"]["radiation_model"]["radiation_brightness"],
            a2["sample_002"]["radiation_model"]["radiation_brightness"],
        )

    def test_expected_wav_filenames(self) -> None:
        names = [expected_wav_filename(s, n) for s in SAMPLE_SET for n in NOTE_SET]
        self.assertEqual(len(names), 9)
        self.assertIn("sample_000_A2_stk_guitar.wav", names)
        self.assertIn("sample_002_E5_stk_guitar.wav", names)
        paths = expected_wav_paths(REPO)
        self.assertEqual(len(paths), 9)
        self.assertTrue(all("pgsm_stk_guitar_demo" in str(p) for p in paths))

    def test_no_python_wav_synthesis_in_module(self) -> None:
        src = (REPO / "gui" / "pgsm_stk_parameter_export.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = {"write_wav_mono", "synthesize_modal_body_response", "synthesize_plucked_string", "FileWvOut"}
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
        for bad in forbidden:
            self.assertNotIn(bad, names, msg=f"forbidden symbol {bad} in export module")

    def test_write_json_roundtrip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pgsm_stk_demo_parameters.json"
            path = write_parameter_export(out, repo_root=REPO)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["export_version"], doc_ver := loaded["export_version"])
            self.assertEqual(len(loaded["renders"]), 9)
            self.assertEqual(loaded.get("expected_render_count"), 9)
            self.assertEqual(doc_ver, "pgsm_stk_parameter_export_v1")


if __name__ == "__main__":
    unittest.main()
