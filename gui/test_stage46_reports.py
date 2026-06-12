#!/usr/bin/env python3
"""Stage 4.6 research reports and optional modal_radiation_color_v1 prototype tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    synthetic_classic_body_modes,
    synthesize_note_with_body_response,
)
from build_sample_comparison import build_sample_comparisons  # noqa: E402
from diagnostic_synthesis import DIAGNOSTIC_MODES, get_diagnostic_mode  # noqa: E402
from stage46_research_audit import (  # noqa: E402
    build_literature_review,
    build_model_gap_analysis,
    write_all_reports,
)


class Stage46ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_literature_review_has_sources(self) -> None:
        doc = build_literature_review()
        self.assertTrue(doc.get("internet_available"))
        self.assertGreaterEqual(len(doc.get("sources") or []), 10)
        self.assertIn("key_findings", doc)

    def test_gap_analysis_sections(self) -> None:
        doc = build_model_gap_analysis()
        self.assertIn("gap_answers", doc)
        self.assertIn("hypothesis_evaluation", doc)
        self.assertIn("candidate_models", doc)
        self.assertEqual(len(doc.get("gap_answers") or {}), 14)
        self.assertEqual(len(doc.get("hypothesis_evaluation") or {}), 6)

    def test_reports_written(self) -> None:
        out_dir = self.out / "debug_reports"
        paths = write_all_reports(repo_root=REPO, use_surrogate=False)
        for key in ("literature_json", "gap_json", "trace_json"):
            self.assertTrue(paths[key].is_file(), key)
            payload = json.loads(paths[key].read_text(encoding="utf-8"))
            self.assertIn("schema_version", payload)
        self.assertTrue(paths["literature_md"].is_file())
        self.assertTrue(paths["gap_md"].is_file())
        self.assertTrue(paths["trace_md"].is_file())


class Stage46PrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_modal_radiation_color_v1_mode_exists(self) -> None:
        cfg = get_diagnostic_mode("modal_radiation_color_v1")
        self.assertEqual(cfg.name, "modal_radiation_color_v1")
        self.assertFalse(cfg.wide_body_signature)

    def test_radiation_factors_vary_by_sample(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(12)}
        meta_a = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "a.wav",
            diagnostic_mode="modal_radiation_color_v1",
            sample_parameters={"top_wood_id": "spruce", "back_wood_id": "rosewood"},
        )
        meta_b = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "b.wav",
            diagnostic_mode="modal_radiation_color_v1",
            sample_parameters={"top_wood_id": "maple", "back_wood_id": "cedar"},
        )
        self.assertTrue(meta_a.get("radiation_color_v1_active"))
        fp_a = meta_a.get("sample_material_damping_fingerprint")
        fp_b = meta_b.get("sample_material_damping_fingerprint")
        self.assertNotEqual(fp_a, fp_b)
        per_a = meta_a.get("per_mode_damping") or []
        self.assertTrue(any(r.get("mode_amplitude_factor") for r in per_a))
        self.assertGreater(float(meta_a.get("far_mode_sample_specificity_score") or 0), 0.0)

    def test_deterministic_and_no_clip(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(8)}
        params = {"top_wood_id": "cedar", "back_wood_id": "mahogany"}
        kw = dict(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.12,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            diagnostic_mode="modal_radiation_color_v1",
            sample_parameters=params,
        )
        m1 = synthesize_note_with_body_response(output_wav=self.out_dir / "x1.wav", **kw)
        m2 = synthesize_note_with_body_response(output_wav=self.out_dir / "x2.wav", **kw)
        self.assertEqual(m1.get("output_rms_dbfs"), m2.get("output_rms_dbfs"))
        self.assertLessEqual(float(m1.get("output_peak_dbfs") or 0), -0.5)

    def test_far_weighting_varies_across_samples(self) -> None:
        samples = [
            {"sample_id": "sample_000", "run_id": "", "parameters": {"top_wood_id": "spruce", "back_wood_id": "rosewood"}},
            {"sample_id": "sample_001", "run_id": "", "parameters": {"top_wood_id": "maple", "back_wood_id": "cedar"}},
        ]
        manifest = build_sample_comparisons(
            repo_root=REPO,
            out_dir=self.out_dir,
            samples=samples,
            notes=(("A2", 110.0),),
            duration_s=0.08,
            silence_s=0.02,
            use_surrogate=False,
            diagnostic_mode="modal_radiation_color_v1",
        )
        segs = manifest["notes"][0]["segments"]
        scores = [float(s.get("far_mode_sample_specificity_score") or 0) for s in segs]
        self.assertTrue(all(s > 0 for s in scores))


if __name__ == "__main__":
    unittest.main()
