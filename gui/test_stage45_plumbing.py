#!/usr/bin/env python3
"""Stage 4.5 damping plumbing and evidence tests."""
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
from diagnostic_synthesis import active_sample_parameters, summarize_comparison_note, use_diagnostic_mode  # noqa: E402
from modal_damping import compute_per_mode_damping  # noqa: E402
from stage45_damping_dataflow import (  # noqa: E402
    build_dataflow_report,
    validate_diagnostic_evidence,
    write_dataflow_reports,
)


class Stage45PlumbingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sample_parameters_active_without_diagnostic_mode(self) -> None:
        with use_diagnostic_mode(None, sample_parameters={"top_wood_id": "maple", "back_wood_id": "cedar"}):
            params = active_sample_parameters()
        self.assertEqual(params.get("top_wood_id"), "maple")
        self.assertEqual(params.get("back_wood_id"), "cedar")

    def test_damping_q_summary_nested_in_metadata(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(12)}
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "t.wav",
            sample_parameters={"top_wood_id": "maple", "back_wood_id": "cedar"},
        )
        dqs = meta.get("damping_q_summary") or {}
        self.assertIn("mode_q_spread", dqs)
        self.assertGreater(float(dqs["mode_q_spread"]), 0.0)
        self.assertIn("material_damping_spread", dqs)
        self.assertGreater(float(dqs["material_damping_spread"]), 0.0)
        self.assertTrue(meta.get("per_mode_tau_used_in_time_decay"))

    def test_different_woods_different_q_same_geometry(self) -> None:
        mode = synthetic_classic_body_modes(1)[0]
        mode["top_share"] = 0.5
        mode["back_share"] = 0.5
        geom = {"geometry.length": 0.48, "geometry.width": 0.37}
        a = compute_per_mode_damping(
            mode, float(mode["frequency_hz"]), {**geom, "top_wood_id": "spruce", "back_wood_id": "mahogany"}
        )
        b = compute_per_mode_damping(
            mode, float(mode["frequency_hz"]), {**geom, "top_wood_id": "cedar", "back_wood_id": "rosewood"}
        )
        self.assertNotAlmostEqual(a["mode_q"], b["mode_q"], places=3)

    def test_cross_sample_spread_in_summary(self) -> None:
        samples = [
            {
                "sample_id": "sample_000",
                "run_id": "",
                "parameters": {
                    "top_wood_id": "spruce",
                    "back_wood_id": "rosewood",
                    "geometry.length": 0.48,
                    "geometry.width": 0.37,
                    "geometry.depth": 0.10,
                },
            },
            {
                "sample_id": "sample_001",
                "run_id": "",
                "parameters": {
                    "top_wood_id": "maple",
                    "back_wood_id": "cedar",
                    "geometry.length": 0.36,
                    "geometry.width": 0.28,
                    "geometry.depth": 0.11,
                },
            },
        ]
        manifest = build_sample_comparisons(
            repo_root=REPO,
            out_dir=self.out_dir,
            samples=samples,
            notes=(("A2", 110.0),),
            duration_s=0.08,
            silence_s=0.02,
            use_surrogate=False,
            diagnostic_mode="baseline_current",
        )
        segs = manifest["notes"][0]["segments"]
        summary = summarize_comparison_note(segs)
        self.assertGreater(summary["mode_q_spread_mean"], 0.0)
        self.assertGreater(summary["cross_sample_material_fingerprint_spread"], 0.0)
        self.assertGreater(summary["within_sample_mode_q_spread_mean"], 0.0)

    def test_validate_evidence_fails_zero_spread(self) -> None:
        segs = [
            {
                "mode_q_median": 40.0,
                "sample_mode_q_fingerprint": 40.0,
                "material_damping_median": 1.0,
                "sample_material_damping_fingerprint": 1.0,
                "modal_source": "m4_surrogate",
            },
            {
                "mode_q_median": 40.0,
                "sample_mode_q_fingerprint": 40.0,
                "material_damping_median": 1.0,
                "sample_material_damping_fingerprint": 1.0,
                "modal_source": "m4_surrogate",
            },
        ]
        with self.assertRaises(RuntimeError):
            validate_diagnostic_evidence(segs, require_m4=True)

    def test_validate_evidence_fails_synthetic(self) -> None:
        segs = [
            {
                "mode_q_median": 40.0,
                "sample_mode_q_fingerprint": 40.0,
                "material_damping_median": 1.0,
                "sample_material_damping_fingerprint": 1.0,
                "modal_source": "synthetic_fallback",
            },
            {
                "mode_q_median": 41.0,
                "sample_mode_q_fingerprint": 41.0,
                "material_damping_median": 1.1,
                "sample_material_damping_fingerprint": 1.1,
                "modal_source": "synthetic_fallback",
            },
        ]
        with self.assertRaises(RuntimeError):
            validate_diagnostic_evidence(segs, require_m4=True)

    def test_dataflow_report_writes(self) -> None:
        paths = write_dataflow_reports(self.out_dir, repo_root=REPO, max_samples=3, use_surrogate=False)
        doc = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertIn("pass_fail", doc)
        self.assertGreaterEqual(len(doc.get("per_sample") or []), 1)

    def test_lhs_report_cross_sample_spread_offline(self) -> None:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": p}
            for i, p in enumerate(
                [
                    {
                        "top_wood_id": "spruce",
                        "back_wood_id": "rosewood",
                        "geometry.length": 0.48,
                        "geometry.width": 0.37,
                    },
                    {
                        "top_wood_id": "maple",
                        "back_wood_id": "cedar",
                        "geometry.length": 0.40,
                        "geometry.width": 0.32,
                    },
                    {
                        "top_wood_id": "cedar",
                        "back_wood_id": "mahogany",
                        "geometry.length": 0.36,
                        "geometry.width": 0.28,
                    },
                ]
            )
        ]
        report = build_dataflow_report(repo_root=REPO, samples=samples, use_surrogate=False)
        self.assertGreater(report["cross_sample_material_damping_median_spread"], 0.0)


if __name__ == "__main__":
    unittest.main()
