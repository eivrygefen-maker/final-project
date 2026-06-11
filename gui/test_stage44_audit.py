#!/usr/bin/env python3
"""Stage 4.4 audit, material damping, and 60/40 modal mode tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from audit_synthesis_model import build_audit_report, write_audit_reports  # noqa: E402
from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    synthetic_classic_body_modes,
    synthesize_note_with_body_response,
)
from build_sample_comparison import build_sample_comparisons, parse_notes_arg  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from modal_damping import (  # noqa: E402
    WOOD_DAMPING_COEFF,
    compute_material_damping_components,
    compute_per_mode_damping,
    list_wood_damping_constants,
)


class Stage44AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_audit_report_generated(self) -> None:
        paths = write_audit_reports(self.out_dir)
        self.assertTrue(paths["json"].is_file())
        self.assertTrue(paths["markdown"].is_file())
        doc = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertIn("modal_usage", doc)
        self.assertIn("material_wood_usage", doc)

    def test_wood_constants_used(self) -> None:
        coeffs = list_wood_damping_constants()
        self.assertIn("spruce", coeffs)
        self.assertNotEqual(coeffs["cedar"], coeffs["maple"])

    def test_different_woods_different_q(self) -> None:
        mode = synthetic_classic_body_modes(1)[0]
        mode["top_share"] = 0.5
        mode["back_share"] = 0.5
        geom = {"geometry": {"top_thickness": 0.003}}
        a = compute_per_mode_damping(
            mode, float(mode["frequency_hz"]), {**geom, "top_wood_id": "spruce", "back_wood_id": "mahogany"}
        )
        b = compute_per_mode_damping(
            mode, float(mode["frequency_hz"]), {**geom, "top_wood_id": "cedar", "back_wood_id": "rosewood"}
        )
        self.assertNotAlmostEqual(a["mode_q"], b["mode_q"], places=2)

    def test_mixed_top_back_weighted_damping(self) -> None:
        mode = synthetic_classic_body_modes(1)[0]
        mode["top_share"] = 0.5
        mode["back_share"] = 0.5
        mode["air_share"] = 0.0
        params = {"top_wood_id": "maple", "back_wood_id": "cedar", "geometry": {}}
        mat = compute_material_damping_components(mode, params)
        expected = (
            mat["top_wood_damping_component"]
            + mat["back_wood_damping_component"]
            + mat["air_damping_component"]
            + mat["coupled_damping_component"]
        )
        self.assertAlmostEqual(mat["mode_material_damping"], expected, places=4)
        self.assertGreater(mat["top_wood_damping_component"], mat["back_wood_damping_component"])
        self.assertGreater(mat["top_wood_damping_component"], 0)
        self.assertGreater(mat["back_wood_damping_component"], 0)

    def test_modal_body_60_40_exists(self) -> None:
        self.assertIn("modal_body_60_40_v1", list_diagnostic_modes())
        cfg = get_diagnostic_mode("modal_body_60_40_v1")
        self.assertAlmostEqual(cfg.near_modal_energy_target, 0.60, places=2)
        self.assertAlmostEqual(cfg.far_broad_energy_target, 0.40, places=2)

    def test_60_40_higher_far_than_baseline(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(30)}
        base = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "b.wav",
            diagnostic_mode="baseline_current",
        )
        v1 = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "v1.wav",
            diagnostic_mode="modal_body_60_40_v1",
        )
        self.assertGreater(
            float(v1.get("broad_body_energy_fraction") or 0),
            float(base.get("broad_body_energy_fraction") or 0),
        )

    def test_per_mode_damping_metadata(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(8)}
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "a2.wav",
            sample_parameters={"top_wood_id": "spruce", "back_wood_id": "rosewood"},
        )
        self.assertGreater(meta["per_mode_damping_count"], 0)
        top = meta["top_contributing_modes"][0]
        self.assertIn("mode_bandwidth_hz", top)
        self.assertIn("mode_tau_s", top)

    def test_lightweight_comparison_three_modes(self) -> None:
        samples = [{"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {}} for i in range(3)]
        notes = parse_notes_arg("A2,E5")
        for mode in ("baseline_current", "modal_damping_body_signature_v1", "modal_body_60_40_v1"):
            build_sample_comparisons(
                repo_root=REPO,
                out_dir=self.out_dir / mode,
                samples=samples,
                notes=notes,
                duration_s=0.08,
                silence_s=0.02,
                use_surrogate=False,
                diagnostic_mode=mode,
            )
        self.assertTrue((self.out_dir / "modal_body_60_40_v1" / "A2_26_guitars.wav").is_file())

    def test_deterministic(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(6)}
        m1 = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.08,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "e1.wav",
            diagnostic_mode="modal_body_60_40_v1",
        )
        m2 = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.08,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "e2.wav",
            diagnostic_mode="modal_body_60_40_v1",
        )
        self.assertEqual(m1["note_reward_score"], m2["note_reward_score"])


if __name__ == "__main__":
    unittest.main()
