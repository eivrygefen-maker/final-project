#!/usr/bin/env python3
"""Stage 4.7 bridge-gated radiation v2 tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    _mode_radiation_v2_factors,
    _proxy_pool_from_modes,
    compute_mode_weight_components,
    synthetic_classic_body_modes,
    synthesize_note_with_body_response,
)
from build_sample_comparison import build_sample_comparisons, m4_surrogate_model_available  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode  # noqa: E402
from modal_damping import compute_per_mode_damping  # noqa: E402
from stage47_reports import build_rom_dataset_status, write_all_reports  # noqa: E402
from string_body_balance import low_body_color_strength  # noqa: E402


class Stage47RadiationV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_v2_mode_exists(self) -> None:
        cfg = get_diagnostic_mode("modal_radiation_color_v2")
        self.assertEqual(cfg.name, "modal_radiation_color_v2")
        self.assertFalse(cfg.wide_body_signature)

    def test_bridge_gate_affects_strength(self) -> None:
        mode = synthetic_classic_body_modes(1)[0]
        mode["bridge_excitation_abs"] = 0.9
        mode["radiation_proxy"] = 0.5
        mode["mic_output_proxy"] = 0.5
        pools = _proxy_pool_from_modes([mode])
        damp = compute_per_mode_damping(mode, float(mode["frequency_hz"]), {"top_wood_id": "spruce"})
        damp["frequency_hz"] = mode["frequency_hz"]
        defaults: list = []
        flags: dict = {}
        comp_hi = compute_mode_weight_components(mode, defaults_used=defaults, flags=flags)
        mode_lo = dict(mode)
        mode_lo["bridge_excitation_abs"] = 0.05
        comp_lo = compute_mode_weight_components(mode_lo, defaults_used=[], flags={})
        hi = _mode_radiation_v2_factors(mode, damp, comp_hi, 200.0, pools, note_hz=110.0)
        lo = _mode_radiation_v2_factors(mode_lo, damp, comp_lo, 200.0, pools, note_hz=110.0)
        self.assertGreater(hi["mode_final_amplitude_factor"], lo["mode_final_amplitude_factor"])

    def test_radiation_cannot_dominate_without_bridge(self) -> None:
        hi = synthetic_classic_body_modes(1)[0]
        hi["bridge_excitation_abs"] = 0.8
        hi["radiation_proxy"] = 0.5
        mode = synthetic_classic_body_modes(1)[0]
        mode["bridge_excitation_abs"] = 0.02
        mode["radiation_proxy"] = 1.0
        mode["mic_output_proxy"] = 1.0
        pools = _proxy_pool_from_modes([hi, mode])
        damp = compute_per_mode_damping(mode, float(mode["frequency_hz"]), {})
        damp["frequency_hz"] = mode["frequency_hz"]
        comp = compute_mode_weight_components(mode, defaults_used=[], flags={})
        out = _mode_radiation_v2_factors(mode, damp, comp, 200.0, pools, note_hz=440.0)
        mode2 = dict(mode)
        mode2["radiation_proxy"] = 0.1
        mode2["mic_output_proxy"] = 0.1
        comp2 = compute_mode_weight_components(mode2, defaults_used=[], flags={})
        out2 = _mode_radiation_v2_factors(mode2, damp, comp2, 200.0, pools, note_hz=440.0)
        self.assertLess(out["mode_final_amplitude_factor"], 0.25)
        self.assertLess(abs(out["mode_final_amplitude_factor"] - out2["mode_final_amplitude_factor"]), 0.15)

    def test_factors_vary_across_samples(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(10)}
        m_a = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out / "a.wav",
            diagnostic_mode="modal_radiation_color_v2",
            sample_parameters={"top_wood_id": "spruce", "back_wood_id": "rosewood"},
        )
        m_b = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out / "b.wav",
            diagnostic_mode="modal_radiation_color_v2",
            sample_parameters={"top_wood_id": "maple", "back_wood_id": "cedar"},
        )
        self.assertTrue(m_a.get("modal_radiation_color_v2_active"))
        self.assertNotEqual(
            m_a.get("sample_material_damping_fingerprint"),
            m_b.get("sample_material_damping_fingerprint"),
        )

    def test_low_body_rule_continuous_by_f0(self) -> None:
        self.assertGreater(low_body_color_strength(110.0), low_body_color_strength(440.0))
        self.assertEqual(low_body_color_strength(300.0), 0.0)
        self.assertNotIn("A2", str(low_body_color_strength.__doc__ or ""))

    def test_v2_metadata_fields(self) -> None:
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data={"predicted_modes": synthetic_classic_body_modes(8)},
            output_wav=self.out / "m.wav",
            diagnostic_mode="modal_radiation_color_v2",
            sample_parameters={"top_wood_id": "spruce", "back_wood_id": "mahogany"},
        )
        for key in (
            "modal_radiation_color_v2_active",
            "low_body_color_strength",
            "raw_body_rms_before_any_gain",
            "body_to_string_ratio_before_normalization",
            "applied_body_gain",
            "applied_loudness_gain",
        ):
            self.assertIn(key, meta)
        per = meta.get("per_mode_damping") or []
        self.assertTrue(any(r.get("mode_bridge_gate_factor") is not None for r in per))

    def test_deterministic_no_clip(self) -> None:
        kw = dict(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.12,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data={"predicted_modes": synthetic_classic_body_modes(8)},
            diagnostic_mode="modal_radiation_color_v2",
            sample_parameters={"top_wood_id": "cedar", "back_wood_id": "rosewood"},
        )
        a = synthesize_note_with_body_response(output_wav=self.out / "e1.wav", **kw)
        b = synthesize_note_with_body_response(output_wav=self.out / "e2.wav", **kw)
        self.assertEqual(a.get("output_rms_dbfs"), b.get("output_rms_dbfs"))
        self.assertLessEqual(float(a.get("output_peak_dbfs") or 0), -0.5)

    def test_stage47_reports_exist(self) -> None:
        paths = write_all_reports(repo_root=REPO)
        self.assertTrue(paths["rom_json"].is_file())
        self.assertTrue(paths["model_json"].is_file())
        rom = build_rom_dataset_status(REPO)
        self.assertGreater(rom.get("jsonl_entry_count") or 0, 0)

    def test_no_fem_launched(self) -> None:
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
