#!/usr/bin/env python3
"""Stage 4.3 structural STK / per-mode damping tests (no FEM)."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    synthetic_classic_body_modes,
    synthesize_note_with_body_response,
)
from build_body_difference_diagnostics import main as build_diag_main  # noqa: E402
from build_sample_comparison import build_sample_comparisons, parse_notes_arg  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from modal_damping import compute_per_mode_damping  # noqa: E402


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    count = len(raw) // 2
    return np.asarray(struct.unpack(f"<{count}h", raw), dtype=np.float64) / 32767.0


class Stage43StructuralTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_per_mode_damping_arrays(self) -> None:
        mode = synthetic_classic_body_modes(1)[0]
        params = {"top_wood_id": "cedar", "geometry": {"top_thickness": 0.0035}}
        rec = compute_per_mode_damping(mode, float(mode["frequency_hz"]), params)
        for key in ("mode_q", "mode_damping", "mode_tau_s", "mode_bandwidth_hz", "mode_category"):
            self.assertIn(key, rec)
        self.assertGreater(rec["mode_q"], 0)
        self.assertGreater(rec["mode_bandwidth_hz"], 0)

    def test_synthesis_per_mode_damping_metadata(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(12)}
        wav = self.out_dir / "a2.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.15,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
            sample_parameters={"top_wood_id": "maple", "back_wood_id": "rosewood"},
        )
        self.assertGreaterEqual(meta["per_mode_damping_count"], 12)
        self.assertIn("mode_q_spread", meta.get("damping_q_summary") or {})
        self.assertIn("mode_bandwidth_hz_median", meta.get("damping_q_summary") or {})
        top = meta["top_contributing_modes"][0]
        self.assertIn("mode_tau_s", top)
        self.assertIn("mode_bandwidth_hz", top)

    def test_structural_mode_differs_from_baseline(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(20)}
        params = {"top_wood_id": "spruce", "geometry": {"width": 0.39, "depth": 0.105}}
        base = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.12,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "e5_base.wav",
            diagnostic_mode="baseline_current",
            sample_parameters=params,
        )
        structural = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.12,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "e5_struct.wav",
            diagnostic_mode="modal_damping_body_signature_v1",
            sample_parameters=params,
        )
        self.assertLess(
            structural["high_note_string_direct_scale_applied"],
            base.get("high_note_string_direct_scale_applied", 1.0),
        )
        self.assertGreater(
            structural.get("broad_body_energy_fraction") or 0.0,
            base.get("broad_body_energy_fraction") or 0.0,
        )

    def test_lightweight_comparison_cli(self) -> None:
        out = self.out_dir / "diag"
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {"top_wood_id": "spruce"}}
            for i in range(3)
        ]
        notes = parse_notes_arg("A2,E5")
        for mode in ("baseline_current", "modal_damping_body_signature_v1"):
            build_sample_comparisons(
                repo_root=REPO,
                out_dir=out / mode,
                samples=samples,
                notes=notes,
                duration_s=0.08,
                silence_s=0.02,
                use_surrogate=False,
                diagnostic_mode=mode,
            )
        self.assertTrue((out / "baseline_current" / "A2_26_guitars.wav").is_file())
        segs = json.loads((out / "modal_damping_body_signature_v1" / "comparison_manifest.json").read_text())[
            "notes"
        ][0]["segments"]
        self.assertEqual(len(segs), 3)
        self.assertIn("note_reward_score", segs[0])

    def test_no_clipping(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(30)}
        wav = self.out_dir / "clip.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.2,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
            diagnostic_mode="modal_damping_body_signature_v1",
        )
        samples = _read_wav(wav)
        self.assertLessEqual(float(np.max(np.abs(samples))), 1.0 + 1e-6)
        self.assertLessEqual(meta["output_peak_dbfs"], -0.5)

    def test_modal_damping_body_signature_v1_registered(self) -> None:
        self.assertIn("modal_damping_body_signature_v1", list_diagnostic_modes())
        cfg = get_diagnostic_mode("modal_damping_body_signature_v1")
        self.assertTrue(cfg.all_mode_broad_contribution)
        self.assertGreater(cfg.per_mode_damping_strength, 0.0)

    def test_deterministic(self) -> None:
        modal = {"predicted_modes": synthetic_classic_body_modes(8)}
        m1 = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "d1.wav",
            diagnostic_mode="modal_damping_body_signature_v1",
        )
        m2 = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.1,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "d2.wav",
            diagnostic_mode="modal_damping_body_signature_v1",
        )
        self.assertEqual(m1["note_reward_score"], m2["note_reward_score"])


if __name__ == "__main__":
    unittest.main()
