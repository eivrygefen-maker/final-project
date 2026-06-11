#!/usr/bin/env python3
"""Body-response synthesis tests (modal transfer-function model, no FEM)."""
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
    FULL_MODAL_BAND_HZ,
    STRING_PLUCK_GAIN,
    concatenate_audio_with_crossfade,
    high_note_pluck_softening_gain,
    load_modal_data_from_path,
    modes_in_validated_band,
    parse_modal_modes,
    synthetic_classic_body_modes,
    synthesize_note_with_body_response,
)

ROM_STK_JSON = REPO / "FEM" / "outputs" / "rom_stk_body.json"


def _read_wav_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width == 2:
        count = len(raw) // 2
        samples = struct.unpack(f"<{count}h", raw)
        return np.asarray(samples, dtype=np.float64) / 32767.0
    raise ValueError(f"unsupported sample width {width}")


class BodyResponseSynthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _spread_modal_fixture(self, n: int = 55):
        return {"predicted_modes": synthetic_classic_body_modes(n)}

    def test_e2_a4_e5_smoke(self) -> None:
        modal = self._spread_modal_fixture()
        cases = (("E2", 82.41), ("A2", 110.0), ("A4", 440.0), ("E5", 659.25))
        lo, hi = FULL_MODAL_BAND_HZ
        decay_by_note: dict[str, dict] = {}
        for name, hz in cases:
            wav = self.out_dir / f"{name}.wav"
            meta_path = self.out_dir / f"{name}.json"
            meta = synthesize_note_with_body_response(
                frequency_hz=hz,
                note_name=name,
                duration_s=3.0,
                sample_rate=DEFAULT_SAMPLE_RATE,
                modal_data=modal,
                output_wav=wav,
                output_metadata_json=meta_path,
            )
            decay_by_note[name] = meta
            self.assertTrue(wav.is_file() and wav.stat().st_size > 44, name)
            samples = _read_wav_samples(wav)
            self.assertGreater(samples.size, 0, name)
            self.assertTrue(np.all(np.isfinite(samples)), name)
            self.assertLessEqual(float(np.max(np.abs(samples))), 1.0 + 1e-6, name)
            self.assertEqual(meta["available_modal_count"], 55)
            self.assertEqual(meta["evaluated_modal_count"], 55)
            self.assertAlmostEqual(meta["available_modal_frequency_min_hz"], lo, places=0)
            self.assertAlmostEqual(meta["available_modal_frequency_max_hz"], hi, places=0)
            self.assertEqual(meta["evaluated_modal_frequency_min_hz"], lo)
            self.assertEqual(meta["evaluated_modal_frequency_max_hz"], hi)
            self.assertTrue(meta["pitch_preserved"])
            self.assertGreater(meta["body_to_string_rms_ratio_before_loudness"], 2.5, name)
            self.assertLess(meta["body_to_string_rms_ratio_before_loudness"], 8.0, name)
            self.assertTrue(meta["anti_click_taper_applied"], name)
            self.assertLess(abs(samples[0]), 0.05, name)
            self.assertLess(abs(samples[-1]), 0.01, name)
            self.assertGreater(
                meta["body_rms_before_mix"] / max(meta["string_rms_before_mix"], 1e-9),
                3.0,
                name,
            )
            rms_floor = -28.0 if name in ("A4", "E5") else -25.5
            self.assertGreater(meta["output_rms_dbfs"], rms_floor, name)
            self.assertLessEqual(meta["output_peak_dbfs"], -0.5, name)
            self.assertIn("limiter_used", meta)
            self.assertIn("target_rms_dbfs", meta)
            self.assertIn("output_decay_slope_db_per_s", meta)
            self.assertIn("late_to_early_rms_db", meta)
            self.assertIn("note_decay_tau_s", meta)
            self.assertIn("body_decay_tau_s", meta)
            self.assertEqual(meta["body_modal_richness_gain"], 1.2, name)
            self.assertAlmostEqual(
                meta["body_rms_after_richness_gain"],
                meta["body_rms_before_richness_gain"] * 1.2,
                places=6,
                msg=name,
            )
            self.assertLess(meta["late_to_early_rms_db"], -3.0, name)
            if name in ("E2", "A2"):
                self.assertTrue(meta["fundamental_anchor_used"], name)
            if name in ("A4", "E5"):
                self.assertTrue(meta["high_note_decay_applied"], name)
            if name == "E5":
                self.assertTrue(meta["high_frequency_fallback_used"])
            else:
                self.assertFalse(meta["high_frequency_fallback_used"])

        self.assertGreater(
            decay_by_note["E2"]["note_decay_tau_s"],
            decay_by_note["A4"]["note_decay_tau_s"],
        )
        self.assertGreater(
            decay_by_note["A4"]["note_decay_tau_s"],
            decay_by_note["E5"]["note_decay_tau_s"],
        )
        self.assertLess(
            decay_by_note["E5"]["late_to_early_rms_db"],
            decay_by_note["E2"]["late_to_early_rms_db"],
        )
        self.assertLess(
            decay_by_note["E5"]["output_decay_slope_db_per_s"],
            decay_by_note["E2"]["output_decay_slope_db_per_s"],
        )

    def test_a2_evaluates_all_band_modes_not_narrow_subset(self) -> None:
        modal = self._spread_modal_fixture(40)
        wav = self.out_dir / "A2.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.35,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
        )
        self.assertEqual(meta["evaluated_modal_count"], 40)
        self.assertGreater(meta["available_modal_frequency_max_hz"], 400.0)
        self.assertLess(meta["available_modal_frequency_min_hz"], 100.0)
        top = meta["top_contributing_modes"]
        self.assertGreater(len(top), 0)
        nearest = {row["nearest_harmonic_hz"] for row in top}
        self.assertTrue(any(abs(h - 110.0) < 1.0 or abs(h - 220.0) < 2.0 for h in nearest))

    def test_legacy_modes_hz_json(self) -> None:
        modal = {
            "modes_hz": [95.0, 180.0, 260.0, 410.0],
            "mode_weights": [1.0, 0.8, 0.6, 0.4],
            "analysis": "rom_online_body",
        }
        modes, _ = parse_modal_modes(modal)
        self.assertEqual(len(modes), 4)
        self.assertEqual(len(modes_in_validated_band(modes)), 4)
        wav = self.out_dir / "legacy.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.25,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
        )
        self.assertTrue(wav.is_file())
        self.assertEqual(meta["evaluated_modal_count"], 4)
        self.assertEqual(meta["available_modal_count"], 4)

    def test_consumes_gui_rom_stk_json_if_present(self) -> None:
        if not ROM_STK_JSON.is_file():
            self.skipTest(f"no {ROM_STK_JSON}")
        modal = load_modal_data_from_path(ROM_STK_JSON)
        modes, _ = parse_modal_modes(modal)
        wav = self.out_dir / "rom_gui.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.25,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
        )
        self.assertTrue(wav.is_file())
        doc_n = int(modal.get("num_modes") or len(modes))
        self.assertEqual(meta["available_modal_count"], doc_n)
        self.assertEqual(meta["evaluated_modal_count"], len(modes_in_validated_band(modes)))
        self.assertGreater(meta["available_modal_frequency_max_hz"], 200.0)

    def test_high_note_pluck_softening_metadata(self) -> None:
        modal = self._spread_modal_fixture()
        e2 = synthesize_note_with_body_response(
            frequency_hz=82.41,
            note_name="E2",
            duration_s=0.35,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "soft_e2.wav",
        )
        a4 = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.35,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "soft_a4.wav",
        )
        e5 = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.35,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=self.out_dir / "soft_e5.wav",
        )
        self.assertFalse(e2["high_note_pluck_softening_applied"])
        self.assertAlmostEqual(e2["high_note_pluck_softening_gain"], 1.0, places=4)
        self.assertAlmostEqual(e2["string_pluck_gain"], STRING_PLUCK_GAIN, places=6)
        self.assertTrue(a4["high_note_pluck_softening_applied"])
        self.assertTrue(e5["high_note_pluck_softening_applied"])
        self.assertLess(e5["high_note_pluck_softening_gain"], a4["high_note_pluck_softening_gain"])
        self.assertLess(a4["string_pluck_gain"], STRING_PLUCK_GAIN)
        self.assertLess(e5["string_hf_rolloff_factor"], 1.0)
        self.assertGreater(e2["body_to_string_rms_ratio_before_loudness"], 2.5)

    def test_metadata_fields(self) -> None:
        wav = self.out_dir / "meta.wav"
        meta_path = self.out_dir / "meta.json"
        meta = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.2,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=self._spread_modal_fixture(),
            output_wav=wav,
            output_metadata_json=meta_path,
        )
        required = {
            "note_name",
            "frequency_hz",
            "pitch_preserved",
            "synthesis_model",
            "available_modal_count",
            "available_modal_frequency_min_hz",
            "available_modal_frequency_max_hz",
            "evaluated_modal_count",
            "active_modal_count_after_threshold",
            "selected_or_pruned_policy",
            "harmonics_used_hz",
            "top_contributing_modes",
            "high_frequency_fallback_used",
            "dry_mix",
            "wet_mix",
            "direct_string_gain",
            "body_filter_gain",
            "dry_rms_before_mix",
            "body_rms_before_mix",
            "dry_gain_applied",
            "body_gain_applied",
            "body_modal_richness_gain",
            "body_rms_before_richness_gain",
            "body_rms_after_richness_gain",
            "final_dry_to_body_rms_ratio",
            "final_peak_normalization_gain",
            "output_rms_dbfs",
            "output_peak_dbfs",
            "target_rms_dbfs",
            "limiter_used",
            "limiter_gain_reduction_db",
            "final_peak_ceiling_dbfs",
            "fundamental_anchor_used",
            "string_pluck_gain",
            "body_modal_gain",
            "body_to_string_rms_ratio_before_loudness",
            "body_modal_bandwidth_widening",
            "anti_click_taper_applied",
            "fade_in_ms",
            "fade_out_ms",
            "output_decay_slope_db_per_s",
            "early_rms_dbfs",
            "late_rms_dbfs",
            "late_to_early_rms_db",
            "note_decay_tau_s",
            "body_decay_tau_s",
            "harmonic_decay_model",
            "high_note_decay_applied",
            "high_note_pluck_softening_applied",
            "high_note_pluck_softening_gain",
            "string_hf_rolloff_factor",
            "effective_string_pluck_gain",
            "q_min",
            "q_median",
            "q_max",
            "output_wav",
        }
        self.assertEqual(meta["direct_string_role"], "string_pluck_plus_pitch_layer")
        self.assertTrue(required.issubset(meta.keys()))
        self.assertTrue(meta_path.is_file())
        row = meta["top_contributing_modes"][0]
        self.assertIn("bridge_weight", row)
        self.assertIn("nearest_harmonic_hz", row)

    def test_concatenation_crossfade_low_discontinuity(self) -> None:
        modal = self._spread_modal_fixture()
        sr = DEFAULT_SAMPLE_RATE
        paths = []
        for name, hz in (("E2", 82.41), ("A2", 110.0)):
            wav = self.out_dir / f"{name}_xf.wav"
            synthesize_note_with_body_response(
                frequency_hz=hz,
                note_name=name,
                duration_s=0.35,
                sample_rate=sr,
                modal_data=modal,
                output_wav=wav,
            )
            paths.append(_read_wav_samples(wav))
        joined = concatenate_audio_with_crossfade(paths, sr, crossfade_ms=10.0, silence_ms=20.0)
        self.assertGreater(joined.size, 0)
        self.assertLess(abs(float(joined[0])), 0.05)
        self.assertLess(abs(float(joined[-1])), 0.01)
        mid = len(paths[0]) + 10
        if mid + 5 < joined.size:
            jump = float(np.max(np.abs(np.diff(joined[mid : mid + 5]))))
            self.assertLess(jump, 0.35)


if __name__ == "__main__":
    unittest.main()
