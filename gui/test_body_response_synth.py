#!/usr/bin/env python3
"""Stage-1 body-response synthesis tests (no FEM)."""
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
    load_modal_data_from_path,
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

    def _modal_fixture(self):
        return {"predicted_modes": synthetic_classic_body_modes()}

    def test_e2_a4_e5_smoke(self) -> None:
        modal = self._modal_fixture()
        cases = (("E2", 82.41), ("A4", 440.0), ("E5", 659.25))
        for name, hz in cases:
            wav = self.out_dir / f"{name}.wav"
            meta_path = self.out_dir / f"{name}.json"
            meta = synthesize_note_with_body_response(
                frequency_hz=hz,
                note_name=name,
                duration_s=0.5,
                sample_rate=DEFAULT_SAMPLE_RATE,
                modal_data=modal,
                output_wav=wav,
                output_metadata_json=meta_path,
            )
            self.assertTrue(wav.is_file() and wav.stat().st_size > 44, name)
            samples = _read_wav_samples(wav)
            self.assertGreater(samples.size, 0, name)
            self.assertTrue(np.all(np.isfinite(samples)), name)
            self.assertLessEqual(float(np.max(np.abs(samples))), 1.0 + 1e-6, name)
            self.assertEqual(meta["note_name"], name)
            self.assertGreater(meta["modal_mode_count_used"], 0)
            if name == "E5":
                self.assertTrue(meta["high_frequency_fallback_used"])
            else:
                self.assertFalse(meta["high_frequency_fallback_used"])

    def test_legacy_modes_hz_json(self) -> None:
        modal = {
            "modes_hz": [95.0, 180.0, 260.0, 410.0],
            "mode_weights": [1.0, 0.8, 0.6, 0.4],
            "analysis": "rom_online_body",
        }
        modes, _ = parse_modal_modes(modal)
        self.assertEqual(len(modes), 4)
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
        self.assertEqual(meta["modal_mode_count_used"], 4)

    def test_consumes_gui_rom_stk_json_if_present(self) -> None:
        if not ROM_STK_JSON.is_file():
            self.skipTest(f"no {ROM_STK_JSON}")
        modal = load_modal_data_from_path(ROM_STK_JSON)
        wav = self.out_dir / "rom_gui.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=82.41,
            note_name="E2",
            duration_s=0.25,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal,
            output_wav=wav,
        )
        self.assertTrue(wav.is_file())
        self.assertGreater(meta["modal_mode_count_available"], 0)

    def test_metadata_fields(self) -> None:
        wav = self.out_dir / "meta.wav"
        meta_path = self.out_dir / "meta.json"
        meta = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.2,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=self._modal_fixture(),
            output_wav=wav,
            output_metadata_json=meta_path,
        )
        required = {
            "note_name",
            "frequency_hz",
            "duration_s",
            "sample_rate",
            "modal_mode_count_available",
            "modal_mode_count_used",
            "full_modal_band_hz",
            "high_frequency_fallback_used",
            "bridge_weighting_used",
            "mic_proxy_used",
            "radiation_proxy_used",
            "q_or_damping_used",
            "defaults_used",
            "output_wav",
        }
        self.assertTrue(required.issubset(meta.keys()))
        self.assertEqual(meta["full_modal_band_hz"], list(FULL_MODAL_BAND_HZ))
        self.assertTrue(meta_path.is_file())
        on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["note_name"], "A4")


if __name__ == "__main__":
    unittest.main()
