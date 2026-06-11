#!/usr/bin/env python3
"""Synthesis preset A/B tests (no FEM)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    synthesize_note_with_body_response,
    synthetic_classic_body_modes,
)
from synthesis_presets import get_synthesis_preset, list_synthesis_preset_names  # noqa: E402


class SynthesisPresetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)
        self.modal = {"predicted_modes": synthetic_classic_body_modes(40)}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_all_presets_produce_metadata(self) -> None:
        for name in list_synthesis_preset_names():
            meta = synthesize_note_with_body_response(
                frequency_hz=659.25,
                note_name="E5",
                duration_s=0.2,
                sample_rate=DEFAULT_SAMPLE_RATE,
                modal_data=self.modal,
                output_wav=self.out_dir / f"{name}.wav",
                synthesis_preset=name,
            )
            self.assertEqual(meta["synthesis_preset"], name)
            self.assertIn("synthesis_tuning", meta)

    def test_less_metal_highs_softer_than_current(self) -> None:
        current = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.25,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=self.modal,
            output_wav=self.out_dir / "current.wav",
            synthesis_preset="current",
        )
        less_metal = synthesize_note_with_body_response(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.25,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=self.modal,
            output_wav=self.out_dir / "less_metal.wav",
            synthesis_preset="less_metal_highs",
        )
        self.assertLess(
            less_metal["high_note_pluck_softening_gain"],
            current["high_note_pluck_softening_gain"],
        )
        self.assertLess(
            get_synthesis_preset("less_metal_highs").high_note_pluck_gain_floor,
            get_synthesis_preset("current").high_note_pluck_gain_floor,
        )


if __name__ == "__main__":
    unittest.main()
