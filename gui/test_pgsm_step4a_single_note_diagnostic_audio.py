#!/usr/bin/env python3
"""PGSM Step 4A — diagnostic audio tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step3d_pre_synthesis_contract import STEP4A_READINESS  # noqa: E402
from pgsm_step4a_single_note_diagnostic_audio import (  # noqa: E402
    DIAGNOSTIC_LABEL,
    PGSM_STEP4A_VERSION,
    READINESS_STEP4B,
    build_pgsm_step4a_report,
    load_step3d_contract,
    write_pgsm_step4a_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep4aSingleNoteDiagnosticAudio(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.audio_dir = self.tmp / "pgsm_step4a_diagnostic_audio"
        self._report_cache: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step4a_report(
                repo_root=REPO,
                audio_dir=self.audio_dir,
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
        return self._report_cache

    def _wav_peak(self, path: Path) -> float:
        with wave.open(str(path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        import numpy as np

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32767.0
        return float(np.max(np.abs(samples)))

    def test_step3d_contract_loads(self) -> None:
        doc = load_step3d_contract(REPO)
        self.assertEqual(
            (doc.get("readiness_after_step3d") or {}).get("current_status"),
            STEP4A_READINESS,
        )

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step4a_reports(
            repo_root=REPO,
            json_path=self.tmp / "step4a.json",
            md_path=self.tmp / "step4a.md",
            audio_dir=self.audio_dir,
            max_modes=MAX_MODES_TEST,
        )
        self.assertTrue((self.tmp / "step4a.json").is_file())
        doc = json.loads((self.tmp / "step4a.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP4A_VERSION)
        self.assertTrue(doc["diagnostic_audio_generated"])
        self.assertIn("diagnostic audio only", doc["explicit_statement"])

    def test_exactly_one_main_diagnostic_wav(self) -> None:
        self._report()
        main_wavs = list(self.audio_dir.glob("*_diagnostic.wav"))
        self.assertEqual(len(main_wavs), 1)

    def test_stems_generated_as_diagnostics(self) -> None:
        self._report()
        self.assertTrue((self.audio_dir / "sample_000_A4_body_stem.wav").is_file())
        self.assertTrue((self.audio_dir / "sample_000_A4_excitation_stem.wav").is_file())

    def test_no_wav_outside_audio_dir_in_tmp(self) -> None:
        self._report()
        all_wavs = list(self.tmp.rglob("*.wav"))
        for p in all_wavs:
            self.assertIn("pgsm_step4a_diagnostic_audio", str(p))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step4a_report(
                repo_root=REPO,
                audio_dir=self.audio_dir,
                write_wav=False,
                max_modes=MAX_MODES_TEST,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_step4a_single_note_diagnostic_audio as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("stk_body_transfer", src.lower().replace("stk_body_transfer_final_v1", ""))

    def test_final_synthesis_ready_false(self) -> None:
        report = self._report()
        self.assertFalse(report.get("final_synthesis_ready"))
        rg = report.get("readiness_after_step4a") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_diagnostic_label_in_report(self) -> None:
        report = self._report()
        self.assertEqual(report.get("diagnostic_label"), DIAGNOSTIC_LABEL)

    def test_peak_amplitude_below_0p3_fs(self) -> None:
        self._report()
        main = self.audio_dir / "sample_000_A4_diagnostic.wav"
        peak = self._wav_peak(main)
        self.assertLessEqual(peak, 0.3)

    def test_no_clipping(self) -> None:
        report = self._report()
        norm = (report.get("normalization_summary") or {}).get("body") or {}
        self.assertFalse(norm.get("clipping"))

    def test_envelope_consistency_computed(self) -> None:
        report = self._report()
        env = report.get("envelope_consistency") or {}
        self.assertIn("envelope_correlation_vs_step3c", env)
        self.assertTrue(env.get("pass"))

    def test_no_delayed_body_onset(self) -> None:
        report = self._report()
        self.assertTrue((report.get("envelope_consistency") or {}).get("no_delayed_body_onset"))

    def test_no_end_rise(self) -> None:
        report = self._report()
        self.assertTrue((report.get("envelope_consistency") or {}).get("no_end_rise"))

    def test_no_hard_gate(self) -> None:
        report = self._report()
        self.assertTrue((report.get("envelope_consistency") or {}).get("no_hard_gate"))

    def test_spectral_modal_consistency_computed(self) -> None:
        report = self._report()
        spec = report.get("spectral_modal_consistency") or {}
        self.assertIn("modal_peaks_aligned_count", spec)
        self.assertTrue(spec.get("pass"))

    def test_artifact_guard_forbids_v6_patterns(self) -> None:
        report = self._report()
        art = report.get("artifact_guard_results") or {}
        self.assertFalse(art.get("body_tail_stem_used"))
        self.assertFalse(art.get("helmholtz_echo_ir_used"))
        self.assertFalse(art.get("artificial_reverb"))
        self.assertFalse(art.get("second_pluck_onset"))
        self.assertTrue(art.get("pass"))

    def test_exact_open_string_claim_blocked(self) -> None:
        report = self._report()
        exc = report.get("excitation_proxy_summary") or {}
        self.assertFalse(exc.get("exact_open_string_claim_allowed"))

    def test_readiness_not_stk_website_multiguitar_final(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step4a") or {}
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_readiness_step4b_refinement_only(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))
        rg = report.get("readiness_after_step4a") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP4B)

    def test_safe_next_step_not_production(self) -> None:
        report = self._report()
        nxt = report.get("safe_next_step", "").lower()
        self.assertIn("step 4b", nxt)
        self.assertIn("diagnostic", nxt)
        self.assertTrue("not stk" in nxt or "still not" in nxt)
        self.assertNotIn("website production", nxt)
        self.assertNotIn("replace website", nxt)


if __name__ == "__main__":
    unittest.main()
