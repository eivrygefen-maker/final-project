#!/usr/bin/env python3
"""PGSM Step 5J.1 — slow integration validation (WAV + disk reports). Run before final approval."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5j_1_guitar_articulation_body_balance_repair import (  # noqa: E402
    DOCUMENTED_LIMITATION_TYPE,
    GENERATED_CONTRACT_JSON,
    PGSM_STEP5J_1_VERSION,
    READINESS_DOCUMENTED_E5_COMB_LIMITATION,
    REPORT_JSON,
    SAFE_NEXT_STEP_STEP5K,
    SOURCE_CONTRACT_JSON,
    VALIDATION_MAX_MODES,
    validate_report_internal_consistency,
    write_pgsm_step5j_1_reports,
)
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import DEFAULT_DURATION_S  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPgsmStep5j1GuitarArticulationBodyBalanceRepairIntegration(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None
    _source_contract_hash: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if SOURCE_CONTRACT_JSON.is_file():
            cls._source_contract_hash = _file_sha256(SOURCE_CONTRACT_JSON)
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5j_1_guitar_articulation_body_balance_repair"
        cls._shared_report = write_pgsm_step5j_1_reports(
            repo_root=REPO,
            json_path=REPORT_JSON,
            md_path=REPORT_JSON.with_suffix(".md"),
            data_path=GENERATED_CONTRACT_JSON,
            audio_dir=cls._shared_wav_dir,
            render_audio=True,
            write_outputs=True,
            fast_validation=False,
            max_modes=VALIDATION_MAX_MODES,
            duration_s=DEFAULT_DURATION_S,
        )

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_validation_mode_full(self) -> None:
        vcfg = self._report().get("validation_config") or {}
        self.assertEqual(vcfg.get("validation_mode"), "full")
        self.assertTrue(vcfg.get("render_audio"))
        self.assertTrue(vcfg.get("write_outputs"))
        self.assertFalse(vcfg.get("tracked_source_files_modified"))

    def test_validation_max_modes_full(self) -> None:
        self.assertEqual(self._report().get("validation_max_modes"), VALIDATION_MAX_MODES)

    def test_tracked_source_contract_unmodified(self) -> None:
        if self._source_contract_hash is None:
            self.skipTest("source contract file missing")
        self.assertEqual(_file_sha256(SOURCE_CONTRACT_JSON), self._source_contract_hash)

    def test_generated_contract_written(self) -> None:
        self.assertTrue(GENERATED_CONTRACT_JSON.is_file())

    def test_disk_report_matches_shared_build(self) -> None:
        self.assertTrue(REPORT_JSON.is_file())
        disk = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        shared_obj = self._report().get("objective_test_results") or {}
        disk_obj = disk.get("objective_test_results") or {}
        self.assertEqual(disk_obj.get("all_pass"), shared_obj.get("all_pass"))
        self.assertEqual(
            (disk.get("artifact_guard_results") or {}).get("pass"),
            (self._report().get("artifact_guard_results") or {}).get("pass"),
        )
        self.assertEqual(disk.get("validation_max_modes"), VALIDATION_MAX_MODES)

    def test_report_internal_consistency(self) -> None:
        check = validate_report_internal_consistency(self._report())
        self.assertTrue(check.get("pass"), msg=str(check.get("issues")))

    def test_four_body_balance_v2_wavs(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_body_balance_v2_diagnostic.wav").is_file())

    def test_all_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        stems = (
            "string_force_stem",
            "pluck_attack_stem",
            "top_plate_stem",
            "back_plate_stem",
            "air_cavity_stem",
            "radiation_sum_stem",
            "final_body_balance_stem",
        )
        for note in NOTE_SET:
            for stem in stems:
                self.assertTrue((out / f"sample_000_{note}_{stem}.wav").is_file())

    def test_no_comb_echo_honest_fail_for_e5(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertFalse(art.get("pass"))
        self.assertIn("no_comb_echo", art.get("failed_guard_fields") or [])

    def test_documented_e5_comb_limitation_full_mode(self) -> None:
        self.assertTrue(self._report().get("documented_limitation"))
        self.assertEqual(self._report().get("limitation_type"), DOCUMENTED_LIMITATION_TYPE)
        self.assertEqual(self._report().get("safe_next_step"), SAFE_NEXT_STEP_STEP5K)
        rg = self._report().get("readiness_after_step5j_1") or {}
        self.assertEqual(rg.get("current_status"), READINESS_DOCUMENTED_E5_COMB_LIMITATION)

    def test_objective_all_pass_remains_false(self) -> None:
        self.assertFalse((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_write_reports_to_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports" / "generated_contracts").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5j_top_back_air_radiation_weighting_refinement.json",
                "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json",
                "pgsm_step5h_note_string_fret_contract.json",
                "pgsm_step3c_numeric_calibration.json",
            ):
                src = REPO / "audio" / "debug_reports" / name
                if src.is_file():
                    (root / "audio" / "debug_reports" / name).write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            cd = REPO / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
            if cd.is_file():
                (root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json").write_text(
                    cd.read_text(encoding="utf-8"), encoding="utf-8"
                )
            generated = root / "audio" / "debug_reports" / "generated_contracts" / "contract.generated.json"
            report = write_pgsm_step5j_1_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=generated,
                audio_dir=root / "audio" / "step5j1_test",
                render_audio=True,
                write_outputs=True,
                fast_validation=False,
                max_modes=VALIDATION_MAX_MODES,
                duration_s=DEFAULT_DURATION_S,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5J_1_VERSION)
            self.assertTrue(generated.is_file())
            self.assertTrue(validate_report_internal_consistency(report).get("pass"))


if __name__ == "__main__":
    unittest.main()
