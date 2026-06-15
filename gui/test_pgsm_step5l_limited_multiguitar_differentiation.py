#!/usr/bin/env python3
"""PGSM Step 5L — fast limited multi-guitar differentiation tests."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_step5j_1_guitar_articulation_body_balance_repair import collect_all_previous_audio_fingerprints  # noqa: E402
from pgsm_step5l_limited_multiguitar_differentiation import (  # noqa: E402
    AUDIO_DIR,
    FAST_NOTE_SET,
    FAST_SAMPLE_SET,
    FAST_VALIDATION_DURATION_S,
    FAST_VALIDATION_MAX_MODES,
    READINESS_AFTER,
    READINESS_FAIL,
    READINESS_WEAK,
    SAFE_NEXT_STEP_5M,
    SOURCE_CONTRACT_JSON,
    STEP5K_REPORT_JSON,
    build_multiguitar_contract,
    build_pgsm_step5l_report,
    validate_report_internal_consistency,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_wav_directory(audio_dir: Path) -> dict[str, tuple[int, int]]:
    if not audio_dir.is_dir():
        return {}
    return {
        str(p.resolve()): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(audio_dir.glob("*.wav"))
    }


class TestPgsmStep5lLimitedMultiguitarDifferentiation(unittest.TestCase):
    _shared_report: dict | None = None
    _source_contract_hash: str | None = None
    _step5k_report_hash_before: str | None = None
    _step5k_report_mtime_before: int | None = None
    _audio_snapshot_before: dict[str, tuple[int, int]] | None = None
    _audio_snapshot_after: dict[str, tuple[int, int]] | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if SOURCE_CONTRACT_JSON.is_file():
            cls._source_contract_hash = _file_sha256(SOURCE_CONTRACT_JSON)
        if STEP5K_REPORT_JSON.is_file():
            cls._step5k_report_hash_before = _file_sha256(STEP5K_REPORT_JSON)
            cls._step5k_report_mtime_before = STEP5K_REPORT_JSON.stat().st_mtime_ns
        cls._audio_snapshot_before = _snapshot_wav_directory(AUDIO_DIR)
        cls._shared_report = build_pgsm_step5l_report(
            repo_root=REPO,
            fast_validation=True,
        )
        cls._audio_snapshot_after = _snapshot_wav_directory(AUDIO_DIR)

    def setUp(self) -> None:
        self._prev_fp = collect_all_previous_audio_fingerprints(REPO)

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_validation_mode_fast(self) -> None:
        vcfg = self._report().get("validation_config") or {}
        upstream = self._report().get("upstream_step5k_status") or {}
        self.assertEqual(vcfg.get("validation_mode"), "fast")
        self.assertFalse(vcfg.get("render_audio"))
        self.assertFalse(vcfg.get("write_outputs"))
        self.assertEqual(
            vcfg.get("upstream_step5k_rebuild_skipped"),
            upstream.get("step5k_upstream_source") == "disk_json",
        )
        self.assertEqual(vcfg.get("duration_s"), FAST_VALIDATION_DURATION_S)
        self.assertEqual(self._report().get("validation_max_modes"), FAST_VALIDATION_MAX_MODES)

    def test_fast_validation_no_audio_output_files_written(self) -> None:
        self.assertEqual(self._audio_snapshot_before, self._audio_snapshot_after)

    def test_tracked_source_contract_unmodified(self) -> None:
        if self._source_contract_hash is None:
            self.skipTest("source contract file missing")
        self.assertEqual(_file_sha256(SOURCE_CONTRACT_JSON), self._source_contract_hash)

    def test_step5k_upstream_loads(self) -> None:
        upstream = self._report().get("upstream_step5k_status") or {}
        self.assertTrue(upstream.get("pass"))
        self.assertTrue(upstream.get("upstream_step5k_loaded"))
        self.assertIn(
            upstream.get("step5k_upstream_source"),
            ("disk_json", "in_memory_fast_build"),
        )
        self.assertIsInstance(upstream.get("upstream_step5k_fast_validation_used"), bool)
        if upstream.get("step5k_upstream_source") == "in_memory_fast_build":
            self.assertTrue(upstream.get("upstream_step5k_fast_validation_used"))
        else:
            self.assertFalse(upstream.get("upstream_step5k_fast_validation_used"))
        self.assertTrue(upstream.get("step5l_multiguitar_planning_allowed"))

    def test_step5k_report_not_written_during_fast_test(self) -> None:
        if self._step5k_report_hash_before is not None:
            self.assertEqual(_file_sha256(STEP5K_REPORT_JSON), self._step5k_report_hash_before)
            self.assertEqual(STEP5K_REPORT_JSON.stat().st_mtime_ns, self._step5k_report_mtime_before)
        else:
            self.assertFalse(STEP5K_REPORT_JSON.is_file())

    def test_report_internal_consistency(self) -> None:
        check = validate_report_internal_consistency(self._report())
        self.assertTrue(check.get("pass"), msg=str(check.get("issues")))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5l_limited_multiguitar_differentiation as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5l_report(
                repo_root=REPO,
                fast_validation=True,
                max_modes=16,
                duration_s=0.25,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(self._report().get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_previous_audio_preserved(self) -> None:
        after = collect_all_previous_audio_fingerprints(REPO)
        self.assertEqual(self._prev_fp, after)

    def test_sample_set_at_least_three(self) -> None:
        samples = self._report().get("sample_set") or []
        self.assertGreaterEqual(len(samples), 3)
        for sid in FAST_SAMPLE_SET:
            self.assertIn(sid, samples)

    def test_note_set_includes_core_notes(self) -> None:
        notes = self._report().get("note_set") or []
        for note in FAST_NOTE_SET:
            self.assertIn(note, notes)

    def test_per_sample_physical_parameters(self) -> None:
        phys = self._report().get("per_sample_physical_parameters") or {}
        for sid in FAST_SAMPLE_SET:
            row = phys.get(sid) or {}
            self.assertIn("top_wood_id", row)
            self.assertIn("bridge_mobility_proxy", row)
            self.assertIn("body_depth_m", row)

    def test_differentiation_trace_per_sample(self) -> None:
        traces = self._report().get("per_sample_differentiation_trace") or {}
        for sid in FAST_SAMPLE_SET:
            tr = traces.get(sid) or {}
            self.assertGreaterEqual(len(tr.get("physical_drivers_applied") or []), 4)
            self.assertIn("modifiers", tr)
            self.assertIn("sample_id_only_gain", " ".join(tr.get("forbidden_levers_not_used") or []))

    def test_pairwise_metrics_computed(self) -> None:
        pairwise = self._report().get("pairwise_guitar_difference_metrics") or {}
        self.assertGreaterEqual(len(pairwise), 2)
        for row in pairwise.values():
            self.assertIn("overall_differentiation_score", row)

    def test_anti_cheat_checks(self) -> None:
        ac = self._report().get("anti_cheat_checks") or {}
        self.assertTrue(ac.get("physical_driver_trace_per_sample"))
        self.assertTrue(ac.get("no_sample_id_only_gain"))
        self.assertTrue(ac.get("no_randomization"))
        self.assertIn("differences_not_only_loudness", ac)

    def test_loudness_only_difference_rejected(self) -> None:
        ac = self._report().get("anti_cheat_checks") or {}
        loud = self._report().get("loudness_normalization_report") or {}
        self.assertTrue(loud.get("gain_separate_from_physics"))
        if not ac.get("differences_not_only_loudness"):
            self.assertLess(
                float(self._report().get("mean_overall_differentiation_score") or 0),
                0.02,
            )

    def test_per_note_per_sample_metrics(self) -> None:
        metrics = self._report().get("per_note_per_sample_metrics") or {}
        for sid in FAST_SAMPLE_SET:
            self.assertIn(sid, metrics)
            for note in FAST_NOTE_SET:
                m = metrics[sid].get(note) or {}
                self.assertIn("spectral_centroid_hz", m)
                self.assertIn("body_balance", m)
                self.assertIn("bridge_coupling_strength", m)

    def test_readiness_honest(self) -> None:
        rg = self._report().get("readiness_after_step5l") or {}
        ac = self._report().get("anti_cheat_checks") or {}
        art = self._report().get("artifact_guard_results") or {}
        mean_d = float(self._report().get("mean_overall_differentiation_score") or 0)
        if ac.get("pass") and art.get("pass") and mean_d >= 0.10:
            self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        elif ac.get("pass") and art.get("pass") and mean_d >= 0.04:
            self.assertEqual(rg.get("current_status"), READINESS_WEAK)
        elif not ac.get("pass") or not art.get("pass"):
            self.assertEqual(rg.get("current_status"), READINESS_FAIL)

    def test_safe_next_step_step5m(self) -> None:
        self.assertEqual(self._report().get("safe_next_step"), SAFE_NEXT_STEP_5M)

    def test_final_synthesis_and_multiguitar_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5l") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_multiguitar_contract(self) -> None:
        contract = build_multiguitar_contract()
        self.assertGreaterEqual(len(contract.get("physical_drivers") or []), 5)


if __name__ == "__main__":
    unittest.main()
