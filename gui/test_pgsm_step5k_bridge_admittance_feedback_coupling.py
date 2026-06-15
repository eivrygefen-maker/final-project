#!/usr/bin/env python3
"""PGSM Step 5K — fast bridge/admittance coupling diagnostic tests."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5j_1_guitar_articulation_body_balance_repair import (  # noqa: E402
    COMB_ECHO_FAIL_THRESHOLD,
    DOCUMENTED_LIMITATION_TYPE,
    READINESS_DOCUMENTED_E5_COMB_LIMITATION,
    collect_all_previous_audio_fingerprints,
)
from pgsm_step5k_bridge_admittance_feedback_coupling import (  # noqa: E402
    AUDIO_DIR,
    FAST_VALIDATION_DURATION_S,
    FAST_VALIDATION_MAX_MODES,
    READINESS_AFTER,
    READINESS_INSUFFICIENT,
    READINESS_PARTIAL,
    SAFE_NEXT_STEP_5L,
    SOURCE_CONTRACT_JSON,
    build_bridge_admittance_coupling_contract,
    build_pgsm_step5k_report,
    validate_report_internal_consistency,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

STEP5J_1_REPORT = (
    REPO / "audio" / "debug_reports" / "pgsm_step5j_1_guitar_articulation_body_balance_repair.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_wav_directory(audio_dir: Path) -> dict[str, tuple[int, int]]:
    if not audio_dir.is_dir():
        return {}
    return {
        str(p.resolve()): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(audio_dir.glob("*.wav"))
    }


class TestPgsmStep5kBridgeAdmittanceFeedbackCoupling(unittest.TestCase):
    _shared_report: dict | None = None
    _source_contract_hash: str | None = None
    _audio_snapshot_before: dict[str, tuple[int, int]] | None = None
    _audio_snapshot_after: dict[str, tuple[int, int]] | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if not STEP5J_1_REPORT.is_file():
            raise unittest.SkipTest("Step 5J.1 report required for Step 5K upstream load")
        if SOURCE_CONTRACT_JSON.is_file():
            cls._source_contract_hash = _file_sha256(SOURCE_CONTRACT_JSON)
        cls._audio_snapshot_before = _snapshot_wav_directory(AUDIO_DIR)
        cls._shared_report = build_pgsm_step5k_report(
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
        self.assertEqual(vcfg.get("validation_mode"), "fast")
        self.assertFalse(vcfg.get("render_audio"))
        self.assertFalse(vcfg.get("write_outputs"))
        self.assertEqual(vcfg.get("fast_duration_s"), FAST_VALIDATION_DURATION_S)
        self.assertEqual(self._report().get("validation_max_modes"), FAST_VALIDATION_MAX_MODES)

    def test_fast_validation_no_audio_output_files_written(self) -> None:
        self.assertEqual(self._audio_snapshot_before, self._audio_snapshot_after)

    def test_tracked_source_contract_unmodified(self) -> None:
        if self._source_contract_hash is None:
            self.skipTest("source contract file missing")
        self.assertEqual(_file_sha256(SOURCE_CONTRACT_JSON), self._source_contract_hash)

    def test_step5j_1_documented_limitation_loads(self) -> None:
        self.assertTrue(self._report().get("documented_limitation_loaded"))
        upstream = self._report().get("upstream_step5j_1_status") or {}
        self.assertTrue(upstream.get("pass"))
        self.assertTrue(self._report().get("bridge_coupling_plan_allowed"))
        if upstream.get("documented_limitation_explicit") and STEP5J_1_REPORT.is_file():
            step5j_1 = json.loads(STEP5J_1_REPORT.read_text(encoding="utf-8"))
            self.assertEqual(step5j_1.get("limitation_type"), DOCUMENTED_LIMITATION_TYPE)
            self.assertEqual(
                upstream.get("step5j_1_readiness_status"),
                READINESS_DOCUMENTED_E5_COMB_LIMITATION,
            )

    def test_report_internal_consistency(self) -> None:
        check = validate_report_internal_consistency(self._report())
        self.assertTrue(check.get("pass"), msg=str(check.get("issues")))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5k_bridge_admittance_feedback_coupling as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5k_report(
                repo_root=REPO,
                fast_validation=True,
                max_modes=8,
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
        self.assertTrue(self._report().get("no_previous_audio_modified"))

    def test_coupling_contract_exists(self) -> None:
        contract = self._report().get("coupling_contract") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in (
            "bridge_admittance_proxy",
            "mode_coupling_factor",
            "causal_coupling_ir",
            "e5_low_body_mode_comb_target",
        ):
            self.assertIn(term, names)
        built = build_bridge_admittance_coupling_contract()
        self.assertEqual(
            built.get("parameters", {}).get("comb_echo_fail_threshold"),
            COMB_ECHO_FAIL_THRESHOLD,
        )

    def test_coupling_modifies_e5_effective_path(self) -> None:
        e5 = (self._report().get("per_note_coupling_metrics") or {}).get("E5") or {}
        self.assertTrue(e5.get("coupling_applied"))
        self.assertGreater(e5.get("force_delta_l2_relative") or 0, 1e-6)
        self.assertGreater(e5.get("guarded_mode_count") or 0, 0)

    def test_e5_comb_before_after_computed(self) -> None:
        e5 = self._report().get("E5_comb_before_after") or {}
        self.assertTrue(e5.get("applicable"))
        self.assertIsNotNone(e5.get("comb_score_before"))
        self.assertIsNotNone(e5.get("comb_score_after"))
        rad = self._report().get("E5_radiation_sum_before_after") or {}
        self.assertTrue(rad.get("applicable"))
        self.assertIsNotNone(rad.get("before"))
        self.assertIsNotNone(rad.get("after"))

    def test_stem_balance_before_after_computed(self) -> None:
        bal = self._report().get("stem_balance_before_after") or {}
        for note in NOTE_SET:
            self.assertIn(note, bal)
            self.assertIn("before", bal[note])
            self.assertIn("after", bal[note])

    def test_no_comb_echo_honest_fail_or_improved(self) -> None:
        e5_metrics = (self._report().get("per_note_metrics") or {}).get("E5") or {}
        e5_ba = self._report().get("E5_comb_before_after") or {}
        comb_after = float(e5_metrics.get("comb_echo_score") or e5_ba.get("comb_score_after") or 0)
        if not e5_metrics.get("no_comb_echo"):
            self.assertGreaterEqual(comb_after, COMB_ECHO_FAIL_THRESHOLD)
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("e5_comb_before_after_computed"))

    def test_objective_all_pass_not_forced(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        art = self._report().get("artifact_guard_results") or {}
        if not art.get("pass"):
            self.assertFalse(obj.get("all_pass"))
            self.assertFalse(obj.get("artifact_guard_pass"))

    def test_readiness_honest(self) -> None:
        rg = self._report().get("readiness_after_step5k") or {}
        obj = self._report().get("objective_test_results") or {}
        art = self._report().get("artifact_guard_results") or {}
        e5 = self._report().get("E5_comb_before_after") or {}
        if obj.get("all_pass") and art.get("pass"):
            self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        elif e5.get("comb_improved"):
            self.assertEqual(rg.get("current_status"), READINESS_PARTIAL)
        else:
            self.assertEqual(rg.get("current_status"), READINESS_INSUFFICIENT)

    def test_safe_next_step_step5l(self) -> None:
        self.assertEqual(self._report().get("safe_next_step"), SAFE_NEXT_STEP_5L)

    def test_final_synthesis_and_multiguitar_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5k") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertTrue(rg.get("step5l_multiguitar_planning_allowed"))

    def test_bridge_admittance_proxy_summary(self) -> None:
        adm = self._report().get("bridge_admittance_proxy_summary") or {}
        self.assertIn("model", adm)
        self.assertGreater(adm.get("low_body_mode_count") or 0, 0)


if __name__ == "__main__":
    unittest.main()
