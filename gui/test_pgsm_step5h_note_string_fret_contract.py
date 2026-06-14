#!/usr/bin/env python3
"""PGSM Step 5H — note/string/fret contract repair tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step5e_string_driven_bridge_force_repair import collect_previous_audio_fingerprints  # noqa: E402
from pgsm_step5f_string_driven_extended_validation import collect_step5e_fingerprints  # noqa: E402
from pgsm_step5g_physical_tone_model_update_plan import READINESS_AFTER as READINESS_STEP5G  # noqa: E402
from pgsm_step5h_note_string_fret_contract import (  # noqa: E402
    DEFAULT_MAX_FRET,
    PGSM_STEP5H_VERSION,
    READINESS_AFTER,
    SCALE_LENGTH_M,
    STRING_DEFINITIONS,
    build_pgsm_step5h_report,
    compute_effective_length_m,
    compute_frequency_from_fret,
    frequency_error_cents,
    write_pgsm_step5h_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep5hNoteStringFretContract(unittest.TestCase):
    _shared_report: dict | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_report = build_pgsm_step5h_report(repo_root=REPO)

    def setUp(self) -> None:
        self._fp = {
            **collect_step5e_fingerprints(REPO),
            **collect_previous_audio_fingerprints(REPO),
        }

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_step5g_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5g_loaded"))

    def test_step5g_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5g_readiness"), READINESS_STEP5G)
        self.assertTrue(upstream.get("step5g_pass"))
        self.assertTrue(upstream.get("step5g_all_pass"))

    def test_no_audio_generated(self) -> None:
        self.assertTrue(self._report().get("no_audio_generated"))

    def test_no_audio_modified(self) -> None:
        after = {
            **collect_step5e_fingerprints(REPO),
            **collect_previous_audio_fingerprints(REPO),
        }
        self.assertEqual(self._fp, after)
        self.assertTrue(self._report().get("no_audio_modified"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5h_note_string_fret_contract as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5h_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertEqual(report.get("website_default"), STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_base_tuning_includes_standard_open_strings(self) -> None:
        base = self._report().get("classical_guitar_base_contract") or {}
        open_notes = {s["open_note"] for s in base.get("strings") or []}
        for note in ("E2", "A2", "D3", "G3", "B3", "E4"):
            self.assertIn(note, open_notes)

    def test_scale_length_literature_fallback(self) -> None:
        base = self._report().get("classical_guitar_base_contract") or {}
        self.assertAlmostEqual(base.get("scale_length_m"), SCALE_LENGTH_M)
        self.assertEqual(base.get("scale_length_source_level"), "L2_literature_fallback")

    def test_fret_effective_length_equation(self) -> None:
        f0 = 110.0
        fret = 12
        self.assertAlmostEqual(compute_frequency_from_fret(f0, fret), 220.0, places=4)
        self.assertAlmostEqual(compute_effective_length_m(SCALE_LENGTH_M, 0), SCALE_LENGTH_M)
        self.assertAlmostEqual(compute_effective_length_m(SCALE_LENGTH_M, 12), SCALE_LENGTH_M / 2.0, places=5)
        err = frequency_error_cents(220.0, 220.0)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_a2_maps_to_open_string_5(self) -> None:
        preferred = self._report().get("preferred_diagnostic_mapping") or {}
        a2 = preferred.get("A2") or {}
        self.assertEqual(a2.get("string_id"), "string_5")
        self.assertEqual(a2.get("fret"), 0)
        self.assertTrue(a2.get("exact_open_string_claim_allowed"))
        self.assertTrue(a2.get("physical_playable_candidate"))

    def test_a3_has_multiple_playable_candidates(self) -> None:
        candidates = (self._report().get("diagnostic_note_candidates") or {}).get("A3") or []
        playable = [c for c in candidates if c.get("physical_playable_candidate")]
        self.assertGreaterEqual(len(playable), 2)

    def test_a4_has_playable_candidates_not_only_reference(self) -> None:
        candidates = (self._report().get("diagnostic_note_candidates") or {}).get("A4") or []
        playable = [c for c in candidates if c.get("physical_playable_candidate")]
        self.assertGreaterEqual(len(playable), 3)
        preferred = (self._report().get("preferred_diagnostic_mapping") or {}).get("A4") or {}
        self.assertFalse(preferred.get("diagnostic_reference_only"))
        self.assertIsNotNone(preferred.get("string_id"))
        self.assertIsNotNone(preferred.get("fret"))

    def test_e5_maps_to_physical_playable_candidate(self) -> None:
        preferred = (self._report().get("preferred_diagnostic_mapping") or {}).get("E5") or {}
        self.assertTrue(preferred.get("physical_playable_candidate"))
        self.assertEqual(preferred.get("string_id"), "string_1")
        self.assertEqual(preferred.get("fret"), 12)

    def test_preferred_mapping_exists_for_all_notes(self) -> None:
        preferred = self._report().get("preferred_diagnostic_mapping") or {}
        for note in ("A2", "A3", "A4", "E5"):
            self.assertIn(note, preferred)
            self.assertTrue(preferred[note].get("physical_playable_candidate"))

    def test_frequency_error_cents_computed_for_candidates(self) -> None:
        candidates = self._report().get("diagnostic_note_candidates") or {}
        for note, rows in candidates.items():
            for c in rows:
                if c.get("physical_playable_candidate"):
                    self.assertIn("frequency_error_cents", c)
                    self.assertLess(abs(float(c["frequency_error_cents"])), 1.0, msg=note)

    def test_effective_length_le_scale_for_candidates(self) -> None:
        candidates = self._report().get("diagnostic_note_candidates") or {}
        for rows in candidates.values():
            for c in rows:
                if c.get("physical_playable_candidate"):
                    self.assertLessEqual(float(c["effective_length_m"]), SCALE_LENGTH_M + 1e-6)

    def test_diagnostic_reference_only_labeled(self) -> None:
        rejected = self._report().get("rejected_reference_only_mappings") or []
        self.assertTrue(rejected)
        for r in rejected:
            self.assertTrue(r.get("diagnostic_reference_only"))
        a4_candidates = (self._report().get("diagnostic_note_candidates") or {}).get("A4") or []
        ref = next(c for c in a4_candidates if c.get("mapping_id") == "A4_concert_reference")
        self.assertTrue(ref.get("diagnostic_reference_only"))
        self.assertFalse(ref.get("physical_playable_candidate"))

    def test_stk_readiness_contract_blocked(self) -> None:
        stk = self._report().get("stk_readiness_contract") or {}
        self.assertFalse(stk.get("stk_integration_allowed"))
        self.assertEqual(
            stk.get("reason"),
            "note_string_fret_contract_ready_but_damping_and_radiation_updates_pending",
        )
        entries = stk.get("future_stk_note_entries") or []
        self.assertEqual(len(entries), 4)
        for e in entries:
            self.assertIn("string_id", e)
            self.assertIn("effective_length_m", e)
            self.assertIn("pluck_position_ratio_relative_to_string", e)

    def test_readiness_contract_only_not_final(self) -> None:
        rg = self._report().get("readiness_after_step5h") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertTrue(rg.get("contract_only_not_final"))

    def test_objective_all_pass(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))

    def test_step5g_first_target_satisfied(self) -> None:
        tgt = self._report().get("step5g_first_target_validation") or {}
        self.assertTrue(tgt.get("step5g_first_target_is_contract_repair"))
        self.assertTrue(tgt.get("step5h_contract_repair_complete"))
        self.assertTrue(tgt.get("satisfies_step5g_first_target"))

    def test_default_max_fret_conservative(self) -> None:
        base = self._report().get("classical_guitar_base_contract") or {}
        self.assertEqual(base.get("fret_range", {}).get("default_max_fret"), DEFAULT_MAX_FRET)
        preferred = self._report().get("preferred_diagnostic_mapping") or {}
        for note, mapping in preferred.items():
            self.assertLessEqual(int(mapping["fret"]), DEFAULT_MAX_FRET, msg=note)

    def test_string_definitions_count(self) -> None:
        self.assertEqual(len(STRING_DEFINITIONS), 6)

    def test_write_reports_creates_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5g_physical_tone_model_update_plan.json",
                "pgsm_step5f_string_driven_extended_validation.json",
                "pgsm_step5e_string_driven_bridge_force_repair.json",
            ):
                src = REPO / "audio" / "debug_reports" / name
                if src.is_file():
                    (root / "audio" / "debug_reports" / name).write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            report = write_pgsm_step5h_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract.json",
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5H_VERSION)
            self.assertTrue((root / "audio" / "debug_reports" / "out.json").is_file())
            self.assertTrue((root / "data" / "contract.json").is_file())
            loaded = json.loads((root / "data" / "contract.json").read_text(encoding="utf-8"))
            self.assertIn("preferred_diagnostic_mapping", loaded)


if __name__ == "__main__":
    unittest.main()
