#!/usr/bin/env python3
"""Stage 5.2 ROM audio-proxy validation tests (no FEM)."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage52_rom_audio_proxy_candidates import (  # noqa: E402
    CANDIDATE_A,
    apply_audio_proxy_candidate,
    diagnose_audio_proxy_weakness,
    evaluate_candidates_on_comparisons,
)
from stage52_rom_audio_proxy_report import (  # noqa: E402
    _is_valid_loo,
    aggregate_metrics,
    assess_verdicts,
    build_stage52_report,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    VALIDATION_LEAVE_ONE_OUT,
    _write_comparison_csv,
)


def _fake_loo_comparison(sample_id: str = "sample_010") -> dict:
    matches = []
    for i, (f_hz, rad) in enumerate(
        (
            (95.0, 1.0e-5),
            (120.0, 5.0e-5),
            (180.0, 2.0e-4),
            (240.0, 8.0e-4),
            (310.0, 1.5e-3),
            (420.0, 3.0e-4),
        )
    ):
        rom_rad = rad * (1.0 + 0.15 * ((i % 3) - 1))
        matches.append(
            {
                "rom_frequency_hz": round(f_hz * (1.0 + 0.01 * (i % 2)), 6),
                "fom_frequency_hz": f_hz,
                "abs_error_hz": round(abs(f_hz * 0.01), 6),
                "relative_error": 0.01,
                "fom_coupling_class": "top_dominant" if i % 2 == 0 else "back_dominant",
                "rom_coupling_class": "top_dominant" if i % 2 == 0 else "air_dominant",
                "fom_dominant_region": "top" if i % 2 == 0 else "back",
                "rom_dominant_region": "top" if i % 2 == 0 else "back",
                "fom_radiation_proxy": rad,
                "rom_radiation_proxy": rom_rad,
                "fom_mic_output_proxy": rad * 10.0,
                "rom_mic_output_proxy": rom_rad * 9.5,
                "fom_bridge_excitation_abs": rad * 0.5,
                "rom_bridge_excitation_abs": rom_rad * 0.55,
                "fom_top_share": 0.6 if i % 2 == 0 else 0.2,
                "rom_top_share": 0.55 if i % 2 == 0 else 0.25,
                "fom_back_share": 0.2 if i % 2 == 0 else 0.6,
                "rom_back_share": 0.25 if i % 2 == 0 else 0.55,
                "fom_air_share": 0.2,
                "rom_air_share": 0.2,
                "rom_radiation_proxy_log10": -5.0,
                "fom_radiation_proxy_log10": -5.0,
                "rom_radiation_proxy_p95_norm": 0.5,
                "fom_radiation_proxy_p95_norm": 0.55,
            }
        )
    return {
        "schema": "rom_fom_comparison_v4",
        "status": "COMPLETED",
        "sample_id": sample_id,
        "lhs_row_index": 10,
        "run_id": f"{sample_id}_rom_shadow_v1",
        "validation_mode": VALIDATION_LEAVE_ONE_OUT,
        "training_includes_target": False,
        "accuracy_meaningful": True,
        "matched_mode_count": len(matches),
        "median_relative_error": 0.012,
        "mean_relative_error": 0.013,
        "p90_relative_error": 0.018,
        "median_abs_error_hz": 1.8,
        "per_mode_matches": matches,
        "phase2_scalar_metrics": {
            "radiation_proxy_log_mae": 0.38,
            "mic_output_proxy_p95_norm_mae": 0.28,
            "top_k_radiation_overlap": 0.42,
            "radiation_proxy_rank_correlation": 0.71,
            "coupling_class_accuracy": 0.80,
            "dominant_region_accuracy": 0.88,
            "top_share_mae": 0.06,
            "back_share_mae": 0.07,
            "air_share_mae": 0.04,
            "radiation_proxy_relative_error_median": 0.22,
        },
    }


class TestStage52CsvWriter(unittest.TestCase):
    def test_extended_audio_fields_do_not_crash_csv_writer(self) -> None:
        comparison = _fake_loo_comparison()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.csv"
            _write_comparison_csv(path, comparison)
            self.assertTrue(path.is_file())
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), len(comparison["per_mode_matches"]))
            self.assertIn("rom_radiation_proxy", reader.fieldnames or [])
            self.assertIn("fom_top_share", reader.fieldnames or [])
            self.assertIn("rom_radiation_proxy_p95_norm", reader.fieldnames or [])


class TestStage52Report(unittest.TestCase):
    def test_valid_loo_filter(self) -> None:
        comp = _fake_loo_comparison()
        self.assertTrue(_is_valid_loo(comp))
        comp_bad = dict(comp, training_includes_target=True)
        self.assertFalse(_is_valid_loo(comp_bad))

    def test_aggregate_and_verdicts(self) -> None:
        comps = [_fake_loo_comparison("sample_010"), _fake_loo_comparison("sample_020")]
        agg = aggregate_metrics(comps)
        self.assertEqual(agg["sample_count"], 2)
        verdicts = assess_verdicts(agg)
        self.assertIn("overall_rom_readiness", verdicts)
        self.assertIn("answers", verdicts)

    def test_build_report_with_fixture_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            cmp_dir = repo / "ROM" / "classic" / "comparisons"
            cmp_dir.mkdir(parents=True)
            comp = _fake_loo_comparison()
            out_path = cmp_dir / "sample_010__sample_010_rom_shadow_v1_rom_fom_comparison.json"
            out_path.write_text(json.dumps(comp), encoding="utf-8")
            (repo / "ROM" / "classic").mkdir(parents=True, exist_ok=True)
            (repo / "ROM" / "classic" / "lhs_pool.json").write_text(
                json.dumps({"shape_name": "classic", "entries": [{"id": "sample_010", "status": "COMPLETED"}]}),
                encoding="utf-8",
            )
            report = build_stage52_report(
                repo,
                out_json=repo / "audio" / "debug_reports" / "stage52_rom_audio_proxy_report.json",
                out_md=repo / "audio" / "debug_reports" / "stage52_rom_audio_proxy_report.md",
            )
            self.assertEqual(report["loo_validation"]["valid_sample_count"], 1)
            self.assertFalse(report["production_surrogate_overwritten"])
            self.assertFalse(report["fem_launched"])


class TestStage52Candidates(unittest.TestCase):
    def test_candidate_a_runs_on_synthetic_modes(self) -> None:
        modes = [
            {"frequency_hz": 100.0, "radiation_proxy": 1e-4},
            {"frequency_hz": 200.0, "radiation_proxy": 5e-4},
        ]
        neighbors = [
            [
                {"frequency_hz": 99.0, "radiation_proxy": 1.2e-4},
                {"frequency_hz": 201.0, "radiation_proxy": 4.5e-4},
            ]
        ]
        out = apply_audio_proxy_candidate(
            CANDIDATE_A,
            modes,
            neighbor_catalogs=neighbors,
            neighbor_weights=[1.0],
        )
        self.assertEqual(len(out), 2)
        self.assertIsNotNone(out[0].get("radiation_proxy"))

    def test_diagnosis_and_candidate_sweep(self) -> None:
        comp = _fake_loo_comparison()
        diag = diagnose_audio_proxy_weakness([comp])
        self.assertIn("likely_contributors", diag)
        sweep = evaluate_candidates_on_comparisons([comp])
        self.assertIn("candidates", sweep)
        self.assertIn(CANDIDATE_A, sweep["candidates"])


if __name__ == "__main__":
    unittest.main()
