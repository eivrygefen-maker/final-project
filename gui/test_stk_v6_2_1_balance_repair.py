#!/usr/bin/env python3
"""Lightweight STK V6.2.1 balance repair tests (no FEM/ROM batch)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from build_sample_comparison import load_lhs_sample_entries  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import (  # noqa: E402
    DEFAULT_DURATION_S,
    STK_V6_2_MODE,
    V6_2_1_VARIANTS,
    compute_balance_diagnostics,
    load_reference_modal_from_audit,
    synthesize_v6_2_physical_routing,
)

TEST_DURATION_S = 2.5
SAMPLE_RATE = 44100


def _load_sample_context():
    audit = load_audit_report()
    samples = load_lhs_sample_entries(REPO, max_samples=26)
    sample = next(s for s in samples if s["sample_id"] == "sample_000")
    modal = load_reference_modal_from_audit(audit, REPO)
    params = normalize_sample_parameters(sample.get("parameters"))
    return audit, modal, params


class TestStkV621BalanceRepair(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_v621_modes_registered_diagnostic_only(self) -> None:
        modes = list_diagnostic_modes()
        for mode in V6_2_1_VARIANTS:
            self.assertIn(mode, modes)
            self.assertNotEqual(DEFAULT_WEBSITE_STK_MODE, mode)
            cfg = get_diagnostic_mode(mode)
            self.assertIn("V6.2.1", cfg.description)

    def test_v62_original_still_exists(self) -> None:
        self.assertIn(STK_V6_2_MODE, list_diagnostic_modes())

    def test_no_fem_rom_calls_in_builder(self) -> None:
        from build_stk_v6_2_1_diagnostic_audio import build_stk_v6_2_1_diagnostics

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            with patch("build_stk_v6_2_1_diagnostic_audio.synthesize_mode_to_wav") as mock_syn:
                mock_syn.return_value = None
                report = build_stk_v6_2_1_diagnostics(
                    repo_root=REPO,
                    out_dir=out,
                    notes=[("A4", 440.0)],
                    duration_s=0.35,
                )
            self.assertTrue(report.get("no_fem_run"))
            self.assertTrue(report.get("no_rom_run"))

    def test_v621_stems_generated(self) -> None:
        audit, modal, params = _load_sample_context()
        mode = "stk_v6_2_1_balanced_tail_alpha"
        stems, final, meta = synthesize_v6_2_physical_routing(
            frequency_hz=440.0,
            duration_s=TEST_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant=mode,
        )
        for name in (
            "pluck_attack_stem",
            "direct_string_short_stem",
            "top_radiation_stem",
            "soundhole_air_stem",
            "cavity_body_tail_stem",
            "final_mix",
        ):
            self.assertIn(name, stems)
        self.assertEqual(meta.get("diagnostic_mode"), mode)
        self.assertEqual(meta.get("norm_method"), "sustain_window_rms")

    def test_final_mix_duration_2p5s(self) -> None:
        audit, modal, params = _load_sample_context()
        _, final, _ = synthesize_v6_2_physical_routing(
            frequency_hz=659.25,
            duration_s=TEST_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant="stk_v6_2_1_soft_pluck_tail_alpha",
        )
        self.assertAlmostEqual(len(final) / SAMPLE_RATE, TEST_DURATION_S, places=2)

    def test_v621_improves_tail_vs_v62(self) -> None:
        audit, modal, params = _load_sample_context()
        notes = [("A4", 440.0), ("E5", 659.25)]
        for _, freq in notes:
            _, final_v62, meta_v62 = synthesize_v6_2_physical_routing(
                frequency_hz=freq,
                duration_s=TEST_DURATION_S,
                sample_rate=SAMPLE_RATE,
                modal_data=modal,
                sample_parameters=params,
                audit=audit,
                sample_id="sample_000",
                repo_root=REPO,
                variant=STK_V6_2_MODE,
            )
            bal_v62 = meta_v62.get("balance_diagnostics") or {}
            best_tail = -1.0
            best_ratio = 1e9
            for mode in V6_2_1_VARIANTS:
                _, final_v621, meta_v621 = synthesize_v6_2_physical_routing(
                    frequency_hz=freq,
                    duration_s=TEST_DURATION_S,
                    sample_rate=SAMPLE_RATE,
                    modal_data=modal,
                    sample_parameters=params,
                    audit=audit,
                    sample_id="sample_000",
                    repo_root=REPO,
                    variant=mode,
                )
                bal = meta_v621.get("balance_diagnostics") or {}
                best_tail = max(best_tail, float(bal.get("tail_rms_1_2p5s") or 0.0))
                best_ratio = min(best_ratio, float(bal.get("attack_to_tail_ratio") or 1e9))
                peak = float(np.max(np.abs(final_v621)))
                self.assertLessEqual(peak, 1.0)
            v62_tail = float(bal_v62.get("tail_rms_1_2p5s") or 0.0)
            v62_ratio = float(bal_v62.get("attack_to_tail_ratio") or 1e9)
            self.assertGreater(best_tail, v62_tail)
            self.assertLess(best_ratio, v62_ratio)

    def test_no_clipping(self) -> None:
        audit, modal, params = _load_sample_context()
        for mode in V6_2_1_VARIANTS:
            _, final, meta = synthesize_v6_2_physical_routing(
                frequency_hz=659.25,
                duration_s=TEST_DURATION_S,
                sample_rate=SAMPLE_RATE,
                modal_data=modal,
                sample_parameters=params,
                audit=audit,
                sample_id="sample_000",
                repo_root=REPO,
                variant=mode,
            )
            self.assertTrue(meta.get("clipping_avoided"))
            self.assertLessEqual(float(np.max(np.abs(final))), 1.0)

    def test_e5_hf_damping_active(self) -> None:
        audit, modal, params = _load_sample_context()
        _, final, meta = synthesize_v6_2_physical_routing(
            frequency_hz=659.25,
            duration_s=TEST_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant="stk_v6_2_1_balanced_tail_alpha",
        )
        bal = meta.get("balance_diagnostics") or {}
        self.assertLess(float(bal.get("metallicity_index") or 1.0), 0.15)
        _, final_v62, meta_v62 = synthesize_v6_2_physical_routing(
            frequency_hz=659.25,
            duration_s=TEST_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant=STK_V6_2_MODE,
        )
        bal_v62 = meta_v62.get("balance_diagnostics") or {}
        self.assertLessEqual(
            float(bal.get("metallicity_index") or 0.0),
            float(bal_v62.get("metallicity_index") or 0.0) + 0.02,
        )

    def test_balance_diagnostics_fields(self) -> None:
        audit, modal, params = _load_sample_context()
        _, final, _ = synthesize_v6_2_physical_routing(
            frequency_hz=440.0,
            duration_s=TEST_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant="stk_v6_2_1_more_string_body_alpha",
        )
        bal = compute_balance_diagnostics(
            final,
            sample_rate=SAMPLE_RATE,
            frequency_hz=440.0,
            duration_s=TEST_DURATION_S,
        )
        for key in (
            "attack_rms_0_50ms",
            "body_rms_200_800ms",
            "tail_rms_1_2p5s",
            "attack_to_body_ratio",
            "attack_to_tail_ratio",
            "tail_audibility_score",
            "pluck_click_index",
            "drum_tap_risk_score",
            "sustain_body_presence_score",
        ):
            self.assertIn(key, bal)

    def test_builder_writes_reports(self) -> None:
        from build_stk_v6_2_1_diagnostic_audio import build_stk_v6_2_1_diagnostics

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            report = build_stk_v6_2_1_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A4", 440.0)],
                duration_s=0.35,
            )
            self.assertIn("recommended_variant", report)
            self.assertIn(report["recommended_variant"], V6_2_1_VARIANTS)
            wavs = list(out.glob("*.wav"))
            self.assertGreater(len(wavs), 0)


if __name__ == "__main__":
    unittest.main()
