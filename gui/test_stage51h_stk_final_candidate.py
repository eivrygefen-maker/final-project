#!/usr/bin/env python3
"""Stage 5.1H STK final candidate freeze tests."""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    STK_FINAL_CANDIDATE_CANONICAL,
    STK_FINAL_DE_THUMP_CANONICAL,
    STK_FINAL_GUI_LABEL,
    apply_residual_de_thump,
    canonical_stk_final_mode,
    g_config_for_mode,
    is_de_thump_mode,
    is_final_stk_candidate_mode,
    is_g_identity_mode,
    is_v4_1_identity_space_mode,
    requires_identity_contrast_context,
    synthesize_v4_1_identity_space_note,
)
from transient_thump_diagnostics import analyze_transient_thump  # noqa: E402
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from build_stk_final_candidate_diagnostics import (  # noqa: E402
    _check_alias_equivalence,
    build_stk_final_candidate_diagnostics,
)
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402


class TestStage51HStkFinalCandidate(unittest.TestCase):
    def test_final_alias_registered(self) -> None:
        self.assertIn(STK_BODY_TRANSFER_FINAL_V1, list_diagnostic_modes())
        self.assertTrue(is_v4_1_identity_space_mode(STK_BODY_TRANSFER_FINAL_V1))
        self.assertTrue(is_g_identity_mode(STK_BODY_TRANSFER_FINAL_V1))
        self.assertTrue(is_final_stk_candidate_mode(STK_BODY_TRANSFER_FINAL_V1))
        self.assertTrue(requires_identity_contrast_context(STK_BODY_TRANSFER_FINAL_V1))
        cfg = get_diagnostic_mode(STK_BODY_TRANSFER_FINAL_V1)
        self.assertIn("final STK candidate", cfg.description)

    def test_de_thump_modes_registered(self) -> None:
        for mode in (STK_FINAL_DE_THUMP_CANONICAL, STK_BODY_TRANSFER_FINAL_V1_DE_THUMP):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_de_thump_mode(mode))
            self.assertTrue(g_config_for_mode(mode)["de_thump_active"])
        self.assertEqual(
            canonical_stk_final_mode(STK_BODY_TRANSFER_FINAL_V1_DE_THUMP),
            STK_FINAL_DE_THUMP_CANONICAL,
        )

    def test_de_thump_reduces_onset(self) -> None:
        sr = 44100
        t = np.arange(int(0.2 * sr)) / sr
        residual = 0.02 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 8)
        out, meta = apply_residual_de_thump(residual, sample_rate=sr)
        self.assertTrue(meta["de_thump_active"])
        from transient_thump_diagnostics import _band_energy, _slice_ms

        e_in = _band_energy(_slice_ms(residual, sr, 0, 100), sr, 0, 120)
        e_out = _band_energy(_slice_ms(out, sr, 0, 100), sr, 0, 120)
        self.assertLess(e_out, e_in)

    def test_transient_thump_analysis(self) -> None:
        sr = 44100
        n = int(0.5 * sr)
        t = np.arange(n) / sr
        audio = 0.05 * np.sin(2 * np.pi * 110 * t) * np.exp(-t * 3)
        m = analyze_transient_thump(audio, sample_rate=sr, frequency_hz=110.0)
        self.assertTrue(m["valid"])
        self.assertIn("low_thump_ratio", m)
        self.assertIn("transient_peak_dbfs", m)

    def test_alias_maps_to_g30_70(self) -> None:
        self.assertEqual(canonical_stk_final_mode(STK_BODY_TRANSFER_FINAL_V1), STK_FINAL_CANDIDATE_CANONICAL)
        g_alias = g_config_for_mode(STK_BODY_TRANSFER_FINAL_V1)
        g_canon = g_config_for_mode(STK_FINAL_CANDIDATE_CANONICAL)
        self.assertEqual(g_alias["absolute_weight"], 0.30)
        self.assertEqual(g_alias["contrast_weight"], 0.70)
        self.assertFalse(g_alias["decay_active"])
        self.assertFalse(g_alias["bridge_active"])
        self.assertEqual(g_canon["absolute_weight"], 0.30)
        self.assertEqual(g_canon["contrast_weight"], 0.70)

    def test_alias_equivalence_helper(self) -> None:
        a = np.ones(1000) * 0.1
        eq = _check_alias_equivalence(a, a.copy())
        self.assertTrue(eq["passed"])
        b = a.copy()
        b[0] += 1e-8
        eq2 = _check_alias_equivalence(a, b)
        self.assertTrue(eq2["passed"])

    def test_synthesis_final_differs_from_de_thump(self) -> None:
        modal = synthetic_modal_for_sample("sample_004")
        params = {"top_wood_id": "spruce", "back_wood_id": "maple"}
        out_std = REPO / "audio" / "_test_final_std.wav"
        out_de = REPO / "audio" / "_test_final_de.wav"
        for p in (out_std, out_de):
            if p.exists():
                p.unlink()
        kwargs = dict(
            frequency_hz=220.0,
            note_name="A3",
            duration_s=0.2,
            sample_rate=44100,
            modal_data=modal,
            velocity=0.8,
            sample_parameters=params,
            modal_source="synthetic",
            synthesis_preset=None,
            repo_root=REPO,
            sample_id="sample_004",
            output_metadata_json=None,
        )
        synthesize_v4_1_identity_space_note(**kwargs, output_wav=out_std, diagnostic_mode=STK_FINAL_CANDIDATE_CANONICAL)
        synthesize_v4_1_identity_space_note(**kwargs, output_wav=out_de, diagnostic_mode=STK_FINAL_DE_THUMP_CANONICAL)
        from build_sample_comparison import read_wav_float_mono

        a, _ = read_wav_float_mono(out_std)
        b, _ = read_wav_float_mono(out_de)
        self.assertFalse(np.allclose(a, b))

    def test_synthesis_alias_identical(self) -> None:
        modal = synthetic_modal_for_sample("sample_002")
        params = {"top_wood_id": "spruce", "back_wood_id": "mahogany"}
        out_canon = REPO / "audio" / "_test_final_canon.wav"
        out_alias = REPO / "audio" / "_test_final_alias.wav"
        for p in (out_canon, out_alias):
            if p.exists():
                p.unlink()
        kwargs = dict(
            frequency_hz=220.0,
            note_name="A3",
            duration_s=0.15,
            sample_rate=44100,
            modal_data=modal,
            velocity=0.8,
            sample_parameters=params,
            modal_source="synthetic",
            synthesis_preset=None,
            repo_root=REPO,
            sample_id="sample_002",
            output_metadata_json=None,
        )
        synthesize_v4_1_identity_space_note(
            **kwargs,
            output_wav=out_canon,
            diagnostic_mode=STK_FINAL_CANDIDATE_CANONICAL,
        )
        synthesize_v4_1_identity_space_note(
            **kwargs,
            output_wav=out_alias,
            diagnostic_mode=STK_BODY_TRANSFER_FINAL_V1,
        )
        from build_sample_comparison import read_wav_float_mono

        ac, _ = read_wav_float_mono(out_canon)
        aa, _ = read_wav_float_mono(out_alias)
        eq = _check_alias_equivalence(ac, aa)
        self.assertTrue(eq["passed"], msg=str(eq))

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A2", "A3", "A4", "E5", "D4"):
            self.assertNotIn(f'"{token}"', src)

    def test_gui_label_constant(self) -> None:
        self.assertEqual(STK_FINAL_GUI_LABEL, "Physical Body Identity v1")

    def test_de_thump_policy_keeps_final_v1_by_default(self) -> None:
        from transient_thump_diagnostics import build_de_thump_decision_policy, compare_final_vs_de_thump

        per_note = {
            "A3": {
                "modes": {
                    "modal_body_hybrid_v4_1_identity_contrast_g_30_70": {
                        "spectral_differentiation": 0.076,
                        "rms_diff_db_vs_v41_median": -30.0,
                        "likely_audible_vs_v41": True,
                    },
                    "modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump": {
                        "spectral_differentiation": 0.075,
                        "rms_diff_db_vs_v41_median": -31.0,
                        "likely_audible_vs_v41": True,
                    },
                }
            }
        }
        per_note_transient = {
            "A3": {
                "modal_body_hybrid_v4_1_full": {"low_thump_ratio_median": 0.14, "segment_count": 10},
                "modal_body_hybrid_v4_1_identity_contrast_g_30_70": {
                    "low_thump_ratio_median": 0.15,
                    "segment_count": 10,
                    "transient_dominated_count": 2,
                },
                "modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump": {
                    "low_thump_ratio_median": 0.13,
                    "segment_count": 10,
                },
            }
        }
        cmp = compare_final_vs_de_thump(
            per_note=per_note,
            per_note_transient=per_note_transient,
            notes=["A3"],
            final_mode="modal_body_hybrid_v4_1_identity_contrast_g_30_70",
            de_thump_mode="modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump",
            v41_mode="modal_body_hybrid_v4_1_full",
        )
        policy = build_de_thump_decision_policy(
            per_note=per_note,
            per_note_transient=per_note_transient,
            notes=["A3"],
            final_mode="modal_body_hybrid_v4_1_identity_contrast_g_30_70",
            de_thump_mode="modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump",
            v41_mode="modal_body_hybrid_v4_1_full",
            comparison=cmp,
            vm_validated=False,
        )
        self.assertEqual(policy["decision"], "keep_final_v1_default")
        self.assertEqual(policy["website_default_mode"], STK_BODY_TRANSFER_FINAL_V1)

    def test_de_thump_policy_switch_on_listening(self) -> None:
        from transient_thump_diagnostics import build_de_thump_decision_policy, compare_final_vs_de_thump

        per_note = {
            "A3": {
                "modes": {
                    STK_FINAL_CANDIDATE_CANONICAL: {
                        "spectral_differentiation": 0.076,
                        "likely_audible_vs_v41": True,
                    },
                    STK_FINAL_DE_THUMP_CANONICAL: {
                        "spectral_differentiation": 0.075,
                        "likely_audible_vs_v41": True,
                    },
                }
            }
        }
        per_note_transient = {
            "A3": {
                "modal_body_hybrid_v4_1_full": {"low_thump_ratio_median": 0.14, "segment_count": 10},
                STK_FINAL_CANDIDATE_CANONICAL: {
                    "low_thump_ratio_median": 0.20,
                    "segment_count": 10,
                    "transient_dominated_count": 8,
                },
                STK_FINAL_DE_THUMP_CANONICAL: {"low_thump_ratio_median": 0.12, "segment_count": 10},
            }
        }
        cmp = compare_final_vs_de_thump(
            per_note=per_note,
            per_note_transient=per_note_transient,
            notes=["A3"],
            final_mode=STK_FINAL_CANDIDATE_CANONICAL,
            de_thump_mode=STK_FINAL_DE_THUMP_CANONICAL,
            v41_mode="modal_body_hybrid_v4_1_full",
        )
        policy = build_de_thump_decision_policy(
            per_note=per_note,
            per_note_transient=per_note_transient,
            notes=["A3"],
            final_mode=STK_FINAL_CANDIDATE_CANONICAL,
            de_thump_mode=STK_FINAL_DE_THUMP_CANONICAL,
            v41_mode="modal_body_hybrid_v4_1_full",
            comparison=cmp,
            listening_review={"heard_drum_thump": True, "de_thump_sounds_more_natural": True},
            vm_validated=False,
        )
        self.assertEqual(policy["decision"], "switch_default_to_de_thump")
        self.assertEqual(policy["website_default_mode"], STK_BODY_TRANSFER_FINAL_V1_DE_THUMP)

    def test_build_no_fem(self) -> None:
        out = REPO / "audio" / "_test_stk_final"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_final_candidate_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_stk_final_candidate_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    V41 := "modal_body_hybrid_v4_1_full",
                    STK_FINAL_CANDIDATE_CANONICAL,
                    STK_BODY_TRANSFER_FINAL_V1,
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        self.assertTrue(manifest["alias_equivalence_all_passed"])
        report_path = REPO / "audio" / "debug_reports" / "stage51h_stk_final_candidate_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1H")
        policy = report.get("de_thump_decision_policy") or {}
        self.assertEqual(policy.get("decision"), "keep_final_v1_default")
        self.assertIn(report.get("freeze_decision", {}).get("decision"), (
            "freeze_final_stk_candidate",
            "freeze_with_warning",
            "do_not_freeze",
        ))


if __name__ == "__main__":
    unittest.main()
