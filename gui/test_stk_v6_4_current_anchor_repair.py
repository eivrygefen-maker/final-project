#!/usr/bin/env python3
"""STK V6.4 current-anchor repair tests (no FEM/ROM)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import DEFAULT_DURATION_S, load_reference_modal_from_audit  # noqa: E402
from stk_v6_3_artifact_quarantine import ARTIFACT_QUARANTINE, V6_3_MODE  # noqa: E402
from stk_v6_4_current_anchor_repair import (  # noqa: E402
    SOUND_BASE_REJECTED,
    V6_4_MODES,
    compute_v64_metrics,
    render_current_final_v1_anchor,
    repair_current_anchor,
)
from build_sample_comparison import load_lhs_sample_entries  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402


def _ctx():
    audit = load_audit_report()
    modal = load_reference_modal_from_audit(audit, REPO)
    sample = next(s for s in load_lhs_sample_entries(REPO) if s["sample_id"] == "sample_000")
    params = normalize_sample_parameters(sample.get("parameters"))
    return audit, modal, params


class TestStkV64CurrentAnchorRepair(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_v64_modes_registered(self) -> None:
        modes = list_diagnostic_modes()
        for m in V6_4_MODES:
            self.assertIn(m, modes)

    def test_quarantined_not_recommended(self) -> None:
        from build_stk_v6_4_review_audio import build_v64_review

        with tempfile.TemporaryDirectory() as td:
            report = build_v64_review(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v63_review=REPO / "audio" / "stk_v6_3_review_audio",
                v622_review=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=0.35,
            )
            for mode in ARTIFACT_QUARANTINE["rejected_modes"]:
                self.assertIn(mode, report.get("do_not_recommend_modes") or [])
            self.assertIn(V6_3_MODE, report.get("do_not_recommend_modes") or [])

    def test_no_helmholtz_or_body_tail_stem(self) -> None:
        audit, modal, params = _ctx()
        anchor, sr, _ = render_current_final_v1_anchor(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.35,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            repo_root=REPO,
        )
        for mode in V6_4_MODES:
            _, _, _, meta = repair_current_anchor(anchor, sample_rate=sr, duration_s=0.35, variant=mode)
            self.assertFalse(meta.get("uses_helmholtz_ir"))
            self.assertFalse(meta.get("uses_delayed_body_ramp"))
            self.assertFalse(meta.get("uses_independent_body_tail_stem"))

    def test_thump_not_worse_than_anchor(self) -> None:
        audit, modal, params = _ctx()
        anchor, sr, _ = render_current_final_v1_anchor(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=DEFAULT_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            repo_root=REPO,
        )
        am = compute_v64_metrics(anchor, sample_rate=sr, duration_s=DEFAULT_DURATION_S)
        improved = False
        for mode in V6_4_MODES:
            rep, _, _, _ = repair_current_anchor(
                anchor, sample_rate=sr, duration_s=DEFAULT_DURATION_S, variant=mode
            )
            cm = compute_v64_metrics(rep, sample_rate=sr, duration_s=DEFAULT_DURATION_S, anchor=anchor)
            if float(cm.get("drum_tap_risk_score") or 1.0) <= float(am.get("drum_tap_risk_score") or 1.0):
                improved = True
        self.assertTrue(improved)

    def test_no_clipping_and_fade(self) -> None:
        audit, modal, params = _ctx()
        anchor, sr, _ = render_current_final_v1_anchor(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=DEFAULT_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            repo_root=REPO,
        )
        for mode in V6_4_MODES:
            rep, _, _, meta = repair_current_anchor(
                anchor, sample_rate=sr, duration_s=DEFAULT_DURATION_S, variant=mode
            )
            self.assertTrue(meta.get("clipping_avoided"))
            self.assertLessEqual(float(np.max(np.abs(rep))), 1.0)
            d = compute_v64_metrics(rep, sample_rate=sr, duration_s=DEFAULT_DURATION_S, anchor=anchor)
            self.assertTrue(d.get("final_200ms_fade_ok"))

    def test_duration_approx_2p5s(self) -> None:
        audit, modal, params = _ctx()
        anchor, sr, _ = render_current_final_v1_anchor(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=DEFAULT_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            repo_root=REPO,
        )
        rep, _, _, _ = repair_current_anchor(
            anchor, sample_rate=sr, duration_s=DEFAULT_DURATION_S,
            variant="stk_v6_4_current_anchor_soft_attack_alpha",
        )
        self.assertAlmostEqual(len(rep) / sr, DEFAULT_DURATION_S, places=2)

    def test_review_pack_max_8(self) -> None:
        from build_stk_v6_4_review_audio import MAX_WAVS, build_v64_review

        with tempfile.TemporaryDirectory() as td:
            report = build_v64_review(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v63_review=REPO / "audio" / "stk_v6_3_review_audio",
                v622_review=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=DEFAULT_DURATION_S,
            )
            self.assertLessEqual(report["review_wav_count"], MAX_WAVS)

    def test_required_files_exist(self) -> None:
        from build_stk_v6_4_review_audio import build_v64_review

        with tempfile.TemporaryDirectory() as td:
            review = Path(td) / "review"
            build_v64_review(
                repo_root=REPO,
                review_dir=review,
                v63_review=REPO / "audio" / "stk_v6_3_review_audio",
                v622_review=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=DEFAULT_DURATION_S,
            )
            required = [
                "current_final_v1_A4_sample_000.wav",
                "v5_alpha_s20_b80_A4_sample_000.wav",
                "stk_v6_4_current_anchor_soft_attack_alpha_A4_sample_000.wav",
                "stk_v6_4_current_anchor_sustain_smooth_alpha_A4_sample_000.wav",
                "stk_v6_4_current_anchor_soft_attack_alpha_A4_attack_window_debug.wav",
                "stk_v6_4_current_anchor_sustain_smooth_alpha_A4_tail_window_debug.wav",
            ]
            for name in required:
                self.assertTrue((review / name).is_file(), f"missing {name}")

    def test_no_fem_rom(self) -> None:
        from build_stk_v6_4_review_audio import build_v64_review

        with tempfile.TemporaryDirectory() as td:
            report = build_v64_review(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v63_review=REPO / "audio" / "stk_v6_3_review_audio",
                v622_review=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=0.35,
            )
            self.assertTrue(report.get("no_fem_run"))
            self.assertTrue(report.get("no_rom_run"))

    def test_sound_base_rejected_documented(self) -> None:
        self.assertIn(V6_3_MODE, SOUND_BASE_REJECTED["rejected_as_sound_base"])


if __name__ == "__main__":
    unittest.main()
