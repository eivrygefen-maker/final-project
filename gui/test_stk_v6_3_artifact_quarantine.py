#!/usr/bin/env python3
"""STK V6.3 artifact quarantine tests (no FEM/ROM)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from build_sample_comparison import load_lhs_sample_entries  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import DEFAULT_DURATION_S, load_reference_modal_from_audit  # noqa: E402
from stk_v6_3_artifact_quarantine import (  # noqa: E402
    ARTIFACT_QUARANTINE,
    REJECTED_V622_MODES,
    V6_3_MODE,
    scan_artifacts,
    synthesize_v6_3_clean_pluck_body,
)

SAMPLE_RATE = 44100
NOTE_HZ = 440.0


def _ctx():
    audit = load_audit_report()
    samples = load_lhs_sample_entries(REPO, max_samples=26)
    sample = next(s for s in samples if s["sample_id"] == "sample_000")
    from sample_parameters import normalize_sample_parameters

    modal = load_reference_modal_from_audit(audit, REPO)
    params = normalize_sample_parameters(sample.get("parameters"))
    return audit, modal, params


class TestStkV63ArtifactQuarantine(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_rejected_modes_listed(self) -> None:
        q = ARTIFACT_QUARANTINE
        self.assertIn("rejected_modes", q)
        self.assertIn("reason", q)
        self.assertEqual(q.get("allowed_future_use"), "baseline_only")
        for mode in REJECTED_V622_MODES:
            self.assertIn(mode, q["rejected_modes"])
            self.assertIn(mode, q["reason"])

    def test_v622_not_in_recommendations(self) -> None:
        from build_stk_v6_3_review_audio import build_v63_quarantine_report

        with tempfile.TemporaryDirectory() as td:
            report = build_v63_quarantine_report(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v622_review_dir=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=0.35,
            )
            self.assertIsNone(report.get("recommended_candidate"))
            for mode in REJECTED_V622_MODES:
                self.assertIn(mode, report.get("do_not_recommend_modes") or [])

    def test_artifact_metrics_exist(self) -> None:
        audit, modal, params = _ctx()
        _, final, _, _ = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        d = scan_artifacts(final, sample_rate=SAMPLE_RATE, duration_s=DEFAULT_DURATION_S)
        for key in (
            "onset_peak_count_0_250ms",
            "second_onset_ratio",
            "thump_to_body_ratio",
            "tail_continuity_ratio",
            "end_click_or_gate_fail",
            "body_tail_peak_count_80_350ms",
        ):
            self.assertIn(key, d)

    def test_clean_candidate_generated(self) -> None:
        self.assertIn(V6_3_MODE, list_diagnostic_modes())
        audit, modal, params = _ctx()
        stems, final, pre, meta = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        self.assertIn("pluck_stem", stems)
        self.assertIn("body_tail_stem", stems)
        self.assertEqual(meta.get("diagnostic_mode"), V6_3_MODE)
        self.assertTrue(meta.get("design", {}).get("no_helmholtz_resonator"))
        self.assertGreater(len(pre), 0)

    def test_no_clipping(self) -> None:
        audit, modal, params = _ctx()
        _, final, _, meta = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        self.assertTrue(meta.get("clipping_avoided"))
        self.assertLessEqual(float(np.max(np.abs(final))), 1.0)

    def test_final_fade_out(self) -> None:
        audit, modal, params = _ctx()
        _, final, _, _ = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        d = scan_artifacts(final, sample_rate=SAMPLE_RATE, duration_s=DEFAULT_DURATION_S)
        self.assertTrue(d.get("final_100ms_fade_to_zero_check"))

    def test_body_tail_no_strong_pulse(self) -> None:
        audit, modal, params = _ctx()
        stems, _, _, _ = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        d = scan_artifacts(
            stems["body_tail_stem"],
            sample_rate=SAMPLE_RATE,
            duration_s=DEFAULT_DURATION_S,
            is_body_tail_stem=True,
        )
        self.assertFalse(d.get("delayed_body_event_fail"))
        self.assertLessEqual(int(d.get("body_tail_peak_count_80_350ms") or 0), 1)

    def test_no_double_onset_threshold(self) -> None:
        audit, modal, params = _ctx()
        _, final, _, _ = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        d = scan_artifacts(final, sample_rate=SAMPLE_RATE, duration_s=DEFAULT_DURATION_S)
        self.assertFalse(d.get("double_pluck_fail"))

    def test_duration_approx_2p5s(self) -> None:
        audit, modal, params = _ctx()
        _, final, _, _ = synthesize_v6_3_clean_pluck_body(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
        )
        self.assertAlmostEqual(len(final) / SAMPLE_RATE, DEFAULT_DURATION_S, places=2)

    def test_review_pack_max_8(self) -> None:
        from build_stk_v6_3_review_audio import MAX_REVIEW_WAVS, build_v63_quarantine_report

        with tempfile.TemporaryDirectory() as td:
            report = build_v63_quarantine_report(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v622_review_dir=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=DEFAULT_DURATION_S,
            )
            self.assertLessEqual(report["review_wav_count"], MAX_REVIEW_WAVS)

    def test_no_fem_rom(self) -> None:
        from build_stk_v6_3_review_audio import build_v63_quarantine_report

        with tempfile.TemporaryDirectory() as td:
            report = build_v63_quarantine_report(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v622_review_dir=REPO / "audio" / "stk_v6_2_2_review_audio",
                duration_s=0.35,
            )
            self.assertTrue(report.get("no_fem_run"))
            self.assertTrue(report.get("no_rom_run"))


if __name__ == "__main__":
    unittest.main()
