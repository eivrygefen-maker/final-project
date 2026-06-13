#!/usr/bin/env python3
"""Lightweight STK V6.2.2 onset/tail repair tests (no FEM/ROM)."""
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
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402
from stk_v6_2_2_onset_tail_repair import (  # noqa: E402
    V6_2_2_VARIANTS,
    compute_onset_diagnostics,
    compute_tail_continuity_diagnostics,
    compute_thump_diagnostics,
    compute_v622_diagnostics,
    synthesize_v6_2_2_onset_tail_repair,
)
from stk_v6_2_physical_routing import (  # noqa: E402
    DEFAULT_DURATION_S,
    load_reference_modal_from_audit,
    synthesize_v6_2_physical_routing,
)

V621_MODE = "stk_v6_2_1_soft_pluck_tail_alpha"
SAMPLE_RATE = 44100
NOTE_HZ = 440.0


def _ctx():
    audit = load_audit_report()
    samples = load_lhs_sample_entries(REPO, max_samples=26)
    sample = next(s for s in samples if s["sample_id"] == "sample_000")
    modal = load_reference_modal_from_audit(audit, REPO)
    params = normalize_sample_parameters(sample.get("parameters"))
    return audit, modal, params


class TestStkV622OnsetTailRepair(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_v622_modes_registered(self) -> None:
        modes = list_diagnostic_modes()
        for m in V6_2_2_VARIANTS:
            self.assertIn(m, modes)
            self.assertNotEqual(DEFAULT_WEBSITE_STK_MODE, m)

    def test_diagnostics_fields(self) -> None:
        audit, modal, params = _ctx()
        _, final, _ = synthesize_v6_2_2_onset_tail_repair(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
            variant="stk_v6_2_2_single_onset_soft_tail_alpha",
        )
        d = compute_v622_diagnostics(
            final, sample_rate=SAMPLE_RATE, frequency_hz=NOTE_HZ, duration_s=DEFAULT_DURATION_S
        )
        for key in (
            "onset_peak_count_0_250ms",
            "second_onset_ratio",
            "thump_index_0_300ms",
            "tail_continuity_ratio",
            "double_pluck_risk_score",
        ):
            self.assertIn(key, d)
        self.assertIn("onset_coherence_pass", compute_onset_diagnostics(final, sample_rate=SAMPLE_RATE))
        self.assertIn("thump_index_0_300ms", compute_thump_diagnostics(final, sample_rate=SAMPLE_RATE))
        self.assertIn(
            "tail_continuity_ratio",
            compute_tail_continuity_diagnostics(final, sample_rate=SAMPLE_RATE, duration_s=DEFAULT_DURATION_S),
        )

    def test_duration_approx_2p5s(self) -> None:
        audit, modal, params = _ctx()
        _, final, _ = synthesize_v6_2_2_onset_tail_repair(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
            variant="stk_v6_2_2_no_thump_body_tail_alpha",
        )
        self.assertAlmostEqual(len(final) / SAMPLE_RATE, DEFAULT_DURATION_S, places=2)

    def test_no_clipping(self) -> None:
        audit, modal, params = _ctx()
        for mode in V6_2_2_VARIANTS:
            _, final, meta = synthesize_v6_2_2_onset_tail_repair(
                frequency_hz=NOTE_HZ,
                duration_s=DEFAULT_DURATION_S,
                sample_rate=SAMPLE_RATE,
                modal_data=modal,
                sample_parameters=params,
                audit=audit,
                repo_root=REPO,
                variant=mode,
            )
            self.assertTrue(meta.get("clipping_avoided"))
            self.assertLessEqual(float(np.max(np.abs(final))), 1.0)

    def test_improves_vs_v621_soft_pluck(self) -> None:
        audit, modal, params = _ctx()
        _, final_v621, _ = synthesize_v6_2_physical_routing(
            frequency_hz=NOTE_HZ,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=SAMPLE_RATE,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            repo_root=REPO,
            variant=V621_MODE,
        )
        d621 = compute_v622_diagnostics(
            final_v621, sample_rate=SAMPLE_RATE, frequency_hz=NOTE_HZ, duration_s=DEFAULT_DURATION_S
        )
        improved_onset = False
        improved_thump = False
        improved_tail = False
        for mode in V6_2_2_VARIANTS:
            _, final, _ = synthesize_v6_2_2_onset_tail_repair(
                frequency_hz=NOTE_HZ,
                duration_s=DEFAULT_DURATION_S,
                sample_rate=SAMPLE_RATE,
                modal_data=modal,
                sample_parameters=params,
                audit=audit,
                repo_root=REPO,
                variant=mode,
            )
            d = compute_v622_diagnostics(
                final, sample_rate=SAMPLE_RATE, frequency_hz=NOTE_HZ, duration_s=DEFAULT_DURATION_S
            )
            if float(d.get("second_onset_ratio") or 1.0) < float(d621.get("second_onset_ratio") or 0.0):
                improved_onset = True
            if float(d.get("thump_index_0_300ms") or 1e9) < float(d621.get("thump_index_0_300ms") or 0.0):
                improved_thump = True
            if float(d.get("tail_continuity_ratio") or 0.0) > float(d621.get("tail_continuity_ratio") or 0.0):
                improved_tail = True
        self.assertTrue(improved_onset, "no variant improved second_onset_ratio vs V6.2.1")
        self.assertTrue(improved_thump, "no variant improved thump_index vs V6.2.1")
        self.assertTrue(improved_tail, "no variant improved tail_continuity_ratio vs V6.2.1")

    def test_review_pack_max_10_wavs(self) -> None:
        from build_stk_v6_2_2_review_audio import MAX_REVIEW_WAVS, build_stk_v6_2_2_review

        with tempfile.TemporaryDirectory() as td:
            review = Path(td) / "review"
            report = build_stk_v6_2_2_review(
                repo_root=REPO,
                review_dir=review,
                v621_source_dir=Path(td) / "empty",
                duration_s=DEFAULT_DURATION_S,
            )
            self.assertLessEqual(report["review_wav_count"], MAX_REVIEW_WAVS)
            self.assertEqual(report["review_wav_count"], len(list(review.glob("*.wav"))))

    def test_required_review_wavs_exist(self) -> None:
        from build_stk_v6_2_2_review_audio import REVIEW_WAV_SPEC, build_stk_v6_2_2_review

        with tempfile.TemporaryDirectory() as td:
            review = Path(td) / "review"
            build_stk_v6_2_2_review(
                repo_root=REPO,
                review_dir=review,
                v621_source_dir=Path(td) / "empty",
                duration_s=DEFAULT_DURATION_S,
            )
            for fname, _ in REVIEW_WAV_SPEC:
                self.assertTrue((review / fname).is_file(), f"missing {fname}")

    def test_no_fem_rom_in_report(self) -> None:
        from build_stk_v6_2_2_review_audio import build_stk_v6_2_2_review

        with tempfile.TemporaryDirectory() as td:
            report = build_stk_v6_2_2_review(
                repo_root=REPO,
                review_dir=Path(td) / "review",
                v621_source_dir=Path(td) / "empty",
                duration_s=0.35,
            )
            self.assertTrue(report.get("no_fem_run"))
            self.assertTrue(report.get("no_rom_run"))


if __name__ == "__main__":
    unittest.main()
