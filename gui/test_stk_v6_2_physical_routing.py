#!/usr/bin/env python3
"""Lightweight STK V6.2 physical routing tests (no FEM/ROM batch)."""
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
from build_sample_comparison import load_lhs_sample_entries, resolve_modal_data_for_sample  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import get_feature, load_audit_report  # noqa: E402
from stk_v6_2_physical_routing import (  # noqa: E402
    STK_V6_2_MODE,
    DEFAULT_DURATION_S,
    load_reference_modal_from_audit,
    synthesize_v6_2_physical_routing,
)

TEST_DURATION_S = 0.35


class TestStkV62PhysicalRouting(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_diagnostic_mode_registered_not_default(self) -> None:
        self.assertIn(STK_V6_2_MODE, list_diagnostic_modes())
        self.assertNotEqual(DEFAULT_WEBSITE_STK_MODE, STK_V6_2_MODE)
        cfg = get_diagnostic_mode(STK_V6_2_MODE)
        self.assertIn("V6.2", cfg.description)

    def test_get_feature_body_depth(self) -> None:
        audit = load_audit_report()
        from stk_v6_2_audit_features import get_sample_record

        rec = get_feature(get_sample_record(audit, "sample_000"), "body_depth", audit=audit)
        self.assertIsNotNone(rec.get("value"))
        self.assertEqual(rec.get("status"), "available")
        self.assertTrue(rec.get("per_sample"))

    def test_reference_shared_not_per_sample(self) -> None:
        audit = load_audit_report()
        from stk_v6_2_audit_features import get_sample_record

        rec = get_feature(
            get_sample_record(audit, "sample_000"),
            "bridge_to_radiation_strength",
            audit=audit,
        )
        self.assertEqual(rec.get("status"), "reference_shared")
        self.assertFalse(rec.get("per_sample"))

    def test_v62_synth_creates_stems(self) -> None:
        audit = load_audit_report()
        samples = load_lhs_sample_entries(REPO, max_samples=1)
        if not samples:
            self.skipTest("no lhs samples")
        sample = next(s for s in samples if s["sample_id"] == "sample_000")
        modal = load_reference_modal_from_audit(audit, REPO)
        params = normalize_sample_parameters(sample.get("parameters"))
        stems, final, meta = synthesize_v6_2_physical_routing(
            frequency_hz=440.0,
            duration_s=TEST_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        for name in (
            "pluck_attack_stem",
            "direct_string_short_stem",
            "bridge_body_stem",
            "top_radiation_stem",
            "soundhole_air_stem",
            "cavity_body_tail_stem",
        ):
            self.assertIn(name, stems)
            self.assertEqual(len(stems[name]), int(TEST_DURATION_S * 44100))
        self.assertIn("feature_provenance_used", meta)
        self.assertTrue(
            any("multi-guitar" in str(x).lower() for x in meta.get("limitations", []))
        )

    def test_string_not_dominated_in_sustain(self) -> None:
        audit = load_audit_report()
        sample = next(
            s for s in load_lhs_sample_entries(REPO, max_samples=26) if s["sample_id"] == "sample_000"
        )
        modal = load_reference_modal_from_audit(audit, REPO)
        _, _, meta = synthesize_v6_2_physical_routing(
            frequency_hz=440.0,
            duration_s=TEST_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        self.assertLess(meta["string_dominance_ratio_sustain_window"], 0.55)

    def test_cavity_tail_longer_than_pluck(self) -> None:
        audit = load_audit_report()
        sample = next(
            s for s in load_lhs_sample_entries(REPO, max_samples=26) if s["sample_id"] == "sample_000"
        )
        modal = load_reference_modal_from_audit(audit, REPO)
        stems, _, _ = synthesize_v6_2_physical_routing(
            frequency_hz=220.0,
            duration_s=TEST_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        sr = 44100
        pluck_tail = float(np.sqrt(np.mean(stems["pluck_attack_stem"][int(0.2 * sr) :] ** 2)))
        cavity_tail = float(np.sqrt(np.mean(stems["cavity_body_tail_stem"][int(0.2 * sr) :] ** 2)))
        self.assertGreater(cavity_tail, pluck_tail)

    def test_e5_hf_damping_applied(self) -> None:
        audit = load_audit_report()
        sample = next(
            s for s in load_lhs_sample_entries(REPO, max_samples=26) if s["sample_id"] == "sample_000"
        )
        modal = load_reference_modal_from_audit(audit, REPO)
        _, _, meta = synthesize_v6_2_physical_routing(
            frequency_hz=659.25,
            duration_s=TEST_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        self.assertGreater(
            float(meta["pluck_params"].get("pluck_metallic_damping") or 0),
            0.25,
        )

    def test_no_clipping(self) -> None:
        audit = load_audit_report()
        sample = next(
            s for s in load_lhs_sample_entries(REPO, max_samples=26) if s["sample_id"] == "sample_000"
        )
        modal = load_reference_modal_from_audit(audit, REPO)
        _, final, meta = synthesize_v6_2_physical_routing(
            frequency_hz=659.25,
            duration_s=TEST_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        self.assertLessEqual(float(np.max(np.abs(final))), 1.0)
        self.assertTrue(meta.get("clipping_avoided"))

    def test_duration_approx_2p5s(self) -> None:
        audit = load_audit_report()
        sample = next(
            s for s in load_lhs_sample_entries(REPO, max_samples=26) if s["sample_id"] == "sample_000"
        )
        modal = load_reference_modal_from_audit(audit, REPO)
        _, final, _ = synthesize_v6_2_physical_routing(
            frequency_hz=220.0,
            duration_s=DEFAULT_DURATION_S,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            audit=audit,
            sample_id="sample_000",
            repo_root=REPO,
        )
        self.assertAlmostEqual(len(final) / 44100, DEFAULT_DURATION_S, places=2)

    def test_builder_creates_reports_and_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "v62"
            json_r = Path(tmp) / "report.json"
            md_r = Path(tmp) / "report.md"
            from build_stk_v6_2_diagnostic_audio import build_stk_v6_2_diagnostics

            report = build_stk_v6_2_diagnostics(
                repo_root=REPO,
                out_dir=out,
                sample_id="sample_000",
                notes=(("A4", 440.0),),
                duration_s=TEST_DURATION_S,
            )
            self.assertIn("feature_provenance_used", report)
            self.assertTrue(
                (out / "stk_v6_2_physical_routing_alpha_A4_sample_000.wav").is_file()
            )
            self.assertTrue(
                (out / "stk_v6_2_A4_sample_000_top_radiation_stem.wav").is_file()
            )

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run:
            from build_stk_v6_2_diagnostic_audio import build_stk_v6_2_diagnostics

            build_stk_v6_2_diagnostics(
                repo_root=REPO,
                out_dir=REPO / "audio" / "_test_v62_no_fem",
                sample_id="sample_000",
                notes=(("A3", 220.0),),
                duration_s=0.25,
            )
            for call in mock_run.call_args_list:
                cmd = " ".join(str(x) for x in (call[0][0] if call[0] else []))
                self.assertNotIn("fem", cmd.lower())
                self.assertNotIn("rom_batch", cmd.lower())


if __name__ == "__main__":
    unittest.main()
