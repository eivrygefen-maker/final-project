#!/usr/bin/env python3
"""Stage 4.8 timbre decomposition tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from bridge_mobility_proxy import compute_body_mass_proxies, bridge_body_coupling_factor  # noqa: E402
from build_body_timbre_decomposition_stage48 import build_stage48_pack  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from timbre_decomposition import LAYER_NAMES, compute_note_layers  # noqa: E402


class TestStage48TimbreDecomposition(unittest.TestCase):
    def test_diagnostic_mode_registered(self) -> None:
        self.assertIn("body_audibility_balance_probe_v1", list_diagnostic_modes())
        cfg = get_diagnostic_mode("body_audibility_balance_probe_v1")
        self.assertIn("body audibility", cfg.description.lower())

    def test_bridge_mobility_proxy_conservative(self) -> None:
        light = compute_body_mass_proxies(
            {"top_wood_id": "cedar", "geometry.length": 0.44, "geometry.top_thickness": 0.0028}
        )
        heavy = compute_body_mass_proxies(
            {"top_wood_id": "maple", "geometry.length": 0.48, "geometry.top_thickness": 0.0035}
        )
        self.assertGreater(light["bridge_mobility_proxy"], 0.7)
        self.assertLess(heavy["bridge_mobility_proxy"], 1.3)
        coupled, meta = bridge_body_coupling_factor(
            {"top_participation": 0.6, "back_participation": 0.2, "air_participation": 0.1, "coupled_participation": 0.1},
            light,
            existing_bridge=1.0,
        )
        self.assertGreaterEqual(coupled, 0.65)
        self.assertLessEqual(coupled, 1.35)
        self.assertEqual(meta["bridge_mobility_affects"], "amplitude")

    def test_layer_generation_deterministic(self) -> None:
        from build_sample_comparison import synthetic_modal_for_sample

        modal = synthetic_modal_for_sample("sample_003")
        params = {"top_wood_id": "spruce", "back_wood_id": "mahogany", "geometry.length": 0.46}
        kwargs = dict(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.35,
            sample_rate=44100,
            modal_data=modal,
            sample_parameters=params,
            use_bridge_mobility_proxy=True,
            modal_source="synthetic_fallback",
        )
        a = compute_note_layers(**kwargs)
        b = compute_note_layers(**kwargs)
        for layer in LAYER_NAMES:
            np.testing.assert_array_almost_equal(a["layers"][layer], b["layers"][layer])

    def test_string_only_identical_across_guitars(self) -> None:
        from build_sample_comparison import synthetic_modal_for_sample

        modal_a = synthetic_modal_for_sample("sample_001")
        modal_b = synthetic_modal_for_sample("sample_009")
        common = dict(frequency_hz=110.0, note_name="A2", duration_s=0.25, sample_rate=44100)
        ra = compute_note_layers(**common, modal_data=modal_a, sample_parameters={"geometry.length": 0.44})
        rb = compute_note_layers(**common, modal_data=modal_b, sample_parameters={"geometry.length": 0.48})
        np.testing.assert_array_almost_equal(ra["layers"]["string_only"], rb["layers"]["string_only"])

    def test_body_only_differs_across_guitars(self) -> None:
        from build_sample_comparison import synthetic_modal_for_sample

        modal_a = synthetic_modal_for_sample("sample_001")
        modal_b = synthetic_modal_for_sample("sample_009")
        common = dict(frequency_hz=440.0, note_name="A4", duration_s=0.35, sample_rate=44100)
        ra = compute_note_layers(**common, modal_data=modal_a, sample_parameters={"geometry.length": 0.42})
        rb = compute_note_layers(**common, modal_data=modal_b, sample_parameters={"geometry.length": 0.50})
        diff = float(np.max(np.abs(ra["layers"]["body_only_raw_pre_norm"] - rb["layers"]["body_only_raw_pre_norm"])))
        self.assertGreater(diff, 1e-8)

    def test_normalization_audit_metadata(self) -> None:
        from build_sample_comparison import synthetic_modal_for_sample

        result = compute_note_layers(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.3,
            sample_rate=44100,
            modal_data=synthetic_modal_for_sample("sample_004"),
            sample_parameters={"top_wood_id": "cedar"},
        )
        meta = result["metadata"]
        self.assertIn("normalization_audit", meta)
        self.assertIn("body_only_final_norm", meta["normalization_audit"])
        self.assertIn("bridge_mobility_proxy", meta)
        self.assertIn("data_attribution", meta)

    def test_no_clipping(self) -> None:
        from build_sample_comparison import synthetic_modal_for_sample

        result = compute_note_layers(
            frequency_hz=440.0,
            note_name="A4",
            duration_s=0.4,
            sample_rate=44100,
            modal_data=synthetic_modal_for_sample("sample_002"),
        )
        for layer, audio in result["layers"].items():
            peak = float(np.max(np.abs(audio)))
            self.assertLessEqual(peak, 1.05, msg=f"layer {layer} peak={peak}")

    def test_build_pack_no_fem(self) -> None:
        out = REPO / "audio" / "_test_stage48_pack"
        if out.exists():
            import shutil

            shutil.rmtree(out)
        with patch(
            "build_body_timbre_decomposition_stage48.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                __import__("build_sample_comparison", fromlist=["synthetic_modal_for_sample"]).synthetic_modal_for_sample(
                    str(sample["sample_id"])
                ),
                "synthetic_fallback",
            ),
        ):
            manifest = build_stage48_pack(
                repo_root=REPO,
                out_dir=out,
                notes=[("A4", 440.0)],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        self.assertEqual(manifest["layer_count"], len(LAYER_NAMES))
        sample_dir = out / "A4" / manifest["sample_ids"][0]
        self.assertTrue((sample_dir / "metadata.json").is_file())
        meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("data_attribution", meta)
        report = REPO / "audio" / "debug_reports" / "stage48_timbre_decomposition_report.json"
        self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
