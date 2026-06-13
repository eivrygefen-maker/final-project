#!/usr/bin/env python3
"""Pipeline default STK mode + precompute cache tests (no FEM/ROM batch)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    STK_FINAL_CANDIDATE_CANONICAL,
    STK_FINAL_GUI_LABEL,
    canonical_stk_final_mode,
)
from build_note_cache import build_note_cache  # noqa: E402
from stk_final_v1_precompute_cache import (  # noqa: E402
    compute_guitar_signature_hash,
    ensure_stk_precompute_cache,
    load_body_signature_cache,
)
from stk_pipeline_defaults import (  # noqa: E402
    DEFAULT_WEBSITE_STK_LABEL,
    DEFAULT_WEBSITE_STK_MODE,
    lhs_params_to_sample_parameters,
    resolve_pipeline_stk_mode_alias,
    user_label_for_stk_mode,
)


class TestStkPipelineDefaultAndCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        (self.repo / "audio" / "debug_reports").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_stk_mode_is_final_v1(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertEqual(resolve_pipeline_stk_mode_alias(), STK_BODY_TRANSFER_FINAL_V1)
        self.assertNotEqual(resolve_pipeline_stk_mode_alias(), STK_BODY_TRANSFER_FINAL_V1_DE_THUMP)

    def test_gui_label(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_LABEL, STK_FINAL_GUI_LABEL)
        self.assertEqual(user_label_for_stk_mode(STK_BODY_TRANSFER_FINAL_V1), "Physical Body Identity v1")

    def test_de_thump_not_default(self) -> None:
        self.assertNotEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1_DE_THUMP)
        alias = resolve_pipeline_stk_mode_alias(
            override=STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
            developer_debug=False,
        )
        self.assertEqual(alias, STK_BODY_TRANSFER_FINAL_V1)

    def test_de_thump_available_in_debug(self) -> None:
        alias = resolve_pipeline_stk_mode_alias(
            override=STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
            developer_debug=True,
        )
        self.assertEqual(alias, STK_BODY_TRANSFER_FINAL_V1_DE_THUMP)

    def test_alias_resolves_to_canonical(self) -> None:
        self.assertEqual(
            canonical_stk_final_mode(STK_BODY_TRANSFER_FINAL_V1),
            STK_FINAL_CANDIDATE_CANONICAL,
        )

    def test_signature_hash_changes_with_geometry(self) -> None:
        params_a = lhs_params_to_sample_parameters(
            {
                "geometry.length": 0.52,
                "geometry.width": 0.32,
                "materials.top.wood_id": "spruce",
                "materials.back.wood_id": "mahogany",
            }
        )
        params_b = lhs_params_to_sample_parameters(
            {
                "geometry.length": 0.55,
                "geometry.width": 0.32,
                "materials.top.wood_id": "spruce",
                "materials.back.wood_id": "mahogany",
            }
        )
        hash_a = compute_guitar_signature_hash(
            modal_json_sha256="abc",
            geometry_fingerprint="geom1",
            sample_parameters=params_a,
            stk_model_alias=STK_BODY_TRANSFER_FINAL_V1,
        )
        hash_b = compute_guitar_signature_hash(
            modal_json_sha256="abc",
            geometry_fingerprint="geom1",
            sample_parameters=params_b,
            stk_model_alias=STK_BODY_TRANSFER_FINAL_V1,
        )
        self.assertNotEqual(hash_a, hash_b)

    def test_cache_reused_when_identical(self) -> None:
        from body_response_synth import synthetic_classic_body_modes

        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        modal_path = self.repo / "modal.json"
        modal_path.write_text(json.dumps(modal_data), encoding="utf-8")
        params = lhs_params_to_sample_parameters(
            {
                "geometry.length": 0.52,
                "geometry.width": 0.32,
                "materials.top.wood_id": "spruce",
                "materials.back.wood_id": "mahogany",
            }
        )
        bundle1, report1 = ensure_stk_precompute_cache(
            repo_root=self.repo,
            modal_json=modal_path,
            modal_data=modal_data,
            sample_parameters=params,
        )
        bundle2, report2 = ensure_stk_precompute_cache(
            repo_root=self.repo,
            modal_json=modal_path,
            modal_data=modal_data,
            sample_parameters=params,
        )
        self.assertFalse(report1["cache_hit"])
        self.assertTrue(report2["cache_hit"])
        self.assertEqual(bundle1["guitar_signature_hash"], bundle2["guitar_signature_hash"])

    def test_generate_works_without_cache(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal_for_synthetic__.json"),
            out_root=self.repo / "note_cache",
            fret_count=3,
            duration_s=0.15,
            sample_rate=44100,
            force=True,
            repo_root=self.repo,
        )
        self.assertEqual(manifest["stk_model_alias"], STK_BODY_TRANSFER_FINAL_V1)
        cache_root = Path(manifest["cache_root"])
        self.assertTrue((cache_root / "note_manifest.json").is_file())

    def test_corrupt_cache_recomputes(self) -> None:
        from body_response_synth import synthetic_classic_body_modes

        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        modal_path = self.repo / "modal.json"
        modal_path.write_text(json.dumps(modal_data), encoding="utf-8")
        params = lhs_params_to_sample_parameters({"geometry.length": 0.52, "materials.top.wood_id": "spruce"})
        bundle, _ = ensure_stk_precompute_cache(
            repo_root=self.repo,
            modal_json=modal_path,
            modal_data=modal_data,
            sample_parameters=params,
        )
        cache_path = Path(bundle["_cache_path"])
        cache_path.write_text('{"schema_version": "broken"}', encoding="utf-8")
        loaded = load_body_signature_cache(
            self.repo,
            bundle["guitar_signature_hash"],
            modal_json_sha256=bundle["modal_json_sha256"],
            stk_model_alias=STK_BODY_TRANSFER_FINAL_V1,
        )
        self.assertIsNone(loaded)
        _bundle2, report2 = ensure_stk_precompute_cache(
            repo_root=self.repo,
            modal_json=modal_path,
            modal_data=modal_data,
            sample_parameters=params,
            force_recompute=True,
        )
        self.assertFalse(report2["cache_hit"])
        self.assertEqual(_bundle2["model_alias"], STK_BODY_TRANSFER_FINAL_V1)

    def test_no_fem_rom_in_pipeline_tests(self) -> None:
        fem_targets = [
            "FEM.run_fom",
            "run_fom_acoustics",
            "build_m4_rom_from_completed_fom",
        ]
        rom_batch_targets = [
            "ROM.classic.build_rom",
            "run_rom_batch",
        ]
        for target in fem_targets + rom_batch_targets:
            module_path = target.split(".")[0]
            if module_path == "FEM":
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 0
                    self.assertIsNotNone(mock_run)
            # Pipeline modules under test do not import FEM runners at import time.
        import build_note_cache as bnc
        import stk_final_v1_precompute_cache as spc

        src_bnc = Path(bnc.__file__).read_text(encoding="utf-8")
        src_spc = Path(spc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("run_fom_acoustics", src_bnc)
        self.assertNotIn("run_rom_batch", src_spc)


if __name__ == "__main__":
    unittest.main()
