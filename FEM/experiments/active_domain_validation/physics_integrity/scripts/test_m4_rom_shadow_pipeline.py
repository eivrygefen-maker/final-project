#!/usr/bin/env python3
"""Tests for official ROM shadow pipeline (no FEM)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_m4_compaction_run_selection import _make_strict_completed_run, _write_json  # noqa: E402
from v2_b3_m4_finalize_completed_run import finalize_completed_run, is_run_already_finalized  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_ROM,
    LEVEL_ROM_PROD,
    MESH_PROFILE_ROM,
)
from v2_b3_m4_official_rom_dataset_lib import (  # noqa: E402
    OFFICIAL_INITIAL_RUN_IDS,
    collect_official_rom_training_rows,
    evaluate_official_rom_run_eligibility,
)
from v2_b3_m4_rom_shadow_pipeline_lib import (  # noqa: E402
    DURABLE_ROM_JSON_NAMES,
    RetrainPolicy,
    build_holdout_official_rom_model,
    build_official_rom_surrogate_from_runs,
    frozen_prediction_internal_path,
    mark_fom_pipeline_started,
    prune_rom_directory_to_durable,
    rom_prediction_summary_path,
    rom_vs_fom_comparison_path,
    run_shadow_rom_compare_nonblocking,
    run_shadow_rom_prepredict_nonblocking,
)
from v2_b3_m4_rom_fom_compare_lib import resolve_sample_context  # noqa: E402


def _guitars_root(repo: Path) -> Path:
    return (
        repo
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
    )


def _make_official_rom_run(
    repo: Path,
    *,
    sample_id: str,
    run_id: str,
    parameters: dict | None = None,
) -> Path:
    run_root = _guitars_root(repo) / sample_id / "runs" / run_id
    _make_strict_completed_run(run_root, sample_id=sample_id, run_id=run_id)
    params = parameters or {
        "geometry.length": 0.48,
        "geometry.width": 0.325,
        "geometry.depth": 0.1,
        "geometry.top_thickness": 0.003,
        "geometry.hole_radius": 0.047,
        "geometry.back_thickness": 0.0033,
        "top_wood_id": "spruce",
        "back_wood_id": "rosewood",
    }
    _write_json(
        run_root / "sample" / "sample_input.json",
        {
            "sample_id": sample_id,
            "run_id": run_id,
            "shape_name": "classic",
            "parameters": params,
            "mesh_profile": MESH_PROFILE_ROM,
            "mesh_level_id": LEVEL_ROM_PROD,
            "dataset_version": DATASET_VERSION_ROM,
            "lhs_row_index": int(sample_id.split("_")[1]),
        },
    )
    _write_json(
        run_root / "compaction" / "compaction_manifest.json",
        {"status": "completed", "deleted_bytes": 100},
    )
    _write_json(
        run_root / "cleanup" / "sample_cleanup_barrier.json",
        {
            "status": "completed",
            "verification": {"pass": True},
            "forbidden_heavy_artifact_count": 0,
            "shared_sample_artifact_count": 0,
        },
    )
    catalog = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
            "\n".join(
                json.dumps(
                    {
                        "frequency_hz": 100.0 + i * 10.0,
                        "chunk_id": f"chunk_{i:03d}",
                        "top_share": 0.3,
                        "back_share": 0.6,
                        "air_share": 0.1,
                        "bridge_excitation_coupling": 0.01,
                        "bridge_excitation_abs": 0.02,
                        "radiation_proxy": 0.001,
                        "mic_output_proxy": 0.01,
                        "modal_norm": 1.0,
                        "coupling_class": "top_back_mixed",
                        "dominant_region": "back",
                        "secondary_region": "top",
                    }
                )
                for i in range(12)
            )
            + "\n",
        encoding="utf-8",
    )
    _write_json(
        run_root / "pipeline_run_manifest.json",
        {"terminal_status": "COMPLETED", "production_acceptance_pass": True},
    )
    for rel in ("lprod/checkpoint", "scout/mesh", "scout/checkpoint", "scout/discovery", "worker_results"):
        path = run_root / rel
        if path.exists():
            shutil.rmtree(path)
    return run_root


def _seed_five_official_runs(repo: Path) -> None:
    woods = [
        ("spruce", "rosewood"),
        ("mahogany", "mahogany"),
        ("rosewood", "spruce"),
        ("rosewood", "cedar"),
        ("mahogany", "spruce"),
    ]
    for i, (top, back) in enumerate(woods):
        sid = f"sample_{i:03d}"
        rid = f"{sid}_rom_official_v1"
        _make_official_rom_run(
            repo,
            sample_id=sid,
            run_id=rid,
            parameters={
                "geometry.length": 0.45 + i * 0.02,
                "geometry.width": 0.30 + i * 0.01,
                "geometry.depth": 0.09 + i * 0.005,
                "geometry.top_thickness": 0.003,
                "geometry.hole_radius": 0.04 + i * 0.001,
                "geometry.back_thickness": 0.0032,
                "top_wood_id": top,
                "back_wood_id": back,
            },
        )


class OfficialRomDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _seed_five_official_runs(self.repo)
        legacy_root = _make_official_rom_run(
            self.repo,
            sample_id="sample_099",
            run_id="sample_099_m4prod2",
        )
        _write_json(
            legacy_root / "sample" / "sample_input.json",
            {
                "sample_id": "sample_099",
                "mesh_profile": "reference",
                "mesh_level_id": "L_prod_reference",
                "dataset_version": "m4_geometry_corrected_reference_v1",
            },
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_collector_includes_only_official_runs(self) -> None:
        training, skipped = collect_official_rom_training_rows(
            repo_root=self.repo,
            allowed_run_ids=list(OFFICIAL_INITIAL_RUN_IDS),
        )
        self.assertEqual(len(training), 5)
        run_ids = {r["run_id"] for r in training}
        self.assertEqual(run_ids, set(OFFICIAL_INITIAL_RUN_IDS))
        self.assertTrue(any(s.get("run_id") == "sample_099_m4prod2" for s in skipped))

    def test_legacy_and_reference_excluded(self) -> None:
        eligible, reasons, _ = evaluate_official_rom_run_eligibility(
            _guitars_root(self.repo) / "sample_099" / "runs" / "sample_099_m4prod2"
        )
        self.assertFalse(eligible)
        self.assertTrue(any("mesh_profile" in r or "dataset_version" in r for r in reasons))

    def test_fresh_model_manifest_lists_five_runs(self) -> None:
        _model, training, skipped, report = build_official_rom_surrogate_from_runs(
            repo_root=self.repo,
            shape_name="classic",
            allowed_run_ids=list(OFFICIAL_INITIAL_RUN_IDS),
            min_mode_count=1,
        )
        self.assertEqual(len(training), 5)
        self.assertEqual(report["training_run_ids"], list(OFFICIAL_INITIAL_RUN_IDS))
        manifest = json.loads((self.repo / "ROM/classic/rom_model_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["training_run_ids"], list(OFFICIAL_INITIAL_RUN_IDS))
        self.assertEqual(manifest["maturity"], "integration_only")
        self.assertFalse(manifest["production_accuracy_validated"])


class RomShadowPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _seed_five_official_runs(self.repo)
        build_official_rom_surrogate_from_runs(
            repo_root=self.repo,
            shape_name="classic",
            allowed_run_ids=list(OFFICIAL_INITIAL_RUN_IDS),
            min_mode_count=1,
        )
        self.run_root = _make_official_rom_run(
            self.repo,
            sample_id="sample_005",
            run_id="sample_005_rom_official_v1",
            parameters={
                "geometry.length": 0.52,
                "geometry.width": 0.34,
                "geometry.depth": 0.11,
                "geometry.top_thickness": 0.0031,
                "geometry.hole_radius": 0.045,
                "geometry.back_thickness": 0.0034,
                "top_wood_id": "maple",
                "back_wood_id": "cedar",
            },
        )
        (self.repo / "ROM/classic/lhs_pool.json").write_text(
            json.dumps({"shape_name": "classic", "entries": []}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_holdout_excludes_current_sample(self) -> None:
        model, rows, _ = build_holdout_official_rom_model(
            repo_root=self.repo,
            shape_name="classic",
            exclude_sample_ids=["sample_005"],
            exclude_run_ids=["sample_005_rom_official_v1"],
        )
        ids = {r["sample_id"] for r in rows}
        self.assertNotIn("sample_005", ids)
        self.assertEqual(len(ids), 5)

    def test_prediction_before_fom_and_frozen_sha(self) -> None:
        context = {
            "sample_id": "sample_005",
            "run_id": "sample_005_rom_official_v1",
            "shape_name": "classic",
            "lhs_row_index": 5,
            "parameters": json.loads((self.run_root / "sample/sample_input.json").read_text())["parameters"],
        }
        prep = run_shadow_rom_prepredict_nonblocking(
            repo_root=self.repo,
            run_root=self.run_root,
            context=context,
        )
        self.assertEqual(prep.get("status"), "COMPLETED")
        summary_path = rom_prediction_summary_path(self.run_root)
        self.assertTrue(summary_path.is_file())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frozen_sha = summary["frozen_prediction_sha256"]
        mark_fom_pipeline_started(self.run_root)
        cmp_result = run_shadow_rom_compare_nonblocking(
            repo_root=self.repo,
            run_root=self.run_root,
            context=context,
        )
        self.assertFalse(cmp_result.get("blocking"))
        self.assertTrue(rom_vs_fom_comparison_path(self.run_root).is_file())
        summary_after = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary_after["frozen_prediction_sha256"], frozen_sha)

    def test_low_accuracy_does_not_block(self) -> None:
        context = {
            "sample_id": "sample_005",
            "run_id": "sample_005_rom_official_v1",
            "shape_name": "classic",
            "lhs_row_index": 5,
            "parameters": json.loads((self.run_root / "sample/sample_input.json").read_text())["parameters"],
        }
        run_shadow_rom_prepredict_nonblocking(repo_root=self.repo, run_root=self.run_root, context=context)
        mark_fom_pipeline_started(self.run_root)
        cmp_result = run_shadow_rom_compare_nonblocking(
            repo_root=self.repo, run_root=self.run_root, context=context
        )
        self.assertFalse(cmp_result.get("blocking"))

    def test_cleanup_retains_only_two_rom_json_files(self) -> None:
        context = {
            "sample_id": "sample_005",
            "run_id": "sample_005_rom_official_v1",
            "shape_name": "classic",
            "lhs_row_index": 5,
            "parameters": json.loads((self.run_root / "sample/sample_input.json").read_text())["parameters"],
        }
        run_shadow_rom_prepredict_nonblocking(repo_root=self.repo, run_root=self.run_root, context=context)
        mark_fom_pipeline_started(self.run_root)
        run_shadow_rom_compare_nonblocking(repo_root=self.repo, run_root=self.run_root, context=context)
        self.assertTrue(frozen_prediction_internal_path(self.run_root).is_file())
        removed = prune_rom_directory_to_durable(self.run_root)
        self.assertIn("rom_prediction_frozen_internal.json", removed)
        remaining = {p.name for p in (self.run_root / "rom").iterdir() if p.is_file()}
        self.assertEqual(remaining, set(DURABLE_ROM_JSON_NAMES))

    def test_retrain_policy_default(self) -> None:
        policy = RetrainPolicy(retrain_every_n_new_samples=5)
        self.assertFalse(policy.should_retrain(new_samples_since_last_train=4))
        self.assertTrue(policy.should_retrain(new_samples_since_last_train=5))


class FinalizeIdempotencyTests(unittest.TestCase):
    def test_already_finalized_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = _make_official_rom_run(repo, sample_id="sample_000", run_id="sample_000_rom_official_v1")
            ok, _ = is_run_already_finalized(run_root)
            self.assertTrue(ok)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True, exist_ok=True)
            lhs.write_text(json.dumps({"shape_name": "classic", "entries": []}), encoding="utf-8")
            report = finalize_completed_run(
                repo_root=repo,
                sample_id="sample_000",
                run_id="sample_000_rom_official_v1",
                lhs_path=lhs,
                shared_root=repo / "shared",
                reconcile_bookkeeping=False,
            )
            self.assertEqual(report.get("outcome"), "ALREADY_FINALIZED")


if __name__ == "__main__":
    unittest.main()
