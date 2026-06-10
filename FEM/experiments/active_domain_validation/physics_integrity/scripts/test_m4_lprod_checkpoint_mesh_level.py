#!/usr/bin/env python3
"""Regression: profile mesh paths vs legacy L_prod Stage A export contract (Option A adapter)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_b3_m4_lprod_checkpoint_run as ckpt_mod  # noqa: E402

# Mirror v2_b3_checkpoint_export.py ALLOWED_MESH_LEVELS without importing mpi4py stack.
CHECKPOINT_EXPORTER_ALLOWED_LEVELS = frozenset({"L_mid", "L_dev_dense", "L_prod", "L_scout_coarse"})
from v2_b3_m4_lprod_checkpoint_run import (  # noqa: E402
    _dataset_version_from_run_context,
    _fix_mesh_argv,
    _lprod_mesh_paths_from_plan,
    _stamp_checkpoint_profile_provenance,
    build_execution_plan,
    run_dry_run,
    run_execute,
)
from v2_b3_m4_lprod_interfaces import evaluate_lprod_mesh_checkpoint_readiness  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    CHECKPOINT_EXPORT_MESH_LEVEL,
    DATASET_VERSION_REFERENCE,
    DATASET_VERSION_ROM,
    LEVEL_L_PROD_LEGACY,
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    apply_mesh_profile_to_sample_input,
    checkpoint_export_mesh_level,
    resolve_mesh_profile,
    run_tree_lprod_mesh_path,
    validate_mesh_profile_reuse,
)
from v2_b3_m4_production_contracts import evaluate_production_acceptance  # noqa: E402
from v2_b3_m4_worker_run_lib import verify_lprod_checkpoint  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_checkpoint_inputs(
    *,
    repo_root: Path,
    run_root: Path,
    sample_input: dict,
) -> None:
    write_json_atomic(run_root / "sample" / "sample_input.json", sample_input)
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {
            "sample_id": sample_input["sample_id"],
            "run_id": run_root.name,
            "terminal_status": "SCOUT_PASS_TARGET_PLAN_READY",
            "stages": {"stage3_zones_plan": {"status": "PASS"}},
        },
    )
    write_json_atomic(
        run_root / "lprod" / "lprod_target_plan.json",
        {"coverage_check": {"pass": True}, "targets_hz": [100.0, 200.0]},
    )
    write_json_atomic(
        run_root / "sample" / "resolved_core_config.json",
        {
            "geometry_numeric_parameters": {"length": 0.48, "width": 0.325, "depth": 0.1},
            "solver": {"mesh_file": "mesh.msh", "clamp_ribs": False},
            "dataset_version": sample_input.get("dataset_version"),
        },
    )


def _stage_a_export_level(argv: list[str]) -> str:
    return argv[argv.index("--mesh-level") + 1]


def _stage_a_operator_mesh(argv: list[str]) -> str:
    return argv[argv.index("--operator-mesh-file") + 1]


def _parse_checkpoint_exporter_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-level", choices=sorted(CHECKPOINT_EXPORTER_ALLOWED_LEVELS))
    parser.add_argument("--operator-mesh-file")
    parser.add_argument("--core-config")
    parser.add_argument("--output-dir")
    parser.add_argument("--B3-block-compose-backend")
    parser.add_argument("--B3-synthesis-region-dofs")
    return parser.parse_args(argv)


class LprodCheckpointMeshLevelTest(unittest.TestCase):
    def _plan_for_profile(self, repo: Path, mesh_profile: str) -> tuple[Path, dict, dict]:
        resolved = resolve_mesh_profile(
            mesh_profile=mesh_profile,
            dataset_version=(
                DATASET_VERSION_REFERENCE if mesh_profile == MESH_PROFILE_REFERENCE else DATASET_VERSION_ROM
            ),
        )
        sample_id = "sample_002"
        run_id = f"{sample_id}_{mesh_profile}_ckpt_test"
        run_root = repo / "guitars" / sample_id / "runs" / run_id
        sample_input = apply_mesh_profile_to_sample_input(
            {
                "sample_id": sample_id,
                "geometry": {"length": 0.48, "width": 0.325, "depth": 0.1},
                "top_wood_id": "spruce",
                "back_wood_id": "mahogany",
            },
            resolved,
        )
        _write_checkpoint_inputs(repo_root=repo, run_root=run_root, sample_input=sample_input)
        manifest = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        plan = build_execution_plan(
            repo_root=repo,
            run_root=run_root,
            sample_input=sample_input,
            manifest=manifest,
            prod_python="python",
            force=False,
        )
        _fix_mesh_argv(
            plan,
            repo_root=repo,
            run_root=run_root,
            sample_id=sample_input["sample_id"],
            prod_python="python",
        )
        return run_root, plan, sample_input

    def _prepare_execute_preflight(self, repo: Path, mesh_profile: str) -> tuple[Path, dict, dict]:
        run_root, plan, sample_input = self._plan_for_profile(repo, mesh_profile)
        mesh_path = run_tree_lprod_mesh_path(run_root, "sample_002", plan["mesh_level_id"])
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"x" * 1001)
        (run_root / "logs").mkdir(parents=True, exist_ok=True)
        return run_root, plan, sample_input

    def test_checkpoint_export_mesh_level_is_l_prod(self) -> None:
        self.assertEqual(checkpoint_export_mesh_level(), LEVEL_L_PROD_LEGACY)
        self.assertEqual(CHECKPOINT_EXPORT_MESH_LEVEL, "L_prod")

    def test_rom_profile_mesh_build_and_stage_a_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan, sample_input = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            argv = plan["argv_stage_a"]
            self.assertEqual(plan["mesh_level_id"], LEVEL_ROM_PROD)
            self.assertEqual(
                plan["argv_mesh_build"][plan["argv_mesh_build"].index("--mesh-level-id") + 1],
                LEVEL_ROM_PROD,
            )
            self.assertEqual(_stage_a_export_level(argv), "L_prod")
            self.assertIn("L_rom_prod", _stage_a_operator_mesh(argv))
            self.assertNotIn("L_prod_reference", _stage_a_export_level(argv))
            self.assertNotIn("L_rom_prod", _stage_a_export_level(argv))

    def test_reference_profile_mesh_build_and_stage_a_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan, sample_input = self._plan_for_profile(repo, MESH_PROFILE_REFERENCE)
            argv = plan["argv_stage_a"]
            self.assertEqual(plan["mesh_level_id"], LEVEL_PROD_REFERENCE)
            self.assertEqual(
                plan["argv_mesh_build"][plan["argv_mesh_build"].index("--mesh-level-id") + 1],
                LEVEL_PROD_REFERENCE,
            )
            self.assertEqual(_stage_a_export_level(argv), "L_prod")
            self.assertIn("L_prod_reference", _stage_a_operator_mesh(argv))

    def test_checkpoint_exporter_argparse_accepts_both_profiles(self) -> None:
        forbidden = {LEVEL_ROM_PROD, LEVEL_PROD_REFERENCE}
        for mesh_profile in (MESH_PROFILE_ROM, MESH_PROFILE_REFERENCE):
            with self.subTest(mesh_profile=mesh_profile):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    run_root, plan, _ = self._plan_for_profile(repo, mesh_profile)
                    argv = plan["argv_stage_a"][2:]
                    args = _parse_checkpoint_exporter_argv(argv)
                    self.assertEqual(args.mesh_level, "L_prod")
                    self.assertIsNotNone(args.operator_mesh_file)
                    self.assertNotIn(args.mesh_level, forbidden)

    def test_stage_a_never_passes_profile_levels_to_exporter(self) -> None:
        for mesh_profile in (MESH_PROFILE_ROM, MESH_PROFILE_REFERENCE):
            with self.subTest(mesh_profile=mesh_profile):
                with tempfile.TemporaryDirectory() as tmp:
                    _, plan, _ = self._plan_for_profile(Path(tmp), mesh_profile)
                    export_level = _stage_a_export_level(plan["argv_stage_a"])
                    self.assertEqual(export_level, "L_prod")
                    self.assertNotIn(export_level, (LEVEL_ROM_PROD, LEVEL_PROD_REFERENCE))

    def test_post_export_stamp_writes_profile_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan, sample_input = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            mesh_path = run_tree_lprod_mesh_path(run_root, "sample_002", LEVEL_ROM_PROD)
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh_bytes = b"rom_mesh_stub" * 100
            mesh_path.write_bytes(mesh_bytes)
            ckpt = run_root / "lprod" / "checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            write_json_atomic(ckpt / "built_metadata.json", {"mesh_level": "L_prod", "active_dimension": 1000})
            _stamp_checkpoint_profile_provenance(
                checkpoint_dir=ckpt,
                lprod_mesh=mesh_path,
                plan=plan,
                sample_input=sample_input,
            )
            built = json.loads((ckpt / "built_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(built["mesh_profile"], MESH_PROFILE_ROM)
            self.assertEqual(built["mesh_level_id"], LEVEL_ROM_PROD)
            self.assertEqual(built["dataset_version"], DATASET_VERSION_ROM)
            self.assertEqual(built["mesh_level"], "L_prod")
            self.assertEqual(built["checkpoint_export_mesh_level"], "L_prod")
            self.assertIn("effective_controls_m", built)
            self.assertEqual(built["generated_mesh_sha256"], hashlib.sha256(mesh_bytes).hexdigest())
            self.assertEqual(built["operator_mesh_sha256"], built["generated_mesh_sha256"])

    def test_validate_reuse_accepts_internal_l_prod_with_stamped_profile(self) -> None:
        expected = resolve_mesh_profile(mesh_profile=MESH_PROFILE_ROM, dataset_version=DATASET_VERSION_ROM)
        errors = validate_mesh_profile_reuse(
            expected=expected,
            existing={
                "mesh_profile": "rom",
                "mesh_level_id": LEVEL_ROM_PROD,
                "dataset_version": DATASET_VERSION_ROM,
                "mesh_level": "L_prod",
                "effective_controls_m": expected.effective_controls_m,
            },
            context="test",
        )
        self.assertFalse(errors)

    def test_worker_verify_accepts_stamped_profile_with_internal_l_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint"
            ckpt.mkdir()
            write_json_atomic(
                ckpt / "checkpoint_export_manifest.json",
                {
                    "status": "PASS",
                    "export_pass": True,
                    "matrix_verify_pass": True,
                    "core_config_mode": "override",
                    "mesh_level": "L_prod",
                },
            )
            write_json_atomic(
                ckpt / "built_metadata.json",
                {
                    "mesh_level": "L_prod",
                    "mesh_profile": "rom",
                    "mesh_level_id": LEVEL_ROM_PROD,
                    "dataset_version": DATASET_VERSION_ROM,
                },
            )
            (ckpt / "A_active_csr.npz").write_bytes(b"A" * 4096)
            (ckpt / "M_active_csr.npz").write_bytes(b"M" * 4096)
            ok, detail = verify_lprod_checkpoint(ckpt)
            self.assertTrue(ok, detail)

    def test_acceptance_does_not_fail_internal_l_prod_mesh_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, _, sample_input = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            ckpt = run_root / "lprod" / "checkpoint"
            ckpt.mkdir(parents=True, exist_ok=True)
            mesh_path = run_tree_lprod_mesh_path(run_root, "sample_002", LEVEL_ROM_PROD)
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh_path.write_bytes(b"x" * 2000)
            write_json_atomic(
                ckpt / "built_metadata.json",
                {
                    "mesh_level": "L_prod",
                    "mesh_profile": "rom",
                    "mesh_level_id": LEVEL_ROM_PROD,
                    "dataset_version": DATASET_VERSION_ROM,
                    "effective_controls_m": sample_input["effective_controls_m"],
                    "operator_mesh_matches_generated": True,
                    "generated_mesh_sha256": "abc",
                    "active_dimension": 1000,
                },
            )
            write_json_atomic(
                run_root / "aggregation" / "aggregation_result.json",
                {"status": "AGGREGATION_PASS", "final_aggregation_ready": True},
            )
            (run_root / "aggregation" / "modes_catalog.jsonl").write_text(
                '{"frequency_hz": 120.0, "mic_output_method": "aperture_pressure_rms_proxy_v1"}\n',
                encoding="utf-8",
            )
            result = evaluate_production_acceptance(run_root=run_root, sample_input=sample_input)
            level_failures = [f for f in result.get("failures") or [] if f.startswith("mesh_level_id=")]
            self.assertFalse(level_failures, result.get("failures"))

    def test_readiness_stage_a_command_shows_l_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, _, sample_input = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            readiness = evaluate_lprod_mesh_checkpoint_readiness(
                repo_root=repo,
                run_root=run_root,
                sample_id="sample_002",
                sample_input=sample_input,
                rel_path_fn=lambda p, **kw: str(p),
                mesh_level_id=LEVEL_ROM_PROD,
            )
            cmd = readiness["commands"]["stage_a_export_planned"]
            self.assertIn("--mesh-level L_prod", cmd)
            self.assertNotIn("L_rom_prod", cmd.split("--mesh-level")[0])
            self.assertIn("L_rom_prod", readiness["commands"]["mesh_build_planned"])

    def test_run_execute_region_dof_preflight_log_resolves_profile_context(self) -> None:
        for mesh_profile, expected_ds in (
            (MESH_PROFILE_ROM, DATASET_VERSION_ROM),
            (MESH_PROFILE_REFERENCE, DATASET_VERSION_REFERENCE),
        ):
            with self.subTest(mesh_profile=mesh_profile):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    run_root, plan, _ = self._prepare_execute_preflight(repo, mesh_profile)
                    self.assertEqual(_stage_a_export_level(plan["argv_stage_a"]), "L_prod")
                    with (
                        patch.object(ckpt_mod, "_run_env_probe", return_value=(0, "{}")),
                        patch.object(ckpt_mod, "_verify_stage_a_env_probe", return_value=(True, "ok")),
                        patch.object(ckpt_mod, "_run_subprocess", return_value=1),
                        patch.object(ckpt_mod, "_verify_lprod_checkpoint_export", return_value=(False, "mock", {})),
                    ):
                        rc = run_execute(
                            repo_root=repo,
                            run_root=run_root,
                            prod_python="python",
                            prod_venv="python",
                            force=False,
                        )
                    self.assertEqual(rc, 1)
                    log_path = run_root / "logs" / "stage4_lprod_checkpoint.log"
                    self.assertTrue(log_path.is_file())
                    log_text = log_path.read_text(encoding="utf-8")
                    self.assertIn("region_dof_preflight", log_text)
                    self.assertIn(expected_ds, log_text)

    def test_checkpoint_dry_run_shows_profile_mesh_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, _, _ = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run_dry_run(repo_root=repo, run_root=run_root, prod_python="python", force=False)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("L_rom_prod", out)
            self.assertIn("stage_a:", out)
            self.assertIn("--mesh-level L_prod", out)


if __name__ == "__main__":
    unittest.main()
