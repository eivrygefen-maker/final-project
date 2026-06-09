#!/usr/bin/env python3
"""Regression: checkpoint path resolves mesh_level_id from mesh profile (no stale L_prod)."""
from __future__ import annotations

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
from v2_b3_m4_lprod_checkpoint_run import (  # noqa: E402
    _dataset_version_from_run_context,
    _fix_mesh_argv,
    _lprod_mesh_paths_from_plan,
    build_execution_plan,
    run_dry_run,
    run_execute,
)
from v2_b3_m4_lprod_interfaces import evaluate_lprod_mesh_checkpoint_readiness  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_REFERENCE,
    DATASET_VERSION_ROM,
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    apply_mesh_profile_to_sample_input,
    resolve_mesh_profile,
    run_tree_lprod_mesh_path,
)
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
        },
    )


class LprodCheckpointMeshLevelTest(unittest.TestCase):
    def _prepare_execute_preflight(
        self,
        repo: Path,
        mesh_profile: str,
    ) -> tuple[Path, dict]:
        run_root, plan = self._plan_for_profile(repo, mesh_profile)
        sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
        mesh_path = run_tree_lprod_mesh_path(run_root, "sample_002", plan["mesh_level_id"])
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(b"x" * 1001)
        (run_root / "logs").mkdir(parents=True, exist_ok=True)
        return run_root, plan

    def _plan_for_profile(self, repo: Path, mesh_profile: str) -> tuple[Path, dict]:
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
        return run_root, plan

    def test_rom_checkpoint_plan_resolves_l_rom_prod_mesh_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            expected = run_tree_lprod_mesh_path(run_root, "sample_002", LEVEL_ROM_PROD)
            self.assertEqual(plan["mesh_level_id"], LEVEL_ROM_PROD)
            self.assertEqual(plan["mesh_profile"], MESH_PROFILE_ROM)
            self.assertTrue(str(plan["paths"]["lprod_mesh"]).endswith("lprod/mesh/L_rom_prod/sample_002.msh"))
            self.assertEqual(plan["argv_stage_a"][plan["argv_stage_a"].index("--mesh-level") + 1], LEVEL_ROM_PROD)
            self.assertEqual(
                plan["argv_mesh_build"][plan["argv_mesh_build"].index("--mesh-level-id") + 1],
                LEVEL_ROM_PROD,
            )
            mesh_level, lprod_mesh, lprod_mesh_rel = _lprod_mesh_paths_from_plan(
                repo_root=repo,
                run_root=run_root,
                sample_id="sample_002",
                plan=plan,
                sample_input=json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(mesh_level, LEVEL_ROM_PROD)
            self.assertEqual(lprod_mesh, expected)
            self.assertIn("L_rom_prod/sample_002.msh", lprod_mesh_rel)
            self.assertNotIn("/L_prod/", lprod_mesh_rel.replace("\\", "/"))
            readiness = evaluate_lprod_mesh_checkpoint_readiness(
                repo_root=repo,
                run_root=run_root,
                sample_id="sample_002",
                sample_input=json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8")),
                rel_path_fn=lambda p, **kw: str(p),
                mesh_level_id=LEVEL_ROM_PROD,
            )
            self.assertEqual(readiness["mesh_level_id"], LEVEL_ROM_PROD)
            self.assertIn(
                "L_rom_prod",
                readiness["paths"]["sample_mesh_path"].replace("\\", "/"),
            )
            self.assertTrue(readiness["paths"]["sample_mesh_path"].endswith("sample_002.msh"))

    def test_reference_checkpoint_plan_resolves_l_prod_reference_mesh_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan = self._plan_for_profile(repo, MESH_PROFILE_REFERENCE)
            expected = run_tree_lprod_mesh_path(run_root, "sample_002", LEVEL_PROD_REFERENCE)
            self.assertEqual(plan["mesh_level_id"], LEVEL_PROD_REFERENCE)
            self.assertTrue(str(plan["paths"]["lprod_mesh"]).endswith("lprod/mesh/L_prod_reference/sample_002.msh"))
            mesh_level, lprod_mesh, lprod_mesh_rel = _lprod_mesh_paths_from_plan(
                repo_root=repo,
                run_root=run_root,
                sample_id="sample_002",
                plan=plan,
                sample_input=json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(mesh_level, LEVEL_PROD_REFERENCE)
            self.assertEqual(lprod_mesh, expected)
            self.assertIn("L_prod_reference/sample_002.msh", lprod_mesh_rel)
            self.assertNotIn("/L_prod/", lprod_mesh_rel.replace("\\", "/"))

    def test_checkpoint_dry_run_does_not_reference_legacy_l_prod_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, _ = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run_dry_run(repo_root=repo, run_root=run_root, prod_python="python", force=False)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("L_rom_prod", out)
            self.assertNotIn("lprod/mesh/L_prod/", out.replace("\\", "/"))

    def test_rom_prod_002_suffix_resolves_l_rom_prod_mesh_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            self.assertEqual(plan["mesh_level_id"], LEVEL_ROM_PROD)
            self.assertIn("L_rom_prod/sample_002.msh", plan["paths"]["lprod_mesh"])
            self.assertNotIn("lprod/mesh/L_prod/", plan["paths"]["lprod_mesh"].replace("\\", "/"))

    def test_dataset_version_from_run_context_rom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan = self._plan_for_profile(repo, MESH_PROFILE_ROM)
            sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
            self.assertEqual(_dataset_version_from_run_context(plan, sample_input), DATASET_VERSION_ROM)
            line = (
                f"region_dof_preflight: mode=best_effort "
                f"dataset={_dataset_version_from_run_context(plan, sample_input)}\n"
            )
            self.assertIn(DATASET_VERSION_ROM, line)

    def test_dataset_version_from_run_context_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root, plan = self._plan_for_profile(repo, MESH_PROFILE_REFERENCE)
            sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
            self.assertEqual(_dataset_version_from_run_context(plan, sample_input), DATASET_VERSION_REFERENCE)

    def test_run_execute_region_dof_preflight_log_resolves_profile_context(self) -> None:
        for mesh_profile, expected_ds in (
            (MESH_PROFILE_ROM, DATASET_VERSION_ROM),
            (MESH_PROFILE_REFERENCE, DATASET_VERSION_REFERENCE),
        ):
            with self.subTest(mesh_profile=mesh_profile):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    run_root, plan = self._prepare_execute_preflight(repo, mesh_profile)
                    self.assertEqual(plan["mesh_profile"], mesh_profile)
                    self.assertEqual(plan["dataset_version"], expected_ds)
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
                    self.assertTrue(log_path.is_file(), f"missing checkpoint log for {mesh_profile}")
                    log_text = log_path.read_text(encoding="utf-8")
                    self.assertIn("region_dof_preflight", log_text)
                    self.assertIn(expected_ds, log_text)
                    self.assertNotIn("DATASET_VERSION", log_text)


if __name__ == "__main__":
    unittest.main()
