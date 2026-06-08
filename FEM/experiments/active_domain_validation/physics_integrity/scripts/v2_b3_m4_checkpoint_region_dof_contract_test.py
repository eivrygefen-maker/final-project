#!/usr/bin/env python3
"""Regression: corrected-dataset L_prod region-DOF export contract."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import LPROD_SYNTHESIS_REGION_DOFS_DEFAULT  # noqa: E402
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    aperture_export_required_for_core_config,
    resolve_production_region_dofs_mode,
    validate_pre_operator_build_region_dof_contract,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_core_config(path: Path, *, dataset_version: str | None) -> None:
    body: dict = {
        "geometry_numeric_parameters": {"length": 0.48, "width": 0.325, "depth": 0.1},
        "solver": {"mesh_file": "mesh.msh"},
    }
    if dataset_version:
        body["dataset_version"] = dataset_version
        body["m4_run_metadata"] = {"dataset_version": dataset_version}
    write_json_atomic(path, body)


def test_corrected_dataset_forces_best_effort() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "resolved_core_config.json"
        _write_core_config(cfg, dataset_version=DATASET_VERSION)
        mode = resolve_production_region_dofs_mode("off", core_config_path=cfg)
        assert mode == "best_effort"
        assert aperture_export_required_for_core_config(cfg) is True


def test_legacy_dataset_allows_off() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "legacy.json"
        _write_core_config(cfg, dataset_version=None)
        os.environ.pop("B3_REQUIRE_APERTURE_MASK", None)
        os.environ["B3_REQUIRE_APERTURE_MASK"] = "0"
        os.environ["B3_ALLOW_CAVITY_MAX_MIC_FALLBACK"] = "1"
        try:
            mode = resolve_production_region_dofs_mode("off", core_config_path=cfg)
            assert mode == "off"
            assert aperture_export_required_for_core_config(cfg) is False
        finally:
            os.environ.pop("B3_ALLOW_CAVITY_MAX_MIC_FALLBACK", None)


def test_corrected_off_mode_fails_preflight() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "resolved_core_config.json"
        _write_core_config(cfg, dataset_version=DATASET_VERSION)
        errors = validate_pre_operator_build_region_dof_contract(
            region_dofs_mode="off",
            core_config_path=cfg,
        )
        assert errors
        assert any("forbidden" in e for e in errors)


def test_smoke_and_batch_share_lprod_default_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "resolved_core_config.json"
        _write_core_config(cfg, dataset_version=DATASET_VERSION)
        from v2_b3_m4_lprod_checkpoint_run import build_execution_plan  # noqa: WPS433

        run_root = Path(tmp) / "run"
        run_root.mkdir(parents=True)
        write_json_atomic(
            run_root / "sample" / "sample_input.json",
            {"sample_id": "sample_000", "geometry": {"length": 0.48}},
        )
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"sample_id": "sample_000", "run_id": "sample_000_test", "terminal_status": "SCOUT_PASS_TARGET_PLAN_READY"},
        )
        write_json_atomic(run_root / "lprod" / "lprod_target_plan.json", {"coverage_check": {"pass": True}})
        write_json_atomic(run_root / "sample" / "resolved_core_config.json", json.loads(cfg.read_text(encoding="utf-8")))
        plan = build_execution_plan(
            repo_root=Path(tmp),
            run_root=run_root,
            sample_input=json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8")),
            manifest=json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8")),
            prod_python="python",
            force=False,
        )
        argv = plan["argv_stage_a"]
        assert "--B3-synthesis-region-dofs" in argv
        idx = argv.index("--B3-synthesis-region-dofs")
        assert argv[idx + 1] == LPROD_SYNTHESIS_REGION_DOFS_DEFAULT
        mode = resolve_production_region_dofs_mode(
            argv[idx + 1],
            core_config_path=run_root / "lprod" / "resolved_core_config.json",
        )
        assert mode == "best_effort"


def main() -> int:
    tests = [
        test_corrected_dataset_forces_best_effort,
        test_legacy_dataset_allows_off,
        test_corrected_off_mode_fails_preflight,
        test_smoke_and_batch_share_lprod_default_mode,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
