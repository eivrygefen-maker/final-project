#!/usr/bin/env python3
"""Tests for unified ShapeContext and shape-agnostic M4 stage contracts."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from m4_shape_context import (  # noqa: E402
    ShapeContext,
    ShapeContextError,
    apply_shape_context_to_resolved_config,
    resolve_shape_context,
    resolve_shape_context_from_sample_input,
)
from m4_shape_registry import M4ShapeConfig, resolve_shape_config  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import build_sample_input  # noqa: E402
from v2_b3_m4_lprod_interfaces import apply_sample_geometry_to_resolved_config  # noqa: E402
from v2_b3_m4_mesh_manifest_lib import resolve_case_shape_metadata, validate_mesh_reuse  # noqa: E402
from v2_b3_m4_pipeline_run_scout import resolve_m4_sample  # noqa: E402
from v2_b3_m4_stage_artifact_contract import (  # noqa: E402
    SCOUT_TERMINAL_ARTIFACTS,
    validate_scout_terminal_artifacts,
    validate_worker_plan_artifacts,
)


def test_classic_resolves_to_classical() -> None:
    ctx = resolve_shape_context("classic", lhs_path="ROM/classic/lhs_pool.json")
    assert ctx.shape_name == "classic"
    assert ctx.geometry_shape_type == "Classical"
    assert ctx.gmsh_shape_type == "Classical"
    assert ctx.scout_density_policy == "intrinsic_discovered_modes_v1"


def test_box_resolves_to_box() -> None:
    ctx = resolve_shape_context("box", lhs_path="ROM/box/lhs_pool.json")
    assert ctx.geometry_shape_type == "Box"
    assert ctx.gmsh_shape_type == "Box"
    assert ctx.scout_density_policy == "box_discovered_modes_v2"


def test_acoustic_resolves_to_acoustic() -> None:
    ctx = resolve_shape_context("acoustic", lhs_path="ROM/acoustic/lhs_pool.json")
    assert ctx.geometry_shape_type == "Acoustic"
    assert ctx.gmsh_shape_type == "Acoustic"


def test_build_sample_input_carries_shape_context() -> None:
    pool = {
        "shape_name": "box",
        "entries": [
            {
                "id": "box_sample_000",
                "parameters": {
                    "geometry.length": 0.46,
                    "geometry.shape_type": "Box",
                },
            }
        ],
    }
    body = build_sample_input(
        pool=pool,
        entry=pool["entries"][0],
        lhs_row_index=0,
        batch_id="batch_test",
        lhs_source_path="ROM/box/lhs_pool.json",
    )
    assert body["shape_name"] == "box"
    assert body["geometry_shape_type"] == "Box"
    assert body["gmsh_shape_type"] == "Box"
    assert body["parameters"]["geometry.shape_type"] == "Box"
    assert body["scout_density_policy"] == "box_discovered_modes_v2"


def test_resolve_m4_sample_uses_geometry_shape_type_not_shape_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = repo_root / "run"
        scout_mesh_rel = "scout/mesh/L_scout_coarse/box_sample_000.msh"
        mesh_abs = repo_root / scout_mesh_rel
        mesh_abs.parent.mkdir(parents=True)
        mesh_abs.write_text("x" * 2000, encoding="utf-8")
        sample = {
            "sample_id": "box_sample_000",
            "shape_name": "box",
            "geometry_shape_type": "Box",
            "gmsh_shape_type": "Box",
            "geometry": {"length": 0.46, "width": 0.36, "depth": 0.10, "top_thickness": 0.003},
            "parameters": {"geometry.shape_type": "Box"},
        }
        resolved, _, _ = resolve_m4_sample(
            sample,
            repo_root=repo_root,
            run_root=run_root,
            scout_mesh_rel=scout_mesh_rel,
            force=True,
            skip_mesh=True,
        )
        assert resolved["geometry"]["shape_type"] == "Box"
        assert resolved["shape_name"] == "box"
        assert resolved["geometry_shape_type"] == "Box"


def test_box_rejects_classical_geometry_when_strict() -> None:
    sample = {
        "sample_id": "box_sample_000",
        "shape_name": "box",
        "geometry_shape_type": "Classical",
        "gmsh_shape_type": "Classical",
    }
    try:
        resolve_shape_context_from_sample_input(sample, legacy_classic_default=False)
        raise AssertionError("expected ShapeContextError")
    except ShapeContextError:
        pass


def test_mesh_metadata_box_not_classical() -> None:
    meta = resolve_case_shape_metadata(
        {"shape_name": "box", "geometry_shape_type": "Box", "gmsh_shape_type": "Box"},
    )
    assert meta["geometry_shape_type"] == "Box"
    assert meta["shape_name"] == "box"


def test_scout_and_lprod_share_shape_from_sample_input() -> None:
    sample = {
        "sample_id": "box_sample_000",
        "shape_name": "box",
        "geometry_shape_type": "Box",
        "gmsh_shape_type": "Box",
        "geometry": {"length": 0.46, "width": 0.36, "depth": 0.10, "top_thickness": 0.003},
        "parameters": {"geometry.shape_type": "Box"},
    }
    scout_meta = resolve_case_shape_metadata(sample, sample_input=sample)
    resolved: Dict[str, Any] = {"geometry": {}, "m4_run_metadata": {}}
    lprod_resolution = apply_sample_geometry_to_resolved_config(resolved, sample_input=sample)
    assert scout_meta["geometry_shape_type"] == "Box"
    assert lprod_resolution["geometry_shape_type"] == "Box"
    assert resolved["geometry"]["shape_type"] == "Box"


def test_stage_contract_is_shape_agnostic() -> None:
    assert any("worker_chunk_plan.preview.json" in rel for rel in SCOUT_TERMINAL_ARTIFACTS)
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        (run_root / "scout").mkdir()
        (run_root / "lprod").mkdir()
        ok, errors = validate_scout_terminal_artifacts(run_root)
        assert ok is False
        assert any("worker_chunk_plan.preview.json" in e for e in errors)
        for rel in SCOUT_TERMINAL_ARTIFACTS:
            path = run_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        ok2, errors2 = validate_scout_terminal_artifacts(run_root)
        assert ok2 is True
        assert errors2 == []


def test_worker_plan_contract_requires_chunk_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        (run_root / "lprod").mkdir()
        ok, errors = validate_worker_plan_artifacts(run_root)
        assert ok is False
        assert any("worker_chunk_plan.preview.json" in e for e in errors)


def test_future_shape_via_registry_without_pipeline_changes() -> None:
    dummy = M4ShapeConfig(
        shape_key="trapezoid_test",
        display_name="TrapezoidTest",
        lhs_pool_rel="ROM/trapezoid_test/lhs_pool.json",
        rom_dir_rel="ROM/trapezoid_test",
        shared_export_key="trapezoid_test",
        geometry_shape_type="TrapezoidTest",
        gmsh_shape_type="TrapezoidTest",
        sample_id_prefix="trap_sample_",
        default_lhs_count=10,
        has_soundhole=False,
        requires_aperture_mask=False,
        soundhole_note="test-only",
        scout_density_policy="intrinsic_discovered_modes_v1",
    )
    import m4_shape_registry as reg  # noqa: E402

    with mock.patch.dict(reg._SHAPE_REGISTRY, {"trapezoid_test": dummy}, clear=False):
        ctx = resolve_shape_context("trapezoid_test")
    assert ctx.geometry_shape_type == "TrapezoidTest"
    assert ctx.shape_name == "trapezoid_test"


def test_classic_legacy_default_when_shape_missing() -> None:
    ctx = resolve_shape_context_from_sample_input(
        {"sample_id": "sample_001", "parameters": {"geometry.shape_type": "Classical"}},
        legacy_classic_default=True,
    )
    assert ctx.shape_name == "classic"
    assert ctx.geometry_shape_type == "Classical"


def main() -> int:
    tests = [
        test_classic_resolves_to_classical,
        test_box_resolves_to_box,
        test_acoustic_resolves_to_acoustic,
        test_build_sample_input_carries_shape_context,
        test_resolve_m4_sample_uses_geometry_shape_type_not_shape_name,
        test_box_rejects_classical_geometry_when_strict,
        test_mesh_metadata_box_not_classical,
        test_scout_and_lprod_share_shape_from_sample_input,
        test_stage_contract_is_shape_agnostic,
        test_worker_plan_contract_requires_chunk_plan,
        test_future_shape_via_registry_without_pipeline_changes,
        test_classic_legacy_default_when_shape_missing,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_shape_context] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
