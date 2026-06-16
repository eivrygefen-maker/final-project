#!/usr/bin/env python3
"""Regression: production core-config parsing must tolerate metadata alongside numeric geometry."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    GEOMETRY_NUMERIC_KEYS,
    GeometryNumericCoercionError,
    apply_sample_geometry_to_resolved_config,
    coerce_geometry_numeric,
    extract_geometry_dict,
    extract_run_metadata,
    geometry_fingerprint,
)
from v2_b3_m4_lprod_checkpoint_run import resolve_lprod_core_config  # noqa: E402
from v2_b3_m4_production_contracts import DATASET_VERSION, geometry_from_core_config  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _production_like_config() -> dict:
    return {
        "shape_name": "classic",
        "dataset_version": DATASET_VERSION,
        "m4_run_metadata": {
            "shape_name": "classic",
            "dataset_version": DATASET_VERSION,
            "shape_type": "classic",
            "mesh_mode": "fom",
        },
        "geometry_numeric_parameters": {
            "length": 0.48,
            "width": 0.325,
            "depth": 0.1,
            "hole_radius": 0.047,
            "top_thickness": 0.003,
            "back_thickness": 0.0033,
        },
        "geometry": {
            "shape_type": "classic",
            "mesh_mode": "fom",
            "length": 0.48,
            "width": 0.325,
            "depth": 0.1,
            "hole_radius": 0.047,
            "top_thickness": 0.003,
            "back_thickness": 0.0033,
        },
        "parameters": {
            "geometry.length": 0.48,
            "geometry.width": 0.325,
            "geometry.depth": 0.1,
            "geometry.hole_radius": 0.047,
            "geometry.top_thickness": 0.003,
            "geometry.back_thickness": 0.0033,
            "materials.top_wood_id": "spruce_sitka",
            "materials.back_wood_id": "rosewood_indian",
        },
        "solver": {
            "mesh_file": "scout/mesh/L_scout_coarse/sample_001.msh",
            "clamp_ribs": False,
        },
    }


def test_extract_geometry_skips_metadata() -> None:
    cfg = _production_like_config()
    geom = extract_geometry_dict(cfg)
    assert set(geom.keys()) == set(GEOMETRY_NUMERIC_KEYS)
    assert geom["length"] == 0.48
    assert geom["back_thickness"] == 0.0033


def test_extract_run_metadata() -> None:
    cfg = _production_like_config()
    meta = extract_run_metadata(cfg)
    assert meta["shape_name"] == "classic"
    assert meta["dataset_version"] == DATASET_VERSION
    assert meta["shape_type"] == "classic"
    assert meta["mesh_mode"] == "fom"


def test_geometry_from_core_config_file() -> None:
    cfg = _production_like_config()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resolved_core_config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        geom = geometry_from_core_config(path)
    assert geom["hole_radius"] == 0.047
    assert geometry_fingerprint(geom)


def test_coerce_geometry_numeric_reports_key() -> None:
    try:
        coerce_geometry_numeric("shape_type", "classic")
    except GeometryNumericCoercionError as exc:
        assert exc.key == "shape_type"
        assert exc.value == "classic"
        assert "classic" in str(exc)
    else:
        raise AssertionError("expected GeometryNumericCoercionError")


def test_lhs_sample_input_shape() -> None:
    sample = {
        "schema": "m4_sample_input_v1",
        "sample_id": "sample_001",
        "shape_name": "classic",
        "parameters": {
            "geometry.length": 0.48,
            "geometry.width": 0.325,
            "geometry.depth": 0.1,
            "geometry.hole_radius": 0.047,
            "geometry.top_thickness": 0.003,
            "geometry.back_thickness": 0.0033,
        },
    }
    geom = extract_geometry_dict(sample)
    meta = extract_run_metadata(sample)
    assert meta["shape_name"] == "classic"
    assert len(geom) == 6


def test_box_sample_input_skips_shape_type_float() -> None:
    sample = {
        "schema": "m4_sample_input_v1",
        "sample_id": "box_sample_000",
        "shape_name": "box",
        "geometry_shape_type": "Box",
        "gmsh_shape_type": "Box",
        "lhs_path": "ROM/box/lhs_pool.json",
        "parameters": {
            "geometry.length": 0.401307,
            "geometry.width": 0.365581,
            "geometry.depth": 0.114949,
            "geometry.hole_radius": 0.038594,
            "geometry.top_thickness": 0.003483,
            "geometry.back_thickness": 0.0038313,
            "geometry.shape_type": "Box",
            "top_wood_id": "mahogany",
            "back_wood_id": "cedar",
        },
    }
    geom = extract_geometry_dict(sample)
    meta = extract_run_metadata(sample)
    assert "shape_type" not in geom
    assert set(geom.keys()) == set(GEOMETRY_NUMERIC_KEYS)
    assert meta["geometry_shape_type"] == "Box"
    assert meta["shape_name"] == "box"
    resolved = {"solver": {"mesh_file": "lprod/mesh/L_prod/box_sample_000.msh"}}
    resolution = apply_sample_geometry_to_resolved_config(resolved, sample_input=sample)
    assert resolution["numeric_parameter_count"] == 6
    assert resolution["geometry_shape_type"] == "Box"
    assert resolved["parameters"]["geometry.shape_type"] == "Box"
    assert isinstance(resolved["parameters"]["geometry.length"], float)


def test_resolve_lprod_core_config_box_shape_type() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = repo_root / "guitars" / "box_sample_000" / "runs" / "box_sample_000_box_fom_v1"
        sample_dir = run_root / "sample"
        lprod_dir = run_root / "lprod"
        sample_dir.mkdir(parents=True)
        lprod_dir.mkdir(parents=True)
        sample = {
            "sample_id": "box_sample_000",
            "shape_name": "box",
            "geometry_shape_type": "Box",
            "gmsh_shape_type": "Box",
            "lhs_path": "ROM/box/lhs_pool.json",
            "parameters": {
                "geometry.length": 0.401307,
                "geometry.width": 0.365581,
                "geometry.depth": 0.114949,
                "geometry.hole_radius": 0.038594,
                "geometry.top_thickness": 0.003483,
                "geometry.back_thickness": 0.0038313,
                "geometry.shape_type": "Box",
                "top_wood_id": "mahogany",
                "back_wood_id": "cedar",
            },
        }
        write_json_atomic(sample_dir / "sample_input.json", sample)
        write_json_atomic(
            sample_dir / "resolved_core_config.json",
            {"solver": {"mesh_file": "scout/mesh/L_scout_coarse/box_sample_000.msh"}},
        )
        out = resolve_lprod_core_config(
            repo_root=repo_root,
            run_root=run_root,
            sample_id="box_sample_000",
            lprod_mesh_rel="lprod/mesh/L_prod/box_sample_000.msh",
            force=True,
            sample_input=sample,
        )
        body = json.loads(out.read_text(encoding="utf-8"))
        assert body["parameters"]["geometry.shape_type"] == "Box"
        assert body["lprod_config_resolution"]["geometry_shape_type"] == "Box"
        assert body["lprod_config_resolution"]["numeric_parameter_count"] == 6


def main() -> int:
    tests = [
        test_extract_geometry_skips_metadata,
        test_extract_run_metadata,
        test_geometry_from_core_config_file,
        test_coerce_geometry_numeric_reports_key,
        test_lhs_sample_input_shape,
        test_box_sample_input_skips_shape_type_float,
        test_resolve_lprod_core_config_box_shape_type,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_geometry_numeric_contract] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
