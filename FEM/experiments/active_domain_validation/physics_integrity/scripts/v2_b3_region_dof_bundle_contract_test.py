#!/usr/bin/env python3
"""Regression: region_dof_indices.npz index/metadata schema and loader contract."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_rich_modal_lib import (  # noqa: E402
    REGION_DOF_INDICES_NPZ,
    REGION_DOF_METADATA_JSON,
    RegionDofSchemaError,
    coerce_region_index_array,
    load_region_dof_bundle,
    split_region_dof_mixed_npz_inplace,
    validate_region_dof_contract,
    write_region_dof_metadata_json,
)


def _built_meta(*, n_w: int = 200) -> dict:
    return {
        "n_w": n_w,
        "n_u_b3": 50,
        "p_idx": list(range(100, 110)),
        "active_dimension": 194273,
    }


def test_legacy_integer_only_npz() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        np.savez_compressed(
            ckpt / REGION_DOF_INDICES_NPZ,
            u_idx_top=np.array([0, 1, 2], dtype=np.int32),
            u_idx_back=np.array([10, 11], dtype=np.int32),
            u_idx_ribs=np.array([20], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([100, 101], dtype=np.int32),
            p_idx_all=np.array([100, 101], dtype=np.int32),
            u_idx_all=np.arange(50, dtype=np.int32),
            layout=np.asarray(["B3_W_global_row_indices_via_u_idx_p_idx"]),
            region_dof_source=np.asarray(["dolfinx_region_dof_export"]),
            back_includes_ribs=np.asarray([True]),
        )
        ctx = load_region_dof_bundle(ckpt, _built_meta(), validate_aperture=False)
        assert ctx["npz_present"]
        assert int(ctx["region"]["u_idx_top"].size) == 3
        assert ctx["region_dof_source"] == "dolfinx_region_dof_export"


def test_mixed_npz_string_metadata_backward_compat() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        np.savez_compressed(
            ckpt / REGION_DOF_INDICES_NPZ,
            u_idx_top=np.array([0, 1], dtype=np.int32),
            u_idx_back=np.array([10], dtype=np.int32),
            u_idx_ribs=np.array([], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([100, 101, 102], dtype=np.int32),
            p_idx_all=np.array([100, 101, 102], dtype=np.int32),
            p_idx_aperture=np.array([100, 101], dtype=np.int32),
            u_idx_all=np.arange(50, dtype=np.int32),
            aperture_selection_method=np.asarray(["facet_adjacent_air_cell_dofs_v1"]),
            p_idx_aperture_count=np.asarray([2]),
            mic_output_method=np.asarray(["aperture_pressure_rms_proxy_v1"]),
            aperture_coordinate_bounds_min=np.array([0.0, -0.05, 0.08], dtype=np.float64),
            aperture_coordinate_bounds_max=np.array([0.1, 0.05, 0.12], dtype=np.float64),
            aperture_facet_count=np.asarray([24]),
            adjacent_air_cell_count=np.asarray([18]),
            layout=np.asarray(["B3_W_global_row_indices_via_u_idx_p_idx"]),
            region_dof_source=np.asarray(["operator_build_context"]),
            back_includes_ribs=np.asarray([True]),
        )
        ctx = load_region_dof_bundle(ckpt, _built_meta(n_w=200))
        assert ctx["metadata"]["aperture_selection_method"] == "facet_adjacent_air_cell_dofs_v1"
        assert int(ctx["region"]["p_idx_aperture"].size) == 2
        bounds = ctx["metadata"]["aperture_coordinate_bounds"]
        assert len(bounds["min"]) == 3
        assert bounds["max"][2] == 0.12


def test_split_npz_plus_metadata_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        np.savez_compressed(
            ckpt / REGION_DOF_INDICES_NPZ,
            u_idx_top=np.array([0], dtype=np.int32),
            u_idx_back=np.array([1], dtype=np.int32),
            u_idx_ribs=np.array([], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([100], dtype=np.int32),
            p_idx_all=np.array([100], dtype=np.int32),
            p_idx_aperture=np.array([100], dtype=np.int32),
            u_idx_all=np.arange(10, dtype=np.int32),
        )
        write_region_dof_metadata_json(
            ckpt,
            {
                "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
                "p_idx_aperture_count": 1,
                "region_dof_source": "operator_build_context",
                "aperture_coordinate_bounds": {
                    "min": [0.0, 0.0, 0.08],
                    "max": [0.05, 0.05, 0.10],
                },
            },
        )
        ctx = load_region_dof_bundle(ckpt, _built_meta(n_w=200))
        assert ctx["region_dof_metadata_present"]
        assert ctx["aperture_selection_method"] == "facet_adjacent_air_cell_dofs_v1"


def test_invalid_string_under_index_key_fails() -> None:
    try:
        coerce_region_index_array("p_idx_aperture", np.asarray(["facet_adjacent_air_cell_dofs_v1"]))
    except RegionDofSchemaError as exc:
        assert "p_idx_aperture" in str(exc)
        assert "facet_adjacent_air_cell_dofs_v1" in str(exc)
    else:
        raise AssertionError("expected RegionDofSchemaError")


def test_validate_contract_requires_selection_method() -> None:
    region = {"p_idx_aperture": np.array([100, 101], dtype=np.int32)}
    failures = validate_region_dof_contract(region, {}, _built_meta(n_w=200), require_aperture=True)
    assert any("aperture_selection_method_missing" in f for f in failures)


def test_split_mixed_npz_inplace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        np.savez_compressed(
            ckpt / REGION_DOF_INDICES_NPZ,
            u_idx_top=np.array([0], dtype=np.int32),
            u_idx_back=np.array([1], dtype=np.int32),
            u_idx_ribs=np.array([], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([100], dtype=np.int32),
            p_idx_all=np.array([100], dtype=np.int32),
            p_idx_aperture=np.array([100], dtype=np.int32),
            u_idx_all=np.arange(5, dtype=np.int32),
            aperture_selection_method=np.asarray(["facet_adjacent_air_cell_dofs_v1"]),
            p_idx_aperture_count=np.asarray([1]),
        )
        result = split_region_dof_mixed_npz_inplace(ckpt)
        assert result["status"] == "PASS"
        assert (ckpt / REGION_DOF_METADATA_JSON).is_file()
        with np.load(ckpt / REGION_DOF_INDICES_NPZ, allow_pickle=False) as z:
            assert "aperture_selection_method" not in z.files
            assert "p_idx_aperture" in z.files
        ctx = load_region_dof_bundle(ckpt, _built_meta(n_w=200))
        assert ctx["metadata"]["aperture_selection_method"] == "facet_adjacent_air_cell_dofs_v1"


def main() -> int:
    tests = [
        test_legacy_integer_only_npz,
        test_mixed_npz_string_metadata_backward_compat,
        test_split_npz_plus_metadata_json,
        test_invalid_string_under_index_key_fails,
        test_validate_contract_requires_selection_method,
        test_split_mixed_npz_inplace,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_region_dof_bundle_contract] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
