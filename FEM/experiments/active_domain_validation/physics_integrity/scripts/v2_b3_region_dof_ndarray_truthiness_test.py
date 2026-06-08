#!/usr/bin/env python3
"""Regression: region-DOF export must not use ndarray truth-value tests."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_rich_modal_lib import build_region_dof_metadata_from_operator_build  # noqa: E402
from v2_b3_synthesis_export import export_region_dof_indices_from_operator_build  # noqa: E402


def test_build_region_dof_metadata_with_multi_element_ndarray() -> None:
    p_ap = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int32)
    region_dof_build = {
        "u_idx_top": np.array([0, 1], dtype=np.int32),
        "u_idx_back": np.array([2], dtype=np.int32),
        "u_idx_ribs": np.array([], dtype=np.int32),
        "u_idx_soundhole": np.array([], dtype=np.int32),
        "p_idx_air": np.array([10, 11, 12, 13], dtype=np.int32),
        "p_idx_all": np.array([10, 11, 12, 13], dtype=np.int32),
        "u_idx_all": np.arange(5, dtype=np.int32),
        "p_idx_aperture": p_ap,
        "region_dof_source": "operator_build_context",
        "region_dof_mesh_file": "/tmp/sample.msh",
        "layout": "B3_W_global_row_indices_via_u_idx_p_idx",
        "back_includes_ribs": True,
        "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
        "mic_output_method": "aperture_pressure_rms_proxy_v1",
        "aperture_facet_count": 12,
        "adjacent_air_cell_count": 8,
    }
    meta = build_region_dof_metadata_from_operator_build(region_dof_build)
    assert meta["p_idx_aperture_count"] == int(p_ap.size)
    assert meta["aperture_selection_method"] == "facet_adjacent_air_cell_dofs_v1"


def test_export_region_dof_indices_from_operator_build_with_ndarray_masks() -> None:
    region_dof_build = {
        "u_idx_top": np.array([0, 1], dtype=np.int32),
        "u_idx_back": np.array([2], dtype=np.int32),
        "u_idx_ribs": np.array([], dtype=np.int32),
        "u_idx_soundhole": np.array([], dtype=np.int32),
        "p_idx_air": np.array([10, 11, 12, 13], dtype=np.int32),
        "p_idx_all": np.array([10, 11, 12, 13], dtype=np.int32),
        "u_idx_all": np.arange(5, dtype=np.int32),
        "p_idx_aperture": np.array([10, 11, 12, 13], dtype=np.int32),
        "region_dof_source": "operator_build_context",
        "region_dof_mesh_file": "/tmp/sample.msh",
        "layout": "B3_W_global_row_indices_via_u_idx_p_idx",
        "back_includes_ribs": True,
        "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
        "mic_output_method": "aperture_pressure_rms_proxy_v1",
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp)
        status, err = export_region_dof_indices_from_operator_build(
            ckpt,
            region_dof_build=region_dof_build,
        )
        assert status == "BEST_EFFORT_PASS", err
        assert (ckpt / "region_dof_indices.npz").is_file()
        assert (ckpt / "region_dof_metadata.json").is_file()


def main() -> int:
    tests = [
        test_build_region_dof_metadata_with_multi_element_ndarray,
        test_export_region_dof_indices_from_operator_build_with_ndarray_masks,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
