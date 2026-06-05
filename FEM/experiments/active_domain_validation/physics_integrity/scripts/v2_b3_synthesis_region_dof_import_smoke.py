#!/usr/bin/env python3
"""Smoke check: synthesis export API + operator-build region DOF npz write (no full checkpoint)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_REQUIRED = (
    "export_region_dof_indices_from_operator_build",
    "write_stage_a_synthesis_artifacts",
    "region_dof_status_is_pass",
    "REGION_DOF_SOURCE_OPERATOR_BUILD",
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Synthesis region-DOF smoke check.")
    parser.add_argument(
        "--require-fem",
        action="store_true",
        help="Also import fem_main_3d (needs production .venv / petsc4py on VM).",
    )
    args = parser.parse_args(argv)

    import v2_b3_synthesis_export as mod

    missing = [name for name in _REQUIRED if not hasattr(mod, name)]
    if missing:
        print(f"FAIL missing exports: {missing}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="b3_region_dof_smoke_") as tmp:
        ckpt = Path(tmp)
        region_dof_build = {
            "u_idx_top": np.array([0, 1, 2], dtype=np.int32),
            "u_idx_back": np.array([10, 11], dtype=np.int32),
            "u_idx_ribs": np.array([20], dtype=np.int32),
            "u_idx_soundhole": np.array([], dtype=np.int32),
            "p_idx_air": np.array([100, 101], dtype=np.int32),
            "p_idx_all": np.array([100, 101], dtype=np.int32),
            "u_idx_all": np.arange(50, dtype=np.int32),
            "region_dof_source": mod.REGION_DOF_SOURCE_OPERATOR_BUILD,
            "region_dof_mesh_file": "/tmp/smoke.msh",
            "layout": mod.REGION_DOF_LAYOUT,
            "back_includes_ribs": True,
            "counts": {"u_idx_top": 3, "u_idx_back": 2, "u_idx_ribs": 1, "p_idx_air": 2},
        }
        status, err = mod.export_region_dof_indices_from_operator_build(
            ckpt,
            region_dof_build=region_dof_build,
        )
        if not mod.region_dof_status_is_pass(status):
            print(f"FAIL operator_build export status={status} err={err}", file=sys.stderr)
            return 1
        npz = ckpt / mod.REGION_DOF_INDICES_NPZ
        if not npz.is_file():
            print("FAIL region_dof_indices.npz not written", file=sys.stderr)
            return 1
        with np.load(npz, allow_pickle=False) as z:
            if int(z["u_idx_top"].size) != 3:
                print("FAIL u_idx_top size", file=sys.stderr)
                return 1
            src = str(np.asarray(z["region_dof_source"]).ravel()[0])
            if src != mod.REGION_DOF_SOURCE_OPERATOR_BUILD:
                print(f"FAIL region_dof_source={src}", file=sys.stderr)
                return 1

    print("PASS export_region_dof_indices_from_operator_build writes npz")
    print(f"PASS region_dof_source={mod.REGION_DOF_SOURCE_OPERATOR_BUILD}")

    if args.require_fem:
        repo = mod.bootstrap_fem_import_paths(start=SCRIPT_DIR)
        try:
            import fem_main_3d as fem3d  # noqa: F401
        except ImportError as exc:
            print(f"FAIL fem_main_3d import: {exc}", file=sys.stderr)
            return 1
        print(f"PASS fem_main_3d import repo_root={repo}")
    else:
        print("SKIP fem_main_3d import (pass --require-fem on production VM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
