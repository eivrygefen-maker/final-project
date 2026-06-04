#!/usr/bin/env python3
"""Smoke check: synthesis export API and fem_main_3d import bootstrap (no checkpoint build)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_REQUIRED = (
    "bootstrap_fem_import_paths",
    "region_dof_subprocess_env",
    "export_region_dof_indices_npz",
    "export_region_dof_indices_isolated",
    "write_stage_a_synthesis_artifacts",
    "region_dof_status_is_pass",
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Synthesis region-DOF import smoke check.")
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

    repo = mod.bootstrap_fem_import_paths(start=SCRIPT_DIR)
    env = mod.region_dof_subprocess_env(repo_root=repo)
    py_path = env.get("PYTHONPATH", "").replace("\\", "/")
    if "FEM/scripts" not in py_path:
        print(f"FAIL PYTHONPATH missing FEM/scripts: {py_path[:200]}", file=sys.stderr)
        return 1

    print(f"PASS repo_root={repo}")
    print(f"PASS bootstrap_fem_import_paths defined and callable")
    print(f"PASS region_dof_subprocess_env PYTHONPATH includes FEM/scripts")

    if args.require_fem:
        try:
            import fem_main_3d as fem3d  # noqa: F401
        except ImportError as exc:
            print(f"FAIL fem_main_3d import: {exc}", file=sys.stderr)
            return 1
        print("PASS fem_main_3d import")
    else:
        print("SKIP fem_main_3d import (pass --require-fem on production VM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
