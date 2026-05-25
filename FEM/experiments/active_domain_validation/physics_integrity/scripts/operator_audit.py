#!/usr/bin/env python3
"""Assemble coupled A/M and capture FSI operator audit (no SLEPc solve)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from mpi4py import MPI


def _resolve_mesh(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, EXPERIMENT_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=PHYSICS_ROOT / "comparison" / "coupling_audit.json",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[operator_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["active_domain_experiment"] = {"enabled": False}
    mesh_file = _resolve_mesh(cfg, args.config.resolve())

    sorting = PHYSICS_ROOT / "coupled_nominal" / "sorting_audit"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    msh, W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    audit = dict(cfg.get("_physics_integrity") or {})
    try:
        audit["matrix_norms"] = {
            "A_F": float(A.norm()),
            "M_F": float(M.norm()),
        }
    except Exception:
        pass
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if MPI.COMM_WORLD.rank == 0:
        print(f"[operator_audit] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
