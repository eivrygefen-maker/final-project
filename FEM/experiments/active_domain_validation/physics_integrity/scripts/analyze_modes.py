#!/usr/bin/env python3
"""
Recompute full mode physics diagnostics for saved eigenvectors (mixed cases).

Use after --ingest-baseline or when scaling metadata was only in logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, load_mode_column_any
from fem_worker_single import hz_result_tag
from mpi4py import MPI

from mode_diagnostics import (
    diagnose_mixed_mode,
    merge_scaling_metadata,
    write_mode_diagnostics_json,
)


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
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=202.0)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        return 2

    case_dir = args.case_dir.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mesh_file = _resolve_mesh(cfg, args.config.resolve())

    # Load mesh only and build collapse maps (no solve).
    msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    solver_cfg = cfg.get("solver", {})
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    u_el = fem3d._displacement_element(msh, 1)
    from basix.ufl import element, mixed_element

    p_el = element("Lagrange", msh.basix_cell(), 1)
    W_el = mixed_element([u_el, p_el])
    from dolfinx import fem

    W = fem.functionspace(msh, W_el)
    V_u, u_to_W = W.sub(0).collapse()
    V_p, p_to_W = W.sub(1).collapse()

    hz_tag = hz_result_tag(args.target_hz)
    result_path = case_dir / "results" / f"result_{hz_tag}.json"
    if not result_path.is_file():
        result_path = next((case_dir / "results").glob("result_*.json"), None)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path and result_path.is_file() else {}
    gnhep = merge_scaling_metadata(case_dir, result)

    mode_rows: List[Dict[str, Any]] = []
    modes_dir = case_dir / "modes"
    for mode_file in sorted(modes_dir.glob(f"mode_{hz_tag}_*{MODE_VECTOR_FILE_SUFFIX}")):
        vec = load_mode_column_any(mode_file)
        # Parse index from filename
        stem = mode_file.stem
        try:
            j = int(stem.split("_")[-1])
        except ValueError:
            j = len(mode_rows)
        cand = next((c for c in result.get("candidates", []) if int(c.get("id", -1)) == j), {})
        fj = float(cand.get("hz", 0.0))
        diag = diagnose_mixed_mode(
            vec,
            u_to_W=np.asarray(u_to_W, dtype=np.int32),
            p_to_W=np.asarray(p_to_W, dtype=np.int32),
            gnhep=gnhep,
            wood_top=float(cand.get("wood_participation", 0.0)) * 0.5,
            wood_back=float(cand.get("wood_participation", 0.0)) * 0.5,
            frequency_hz=fj,
        )
        diag["mode_index"] = j
        diag["p_frac_production"] = float(cand.get("p_frac", diag["p_frac_raw"]))
        diag["vector_path"] = str(mode_file.relative_to(case_dir)).replace("\\", "/")
        mode_rows.append(diag)

    if not mode_rows:
        print(f"[analyze_modes] No mode files in {modes_dir}", file=sys.stderr)
        return 1

    write_mode_diagnostics_json(case_dir, mode_rows, case_label=case_dir.name, scaling=gnhep)
    print(f"[analyze_modes] Updated {len(mode_rows)} modes in {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
