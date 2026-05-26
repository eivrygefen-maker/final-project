#!/usr/bin/env python3
"""Acoustic-cavity-only EVP locator (mpiexec -n 1) for geometry-changing v2 samples."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from typing import Any, Dict, List, Tuple

from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from v2_sensitivity_mesh import sample_geometry

V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"


def pick_locator_frequency_hz(
    freqs_hz: List[float],
    *,
    band_lo: float,
    band_hi: float,
    reference_hz: float,
) -> Tuple[float, str]:
    """Prefer lowest in-band mode (cavity fundamental); else nearest to reference."""
    in_band = sorted(float(f) for f in freqs_hz if band_lo <= float(f) <= band_hi)
    if in_band:
        return float(in_band[0]), "lowest_in_band_cavity_fundamental"
    if not freqs_hz:
        return float("nan"), "no_modes"
    best = min(freqs_hz, key=lambda f: abs(float(f) - reference_hz))
    return float(best), "nearest_to_reference_outside_band"


def main() -> int:
    parser = argparse.ArgumentParser(description="Acoustic cavity-only locator")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--locator-lo-hz", type=float, default=150.0)
    parser.add_argument("--locator-hi-hz", type=float, default=350.0)
    parser.add_argument("--reference-hz", type=float, default=244.394153389752)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--out-pressure-mode-npy",
        type=Path,
        default=None,
        help="Save selected acoustic-cavity pressure eigenvector (n_p_active).",
    )
    parser.add_argument(
        "--out-pressure-mode-meta-json",
        type=Path,
        default=None,
        help="Metadata for archived pressure eigenvector.",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[v2_locator] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    cfg = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(args.mesh.resolve())
    sc["acoustic_cavity_only_diagnosis"] = True
    sc["couple_fluid"] = False
    sc["structural_only_diagnosis"] = False
    sc["coupled_physical_core_v2_diagnosis"] = False
    sc["coupled_physical_core_v2_coupling_enabled"] = False
    sc["physics_integrity_capture"] = False
    sc["acoustic_cavity_num_modes"] = max(4, int(args.num_modes))
    sc["acoustic_min_mode_hz"] = float(args.locator_lo_hz)
    sc["acoustic_max_mode_hz"] = float(args.locator_hi_hz)
    sc["acoustic_shift_target_hz"] = float(args.reference_hz)
    cfg["geometry"] = sample_geometry(sample)
    nm = max(4, int(args.num_modes))

    _msh, _V_space, freqs_hz, eig_ac, n_u_reported, n_p_reported = fem3d._solve_coupled_evp(
        mesh_file=args.mesh.resolve(),
        config=cfg,
        num_modes=nm,
    )
    if int(n_u_reported) > 0 and int(n_p_reported) == 0:
        payload = {
            "locator_status": "failed",
            "error": "locator dispatched to structural-only branch",
            "n_u_reported": int(n_u_reported),
            "n_p_reported": int(n_p_reported),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1
    if int(n_p_reported) <= 0:
        payload = {
            "locator_status": "failed",
            "error": "locator returned no pressure DOFs",
            "n_u_reported": int(n_u_reported),
            "n_p_reported": int(n_p_reported),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    loc_hz, method = pick_locator_frequency_hz(
        list(freqs_hz),
        band_lo=float(args.locator_lo_hz),
        band_hi=float(args.locator_hi_hz),
        reference_hz=float(args.reference_hz),
    )
    j = int(np.argmin([abs(float(f) - loc_hz) for f in freqs_hz])) if freqs_hz else -1
    payload = {
        "locator_status": "ok" if math.isfinite(loc_hz) else "failed",
        "locator_frequency_hz": loc_hz,
        "locator_selection_method": method,
        "locator_band_hz": [float(args.locator_lo_hz), float(args.locator_hi_hz)],
        "all_locator_frequencies_hz": [float(f) for f in freqs_hz],
        "picked_mode_index": int(j),
        "picked_mode_frequency_hz": float(freqs_hz[j]) if j >= 0 else float("nan"),
        "reference_hz": float(args.reference_hz),
        "sample_id": str(sample.get("id", "")),
        "solver_branch": "acoustic_cavity_only_diagnosis_via_solve_coupled_evp",
        "n_p_reported": int(n_p_reported),
        "acoustic_locator_vector_saved": False,
    }
    if (
        math.isfinite(loc_hz)
        and j >= 0
        and eig_ac is not None
        and getattr(eig_ac, "size", 0) > 0
        and args.out_pressure_mode_npy is not None
    ):
        p_mode = np.asarray(eig_ac[:, j], dtype=np.float64).ravel()
        nrm = float(np.linalg.norm(p_mode))
        if nrm > 0.0:
            p_mode = p_mode / nrm
        args.out_pressure_mode_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.out_pressure_mode_npy.resolve()), p_mode)
        payload["acoustic_locator_vector_saved"] = True
        payload["acoustic_locator_pressure_npy"] = str(args.out_pressure_mode_npy)
        payload["n_p_active_locator_vector"] = int(p_mode.size)
        if args.out_pressure_mode_meta_json is not None:
            pmeta = {
                "locator_frequency_hz": float(loc_hz),
                "picked_mode_index": int(j),
                "picked_mode_frequency_hz": float(freqs_hz[j]),
                "n_p_active": int(p_mode.size),
                "vector_norm": float(np.linalg.norm(p_mode)),
                "locator_pressure_reference_source": "acoustic_only_locator_eigenvector",
                "solver_branch": payload["solver_branch"],
            }
            args.out_pressure_mode_meta_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_pressure_mode_meta_json.write_text(
                json.dumps(pmeta, indent=2), encoding="utf-8"
            )
            payload["acoustic_locator_pressure_meta_json"] = str(args.out_pressure_mode_meta_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[v2_locator] sample={sample.get('id')} "
            f"locator_status={payload['locator_status']} "
            f"locator_frequency_hz={loc_hz:.6f} method={method}",
            flush=True,
        )
    return 0 if math.isfinite(loc_hz) else 1


if __name__ == "__main__":
    raise SystemExit(main())
