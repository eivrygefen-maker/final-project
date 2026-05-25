#!/usr/bin/env python3
"""Acoustic-cavity-only EVP locator (mpiexec -n 1) for geometry-changing v2 samples."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
    sc["coupled_physical_core_v2_diagnosis"] = False
    sc["coupled_physical_core_v2_coupling_enabled"] = False
    sc["physics_integrity_capture"] = False
    cfg["geometry"] = sample_geometry(sample)
    nm = max(4, int(args.num_modes))

    _msh, _Vp, freqs_hz, _eig, _np, _extra = fem3d._solve_acoustic_cavity_only_evp(
        mesh_file=args.mesh.resolve(),
        config=cfg,
        num_modes=nm,
    )
    loc_hz, method = pick_locator_frequency_hz(
        list(freqs_hz),
        band_lo=float(args.locator_lo_hz),
        band_hi=float(args.locator_hi_hz),
        reference_hz=float(args.reference_hz),
    )
    payload = {
        "locator_frequency_hz": loc_hz,
        "locator_selection_method": method,
        "locator_band_hz": [float(args.locator_lo_hz), float(args.locator_hi_hz)],
        "all_locator_frequencies_hz": [float(f) for f in freqs_hz],
        "reference_hz": float(args.reference_hz),
        "sample_id": str(sample.get("id", "")),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[v2_locator] sample={sample.get('id')} "
            f"locator_frequency_hz={loc_hz:.6f} method={method}",
            flush=True,
        )
    return 0 if math.isfinite(loc_hz) else 1


if __name__ == "__main__":
    raise SystemExit(main())
