#!/usr/bin/env python3
"""Build full W-space EPS seed from acoustic-cavity mode at locator frequency (experiment-only)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from v2_sensitivity_locator import pick_locator_frequency_hz
from v2_sensitivity_mesh import sample_geometry
from wood_library import apply_wood_ids_to_config

V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed acoustic cavity mode into coupled W seed")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--locator-hz", type=float, required=True)
    parser.add_argument("--locator-lo-hz", type=float, default=150.0)
    parser.add_argument("--locator-hi-hz", type=float, default=350.0)
    parser.add_argument("--reference-hz", type=float, default=244.394153389752)
    parser.add_argument("--num-modes", type=int, default=24)
    parser.add_argument("--out-npy", type=Path, required=True)
    parser.add_argument("--out-meta-json", type=Path, required=True)
    parser.add_argument(
        "--out-pressure-npy",
        type=Path,
        default=None,
        help="Optional archive of acoustic-cavity pressure eigenvector (n_p).",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        return 2

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    mesh_path = args.mesh.resolve()

    cfg_ac = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    sc_ac = cfg_ac.setdefault("solver", {})
    sc_ac["mesh_file"] = str(mesh_path)
    sc_ac["acoustic_cavity_only_diagnosis"] = True
    sc_ac["couple_fluid"] = False
    sc_ac["structural_only_diagnosis"] = False
    sc_ac["coupled_physical_core_v2_diagnosis"] = False
    sc_ac["physics_integrity_capture"] = False
    sc_ac["acoustic_cavity_num_modes"] = max(4, int(args.num_modes))
    sc_ac["acoustic_min_mode_hz"] = float(args.locator_lo_hz)
    sc_ac["acoustic_max_mode_hz"] = float(args.locator_hi_hz)
    sc_ac["acoustic_shift_target_hz"] = float(args.locator_hz)
    cfg_ac["geometry"] = sample_geometry(sample)

    _m1, _V1, freqs_hz, eig_ac, _nu, n_p_ac = fem3d._solve_coupled_evp(
        mesh_file=mesh_path, config=cfg_ac, num_modes=int(args.num_modes)
    )
    if int(n_p_ac) <= 0 or eig_ac.size == 0:
        raise RuntimeError("acoustic cavity solve produced no pressure modes for seed")

    loc_hz, _sel = pick_locator_frequency_hz(
        list(freqs_hz),
        band_lo=float(args.locator_lo_hz),
        band_hi=float(args.locator_hi_hz),
        reference_hz=float(args.reference_hz),
    )
    if not math.isfinite(loc_hz):
        raise RuntimeError("cannot pick locator frequency for seed")

    j = int(np.argmin([abs(float(f) - loc_hz) for f in freqs_hz]))
    p_mode = np.asarray(eig_ac[:, j], dtype=np.float64).ravel()
    n_p_mode = int(p_mode.size)

    cfg_c = copy.deepcopy(json.loads(V2_CONFIG.read_text(encoding="utf-8")))
    sc_c = cfg_c.setdefault("solver", {})
    sc_c["mesh_file"] = str(mesh_path)
    sc_c["coupled_physical_core_v2_diagnosis"] = True
    sc_c["coupled_physical_core_v2_coupling_enabled"] = True
    sc_c["fsi_coupling_gain"] = 1.0
    sc_c["fsi_nitsche_enable"] = False
    sc_c["physics_integrity_capture"] = True
    sc_c["coupled_air_pressure_restriction_diagnosis"] = True
    cfg_c["geometry"] = sample_geometry(sample)
    mats = sample.get("materials") or {}
    if mats.get("top_wood_id") or mats.get("back_wood_id"):
        apply_wood_ids_to_config(
            cfg_c,
            top_wood_id=mats.get("top_wood_id"),
            back_wood_id=mats.get("back_wood_id"),
        )

    _m2, _W2, A_replay, M_replay = fem3d._solve_coupled_evp(
        mesh_file=mesh_path, config=cfg_c, num_modes=0, solve_evp=False
    )
    try:
        A_replay.destroy()
        M_replay.destroy()
    except Exception:
        pass
    u_to_W = np.asarray(cfg_c["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg_c["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    n_W = int((cfg_c.get("_coupled_air_pressure_restriction") or {}).get("n_reduced_W", 0))
    if n_W <= 0:
        n_W = int(max(u_to_W.max(), p_to_W.max()) + 1) if u_to_W.size and p_to_W.size else 0
    if n_W <= 0:
        raise RuntimeError("coupled replay did not yield n_reduced_W")

    seed = np.zeros(int(n_W), dtype=np.float64)
    n_map = min(int(p_mode.size), int(p_to_W.size))
    if n_map != int(p_mode.size) or n_map != int(p_to_W.size):
        print(
            f"[seed] WARN acoustic n_p={n_p_mode} coupled n_p={p_to_W.size}; mapping min={n_map}",
            flush=True,
        )
    for k in range(n_map):
        seed[int(p_to_W[k])] = float(p_mode[k])
    norm = float(np.linalg.norm(seed))
    if norm > 0.0:
        seed /= norm

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out_npy.resolve()), seed)
    p_norm = float(np.linalg.norm(p_mode))
    if p_norm > 0.0:
        p_arch = np.asarray(p_mode, dtype=np.float64).ravel() / p_norm
    else:
        p_arch = np.asarray(p_mode, dtype=np.float64).ravel()
    if args.out_pressure_npy is not None:
        args.out_pressure_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.out_pressure_npy.resolve()), p_arch)
    seed_layout_valid = int(seed.size) == int(n_W) and float(np.linalg.norm(seed)) > 0.0
    meta = {
        "locator_frequency_hz": float(loc_hz),
        "picked_mode_index": j,
        "picked_mode_frequency_hz": float(freqs_hz[j]),
        "n_reduced_W": int(n_W),
        "n_p_acoustic_mode": n_p_mode,
        "n_p_coupled_active": int(p_to_W.size),
        "n_u_active": int(u_to_W.size),
        "seed_norm": float(np.linalg.norm(seed)),
        "acoustic_locator_vector_saved": True,
        "locator_pressure_reference_source": "acoustic_only_locator_eigenvector",
        "seed_build_success": True,
        "seed_layout_valid": bool(seed_layout_valid),
        "seed_vector_length": int(seed.size),
        "mapping_note": (
            "Pressure-only acoustic cavity mode embedded on coupled p_to_W indices; "
            "u block zero. Experiment-only EPS initial space — no v2 assembly change."
        ),
    }
    if args.out_pressure_npy is not None:
        meta["acoustic_locator_pressure_npy"] = str(args.out_pressure_npy)
    args.out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[seed] wrote {args.out_npy} f_mode={freqs_hz[j]:.4f} loc_hz={loc_hz:.4f} n_W={n_W}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
