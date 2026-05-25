#!/usr/bin/env python3
"""MPI worker: one v2 sensitivity sample eigen solve (invoked via mpiexec -n 1)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PHYSICS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PHYSICS_ROOT / "scripts"))

import fem_main_3d as fem3d
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, dense_to_csr_f32_column, save_mode_csr
from fem_worker_single import _apply_master_worker_solver_profile, hz_result_tag
from mode_diagnostics import (
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
)
from v2_sensitivity_mesh import sample_geometry

SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
BAND_LO = 220.0
BAND_HI = 265.0
BASELINE_F_HZ = 244.39159990162557
ENERGY_ACOUSTIC_THRESHOLD = 0.85


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _classify_phys_energy(p_frac: float) -> str:
    if float(p_frac) >= ENERGY_ACOUSTIC_THRESHOLD:
        return "acoustic_dominated"
    if float(p_frac) <= 0.15:
        return "structural_dominated"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 sensitivity MPI solve worker")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=244.39)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[v2_sensitivity_solve] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    cfg_base = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    mesh_path = args.mesh.resolve()
    sample_id = str(args.sample_id)
    target_hz = float(args.target_hz)
    case_dir = SENS_ROOT / "samples" / sample_id
    sorting = case_dir / "sorting"
    for d in (sorting, case_dir / "logs", case_dir / "modes", case_dir / "diagnostics"):
        d.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(mesh_path)
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = True
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["physics_integrity_branch"] = f"v2-sensitivity-{sample_id}"
    sc["_worker_target_hz"] = target_hz
    sc["_worker_harvest_lo_hz"] = BAND_LO
    sc["_worker_harvest_hi_hz"] = BAND_HI
    cfg["geometry"] = sample_geometry(sample)
    mo = sample.get("materials_override") or {}
    top = mo.get("top") or {}
    scale = float(top.get("E_L_scale", 1.0))
    if abs(scale - 1.0) > 1.0e-12:
        mat = cfg.setdefault("materials", {}).setdefault("top", {})
        mat["E_L"] = float(mat.get("E_L", 0.0)) * scale

    eps_band_solver = str(sc.get("eps_band_solver", "shift_invert")).strip() or "shift_invert"
    nm = _apply_master_worker_solver_profile(
        cfg,
        num_modes=int(sc.get("num_modes", 12)),
        structural_only=False,
        eps_band_solver=eps_band_solver,
    )
    lam_t = (2.0 * math.pi * target_hz) ** 2
    sc["_worker_eps_target_lambda"] = lam_t
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_num_modes"] = nm

    t0 = time.perf_counter()
    _msh, _W, freqs_hz, eigvecs, _nu, _np = fem3d._solve_coupled_evp(
        mesh_file=mesh_path,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0

    restr = cfg.get("_coupled_air_pressure_restriction") or {}
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    cfg_am = copy.deepcopy(cfg)
    cfg_am.setdefault("solver", {})["coupled_air_pressure_restriction_replay_audit"] = True
    sorting_am = case_dir / "sorting_energy"
    sorting_am.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting_am.resolve())
    _m2, _W2, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_path,
        config=cfg_am,
        num_modes=0,
        solve_evp=False,
    )
    fem3d.set_sorting_root(sorting.resolve())

    hz_tag = hz_result_tag(target_hz)
    mode_rows: List[Dict[str, Any]] = []
    in_band: List[Dict[str, Any]] = []
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = case_dir / "modes" / f"mode_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
        diag = diagnose_mixed_mode(
            vec, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep, frequency_hz=float(freqs_hz[j])
        )
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        row = {
            **diag,
            **{k: energy[k] for k in energy if k.endswith("_phys") or k == "p_frac_energy_phys"},
            "mode_index": j,
            "frequency_hz": float(freqs_hz[j]),
            "vector_path": str(mode_path.relative_to(case_dir)).replace("\\", "/"),
            "mode_class_physical_energy": _classify_phys_energy(
                float(energy["p_frac_energy_phys"])
            ),
        }
        mode_rows.append(row)
        if BAND_LO <= float(freqs_hz[j]) <= BAND_HI:
            in_band.append(row)

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    eps_diag = cfg.get("_eps_batch_diagnostics") or sc.get("_eps_batch_diagnostics") or {}
    acoustic_pool = [
        m
        for m in in_band
        if m["mode_class_physical_energy"] == "acoustic_dominated"
        or float(m["p_frac_energy_phys"]) >= 0.35
    ]
    pool = acoustic_pool if acoustic_pool else in_band
    nearest = (
        min(pool, key=lambda m: abs(float(m["frequency_hz"]) - BASELINE_F_HZ)) if pool else None
    )

    result = {
        "sample_id": sample_id,
        "elapsed_s": elapsed,
        "mesh_file": str(mesh_path),
        "n_reduced_W": int(restr.get("n_reduced_W", -1)),
        "n_u_active": int(restr.get("n_u_active", u_to_W.size)),
        "n_p_active": int(restr.get("n_p_active", p_to_W.size)),
        "p_to_W": p_to_W.tolist(),
        "eps_batch_diagnostics": eps_diag,
        "nconv_marked": int(eps_diag.get("nconv_marked", -1)),
        "v2_converged": int(eps_diag.get("nconv_marked", -1)) > 0,
        "in_band_modes": in_band,
        "nearest_acoustic_branch": nearest,
        "num_modes_saved": n_modes,
        "gnhep_scales": {k: float(gnhep.get(k, 1.0)) for k in ("s_uu", "s_pp", "s_couple")},
    }
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    _write_json(case_dir / "diagnostics" / "mode_energy_summary.json", {"modes": mode_rows})
    if MPI.COMM_WORLD.rank == 0:
        print(f"[v2_sensitivity_solve] sample={sample_id} v2_converged={result['v2_converged']}")
    return 0 if result["v2_converged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
