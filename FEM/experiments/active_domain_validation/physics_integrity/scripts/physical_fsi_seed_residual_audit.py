#!/usr/bin/env python3
"""
No-eigensolve seeded residual / Rayleigh audit for physical-FSI continuation (alpha=0.01).

Assembles the same reduced active-pressure GNHEP operators as the continuation pilot,
loads the saved alpha=0 decoupled acoustic seed, and evaluates eigen-residual and Rayleigh
metrics without calling SLEPc/EPS.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from petsc4py import PETSc

from coupled_participation_audit import (
    _acoustic_reference_hz,
    _load_coupled_mode_dense_vector,
    _load_modes,
    _resolve_mesh,
    _write_json,
)
from physical_fsi_continuation_post import (
    ALPHA0_HZ,
    DECOUPLED_CASE,
    PILOT_CASE,
    PILOT_CONFIG,
    _select_reference_mode,
)
from physical_fsi_participation_audit import _catalog_saved_modes

SEED_F_HZ = 244.391600
ALPHA_PILOT = 0.01
REL_GOOD_MAX = 0.10
DELTA_HZ_GOOD_MAX = 8.0
REL_PERTURB_MIN = 0.25
DELTA_HZ_PERTURB_MIN = 15.0
BASELINE_REL_EXPECT_MAX = 0.05
N_U_ACTIVE_EXPECT = 102102
N_P_ACTIVE_EXPECT = 9998
N_REDUCED_W_EXPECT = 112100


def _map_crc32(arr: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(arr, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _map_fingerprint(u_to_W: np.ndarray, p_to_W: np.ndarray) -> Dict[str, Any]:
    u = np.asarray(u_to_W, dtype=np.int32).ravel()
    p = np.asarray(p_to_W, dtype=np.int32).ravel()
    return {
        "len_u_to_W": int(u.size),
        "len_p_to_W": int(p.size),
        "crc32_u_to_W": _map_crc32(u),
        "crc32_p_to_W": _map_crc32(p),
    }


def _validate_reduced_layout(
    A: PETSc.Mat,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    restr: Dict[str, Any],
    *,
    seed_length: int,
    alpha_fsi: float,
) -> Dict[str, Any]:
    n_op = int(A.getSize()[0])
    n_u = int(np.asarray(u_to_W).size)
    n_p = int(np.asarray(p_to_W).size)
    n_W = int(restr.get("n_reduced_W", n_u + n_p))
    checks = {
        "operator_size": n_op,
        "len_u_to_W": n_u,
        "len_p_to_W": n_p,
        "seed_length": int(seed_length),
        "n_reduced_W_metadata": n_W,
    }
    expected = {
        "operator_size": N_REDUCED_W_EXPECT,
        "len_u_to_W": N_U_ACTIVE_EXPECT,
        "len_p_to_W": N_P_ACTIVE_EXPECT,
        "seed_length": N_REDUCED_W_EXPECT,
    }
    failures = [
        f"{k}: got {checks[k]} expected {expected[k]}"
        for k in expected
        if checks[k] != expected[k]
    ]
    if failures:
        raise RuntimeError(
            "physical_fsi_seed_residual_audit: reduced active-pressure layout check failed "
            f"at alpha_fsi={alpha_fsi}: " + "; ".join(failures)
        )
    if not restr:
        raise RuntimeError(
            f"physical_fsi_seed_residual_audit: missing _coupled_air_pressure_restriction "
            f"at alpha_fsi={alpha_fsi} (replay audit did not reduce the operator)"
        )
    return {
        **checks,
        "dropped_inactive_p": int(restr.get("dropped_inactive_p", -1)),
        "soundhole_p_active": int(restr.get("soundhole_p_active", -1)),
        "layout_ok": True,
    }


def _assemble_reduced_continuation_operator(
    cfg_base: dict,
    config_path: Path,
    *,
    alpha_fsi: float,
    sorting_subdir: str,
) -> Tuple[PETSc.Mat, PETSc.Mat, dict, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    No-solve replay: same reduced active-pressure GNHEP as continuation pilot.
    Returns (A, M, cfg, u_to_W, p_to_W, restriction_metadata).
    """
    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    sc["coupled_physical_fsi_continuation_diagnosis"] = True
    sc["coupled_physical_fsi_only_diagnosis"] = False
    sc["coupled_decoupled_union_diagnosis"] = False
    sc["physical_fsi_alpha"] = float(alpha_fsi)
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["coupled_air_pressure_restriction_replay_audit"] = True

    sorting = PILOT_CASE / sorting_subdir
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    mesh_file = _resolve_mesh(cfg, config_path)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_seed_residual_audit] "
            f"Assembling A/M at alpha_fsi={alpha_fsi:.6g} "
            "(reduced replay, no SLEPc solve)...",
            flush=True,
        )
    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )

    if "_coupled_air_u_to_W_map" not in cfg or "_coupled_air_p_to_W_map" not in cfg:
        raise RuntimeError(
            "physical_fsi_seed_residual_audit: reduced maps missing on config after "
            f"assembly at alpha_fsi={alpha_fsi}; ensure "
            "coupled_air_pressure_restriction_replay_audit=True"
        )

    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    return A, M, cfg, u_to_W, p_to_W, restr


def _petsc_vec_from_array(mat: PETSc.Mat, arr: np.ndarray) -> PETSc.Vec:
    v = mat.createVecRight()
    v.setArray(np.asarray(arr, dtype=np.float64).copy())
    try:
        v.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )
    except Exception:
        pass
    return v


def _petsc_matvec(mat: PETSc.Mat, x: PETSc.Vec) -> Tuple[np.ndarray, PETSc.Vec]:
    y = mat.createVecRight()
    mat.mult(x, y)
    try:
        y.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )
    except Exception:
        pass
    arr = np.asarray(y.getArray(readonly=True), dtype=np.float64).copy()
    return arr, y


def _mask_on_indices(n: int, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int32).ravel()
    if idx.size:
        out[idx] = np.asarray(x, dtype=np.float64).ravel()[idx]
    return out


def _block_residual_contributions(
    A: PETSc.Mat,
    M: PETSc.Mat,
    x0: np.ndarray,
    *,
    lam0: float,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
) -> Dict[str, Any]:
    n = int(x0.size)
    x = np.asarray(x0, dtype=np.float64).ravel()
    x_u = _mask_on_indices(n, x, u_idx)
    x_p = _mask_on_indices(n, x, p_idx)

    vx = vx_u = vx_p = None
    ay = ay_u = ay_p = None
    my = my_u = my_p = None
    try:
        vx = _petsc_vec_from_array(A, x)
        vx_u = _petsc_vec_from_array(A, x_u)
        vx_p = _petsc_vec_from_array(A, x_p)
        Ax, ay = _petsc_matvec(A, vx)
        Ax_u, ay_u = _petsc_matvec(A, vx_u)
        Ax_p, ay_p = _petsc_matvec(A, vx_p)
        Mx, my = _petsc_matvec(M, vx)
        Mx_u, my_u = _petsc_matvec(M, vx_u)
        Mx_p, my_p = _petsc_matvec(M, vx_p)
    finally:
        for obj in (my_p, my_u, my, ay_p, ay_u, ay, vx_p, vx_u, vx):
            if obj is not None:
                obj.destroy()

    r_total = Ax - float(lam0) * Mx
    r_uu = np.zeros(n, dtype=np.float64)
    r_pp = np.zeros(n, dtype=np.float64)
    if u_idx.size:
        r_uu[u_idx] = (Ax_u - float(lam0) * Mx_u)[u_idx]
    if p_idx.size:
        r_pp[p_idx] = (Ax_p - float(lam0) * Mx_p)[p_idx]
    r_uu_pp = r_uu + r_pp

    r_a_up = np.zeros(n, dtype=np.float64)
    if u_idx.size:
        r_a_up[u_idx] = Ax_p[u_idx]

    r_a_pu = np.zeros(n, dtype=np.float64)
    if p_idx.size:
        r_a_pu[p_idx] = Ax_u[p_idx]

    r_m_pu = np.zeros(n, dtype=np.float64)
    if p_idx.size:
        r_m_pu[p_idx] = -float(lam0) * Mx_u[p_idx]

    r_fsi = r_a_up + r_a_pu + r_m_pu

    def _norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    nr = _norm(r_total)
    n_ax = _norm(Ax)
    n_mx = _norm(Mx)
    denom = n_ax + abs(float(lam0)) * n_mx
    rel = nr / max(denom, 1.0e-30)

    parts = {
        "uu_pp": r_uu_pp,
        "A_up": r_a_up,
        "A_pu": r_a_pu,
        "M_pu": r_m_pu,
        "fsi_combined": r_fsi,
    }
    block_rows: Dict[str, Any] = {}
    for label, vec in parts.items():
        nv = _norm(vec)
        block_rows[label] = {
            "residual_norm": nv,
            "fraction_of_total_residual_norm": nv / max(nr, 1.0e-30),
            "fraction_of_denominator": nv / max(denom, 1.0e-30),
        }

    return {
        "lambda0_rad2_s2": float(lam0),
        "residual_norm": nr,
        "Ax_norm": n_ax,
        "Mx_norm": n_mx,
        "denominator_norm": denom,
        "relative_residual": rel,
        "block_residual_contributions": block_rows,
    }


def _rayleigh_metrics(
    A: PETSc.Mat,
    M: PETSc.Mat,
    x0: np.ndarray,
    *,
    seed_f_hz: float,
) -> Dict[str, float]:
    vx = ay = my = None
    try:
        vx = _petsc_vec_from_array(A, x0)
        Ax, ay = _petsc_matvec(A, vx)
        Mx, my = _petsc_matvec(M, vx)
    finally:
        for obj in (my, ay, vx):
            if obj is not None:
                obj.destroy()
    x = np.asarray(x0, dtype=np.float64).ravel()
    num = np.vdot(x, Ax)
    den = np.vdot(x, Mx)
    if abs(den) < 1.0e-30:
        lam_r = float("nan")
        f_hz = float("nan")
    else:
        lam_r = float(np.real(num / den))
        f_hz = math.sqrt(max(lam_r, 0.0)) / (2.0 * math.pi)
    return {
        "rayleigh_lambda": lam_r,
        "rayleigh_f_hz": f_hz,
        "delta_rayleigh_from_seed_hz": f_hz - float(seed_f_hz),
        "xH_Ax": float(np.real(num)),
        "xH_Mx": float(np.real(den)),
    }


def _evaluate_at_alpha(
    cfg_base: dict,
    config_path: Path,
    *,
    alpha_fsi: float,
    x0: np.ndarray,
    lam0: float,
    seed_f_hz: float,
    sorting_subdir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """Assemble reduced operator, validate layout, return metrics + map fingerprints."""
    A, M, cfg, u_to_W, p_to_W, restr = _assemble_reduced_continuation_operator(
        cfg_base,
        config_path,
        alpha_fsi=alpha_fsi,
        sorting_subdir=sorting_subdir,
    )
    layout = _validate_reduced_layout(
        A,
        u_to_W,
        p_to_W,
        restr,
        seed_length=int(x0.size),
        alpha_fsi=alpha_fsi,
    )
    maps_fp = _map_fingerprint(u_to_W, p_to_W)

    residual = _block_residual_contributions(
        A, M, x0, lam0=lam0, u_idx=u_to_W, p_idx=p_to_W
    )
    rayleigh = _rayleigh_metrics(A, M, x0, seed_f_hz=seed_f_hz)
    cont = cfg.get("_coupled_physical_fsi_continuation_diagnosis") or {}

    out = {
        "alpha_fsi": float(alpha_fsi),
        "n_u_active": layout["len_u_to_W"],
        "n_p_active": layout["len_p_to_W"],
        "n_reduced_W": layout["operator_size"],
        "dropped_inactive_p": layout["dropped_inactive_p"],
        "soundhole_p_active": layout["soundhole_p_active"],
        "nitsche_disabled": True,
        "pressure_restriction_replay": True,
        "reduced_layout": layout,
        "map_fingerprint": maps_fp,
        "continuation_diagnosis": cont,
        **residual,
        **rayleigh,
    }
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass
    return out, maps_fp, u_to_W, p_to_W


def _load_alpha0_seed(target_hz: float, n_W: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta, files = _load_modes(DECOUPLED_CASE, target_hz)
    catalog = _catalog_saved_modes(DECOUPLED_CASE, meta, files, target_hz=target_hz)
    ref_entry = _select_reference_mode(
        catalog, target_hz=ALPHA0_HZ, prefer_acoustic=True
    )
    if ref_entry is None:
        raise RuntimeError(
            "seed residual audit: no decoupled-union acoustic mode at alpha=0"
        )
    vec, load_meta = _load_coupled_mode_dense_vector(
        ref_entry["path"],
        n_coupled_W=n_W,
        mode_index=int(ref_entry["mode_index"]),
    )
    seed_info = {
        "continuation_seed_source": "alpha0_decoupled_acoustic",
        "seed_f_hz": float(ref_entry["frequency_hz"]),
        "seed_vector_path": ref_entry["vector_path"],
        "seed_mode_index": int(ref_entry["mode_index"]),
        "seed_vector_length": int(vec.size),
        "load_meta": load_meta,
    }
    return vec, seed_info


def _verdict(
    baseline: Dict[str, Any],
    alpha_pilot: Dict[str, Any],
) -> Dict[str, Any]:
    rel0 = float(baseline["relative_residual"])
    rel1 = float(alpha_pilot["relative_residual"])
    dhz = float(alpha_pilot["delta_rayleigh_from_seed_hz"])
    baseline_ok = rel0 <= BASELINE_REL_EXPECT_MAX

    if (
        rel1 <= REL_GOOD_MAX
        and abs(dhz) <= DELTA_HZ_GOOD_MAX
        and rel1 <= max(5.0 * rel0 + 0.02, REL_GOOD_MAX)
    ):
        outcome = "SEED_REMAINS_GOOD_AT_ALPHA_0P01"
        note = (
            "Seed vector remains a modest eigen-residual/Rayleigh match at alpha=0.01; "
            "EPS/ST non-convergence is likely a solver/shift issue, not strong operator perturbation."
        )
    elif (
        rel1 >= REL_PERTURB_MIN
        or abs(dhz) >= DELTA_HZ_PERTURB_MIN
        or rel1 > max(20.0 * rel0, 0.12)
    ):
        outcome = "PHYSICAL_FSI_PERTURBS_SEED_STRONGLY_AT_ALPHA_0P01"
        note = (
            "alpha=0.01 operator strongly perturbs the saved decoupled seed "
            "(residual and/or Rayleigh shift); branch/EPS interpretation premature."
        )
    elif rel1 <= REL_GOOD_MAX and abs(dhz) <= DELTA_HZ_GOOD_MAX:
        outcome = "SEED_REMAINS_GOOD_AT_ALPHA_0P01"
        note = "Seed remains numerically close at alpha=0.01 despite marginal baseline scaling."
    else:
        outcome = "PHYSICAL_FSI_PERTURBS_SEED_STRONGLY_AT_ALPHA_0P01"
        note = (
            "Intermediate residual/Rayleigh metrics; treated as strong perturbation "
            "pending clearer EPS convergence."
        )

    return {
        "outcome": outcome,
        "note": note,
        "baseline_relative_residual": rel0,
        "alpha_0p01_relative_residual": rel1,
        "baseline_ok_for_saved_eigenvector": baseline_ok,
        "no_eps_st_branch_verdict": True,
    }


def _prepare_rayleigh_sigma_config(
    cfg_base: dict,
    *,
    f_rayleigh_hz: float,
    seed_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepared follow-up solve config (not executed by this audit)."""
    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    hz = float(f_rayleigh_hz)
    sc["shift_invert_target_hz"] = hz
    sc["_worker_target_hz"] = hz
    sc["eps_st_sigma_try_target_first"] = True
    sc["eps_st_sigma_primary_offset_hz"] = 0.0
    sc["eps_st_sigma_retry_offsets_hz"] = [0.0]
    sc["eps_st_sigma_include_target_in_ladder"] = True
    sc["physics_integrity_branch"] = (
        "coupled-physical-fsi-continuation-alpha-0.01-rayleigh-sigma-prepared"
    )
    sc["continuation_sigma_prepared_from"] = "physical_fsi_seed_residual_audit"
    sc["continuation_prepared_sigma_hz"] = hz
    return {
        "prepared": True,
        "auto_run": False,
        "recommended_shift_invert_sigma_hz": hz,
        "note": (
            "Use this config for a follow-up seeded EPS only if "
            "SEED_REMAINS_GOOD_AT_ALPHA_0P01; shifts ST sinvert to Rayleigh frequency "
            "instead of the prior ~273.72 Hz retry ladder."
        ),
        "seed_meta": seed_meta,
        "config": cfg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-eigensolve seeded residual/Rayleigh audit (alpha=0.01 continuation)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PILOT_CONFIG,
        help="Continuation pilot config (alpha=0.01 assembly template)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PILOT_CASE / "diagnostics",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[physical_fsi_seed_residual_audit] Requires mpiexec -n 1",
                file=sys.stderr,
            )
        return 2

    config_path = args.config.resolve()
    cfg_base = json.loads(config_path.read_text(encoding="utf-8"))
    target_hz = float(cfg_base.get("solver", {}).get("_worker_target_hz", 244.39))
    acoustic_ref_hz = _acoustic_reference_hz(cfg_base.get("solver", {}))
    lam0 = (2.0 * math.pi * SEED_F_HZ) ** 2

    n_W_hint = int(cfg_base.get("solver", {}).get("continuation_seed_vector_length", 112100))
    x0, seed_meta = _load_alpha0_seed(target_hz, n_W_hint)

    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_seed_residual_audit] "
            f"seed_f_hz={SEED_F_HZ:.6f} lambda0={lam0:.6e} seed_length={x0.size}",
            flush=True,
        )

    baseline, maps0, u0, p0 = _evaluate_at_alpha(
        cfg_base,
        config_path,
        alpha_fsi=0.0,
        x0=x0,
        lam0=lam0,
        seed_f_hz=SEED_F_HZ,
        sorting_subdir="sorting_seed_residual_alpha0",
    )
    alpha_eval, maps1, u1, p1 = _evaluate_at_alpha(
        cfg_base,
        config_path,
        alpha_fsi=ALPHA_PILOT,
        x0=x0,
        lam0=lam0,
        seed_f_hz=SEED_F_HZ,
        sorting_subdir="sorting_seed_residual_alpha0p01",
    )
    maps_identical = bool(np.array_equal(u0, u1) and np.array_equal(p0, p1))
    map_compare = {
        "maps_identical": maps_identical,
        "alpha_0": maps0,
        "alpha_0p01": maps1,
        "crc32_match_u": maps0["crc32_u_to_W"] == maps1["crc32_u_to_W"],
        "crc32_match_p": maps0["crc32_p_to_W"] == maps1["crc32_p_to_W"],
    }
    if not maps_identical and MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_seed_residual_audit][warn] u_to_W/p_to_W differ between "
            "alpha=0 and alpha=0.01 assemblies",
            flush=True,
        )

    diag_verdict = _verdict(baseline, alpha_eval)
    prepared_sigma: Optional[Dict[str, Any]] = None
    if diag_verdict["outcome"] == "SEED_REMAINS_GOOD_AT_ALPHA_0P01":
        f_ray = float(alpha_eval.get("rayleigh_f_hz", float("nan")))
        if math.isfinite(f_ray):
            prepared_sigma = _prepare_rayleigh_sigma_config(
                cfg_base, f_rayleigh_hz=f_ray, seed_meta=seed_meta
            )

    report: Dict[str, Any] = {
        "experiment": "physical_fsi_seed_residual_rayleigh_audit",
        "no_eigensolve": True,
        "acoustic_reference_hz": acoustic_ref_hz,
        "seed": seed_meta,
        "lambda0_rad2_s2": lam0,
        "lambda0_from_seed_f_hz": SEED_F_HZ,
        "alpha_pilot": ALPHA_PILOT,
        "reduced_map_comparison": map_compare,
        "evaluations": {
            "alpha_0_baseline": baseline,
            "alpha_0p01": alpha_eval,
        },
        "verdict": diag_verdict,
        "thresholds": {
            "REL_GOOD_MAX": REL_GOOD_MAX,
            "DELTA_HZ_GOOD_MAX": DELTA_HZ_GOOD_MAX,
            "REL_PERTURB_MIN": REL_PERTURB_MIN,
            "DELTA_HZ_PERTURB_MIN": DELTA_HZ_PERTURB_MIN,
            "BASELINE_REL_EXPECT_MAX": BASELINE_REL_EXPECT_MAX,
        },
        "prior_eps_observation": {
            "st_sigma_retry_hz_approx": 273.72,
            "target_hz": target_hz,
            "note": (
                "Pilot EPS used ST sinvert sigma away from seed/Rayleigh; "
                "this audit does not judge EPS/ST or branch continuity."
            ),
        },
        "prepared_rayleigh_sigma_solve": prepared_sigma,
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "physical_fsi_seed_residual_rayleigh_audit.json"
    _write_json(json_path, report)

    if prepared_sigma is not None:
        cfg_path = (
            PHYSICS_ROOT
            / "configs"
            / "coupled_physical_fsi_alpha_pilot_rayleigh_sigma.PREPARED.json"
        )
        cfg_path.write_text(
            json.dumps(prepared_sigma["config"], indent=2) + "\n",
            encoding="utf-8",
        )
        prepared_sigma["prepared_config_path"] = str(
            cfg_path.relative_to(PHYSICS_ROOT)
        ).replace("\\", "/")

    if MPI.COMM_WORLD.rank == 0:
        md = [
            "# Physical-FSI seed residual / Rayleigh audit (no eigensolve)",
            "",
            f"**Verdict:** `{diag_verdict['outcome']}`",
            "",
            diag_verdict.get("note", ""),
            "",
            f"- Seed: `{seed_meta['seed_vector_path']}` @ {seed_meta['seed_f_hz']:.6f} Hz",
            f"- lambda0 = (2*pi*{SEED_F_HZ})^2 = {lam0:.6e}",
            "",
            "## Reduced map ordering (alpha=0 vs alpha=0.01)",
            f"- maps_identical = {map_compare['maps_identical']}",
            f"- crc32 u_to_W match = {map_compare['crc32_match_u']}",
            f"- crc32 p_to_W match = {map_compare['crc32_match_p']}",
            "",
            "## alpha=0 baseline (decoupled continuation assembly)",
            f"- relative_residual = {baseline['relative_residual']:.6e}",
            f"- rayleigh_f_hz = {baseline['rayleigh_f_hz']:.6f}",
            "",
            "## alpha=0.01",
            f"- relative_residual = {alpha_eval['relative_residual']:.6e}",
            f"- rayleigh_f_hz = {alpha_eval['rayleigh_f_hz']:.6f}",
            f"- delta_rayleigh_from_seed_hz = {alpha_eval['delta_rayleigh_from_seed_hz']:+.4f}",
            "",
            "### Block residual fractions (alpha=0.01)",
        ]
        for label, row in (alpha_eval.get("block_residual_contributions") or {}).items():
            md.append(
                f"- {label}: ||r||/||r_total|| = "
                f"{row['fraction_of_total_residual_norm']:.4f}"
            )
        if prepared_sigma:
            md.extend(
                [
                    "",
                    "## Prepared (not run) Rayleigh-centered sigma solve",
                    f"- sigma_hz ≈ {prepared_sigma['recommended_shift_invert_sigma_hz']:.6f}",
                    f"- config: `{prepared_sigma.get('prepared_config_path')}`",
                ]
            )
        (out_dir / "physical_fsi_seed_residual_rayleigh_audit.md").write_text(
            "\n".join(md) + "\n",
            encoding="utf-8",
        )
        print(f"[physical_fsi_seed_residual_audit] outcome={diag_verdict['outcome']}")
        print(f"[physical_fsi_seed_residual_audit] wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
