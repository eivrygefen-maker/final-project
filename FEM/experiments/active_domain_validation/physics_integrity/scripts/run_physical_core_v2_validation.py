#!/usr/bin/env python3
"""
Experiment-only coupled_physical_core_v2 validation (two subcases, one runner).

Subcases:
  coupling_disabled  — block-diagonal uu/pp on reduced domain (acoustic reference)
  physical_coupling_enabled — physical interface blocks, no empirical v1 gain
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
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
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, dense_to_csr_f32_column, save_mode_csr
from fem_worker_single import _apply_master_worker_solver_profile, hz_result_tag
from mode_diagnostics import (
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
    pressure_subspace_mac,
    write_mode_diagnostics_json,
)

V2_CASE = PHYSICS_ROOT / "coupled_physical_core_v2"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
ACOUSTIC_REF_HZ = 244.3916
BAND_LO = 220.0
BAND_HI = 265.0
REF_TOL_HZ = 1.0


def _resolve_mesh(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, EXPERIMENT_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _acoustic_reference_hz() -> float:
    path = PHYSICS_ROOT / "acoustic_only" / "results" / "result_acoustic.json"
    if path.is_file():
        freqs = json.loads(path.read_text(encoding="utf-8")).get("frequencies_hz") or []
        if freqs:
            return float(freqs[0])
    return ACOUSTIC_REF_HZ


def _reciprocity_check(
    A: Any,
    x0: np.ndarray,
    *,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
) -> Dict[str, float]:
    """Sign/reciprocity sanity: compare u^T A_up p pattern vs p^T A_pu u pattern."""
    from physical_fsi_seed_residual_audit import (
        _mask_on_indices,
        _petsc_matvec,
        _petsc_vec_from_array,
    )

    n = int(x0.size)
    x_p = _mask_on_indices(n, x0, p_idx)
    x_u = _mask_on_indices(n, x0, u_idx)
    vp = _petsc_vec_from_array(A, x_p)
    vu = _petsc_vec_from_array(A, x_u)
    try:
        Ap, _ = _petsc_matvec(A, vp)
        Au, _ = _petsc_matvec(A, vu)
    finally:
        vp.destroy()
        vu.destroy()
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    p_idx = np.asarray(p_idx, dtype=np.int32).ravel()
    up = float(np.vdot(x0[u_idx], Ap[u_idx]))
    pu = float(np.vdot(x0[p_idx], Au[p_idx]))
    nu = float(np.linalg.norm(Ap[u_idx]))
    np_ = float(np.linalg.norm(Au[p_idx]))
    ratio = abs(pu) / max(abs(up), 1.0e-30)
    return {
        "bilinear_up": up,
        "bilinear_pu": pu,
        "reciprocity_ratio_abs_pu_over_up": ratio,
        "reciprocity_balanced": abs(math.log(max(ratio, 1.0e-30))) < math.log(10.0),
    }


def _run_subcase(
    cfg_base: dict,
    config_path: Path,
    *,
    subcase: str,
    coupling_enabled: bool,
    target_hz: float,
) -> Dict[str, Any]:
    case_dir = V2_CASE / subcase
    sorting = case_dir / "sorting"
    for d in (sorting, case_dir / "logs", case_dir / "modes", case_dir / "diagnostics"):
        d.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = bool(coupling_enabled)
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["eps_harvest_rank_by_wood"] = False
    sc["eps_harvest_rank_by_p_frac"] = False
    sc["eps_reject_sigma_spurious"] = False
    sc["eps_reject_decoupled_u_only"] = False
    sc["physics_integrity_branch"] = f"coupled-physical-core-v2-{subcase}"

    nm = _apply_master_worker_solver_profile(
        cfg, num_modes=int(sc.get("num_modes", 12)), structural_only=False
    )
    harvest_lo = float(sc.get("_worker_harvest_lo_hz", BAND_LO))
    harvest_hi = float(sc.get("_worker_harvest_hi_hz", BAND_HI))
    lam_t = (2.0 * math.pi * target_hz) ** 2
    sc["_worker_target_hz"] = target_hz
    sc["_worker_eps_target_lambda"] = lam_t
    sc["_worker_harvest_lo_hz"] = harvest_lo
    sc["_worker_harvest_hi_hz"] = harvest_hi
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_num_modes"] = nm

    mesh_file = _resolve_mesh(cfg, config_path)
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_core_v2] subcase={subcase} coupling_enabled={coupling_enabled} "
            f"num_modes={nm}",
            flush=True,
        )
    t0 = time.perf_counter()
    _msh, W, freqs_hz, eigvecs, _nu, _np = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0

    restr = cfg.get("_coupled_air_pressure_restriction") or {}
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    n_W = int(restr.get("n_reduced_W", u_to_W.size + p_to_W.size))
    v2_meta = cfg.get("_coupled_physical_core_v2") or {}
    eps_diag = cfg.get("_eps_batch_diagnostics") or sc.get("_eps_batch_diagnostics") or {}

    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    mode_rows: List[Dict[str, Any]] = []
    in_band: List[Dict[str, Any]] = []
    hz_tag = hz_result_tag(target_hz)
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0

    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = case_dir / "modes" / f"mode_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
        diag = diagnose_mixed_mode(
            vec,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            gnhep=gnhep,
            frequency_hz=float(freqs_hz[j]),
        )
        diag["mode_index"] = j
        diag["vector_path"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
        mode_rows.append(diag)
        f_hz = float(freqs_hz[j])
        if BAND_LO <= f_hz <= BAND_HI:
            in_band.append({**diag, "frequency_hz": f_hz})

    write_mode_diagnostics_json(case_dir, mode_rows, case_label=subcase, scaling=gnhep)

    reciprocity: Dict[str, float] = {}
    if coupling_enabled and n_modes > 0:
        try:
            sorting_am = case_dir / "sorting_reciprocity"
            sorting_am.mkdir(parents=True, exist_ok=True)
            fem3d.set_sorting_root(sorting_am.resolve())
            cfg_am = copy.deepcopy(cfg)
            _m2, _W2, A, M = fem3d._solve_coupled_evp(
                mesh_file=mesh_file,
                config=cfg_am,
                num_modes=0,
                solve_evp=False,
            )
            reciprocity = _reciprocity_check(
                A, eigvecs[:, 0], u_idx=u_to_W, p_idx=p_to_W
            )
            A.destroy()
            M.destroy()
            fem3d.set_sorting_root(sorting.resolve())
        except Exception as exc:
            reciprocity = {"error": f"{type(exc).__name__}: {exc}"}

    ref_hz = _acoustic_reference_hz()
    best_acoustic = None
    if in_band:
        acoustic = [
            m
            for m in in_band
            if m.get("mode_class") == "acoustic_dominated"
            or float(m.get("p_frac_energy_phys", 0.0)) >= 0.35
        ]
        pool = acoustic if acoustic else in_band
        best_acoustic = min(
            pool, key=lambda m: abs(float(m["frequency_hz"]) - ref_hz)
        )

    result = {
        "subcase": subcase,
        "coupling_enabled": coupling_enabled,
        "elapsed_s": elapsed,
        "n_reduced_W": n_W,
        "n_u_active": int(restr.get("n_u_active", u_to_W.size)),
        "n_p_active": int(restr.get("n_p_active", p_to_W.size)),
        "dropped_inactive_p": int(restr.get("dropped_inactive_p", -1)),
        "soundhole_p_active": int(restr.get("soundhole_p_active", -1)),
        "core_v2_assembly": v2_meta,
        "eps_batch_diagnostics": eps_diag,
        "frequencies_hz": [float(f) for f in freqs_hz],
        "in_band_modes": in_band,
        "acoustic_reference_hz": ref_hz,
        "nearest_acoustic_mode": best_acoustic,
        "reciprocity_sign_check": reciprocity,
        "num_modes_saved": n_modes,
    }
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    return result


def _acceptance(
    disabled: Dict[str, Any],
    enabled: Dict[str, Any],
) -> Dict[str, Any]:
    ref_hz = float(disabled.get("acoustic_reference_hz", ACOUSTIC_REF_HZ))
    near = disabled.get("nearest_acoustic_mode") or {}
    f_dis = float(near.get("frequency_hz", float("nan")))
    decoupled_ok = (
        math.isfinite(f_dis)
        and abs(f_dis - ref_hz) <= REF_TOL_HZ
        and float(near.get("p_frac_energy_phys", 0.0)) >= 0.35
    )
    eps_en = enabled.get("eps_batch_diagnostics") or {}
    nconv = int(eps_en.get("nconv_marked", -1))
    coupled_ok = nconv > 0 and len(enabled.get("in_band_modes") or []) > 0
    milestones = {
        "decoupled_v2_reproduces_acoustic_reference": decoupled_ok,
        "physically_coupled_v2_converges": coupled_ok,
        "modes_reported_with_physical_energy": bool(enabled.get("in_band_modes")),
    }
    passed = milestones["decoupled_v2_reproduces_acoustic_reference"] and (
        milestones["physically_coupled_v2_converges"]
    )
    return {
        "initial_v2_milestone_passed": passed,
        "milestones": milestones,
        "note": (
            "v1 scaled formulation closed; this milestone validates the clean v2 path only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="coupled_physical_core_v2 validation")
    parser.add_argument("--config", type=Path, default=V2_CONFIG)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_core_v2] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    config_path = args.config.resolve()
    cfg_base = json.loads(config_path.read_text(encoding="utf-8"))
    target_hz = float(cfg_base.get("solver", {}).get("_worker_target_hz", 244.39))

    disabled = _run_subcase(
        cfg_base,
        config_path,
        subcase="coupling_disabled",
        coupling_enabled=False,
        target_hz=target_hz,
    )
    enabled = _run_subcase(
        cfg_base,
        config_path,
        subcase="physical_coupling_enabled",
        coupling_enabled=True,
        target_hz=target_hz,
    )
    acceptance = _acceptance(disabled, enabled)

    report = {
        "experiment": "coupled_physical_core_v2",
        "formulation_report": "docs/coupled_physical_core_v2_formulation.md",
        "v1_investigation_closed": True,
        "subcases": {
            "coupling_disabled": disabled,
            "physical_coupling_enabled": enabled,
        },
        "acceptance": acceptance,
    }
    diag = V2_CASE / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    _write_json(diag / "physical_core_v2_validation_report.json", report)

    if MPI.COMM_WORLD.rank == 0:
        md = [
            "# coupled_physical_core_v2 validation",
            "",
            f"**Milestone passed:** `{acceptance['initial_v2_milestone_passed']}`",
            "",
            acceptance.get("note", ""),
            "",
            "## coupling_disabled",
            f"- nearest acoustic f={((disabled.get('nearest_acoustic_mode') or {}).get('frequency_hz', float('nan')))} Hz",
            f"- p_frac_energy_phys="
            f"{((disabled.get('nearest_acoustic_mode') or {}).get('p_frac_energy_phys', float('nan')))}",
            "",
            "## physical_coupling_enabled",
            f"- nconv={((enabled.get('eps_batch_diagnostics') or {}).get('nconv_marked', '?'))}",
            f"- in-band modes={len(enabled.get('in_band_modes') or [])}",
            "",
            "v1 A_up audit is historical closure only; do not extend v1 continuation/Nitsche patches.",
        ]
        (diag / "physical_core_v2_validation_report.md").write_text(
            "\n".join(md) + "\n",
            encoding="utf-8",
        )
        print(
            f"[physical_core_v2] milestone_passed={acceptance['initial_v2_milestone_passed']}"
        )
        print(f"[physical_core_v2] wrote {diag / 'physical_core_v2_validation_report.json'}")

    return 0 if acceptance["initial_v2_milestone_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
