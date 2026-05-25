#!/usr/bin/env python3
"""
Run one physics-integrity validation case (isolated; no production SORTING writes).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, dense_to_csr_f32_column, save_mode_csr
from fem_worker_single import _apply_master_worker_solver_profile, hz_result_tag
from mpi4py import MPI
from wood_library import resolve_plate_thicknesses

from mode_diagnostics import (
    diagnose_mixed_mode,
    diagnose_pressure_only_mode,
    diagnose_structural_mode,
    merge_scaling_metadata,
    write_mode_diagnostics_json,
)


CASES = (
    "coupled_nominal",
    "structural_only",
    "acoustic_only",
    "coupled_low_frequency",
    "coupled_near_acoustic",
)


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
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


def _load_acoustic_reference_hz(solver_cfg: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Prefer completed acoustic_only result; fall back to config reference."""
    ref_hz = float(solver_cfg.get("acoustic_reference_hz", 244.39))
    acoustic_result = PHYSICS_ROOT / "acoustic_only" / "results" / "result_acoustic.json"
    acoustic_payload: Optional[Dict[str, Any]] = None
    if acoustic_result.is_file():
        acoustic_payload = json.loads(acoustic_result.read_text(encoding="utf-8"))
        freqs = [float(f) for f in acoustic_payload.get("frequencies_hz") or []]
        if freqs:
            ref_hz = float(freqs[0])
    return ref_hz, acoustic_payload


def _write_coupled_near_acoustic_report(
    case_dir: Path,
    cfg: dict,
    *,
    target_hz: float,
    harvest_lo: float,
    harvest_hi: float,
    mode_rows: List[Dict[str, Any]],
    freqs_hz: List[float],
    n_u_col: int,
    n_p_col: int,
    n_W: int,
    elapsed: float,
) -> None:
    solver_cfg = cfg.get("solver", {})
    branch = str(solver_cfg.get("physics_integrity_branch", "coupled-near-acoustic-244hz"))
    ref_hz, acoustic_payload = _load_acoustic_reference_hz(solver_cfg)
    ref_tol = float(solver_cfg.get("acoustic_reference_tolerance_hz", 8.0))
    min_p_frac = float(solver_cfg.get("coupled_near_acoustic_min_p_frac", 0.05))

    pi = cfg.get("_physics_integrity") or {}
    soundhole_p = int(pi.get("soundhole_pressure_dof_count", 0))
    acoustic_restr = (acoustic_payload or {}).get("soundhole_p_dof_active")
    if acoustic_restr is None and acoustic_payload:
        acoustic_restr = acoustic_payload.get("soundhole_p_dof_active")

    in_band = [
        m
        for m in mode_rows
        if harvest_lo <= float(m.get("frequency_hz", -1.0)) <= harvest_hi
    ]
    in_band.sort(key=lambda m: float(m.get("frequency_hz", 0.0)))

    def _nearest_to_ref(modes: List[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
        best = None
        best_d = ref_tol
        for m in modes:
            d = abs(float(m.get("frequency_hz", 0.0)) - target)
            if d < best_d:
                best_d = d
                best = m
        return best

    nearest_ref = _nearest_to_ref(in_band, ref_hz)
    pressure_near_ref = [
        m
        for m in in_band
        if abs(float(m.get("frequency_hz", 0.0)) - ref_hz) <= ref_tol
        and float(m.get("p_frac_phys_gnhep", 0.0)) >= min_p_frac
    ]
    has_pressure_coupled_near_ref = len(pressure_near_ref) > 0

    mode_table = [
        {
            "frequency_hz": float(m.get("frequency_hz", 0.0)),
            "p_frac_raw": float(m.get("p_frac_raw", 0.0)),
            "p_frac_phys_gnhep": float(m.get("p_frac_phys_gnhep", 0.0)),
            "wood_participation": float(m.get("wood_participation", 0.0)),
            "mode_class": m.get("mode_class"),
            "delta_from_acoustic_ref_hz": float(m.get("frequency_hz", 0.0)) - ref_hz,
        }
        for m in in_band
    ]

    summary = {
        "branch": branch,
        "solver_branch": "coupled_fsi_evp",
        "soundhole_bc": pi.get("soundhole_bc", solver_cfg.get("soundhole_bc")),
        "pressure_gauge": pi.get("pressure_gauge", solver_cfg.get("pressure_gauge")),
        "soundhole_pressure_dof_count": soundhole_p,
        "acoustic_only_soundhole_p_dof_active": acoustic_restr,
        "n_u_collapsed": n_u_col,
        "n_p_collapsed": n_p_col,
        "n_coupled_W_dofs": n_W,
        "search_band_hz": [harvest_lo, harvest_hi],
        "target_hz": target_hz,
        "acoustic_reference_hz": ref_hz,
        "acoustic_reference_tolerance_hz": ref_tol,
        "coupled_frequencies_in_band_hz": [float(m["frequency_hz"]) for m in mode_table],
        "modes_in_band": mode_table,
        "nearest_mode_to_acoustic_ref": nearest_ref,
        "pressure_participating_modes_near_ref": pressure_near_ref,
        "pressure_coupled_mode_near_acoustic_ref": has_pressure_coupled_near_ref,
        "elapsed_s": elapsed,
        "acoustic_only_result_path": str(
            (PHYSICS_ROOT / "acoustic_only" / "results" / "result_acoustic.json").relative_to(
                REPO_ROOT
            )
        )
        if acoustic_payload
        else None,
    }
    diag = case_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    _write_json(diag / "coupled_near_acoustic_summary.json", summary)

    if MPI.COMM_WORLD.rank != 0:
        return

    print(f"[physics_integrity] branch: {branch}")
    print(
        f"[physics_integrity] soundhole: {summary['soundhole_bc']!r} "
        f"pressure_gauge={summary['pressure_gauge']!r} "
        f"active_soundhole_p_dof={soundhole_p}"
    )
    if acoustic_restr is not None:
        print(
            f"[physics_integrity] acoustic_only reference soundhole_p_dof_active={acoustic_restr}"
        )
    print(
        f"[physics_integrity] DOFs: n_u={n_u_col} n_p={n_p_col} "
        f"n_coupled_W={n_W} (collapsed subspaces)"
    )
    print(
        f"[physics_integrity] search band [{harvest_lo:.1f}, {harvest_hi:.1f}] Hz "
        f"target={target_hz:.2f} Hz → {len(in_band)} modes"
    )
    print(f"[physics_integrity] acoustic-only reference: {ref_hz:.2f} Hz (tol ±{ref_tol:.1f} Hz)")
    for row in mode_table:
        print(
            f"  f={row['frequency_hz']:8.3f} Hz  "
            f"p_frac={row['p_frac_phys_gnhep']:.4f}  "
            f"wood={row['wood_participation']:.4f}  "
            f"class={row['mode_class']}  "
            f"Δref={row['delta_from_acoustic_ref_hz']:+.3f} Hz"
        )
    if nearest_ref:
        print(
            f"[physics_integrity] nearest in-band mode to acoustic ref: "
            f"f={float(nearest_ref['frequency_hz']):.3f} Hz "
            f"p_frac={float(nearest_ref.get('p_frac_phys_gnhep', 0.0)):.4f} "
            f"class={nearest_ref.get('mode_class')}"
        )
    print(
        f"[physics_integrity] pressure-participating coupled mode near "
        f"{ref_hz:.2f} Hz: {'YES' if has_pressure_coupled_near_ref else 'NO'} "
        f"({len(pressure_near_ref)} mode(s) with p_frac>={min_p_frac})"
    )


def _save_physics_audit(cfg: dict, case_dir: Path) -> None:
    audit = cfg.get("_physics_integrity") or cfg.get("solver", {}).get("_physics_integrity")
    if not audit:
        return
    diag = case_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    _write_json(diag / "physics_integrity_audit.json", audit)


def _run_coupled_like(
    cfg: dict,
    config_path: Path,
    case_dir: Path,
    *,
    target_hz: float,
    num_modes: int,
    harvest_lo: float,
    harvest_hi: float,
    eps_broad: float,
    case_label: str = "coupled",
    emit_near_acoustic_report: bool = False,
) -> int:
    sorting_root = case_dir / "sorting"
    temp_modes = sorting_root / "temp_modes"
    temp_results = sorting_root / "temp_results"
    for d in (temp_modes, temp_results, case_dir / "logs", case_dir / "timing", case_dir / "modes"):
        d.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting_root.resolve())

    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["active_domain_experiment"] = {"enabled": False}

    nm = _apply_master_worker_solver_profile(
        cfg,
        num_modes=int(num_modes),
        structural_only=False,
        eps_band_solver="shift_invert",
    )
    cfg["solver"]["eps_reject_sigma_spurious"] = False
    cfg["solver"]["eps_reject_target_locked"] = False
    cfg["solver"]["eps_reject_decoupled_u_only"] = False
    cfg["solver"]["eps_harvest_allow_weak_coupling"] = True
    cfg["solver"]["eps_broad_search_hz"] = float(eps_broad)
    cfg["solver"]["_worker_harvest_lo_hz"] = float(harvest_lo)
    cfg["solver"]["_worker_harvest_hi_hz"] = float(harvest_hi)
    target_lambda = (2.0 * math.pi * target_hz) ** 2
    cfg["solver"]["_worker_target_hz"] = target_hz
    cfg["solver"]["_worker_eps_target_lambda"] = target_lambda
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_eps_target_lambda"] = target_lambda
    cfg["_worker_num_modes"] = nm

    mesh_file = _resolve_mesh_path(cfg, config_path)
    t0 = time.perf_counter()
    msh, W, freqs_hz, eigvecs, n_u_rep, n_p_rep = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0
    _save_physics_audit(cfg, case_dir)

    V_u, u_to_W = W.sub(0).collapse()
    V_p, p_to_W = W.sub(1).collapse()
    n_u_col = int(V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs)
    n_p_col = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)
    n_W = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})
    if "pressure_dof_scale" in pi:
        gnhep["pressure_dof_scale"] = float(pi["pressure_dof_scale"])

    hz_tag = hz_result_tag(target_hz)
    st_sigma = float(cfg.get("_worker_st_sigma_hz", target_hz))
    tag1 = list(cfg.pop("_worker_tag1", []) or [])
    tag3 = list(cfg.pop("_worker_tag3", []) or [])
    p_fracs = list(cfg.pop("_worker_p_frac", []) or [])

    mode_rows: List[Dict[str, Any]] = []
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    modes_dir = case_dir / "modes"
    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = modes_dir / f"mode_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
        diag = diagnose_mixed_mode(
            vec,
            u_to_W=np.asarray(u_to_W, dtype=np.int32),
            p_to_W=np.asarray(p_to_W, dtype=np.int32),
            gnhep=gnhep,
            wood_top=float(tag1[j]) if j < len(tag1) else 0.0,
            wood_back=float(tag3[j]) if j < len(tag3) else 0.0,
            frequency_hz=float(freqs_hz[j]),
        )
        diag["mode_index"] = j
        diag["p_frac_production"] = float(p_fracs[j]) if j < len(p_fracs) else diag["p_frac_raw"]
        diag["vector_path"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
        mode_rows.append(diag)

    write_mode_diagnostics_json(case_dir, mode_rows, case_label=case_label, scaling=gnhep)

    result = {
        "case": case_dir.name,
        "target_hz": target_hz,
        "harvest_lo_hz": harvest_lo,
        "harvest_hi_hz": harvest_hi,
        "num_modes": n_modes,
        "frequencies_hz": [float(f) for f in freqs_hz],
        "elapsed_s": elapsed,
        "n_u_collapsed": n_u_col,
        "n_p_collapsed": n_p_col,
        "n_coupled_W_dofs": n_W,
        "physics_integrity": cfg.get("_physics_integrity"),
        "mode_diagnostics": str((case_dir / "diagnostics" / "mode_physics_diagnostics.json").relative_to(case_dir)),
    }
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    _write_json(case_dir / "timing" / "run_summary.json", result)
    if emit_near_acoustic_report:
        _write_coupled_near_acoustic_report(
            case_dir,
            cfg,
            target_hz=target_hz,
            harvest_lo=harvest_lo,
            harvest_hi=harvest_hi,
            mode_rows=mode_rows,
            freqs_hz=[float(f) for f in freqs_hz],
            n_u_col=n_u_col,
            n_p_col=n_p_col,
            n_W=n_W,
            elapsed=elapsed,
        )
    if MPI.COMM_WORLD.rank == 0:
        print(f"[physics_integrity] {case_dir.name} done modes={n_modes} elapsed={elapsed:.1f}s")
    return 0


def _run_structural(cfg: dict, config_path: Path, case_dir: Path) -> int:
    sorting_root = case_dir / "sorting"
    sorting_root.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting_root.resolve())
    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["structural_only_diagnosis"] = True
    cfg["solver"]["couple_fluid"] = False
    cfg["solver"].pop("acoustic_cavity_only_diagnosis", None)
    nm = int(cfg["solver"].get("structural_only_num_modes", 30))
    mesh_file = _resolve_mesh_path(cfg, config_path)
    t0 = time.perf_counter()
    msh, V_u, freqs_hz, eigvecs, n_u, _ = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0
    _save_physics_audit(cfg, case_dir)

    gnhep = merge_scaling_metadata(case_dir)
    gnhep.setdefault("s_uu", 1.0)
    mode_rows: List[Dict[str, Any]] = []
    modes_dir = case_dir / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    hz_tag = "structural"
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = modes_dir / f"mode_struct_{j:03d}.npz"
        np.savez(mode_path, eigvec=vec)
        diag = diagnose_structural_mode(
            vec,
            wood_top=0.5,
            wood_back=0.5,
            frequency_hz=float(freqs_hz[j]),
            gnhep=gnhep,
        )
        diag["mode_index"] = j
        diag["vector_path"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
        mode_rows.append(diag)

    write_mode_diagnostics_json(case_dir, mode_rows, case_label="structural_only", scaling=gnhep)
    band_lo = float(cfg["solver"].get("structural_expected_hz_min", 120.0))
    band_hi = float(cfg["solver"].get("structural_expected_hz_max", 280.0))
    in_band = [f for f in freqs_hz if band_lo <= f <= band_hi]
    result = {
        "case": "structural_only",
        "frequencies_hz": [float(f) for f in freqs_hz],
        "frequencies_in_wood_band_hz": [float(f) for f in in_band],
        "wood_band_hz": [band_lo, band_hi],
        "num_modes": n_modes,
        "elapsed_s": elapsed,
    }
    _write_json(case_dir / "results" / "result_structural.json", result)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[physics_integrity] structural_only modes={n_modes} in_band={len(in_band)}")
    return 0


def _run_acoustic(cfg: dict, config_path: Path, case_dir: Path) -> int:
    sorting_root = case_dir / "sorting"
    sorting_root.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting_root.resolve())
    cfg.setdefault("solver", {})["physics_integrity_capture"] = True
    cfg["solver"]["acoustic_cavity_only_diagnosis"] = True
    cfg["solver"]["couple_fluid"] = False
    cfg["solver"]["structural_only_diagnosis"] = False
    nm = int(cfg["solver"].get("acoustic_cavity_num_modes", 20))
    mesh_file = _resolve_mesh_path(cfg, config_path)
    t0 = time.perf_counter()
    msh, V_space, freqs_hz, eigvecs, n_u_reported, n_p_reported = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0
    if int(n_u_reported) > 0 and int(n_p_reported) == 0:
        raise RuntimeError(
            "acoustic_only case was dispatched to the structural-only shell solver "
            f"(n_u={n_u_reported}, n_p={n_p_reported}). Check acoustic_cavity_only_diagnosis "
            "and solver branch order."
        )
    if int(n_p_reported) <= 0:
        raise RuntimeError(
            f"acoustic_only case returned no pressure DOFs (n_u={n_u_reported}, n_p={n_p_reported})."
        )
    V_p = V_space
    n_p = int(n_p_reported)
    _save_physics_audit(cfg, case_dir)

    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    mode_rows: List[Dict[str, Any]] = []
    modes_dir = case_dir / "modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = modes_dir / f"mode_acoustic_{j:03d}.npz"
        np.savez(mode_path, eigvec=vec)
        diag = diagnose_pressure_only_mode(vec, gnhep=gnhep, frequency_hz=float(freqs_hz[j]))
        diag["mode_index"] = j
        diag["vector_path"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
        mode_rows.append(diag)

    write_mode_diagnostics_json(case_dir, mode_rows, case_label="acoustic_only", scaling=gnhep)
    restr = (cfg.get("_physics_integrity") or {}).get("acoustic_pressure_restriction") or {}
    result = {
        "case": "acoustic_only",
        "solver_branch": "acoustic_cavity_only_diagnosis",
        "n_u_dofs": int(n_u_reported),
        "n_p_dofs": n_p,
        "n_p_full": int(restr.get("n_p_full", n_p)),
        "n_p_active": int(restr.get("n_p_active", n_p)),
        "soundhole_p_dof_active": int(restr.get("soundhole_p_dof_active", 0)),
        "frequencies_hz": [float(f) for f in freqs_hz],
        "num_modes": n_modes,
        "elapsed_s": elapsed,
        "first_acoustic_candidate_hz": float(freqs_hz[0]) if freqs_hz else None,
    }
    _write_json(case_dir / "results" / "result_acoustic.json", result)
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physics_integrity] acoustic_only (pressure cavity) modes={n_modes} "
            f"n_p={n_p} freqs={[round(f, 2) for f in freqs_hz[:5]]}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Physics-integrity validation case runner")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ingest-baseline", action="store_true", help="Analyze existing ../baseline without solve")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physics_integrity] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    case_dir = PHYSICS_ROOT / args.case
    for sub in ("logs", "results", "modes", "diagnostics", "timing"):
        (case_dir / sub).mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if args.ingest_baseline and args.case == "coupled_nominal":
        return _ingest_baseline(case_dir, cfg)

    if args.case in ("coupled_nominal", "coupled_low_frequency", "coupled_near_acoustic"):
        sc = cfg["solver"]
        defaults = {
            "coupled_nominal": (202.0, 156.0, 248.0, 46.0, 8),
            "coupled_low_frequency": (120.0, 60.0, 200.0, 80.0, 8),
            "coupled_near_acoustic": (244.39, 220.0, 265.0, 45.0, 16),
        }
        t_def, lo_def, hi_def, broad_def, nm_def = defaults[args.case]
        return _run_coupled_like(
            cfg,
            config_path,
            case_dir,
            target_hz=float(sc.get("_worker_target_hz", sc.get("shift_invert_target_hz", t_def))),
            num_modes=int(sc.get("num_modes", nm_def)),
            harvest_lo=float(sc.get("_worker_harvest_lo_hz", lo_def)),
            harvest_hi=float(sc.get("_worker_harvest_hi_hz", hi_def)),
            eps_broad=float(sc.get("eps_broad_search_hz", broad_def)),
            case_label=args.case,
            emit_near_acoustic_report=(args.case == "coupled_near_acoustic"),
        )
    if args.case == "structural_only":
        return _run_structural(cfg, config_path, case_dir)
    if args.case == "acoustic_only":
        return _run_acoustic(cfg, config_path, case_dir)
    return 1


def _ingest_baseline(case_dir: Path, cfg: dict) -> int:
    """Reuse successful active_domain_validation/baseline outputs if present."""
    baseline_root = EXPERIMENT_ROOT / "baseline"
    hz_tag = hz_result_tag(float(cfg["solver"].get("shift_invert_target_hz", 202.0)))
    src_result = baseline_root / "results" / f"result_{hz_tag}.json"
    if not src_result.is_file():
        alt = baseline_root / "timing" / "run_summary.json"
        src_result = alt if alt.is_file() else src_result
    if not src_result.is_file():
        print(
            f"[physics_integrity] No baseline at {src_result}; run coupled_nominal solve instead.",
            file=sys.stderr,
        )
        return 1

    result = json.loads(src_result.read_text(encoding="utf-8"))
    (case_dir / "results").mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "results" / src_result.name, result)

    src_modes = baseline_root / "modes"
    dst_modes = case_dir / "modes"
    if src_modes.is_dir():
        import shutil

        for f in src_modes.glob(f"mode_{hz_tag}_*{MODE_VECTOR_FILE_SUFFIX}"):
            shutil.copy2(f, dst_modes / f.name)

    logs_src = baseline_root / "logs"
    if logs_src.is_dir():
        import shutil

        for f in logs_src.glob("*.log"):
            shutil.copy2(f, case_dir / "logs" / f.name)

    gnhep = merge_scaling_metadata(case_dir, result)
    candidates = result.get("candidates") or []
    mode_rows: List[Dict[str, Any]] = []
    for c in candidates:
        j = int(c.get("id", 0))
        vpath = case_dir / str(c.get("vector_path", ""))
        if not vpath.is_file():
            vpath = dst_modes / Path(str(c.get("vector_path", ""))).name
        if not vpath.is_file():
            continue
        # Mixed baseline modes need collapse maps — re-solve capture or skip full diag
        mode_rows.append(
            {
                "mode_index": j,
                "frequency_hz": float(c.get("hz", 0.0)),
                "p_frac_production": float(c.get("p_frac", 0.0)),
                "wood_participation": float(c.get("wood_participation", 0.0)),
                "vector_path": str(vpath.relative_to(case_dir)).replace("\\", "/"),
                "note": "ingested_from_baseline; run analyze_modes.py with --rebuild-maps for full GNHEP metrics",
            }
        )
    write_mode_diagnostics_json(case_dir, mode_rows, case_label="coupled_nominal_ingested", scaling=gnhep)
    print(f"[physics_integrity] Ingested baseline → {case_dir} ({len(mode_rows)} modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
