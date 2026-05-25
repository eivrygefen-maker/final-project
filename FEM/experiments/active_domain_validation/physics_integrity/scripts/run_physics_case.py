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

    write_mode_diagnostics_json(case_dir, mode_rows, case_label="coupled", scaling=gnhep)

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
        "physics_integrity": cfg.get("_physics_integrity"),
        "mode_diagnostics": str((case_dir / "diagnostics" / "mode_physics_diagnostics.json").relative_to(case_dir)),
    }
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    _write_json(case_dir / "timing" / "run_summary.json", result)
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
    cfg["solver"].pop("structural_only_diagnosis", None)
    nm = int(cfg["solver"].get("acoustic_cavity_num_modes", 20))
    mesh_file = _resolve_mesh_path(cfg, config_path)
    t0 = time.perf_counter()
    msh, V_p, freqs_hz, eigvecs, _, n_p = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0
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
    result = {
        "case": "acoustic_only",
        "frequencies_hz": [float(f) for f in freqs_hz],
        "num_modes": n_modes,
        "elapsed_s": elapsed,
        "first_acoustic_candidate_hz": float(freqs_hz[0]) if freqs_hz else None,
    }
    _write_json(case_dir / "results" / "result_acoustic.json", result)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[physics_integrity] acoustic_only modes={n_modes} freqs={[round(f, 2) for f in freqs_hz[:5]]}")
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

    if args.case in ("coupled_nominal", "coupled_low_frequency"):
        return _run_coupled_like(
            cfg,
            config_path,
            case_dir,
            target_hz=float(cfg["solver"].get("_worker_target_hz", cfg["solver"].get("shift_invert_target_hz", 202.0))),
            num_modes=int(cfg["solver"].get("num_modes", 8)),
            harvest_lo=float(cfg["solver"].get("_worker_harvest_lo_hz", 156.0)),
            harvest_hi=float(cfg["solver"].get("_worker_harvest_hi_hz", 248.0)),
            eps_broad=float(cfg["solver"].get("eps_broad_search_hz", 46.0)),
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
