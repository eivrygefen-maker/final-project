#!/usr/bin/env python3
"""
Isolated coupled solve for active-domain validation (baseline or active_domain).

Does not write to FEM/SORTING production paths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_harvest_filter import HarvestFilterConfig, classify_mode_candidate
from fem_mode_array_utils import (
    MODE_VECTOR_FILE_SUFFIX,
    csr_col_norm,
    csr_u_slice,
    dense_to_csr_f32_column,
    save_mode_csr,
)
from fem_worker_single import _apply_master_worker_solver_profile, hz_result_tag
from mpi4py import MPI
from wood_library import resolve_plate_thicknesses


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Active-domain validation coupled solve")
    parser.add_argument(
        "--variant",
        choices=("baseline", "active_domain"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=202.0)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--harvest-lo-hz", type=float, default=156.0)
    parser.add_argument("--harvest-hi-hz", type=float, default=248.0)
    parser.add_argument("--eps-broad-search-hz", type=float, default=46.0)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[experiment] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    variant_root = EXPERIMENT_ROOT / args.variant
    sorting_root = variant_root / "sorting"
    temp_modes = sorting_root / "temp_modes"
    temp_results = sorting_root / "temp_results"
    timing_dir = variant_root / "timing"
    logs_dir = variant_root / "logs"
    modes_dir = variant_root / "modes"
    for d in (temp_modes, temp_results, timing_dir, logs_dir, modes_dir):
        d.mkdir(parents=True, exist_ok=True)

    fem3d.set_sorting_root(sorting_root.resolve())

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ad = cfg.setdefault("solver", {}).setdefault("active_domain_experiment", {})
    if args.variant == "baseline":
        ad["enabled"] = False
    else:
        ad["enabled"] = True
        ad.setdefault("method", "algebraic_restriction")
        ad["bypass_worker_mode_cap"] = True
        ad["timing_dir"] = str(timing_dir.resolve())

    nm = _apply_master_worker_solver_profile(
        cfg,
        num_modes=int(args.num_modes),
        structural_only=False,
        eps_band_solver="shift_invert",
    )
    cfg["solver"]["eps_reject_sigma_spurious"] = False
    cfg["solver"]["eps_reject_target_locked"] = False
    cfg["solver"]["eps_reject_decoupled_u_only"] = False
    cfg["solver"]["eps_harvest_allow_weak_coupling"] = True
    cfg["solver"]["eps_broad_search_hz"] = float(args.eps_broad_search_hz)
    cfg["solver"]["_worker_harvest_lo_hz"] = float(args.harvest_lo_hz)
    cfg["solver"]["_worker_harvest_hi_hz"] = float(args.harvest_hi_hz)

    target_hz = float(args.target_hz)
    target_lambda = (2.0 * math.pi * target_hz) ** 2
    cfg["solver"]["_worker_target_hz"] = target_hz
    cfg["solver"]["_worker_eps_target_lambda"] = target_lambda
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_eps_target_lambda"] = target_lambda
    cfg["_worker_num_modes"] = nm
    cfg["_worker_eps_max_it"] = int(cfg["solver"].get("eigs_maxiter", 3000))

    mesh_file = _resolve_mesh_path(cfg, config_path)
    mesh_audit_path = EXPERIMENT_ROOT / "mesh" / "mesh_audit.json"
    mesh_sha256 = None
    if mesh_audit_path.is_file():
        try:
            mesh_sha256 = json.loads(mesh_audit_path.read_text(encoding="utf-8")).get("sha256")
        except Exception:
            pass
    if MPI.COMM_WORLD.rank == 0:
        print(f"[experiment] variant={args.variant} config={config_path}")
        print(f"[experiment] mesh={mesh_file} sorting_root={sorting_root}")
        try:
            t_top, t_back = resolve_plate_thicknesses(cfg)
            print(f"[experiment] t_top={t_top*1e3:.3f} mm t_back={t_back*1e3:.3f} mm")
        except Exception as exc:
            print(f"[experiment] thickness: {exc}")

    t0 = time.perf_counter()
    try:
        _msh, W, freqs_hz, eigvecs, n_u_reported, n_p_reported = fem3d._solve_coupled_evp(
            mesh_file=mesh_file,
            config=cfg,
            num_modes=nm,
            status_callback=None,
        )
    except Exception as exc:
        if MPI.COMM_WORLD.rank == 0:
            print(f"[experiment] Solve failed: {exc}", file=sys.stderr)
            _write_json(
                timing_dir / "run_status.json",
                {"ok": False, "error": str(exc), "variant": args.variant},
            )
        return 1

    elapsed_s = time.perf_counter() - t0
    V_u, u_to_W = W.sub(0).collapse()
    V_p, p_to_W = W.sub(1).collapse()
    n_u_col = int(V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs)
    n_p_col = int(V_p.dofmap.index_map.size_global * V_p.dofmap.index_map_bs)
    n_mixed = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    ad_meta = cfg.get("_active_domain") or cfg.get("solver", {}).get("_active_domain")
    operator_meta: Dict[str, Any] = {
        "variant": args.variant,
        "n_mixed_global": n_mixed,
        "n_u_collapsed": n_u_col,
        "n_p_collapsed": n_p_col,
        "n_u_reported": int(n_u_reported),
        "n_p_reported": int(n_p_reported),
        "elapsed_s": elapsed_s,
    }
    if isinstance(ad_meta, dict):
        operator_meta["active_domain"] = ad_meta

    hcfg = HarvestFilterConfig.from_solver_cfg(cfg.get("solver", {}))
    st_sigma_hz = float(cfg.get("_worker_st_sigma_hz", target_hz))
    hz_tag = hz_result_tag(target_hz)
    candidates: List[Dict[str, Any]] = []
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    tag1 = list(cfg.pop("_worker_tag1", []) or [])
    tag3 = list(cfg.pop("_worker_tag3", []) or [])
    p_fracs = list(cfg.pop("_worker_p_frac", []) or [])
    p_block_maxes = list(cfg.pop("_worker_p_block_max", []) or [])

    for j in range(n_modes):
        vec_csr = dense_to_csr_f32_column(eigvecs[:, j])
        u_blk = csr_u_slice(vec_csr, n_u_col)
        mode_path = modes_dir / f"mode_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        save_mode_csr(mode_path, vec_csr)
        fj = float(freqs_hz[j])
        wood = float(tag1[j]) + float(tag3[j]) if j < len(tag1) and j < len(tag3) else 0.0
        h_label, h_rom, h_reason = classify_mode_candidate(
            {
                "hz": fj,
                "p_frac": float(p_fracs[j]) if j < len(p_fracs) else 0.0,
                "p_block_max": float(p_block_maxes[j]) if j < len(p_block_maxes) else 0.0,
                "wood_participation": wood,
            },
            target_hz=target_hz,
            st_sigma_hz=st_sigma_hz,
            cfg=hcfg,
        )
        candidates.append(
            {
                "id": j,
                "hz": fj,
                "vector_path": str(mode_path.relative_to(variant_root)).replace("\\", "/"),
                "n_u_collapsed": n_u_col,
                "u_column_norm": float(csr_col_norm(u_blk)),
                "wood_participation": wood,
                "p_frac": float(p_fracs[j]) if j < len(p_fracs) else 0.0,
                "harvest_class": h_label,
                "rom_ready": bool(h_rom),
                "harvest_reason": h_reason,
            }
        )

    result_payload = {
        "variant": args.variant,
        "target_hz": target_hz,
        "st_sigma_hz": st_sigma_hz,
        "harvest_lo_hz": float(args.harvest_lo_hz),
        "harvest_hi_hz": float(args.harvest_hi_hz),
        "eps_broad_search_hz": float(args.eps_broad_search_hz),
        "num_modes_requested": nm,
        "num_modes_slepc": n_modes,
        "candidates": candidates,
        "soundhole_bc": cfg["solver"].get("soundhole_bc"),
        "pressure_gauge": cfg["solver"].get("pressure_gauge"),
        "mesh_file": str(mesh_file),
        "mesh_sha256": mesh_sha256,
        "operator_meta": operator_meta,
    }
    if timing_dir.is_dir() and (timing_dir / "time_stats.json").is_file():
        try:
            result_payload["time_stats"] = json.loads(
                (timing_dir / "time_stats.json").read_text(encoding="utf-8")
            )
        except Exception:
            pass
    _write_json(temp_results / f"result_{hz_tag}.json", result_payload)
    _write_json(timing_dir / "run_summary.json", result_payload)
    _write_json(
        variant_root / "results" / f"result_{hz_tag}.json",
        result_payload,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[experiment] Done variant={args.variant} elapsed={elapsed_s:.1f}s "
            f"modes={n_modes} n_u_col={n_u_col} n_p_col={n_p_col} "
            f"n_active={operator_meta.get('active_domain', {}).get('n_active', n_mixed)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
