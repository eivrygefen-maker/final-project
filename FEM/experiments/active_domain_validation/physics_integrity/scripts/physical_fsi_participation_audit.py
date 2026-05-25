#!/usr/bin/env python3
"""
No-eigensolve audit: physical-FSI-only mode at ~245.30 Hz vs decoupled-union acoustic at ~244.39 Hz.

Assembles reduced-domain A/M once (physical FSI, Nitsche off), replays saved mode vectors,
computes participation metrics and pressure-subspace MAC, and refreshes experiment verdict.
"""
from __future__ import annotations

import argparse
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
from fem_worker_single import hz_result_tag

from coupled_participation_audit import (
    _acoustic_reference_hz,
    _load_coupled_mode_dense_vector,
    _load_modes,
    _resolve_mesh,
    _write_json,
)
from mode_diagnostics import (
    P_FRAC_FULLY_UNSCALED_DEFINITION,
    P_FRAC_PHYS_GNHEP_DEFINITION,
    P_FRAC_PRODUCTION_DEFINITION,
    block_l2_p_fraction,
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
    pressure_subspace_mac,
    unscale_mixed_mode_vector,
)

DECOUPLED_CASE = PHYSICS_ROOT / "coupled_decoupled_union"
PHYSICAL_CASE = PHYSICS_ROOT / "coupled_physical_fsi_only"
DECOUPLED_CONFIG = PHYSICS_ROOT / "configs" / "coupled_decoupled_union.json"
PHYSICAL_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_fsi_only.json"

WOOD_PARTICIPATION_NOTE = (
    "wood_participation in mode_physics_diagnostics is top_plate_frac + back_plate_frac from "
    "harvest/sifter: each term is the fraction of structural (shell) kinetic energy on that plate "
    "facet tag relative to the total structural energy on top+back, not air-vs-wood volume "
    "participation. With tiny ||u|| and p-dominated modes it often saturates at 1.0 because "
    "essentially all resolved structural energy sits on one plate tag; it does not contradict "
    "pressure dominance in the global mixed eigenvector."
)

MAC_SCALING_NOTE = (
    "Pressure MAC uses only active air-supported pressure DOFs (n_p_active collapse indices mapped "
    "to reduced W). Inner product is np.vdot (complex-conjugate on the first vector); for real "
    "SLEPc coefficients this is a standard dot product. Uniform positive scalings (raw, s_pp, "
    "s_pp/p_scale) cancel in MAC = |<a,b>|/(||a|| ||b||). Three scalings are reported for "
    "documentation; MAC values should agree within numerical tolerance when both vectors use the "
    "same p_to_W ordering."
)

ENERGY_PARTICIPATION_THRESHOLD = 0.35
MAC_BRANCH_MATCH_THRESHOLD = 0.85
BAND_LO_HZ = 220.0
BAND_HI_HZ = 265.0
PRESSURE_DOMINATED_BAND_HZ = (
    222.096964,
    245.299844,
    256.474653,
)


def _load_result_frequencies(case_dir: Path, target_hz: float) -> List[float]:
    """Frequencies from case result/timing JSON (not mode filename tags)."""
    hz_tag = hz_result_tag(target_hz)
    freqs: List[float] = []
    for rel in (
        f"results/result_{hz_tag}.json",
        f"timing/run_summary.json",
    ):
        path = case_dir / rel
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for f in payload.get("frequencies_hz") or []:
            freqs.append(float(f))
        if freqs:
            break
    return freqs


def _summary_hint_frequency(
    case_dir: Path,
    summary_name: str,
    *,
    prefer_hz: Optional[float] = None,
    prefer_keys: Optional[Tuple[str, ...]] = None,
) -> Optional[float]:
    path = case_dir / "diagnostics" / summary_name
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = prefer_keys or (
        "acoustic_recovered_modes",
        "acoustic_candidate_modes",
        "ranking_by_proximity_to_acoustic_ref",
        "modes_in_band",
        "modes_all",
    )
    for key in keys:
        modes = payload.get(key) or []
        if not modes:
            continue
        if prefer_hz is not None:
            best = min(
                modes,
                key=lambda m: abs(float(m.get("frequency_hz", float("nan"))) - prefer_hz),
            )
        else:
            best = min(
                modes,
                key=lambda m: abs(float(m.get("delta_from_acoustic_ref_hz", 0.0))),
            )
        f_hz = float(best.get("frequency_hz", float("nan")))
        if np.isfinite(f_hz):
            return f_hz
    return None


def _catalog_saved_modes(
    case_dir: Path,
    modes_meta: List[Dict[str, Any]],
    mode_files: List[Path],
    *,
    target_hz: float,
) -> List[Dict[str, Any]]:
    """One entry per saved mode file with frequency from diagnostics or result list."""
    result_freqs = _load_result_frequencies(case_dir, target_hz)
    catalog: List[Dict[str, Any]] = []
    for path in mode_files:
        try:
            mode_index = int(path.stem.split("_")[-1])
        except ValueError:
            mode_index = len(catalog)
        meta = next(
            (m for m in modes_meta if int(m.get("mode_index", -1)) == mode_index),
            {},
        )
        f_hz = float(meta.get("frequency_hz", float("nan")))
        if not np.isfinite(f_hz) and mode_index < len(result_freqs):
            f_hz = float(result_freqs[mode_index])
        if not np.isfinite(f_hz):
            continue
        catalog.append(
            {
                "mode_index": int(mode_index),
                "frequency_hz": f_hz,
                "vector_path": path.name,
                "mode_class": meta.get("mode_class"),
                "p_frac_phys_gnhep": float(meta.get("p_frac_phys_gnhep", 0.0)),
                "p_frac_production": float(
                    meta.get("p_frac_production", meta.get("p_frac_raw", 0.0))
                ),
                "wood_participation": float(meta.get("wood_participation", 0.0)),
                "meta": meta,
                "path": path,
            }
        )
    return catalog


def _select_mode_from_catalog(
    catalog: List[Dict[str, Any]],
    *,
    primary_target_hz: float,
    primary_tol_hz: float,
    fallback_target_hz: Optional[float] = None,
    fallback_tol_hz: Optional[float] = None,
    prefer_acoustic: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pick nearest mode to primary_target within primary_tol; else nearest to fallback within
    fallback_tol (e.g. branch-survival ±1 Hz from acoustic reference).
    """
    if not catalog:
        return None, []

    def _dist(entry: Dict[str, Any], target: float) -> float:
        return abs(float(entry["frequency_hz"]) - target)

    considered = sorted(
        [
            {
                "mode_index": e["mode_index"],
                "frequency_hz": e["frequency_hz"],
                "vector_path": e["vector_path"],
                "mode_class": e.get("mode_class"),
                "p_frac_phys_gnhep": e.get("p_frac_phys_gnhep"),
                "p_frac_production": e.get("p_frac_production"),
            }
            for e in catalog
        ],
        key=lambda e: e["frequency_hz"],
    )

    pool = list(catalog)
    if prefer_acoustic:
        acoustic_pool = [
            e
            for e in catalog
            if e.get("mode_class") == "acoustic_dominated"
            or float(e.get("p_frac_phys_gnhep", 0.0)) >= 0.35
            or float(e.get("p_frac_production", 0.0)) >= 0.35
        ]
        if acoustic_pool:
            pool = acoustic_pool

    primary_hits = [e for e in pool if _dist(e, primary_target_hz) <= primary_tol_hz]
    if primary_hits:
        chosen = min(primary_hits, key=lambda e: _dist(e, primary_target_hz))
        return chosen, considered

    if fallback_target_hz is not None and fallback_tol_hz is not None:
        fallback_hits = [
            e for e in catalog if _dist(e, fallback_target_hz) <= fallback_tol_hz
        ]
        if fallback_hits:
            chosen = min(fallback_hits, key=lambda e: _dist(e, fallback_target_hz))
            return chosen, considered

    return None, considered


def _audit_candidate_mode(
    vec: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    M: Any,
    A: Any,
    mode_meta: Dict[str, Any],
    frequency_hz: float,
) -> Dict[str, Any]:
    diag = diagnose_mixed_mode(
        vec,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        wood_top=float(mode_meta.get("top_plate_frac", 0.0)),
        wood_back=float(mode_meta.get("back_plate_frac", 0.0)),
        frequency_hz=frequency_hz,
    )
    p_prod, u_norm, p_norm = block_l2_p_fraction(vec, u_to_W=u_to_W, p_to_W=p_to_W)
    vec_unscaled = unscale_mixed_mode_vector(
        vec, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep, undo_pressure_dof_scale=True
    )
    p_full, _, _ = block_l2_p_fraction(vec_unscaled, u_to_W=u_to_W, p_to_W=p_to_W)
    energy = compute_mass_energy_participation(
        vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
    )
    return {
        **diag,
        "p_frac_production": float(
            mode_meta.get("p_frac_production", mode_meta.get("p_frac_raw", p_prod))
        ),
        "p_frac_raw_audit": float(p_prod),
        "p_frac_fully_unscaled": float(p_full),
        "u_norm_solver_coords": float(u_norm),
        "p_norm_solver_coords": float(p_norm),
        "structural_modal_energy_phys": float(energy["structural_modal_energy_phys"]),
        "acoustic_modal_energy_phys": float(energy["acoustic_modal_energy_phys"]),
        "p_frac_energy_phys": float(energy["p_frac_energy_phys"]),
        "mass_cross_term_phys": float(energy["mass_cross_term_phys"]),
        "mass_cross_u_from_p_gnhep": float(energy["mass_cross_u_from_p_gnhep"]),
        "mass_cross_p_from_u_gnhep": float(energy["mass_cross_p_from_u_gnhep"]),
    }


def _p_index_map_fingerprint(p_to_W: np.ndarray) -> Dict[str, Any]:
    idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    return {
        "length": int(idx.size),
        "crc32": int(zlib.crc32(idx.tobytes()) & 0xFFFFFFFF),
        "first_indices": idx[:10].tolist(),
        "last_indices": idx[-10:].tolist() if idx.size > 10 else idx.tolist(),
        "sample_indices": idx[idx.size // 2 : idx.size // 2 + 5].tolist() if idx.size >= 5 else [],
    }


def _replay_reduced_maps(
    config_path: Path,
    case_dir: Path,
    *,
    sorting_subdir: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    solver_cfg = cfg.setdefault("solver", {})
    sorting = case_dir / sorting_subdir
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())
    solver_cfg["physics_integrity_capture"] = True
    solver_cfg["active_domain_experiment"] = {"enabled": False}
    solver_cfg["coupled_air_pressure_restriction_replay_audit"] = True
    mesh_file = _resolve_mesh(cfg, config_path)
    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass
    u_map = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_map = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    restr = cfg.get("_coupled_air_pressure_restriction") or {}
    return u_map, p_map, restr


def _pressure_mac_report(
    vec_ref: np.ndarray,
    vec_cand: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    *,
    p_to_W_ref: Optional[np.ndarray] = None,
    map_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    p_scale = max(float(gnhep.get("pressure_dof_scale", 30.0)), 1.0e-30)
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    p_ref_idx = np.asarray(p_to_W_ref if p_to_W_ref is not None else p_to_W, dtype=np.int32).ravel()

    pa_raw = np.asarray(vec_ref[p_ref_idx], dtype=np.complex128).ravel()
    pb_raw = np.asarray(vec_cand[p_idx], dtype=np.complex128).ravel()
    pa_spp = pa_raw * s_p
    pb_spp = pb_raw * s_p
    scale_un = s_p / p_scale
    pa_un = pa_raw * scale_un
    pb_un = pb_raw * scale_un

    def _mac(pa: np.ndarray, pb: np.ndarray) -> float:
        na = float(np.linalg.norm(pa))
        nb = float(np.linalg.norm(pb))
        if na <= 0.0 or nb <= 0.0:
            return float("nan")
        return float(abs(np.vdot(pa, pb)) / (na * nb))

    mac_raw = _mac(pa_raw, pb_raw)
    mac_spp = _mac(pa_spp, pb_spp)
    mac_un = _mac(pa_un, pb_un)

    return {
        "mac_pressure_raw": mac_raw,
        "mac_pressure_gnhep_undo_s_pp": mac_spp,
        "mac_pressure_fully_unscaled": mac_un,
        "mac_invariance_abs_diff_raw_vs_s_pp": abs(mac_raw - mac_spp),
        "mac_invariance_abs_diff_raw_vs_unscaled": abs(mac_raw - mac_un),
        "uniform_scaling_should_not_change_mac": True,
        "inner_product": "np.vdot (complex-conjugate on first argument)",
        "vector_norms": {
            "reference_p_raw": float(np.linalg.norm(pa_raw)),
            "candidate_p_raw": float(np.linalg.norm(pb_raw)),
            "reference_p_gnhep_undo_s_pp": float(np.linalg.norm(pa_spp)),
            "candidate_p_gnhep_undo_s_pp": float(np.linalg.norm(pb_spp)),
            "reference_p_fully_unscaled": float(np.linalg.norm(pa_un)),
            "candidate_p_fully_unscaled": float(np.linalg.norm(pb_un)),
        },
        "p_index_map_candidate": _p_index_map_fingerprint(p_to_W),
        "p_index_map_reference": _p_index_map_fingerprint(
            p_to_W_ref if p_to_W_ref is not None else p_to_W
        ),
        "p_index_maps_identical": bool(np.array_equal(p_ref_idx, p_idx)),
        "map_validation": map_validation or {},
        "scaling_documentation": MAC_SCALING_NOTE,
        "n_p_active_dofs": int(p_idx.size),
    }


def _is_pressure_dominated(row: Dict[str, Any], energy: Dict[str, Any]) -> bool:
    return (
        float(energy.get("p_frac_energy_phys", 0.0)) >= ENERGY_PARTICIPATION_THRESHOLD
        or float(row.get("p_frac_production", 0.0)) >= ENERGY_PARTICIPATION_THRESHOLD
        or float(row.get("p_frac_phys_gnhep", 0.0)) >= ENERGY_PARTICIPATION_THRESHOLD
    )


def _split_audit_conclusions(
    *,
    cand_f: float,
    ref_hz: float,
    freq_tol_hz: float,
    p_frac_energy_phys: float,
    mac_spp: float,
    mac_threshold: float,
) -> Dict[str, Any]:
    freq_near_ref = abs(cand_f - ref_hz) <= freq_tol_hz
    pressure_energy = (
        freq_near_ref and float(p_frac_energy_phys) >= ENERGY_PARTICIPATION_THRESHOLD
    )
    mac_confirmed = math.isfinite(mac_spp) and float(mac_spp) >= mac_threshold
    return {
        "PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS": pressure_energy,
        "DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED": bool(mac_confirmed),
        "frequency_within_branch_survival_hz": freq_near_ref,
        "p_frac_energy_phys": float(p_frac_energy_phys),
        "mac_pressure_gnhep_undo_s_pp": float(mac_spp),
        "mac_threshold": float(mac_threshold),
        "energy_threshold": ENERGY_PARTICIPATION_THRESHOLD,
        "interpretation": (
            "A pressure-energy-dominated mode near the reference does not imply the same "
            "acoustic eigenvector shape; DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED requires "
            f"validated pressure MAC >= {mac_threshold:.2f}."
        ),
    }


def _refresh_physical_fsi_summary(
    *,
    audit: Dict[str, Any],
    ref_hz: float,
    freq_tol_hz: float,
) -> None:
    cand = audit["physical_fsi_candidate"]
    overlap = audit.get("pressure_overlap") or cand.get("pressure_overlap_vs_decoupled_acoustic") or {}
    mac = float(overlap.get("mac_pressure_gnhep_undo_s_pp", float("nan")))
    conclusions = audit.get("audit_conclusions") or _split_audit_conclusions(
        cand_f=float(cand["frequency_hz"]),
        ref_hz=ref_hz,
        freq_tol_hz=freq_tol_hz,
        p_frac_energy_phys=float(cand["p_frac_energy_phys"]),
        mac_spp=mac,
        mac_threshold=float(
            (audit.get("mode_selection") or {}).get("mac_threshold", MAC_BRANCH_MATCH_THRESHOLD)
        ),
    )
    summary_path = PHYSICAL_CASE / "diagnostics" / "physical_fsi_only_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "post_audit_refreshed": True,
            "participation_audit_json": "physical_fsi_participation_audit.json",
            "pressure_overlap": overlap,
            "audit_conclusions": conclusions,
            "verdict_note": audit.get("verdict_note"),
            "branch_shift_hz": (audit.get("mode_selection") or {}).get("branch_shift_hz"),
        }
    )
    _write_json(summary_path, summary)
    md = PHYSICAL_CASE / "diagnostics" / "physical_fsi_only_summary.md"
    md.write_text(
        "\n".join(
            [
                "# Physical-FSI-only isolation (post participation audit)",
                "",
                f"- PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS="
                f"**{conclusions['PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS']}**",
                f"- DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED="
                f"**{conclusions['DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED']}**",
                "",
                f"Candidate f={float(cand['frequency_hz']):.6f} Hz | "
                f"p_frac_energy={float(cand['p_frac_energy_phys']):.4e} | "
                f"MAC raw={overlap.get('mac_pressure_raw', float('nan')):.4f} | "
                f"MAC s_pp={mac:.4f} | MAC unscaled={overlap.get('mac_pressure_fully_unscaled', float('nan')):.4f}",
                "",
                conclusions.get("interpretation", ""),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_participation_audit] refreshed summary: "
            f"PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS="
            f"{conclusions['PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS']} "
            f"DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED="
            f"{conclusions['DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED']}"
        )
        print(f"  pressure_overlap={overlap}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical-FSI participation / overlap audit")
    parser.add_argument(
        "--physical-freq-hz",
        type=float,
        default=None,
        help="Override physical-FSI target (default: physical_fsi_only_summary hint or 245.299844)",
    )
    parser.add_argument(
        "--decoupled-freq-hz",
        type=float,
        default=None,
        help="Override decoupled target (default: decoupled_union_summary hint or 244.3916)",
    )
    parser.add_argument(
        "--self-case-match-tol-hz",
        type=float,
        default=0.15,
        help="Max |f - case target| when matching within one case.",
    )
    parser.add_argument(
        "--branch-survival-tol-hz",
        type=float,
        default=None,
        help="Physical-FSI fallback: nearest mode to acoustic ref within this Hz (default: config 1.0).",
    )
    parser.add_argument("--mac-threshold", type=float, default=0.85)
    parser.add_argument("--skip-summary-refresh", action="store_true")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_fsi_participation_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    physical_cfg = json.loads(PHYSICAL_CONFIG.read_text(encoding="utf-8"))
    solver_cfg = physical_cfg.setdefault("solver", {})
    target_hz = float(solver_cfg.get("_worker_target_hz", 244.39))
    ref_hz = _acoustic_reference_hz(solver_cfg)
    freq_tol = float(solver_cfg.get("physical_fsi_acoustic_tol_hz", 1.0))

    sorting = PHYSICAL_CASE / "sorting_physical_fsi_audit"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())
    solver_cfg["physics_integrity_capture"] = True
    solver_cfg["active_domain_experiment"] = {"enabled": False}
    solver_cfg["coupled_air_pressure_restriction_replay_audit"] = True

    mesh_file = _resolve_mesh(physical_cfg, PHYSICAL_CONFIG)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_participation_audit] Assembling physical-FSI-only A/M "
            "(reduced domain, no SLEPc)...",
            flush=True,
        )
    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=physical_cfg,
        num_modes=0,
        solve_evp=False,
    )
    u_to_W = np.asarray(physical_cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(physical_cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    n_W = int(A.getSize()[0])
    restr = physical_cfg.get("_coupled_air_pressure_restriction") or {}

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_fsi_participation_audit] reduced W={n_W} "
            f"n_u_active={restr.get('n_u_active')} n_p_active={restr.get('n_p_active')}",
            flush=True,
        )

    gnhep = merge_scaling_metadata(PHYSICAL_CASE)
    pi = physical_cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    self_tol = float(args.self_case_match_tol_hz)
    branch_tol = float(
        args.branch_survival_tol_hz
        if args.branch_survival_tol_hz is not None
        else freq_tol
    )
    dec_target = float(
        args.decoupled_freq_hz
        if args.decoupled_freq_hz is not None
        else _summary_hint_frequency(
            DECOUPLED_CASE,
            "decoupled_union_summary.json",
            prefer_hz=ref_hz,
            prefer_keys=(
                "acoustic_recovered_modes",
                "ranking_by_acoustic_participation",
                "modes_in_band",
            ),
        )
        or 244.3916
    )
    phys_target = float(
        args.physical_freq_hz
        if args.physical_freq_hz is not None
        else _summary_hint_frequency(
            PHYSICAL_CASE,
            "physical_fsi_only_summary.json",
            prefer_hz=245.299844,
            prefer_keys=(
                "acoustic_candidate_modes",
                "ranking_by_proximity_to_acoustic_ref",
                "modes_in_band",
            ),
        )
        or 245.299844
    )

    phys_meta, phys_files = _load_modes(PHYSICAL_CASE, target_hz)
    dec_meta, dec_files = _load_modes(DECOUPLED_CASE, target_hz)
    if not phys_files or not dec_files:
        print("[physical_fsi_participation_audit] Missing saved mode files", file=sys.stderr)
        return 1

    phys_catalog = _catalog_saved_modes(
        PHYSICAL_CASE, phys_meta, phys_files, target_hz=target_hz
    )
    dec_catalog = _catalog_saved_modes(
        DECOUPLED_CASE, dec_meta, dec_files, target_hz=target_hz
    )
    if not phys_catalog or not dec_catalog:
        print(
            "[physical_fsi_participation_audit] No modes with finite frequencies in catalogs",
            file=sys.stderr,
        )
        return 1

    ref_entry, dec_considered = _select_mode_from_catalog(
        dec_catalog,
        primary_target_hz=dec_target,
        primary_tol_hz=self_tol,
        fallback_target_hz=ref_hz,
        fallback_tol_hz=self_tol,
        prefer_acoustic=True,
    )
    cand_entry, phys_considered = _select_mode_from_catalog(
        phys_catalog,
        primary_target_hz=phys_target,
        primary_tol_hz=self_tol,
        fallback_target_hz=ref_hz,
        fallback_tol_hz=branch_tol,
        prefer_acoustic=False,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_fsi_participation_audit] decoupled catalog: {len(dec_considered)} mode(s), "
            f"target={dec_target:.6f} Hz (self ±{self_tol:.2f})",
            flush=True,
        )
        for row in dec_considered[:16]:
            print(
                f"  decoupled candidate: {row['vector_path']} f={row['frequency_hz']:.6f} Hz "
                f"class={row.get('mode_class')}",
                flush=True,
            )
        print(
            f"[physical_fsi_participation_audit] physical-FSI catalog: {len(phys_considered)} mode(s), "
            f"target={phys_target:.6f} Hz (self ±{self_tol:.2f}, branch fallback ±{branch_tol:.2f} "
            f"from ref {ref_hz:.4f} Hz)",
            flush=True,
        )
        for row in phys_considered[:16]:
            print(
                f"  physical-FSI candidate: {row['vector_path']} f={row['frequency_hz']:.6f} Hz "
                f"class={row.get('mode_class')}",
                flush=True,
            )

    if ref_entry is None or cand_entry is None:
        missing = []
        if ref_entry is None:
            missing.append(
                f"decoupled (need mode within ±{self_tol:.2f} Hz of {dec_target:.4f} or {ref_hz:.4f})"
            )
        if cand_entry is None:
            missing.append(
                f"physical-FSI (need mode within ±{self_tol:.2f} Hz of {phys_target:.4f} "
                f"or ±{branch_tol:.2f} Hz of acoustic ref {ref_hz:.4f})"
            )
        print(
            "[physical_fsi_participation_audit] Could not select saved modes: "
            + "; ".join(missing),
            file=sys.stderr,
        )
        return 1

    ref_path = ref_entry["path"]
    cand_path = cand_entry["path"]
    ref_meta = ref_entry.get("meta") or {}
    cand_meta = cand_entry.get("meta") or {}
    ref_f = float(ref_entry["frequency_hz"])
    cand_f = float(cand_entry["frequency_hz"])
    branch_shift_hz = cand_f - ref_f

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_fsi_participation_audit] selected decoupled: {ref_entry['vector_path']} "
            f"decoupled_f_hz={ref_f:.6f}",
            flush=True,
        )
        print(
            f"[physical_fsi_participation_audit] selected physical-FSI: {cand_entry['vector_path']} "
            f"physical_fsi_f_hz={cand_f:.6f}",
            flush=True,
        )
        print(
            f"[physical_fsi_participation_audit] branch_shift_hz={branch_shift_hz:+.6f} "
            f"(Δ acoustic ref {cand_f - ref_hz:+.6f} Hz)",
            flush=True,
        )

    vec_cand, cand_load = _load_coupled_mode_dense_vector(
        cand_path, n_coupled_W=n_W, mode_index=int(cand_entry["mode_index"])
    )
    vec_ref, ref_load = _load_coupled_mode_dense_vector(
        ref_path, n_coupled_W=n_W, mode_index=int(ref_entry["mode_index"])
    )

    cand_audit = _audit_candidate_mode(
        vec_cand,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        M=M,
        A=A,
        mode_meta=cand_meta or {},
        frequency_hz=cand_f,
    )
    ref_audit = _audit_candidate_mode(
        vec_ref,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        M=M,
        A=A,
        mode_meta=ref_meta or {},
        frequency_hz=ref_f,
    )
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_participation_audit] replaying decoupled-union maps for MAC validation...",
            flush=True,
        )
    _du, p_map_dec, dec_restr = _replay_reduced_maps(
        DECOUPLED_CONFIG,
        DECOUPLED_CASE,
        sorting_subdir="sorting_mac_map_replay_decoupled",
    )
    map_validation = {
        "physical_fsi_p_map": _p_index_map_fingerprint(p_to_W),
        "decoupled_union_p_map": _p_index_map_fingerprint(p_map_dec),
        "p_maps_byte_identical": bool(np.array_equal(p_to_W, p_map_dec)),
        "physical_fsi_n_p_active": int(p_to_W.size),
        "decoupled_union_n_p_active": int(p_map_dec.size),
        "physical_fsi_restriction": restr,
        "decoupled_union_restriction": dec_restr,
    }
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_fsi_participation_audit] p_to_W map check: "
            f"len={p_to_W.size} identical={map_validation['p_maps_byte_identical']} "
            f"crc_phys={map_validation['physical_fsi_p_map']['crc32']} "
            f"crc_dec={map_validation['decoupled_union_p_map']['crc32']}",
            flush=True,
        )

    mac_report = _pressure_mac_report(
        vec_ref,
        vec_cand,
        p_to_W,
        gnhep,
        p_to_W_ref=p_map_dec,
        map_validation=map_validation,
    )
    cand_audit["pressure_overlap_vs_decoupled_acoustic"] = mac_report
    cand_audit["delta_from_decoupled_mode_hz"] = float(branch_shift_hz)
    cand_audit["mode_load"] = cand_load
    ref_audit["mode_load"] = ref_load
    ref_audit["role"] = "decoupled_union_acoustic_reference"

    band_lo = float(solver_cfg.get("_worker_harvest_lo_hz", BAND_LO_HZ))
    band_hi = float(solver_cfg.get("_worker_harvest_hi_hz", BAND_HI_HZ))
    band_mac_rows: List[Dict[str, Any]] = []
    target_band_hz = set(PRESSURE_DOMINATED_BAND_HZ)
    for entry in phys_catalog:
        f_hz = float(entry["frequency_hz"])
        in_band = band_lo <= f_hz <= band_hi
        near_listed = any(abs(f_hz - t) <= self_tol for t in target_band_hz)
        if not in_band and not near_listed:
            continue
        vec_band, _ = _load_coupled_mode_dense_vector(
            entry["path"], n_coupled_W=n_W, mode_index=int(entry["mode_index"])
        )
        energy_band = compute_mass_energy_participation(
            vec_band, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        if not _is_pressure_dominated(entry, energy_band) and not near_listed:
            continue
        mac_band = _pressure_mac_report(
            vec_ref, vec_band, p_to_W, gnhep, p_to_W_ref=p_map_dec, map_validation=map_validation
        )
        band_mac_rows.append(
            {
                "frequency_hz": f_hz,
                "vector_path": entry["vector_path"],
                "mode_index": entry["mode_index"],
                "p_frac_production": float(entry.get("p_frac_production", 0.0)),
                "p_frac_energy_phys": float(energy_band["p_frac_energy_phys"]),
                "delta_from_decoupled_acoustic_hz": f_hz - ref_f,
                "delta_from_acoustic_reference_hz": f_hz - ref_hz,
                "pressure_overlap": mac_band,
            }
        )
    band_mac_rows.sort(key=lambda r: float(r["frequency_hz"]))

    mac_threshold = float(args.mac_threshold)
    conclusions = _split_audit_conclusions(
        cand_f=cand_f,
        ref_hz=ref_hz,
        freq_tol_hz=freq_tol,
        p_frac_energy_phys=float(cand_audit["p_frac_energy_phys"]),
        mac_spp=float(mac_report["mac_pressure_gnhep_undo_s_pp"]),
        mac_threshold=mac_threshold,
    )
    verdict_note = (
        f"branch_shift_hz={branch_shift_hz:+.6f}; "
        f"PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS="
        f"{conclusions['PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS']}; "
        f"DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED="
        f"{conclusions['DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED']}; "
        f"MAC(s_pp)={mac_report['mac_pressure_gnhep_undo_s_pp']:.4f} "
        "(do not proceed to Nitsche isolation until MAC mapping validated)."
    )

    report: Dict[str, Any] = {
        "experiment": "physical_fsi_participation_overlap_audit",
        "no_eigensolve": True,
        "acoustic_reference_hz": ref_hz,
        "mode_selection": {
            "decoupled_primary_target_hz": dec_target,
            "physical_fsi_primary_target_hz": phys_target,
            "self_case_match_tol_hz": self_tol,
            "branch_survival_tol_hz": branch_tol,
            "decoupled_f_hz": ref_f,
            "physical_fsi_f_hz": cand_f,
            "branch_shift_hz": float(branch_shift_hz),
            "decoupled_modes_considered": dec_considered,
            "physical_fsi_modes_considered": phys_considered,
            "selected_decoupled_vector": ref_entry["vector_path"],
            "selected_physical_fsi_vector": cand_entry["vector_path"],
            "mac_threshold": mac_threshold,
        },
        "p_index_map_validation": map_validation,
        "pressure_dominated_modes_in_band_mac": band_mac_rows,
        "audit_conclusions": conclusions,
        "verdict_note": verdict_note,
        "decoupled_union_reference_mode": {
            "frequency_hz": ref_f,
            "vector_path": str(ref_path.relative_to(DECOUPLED_CASE)).replace("\\", "/"),
            **ref_audit,
        },
        "physical_fsi_candidate": {
            "frequency_hz": cand_f,
            "vector_path": str(cand_path.relative_to(PHYSICAL_CASE)).replace("\\", "/"),
            **cand_audit,
        },
        "pressure_overlap": mac_report,
        "wood_participation_interpretation": WOOD_PARTICIPATION_NOTE,
        "metric_definitions": {
            "p_frac_production": P_FRAC_PRODUCTION_DEFINITION,
            "p_frac_phys_gnhep": P_FRAC_PHYS_GNHEP_DEFINITION,
            "p_frac_fully_unscaled": P_FRAC_FULLY_UNSCALED_DEFINITION,
            "p_frac_energy_phys": (
                "E_air_phys / (E_struct + E_air + |cross|) from x^T M x on assembled "
                "physical-FSI-only reduced operator."
            ),
        },
        "scaling_metadata": gnhep,
        "reduced_domain": restr,
        "legacy_combined_verdict_suppressed": (
            "Do not use frequency+energy alone; see audit_conclusions."
        ),
        "metric_conflict_explanation": (
            "p_frac_phys_gnhep divides GNHEP-undone ||p|| by (||u||_gnhep_undo + ||p||_gnhep_undo). "
            "When ||u|| is tiny in solver coordinates but FSI coupling inflates the u-block after "
            "s_uu scaling, the denominator is u-dominated and p_frac_phys_gnhep can be O(1e-4) "
            "even with production p_frac≈1 and ||p||≈1. Energy participation and pressure MAC "
            "against the decoupled acoustic mode are the appropriate branch-survival indicators."
        ),
    }

    diag_dir = PHYSICAL_CASE / "diagnostics"
    out_json = diag_dir / "physical_fsi_participation_audit.json"
    out_md = diag_dir / "physical_fsi_participation_audit.md"
    _write_json(out_json, report)

    if MPI.COMM_WORLD.rank == 0:
        lines = [
            "# Physical-FSI participation / overlap audit",
            "",
            f"- **PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS="
            f"{conclusions['PRESSURE_ENERGY_MODE_NEAR_REFERENCE_EXISTS']}**",
            f"- **DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED="
            f"{conclusions['DECOUPLED_ACOUSTIC_BRANCH_MATCH_CONFIRMED']}**",
            "",
            verdict_note,
            "",
            "## Candidate mode (physical FSI only)",
            "",
            f"- decoupled_f_hz = **{ref_f:.6f}** | physical_fsi_f_hz = **{cand_f:.6f}**",
            f"- branch_shift_hz = **{branch_shift_hz:+.6f}** (Δ acoustic ref = {cand_f - ref_hz:+.6f} Hz)",
            f"- p_frac production = {cand_audit['p_frac_production']:.6e}",
            f"- p_frac GNHEP undo = {cand_audit['p_frac_phys_gnhep']:.6e}",
            f"- p_frac fully unscaled = {cand_audit['p_frac_fully_unscaled']:.6e}",
            f"- p_frac energy (phys) = {cand_audit['p_frac_energy_phys']:.6e}",
            f"- E_air phys = {cand_audit['acoustic_modal_energy_phys']:.6e}",
            f"- E_struct phys = {cand_audit['structural_modal_energy_phys']:.6e}",
            f"- mass cross (phys) = {cand_audit['mass_cross_term_phys']:.6e}",
            f"- ||u||, ||p|| (solver) = {cand_audit['u_norm_solver_coords']:.6e}, "
            f"{cand_audit['p_norm_solver_coords']:.6e}",
            "",
            "## Pressure overlap vs decoupled acoustic (active p only)",
            "",
            f"- MAC raw = {mac_report['mac_pressure_raw']:.6f}",
            f"- MAC gnhep_undo (s_pp) = {mac_report['mac_pressure_gnhep_undo_s_pp']:.6f}",
            f"- MAC fully unscaled = {mac_report['mac_pressure_fully_unscaled']:.6f}",
            f"- MAC invariance |raw-s_pp| = {mac_report['mac_invariance_abs_diff_raw_vs_s_pp']:.3e}",
            f"- ||p_ref|| raw = {mac_report['vector_norms']['reference_p_raw']:.6e} "
            f"||p_cand|| raw = {mac_report['vector_norms']['candidate_p_raw']:.6e}",
            "",
            MAC_SCALING_NOTE,
            "",
            "## p-index map validation",
            "",
            f"- n_p_active = {mac_report['n_p_active_dofs']} | maps identical = "
            f"{map_validation['p_maps_byte_identical']}",
            f"- physical crc32 = {map_validation['physical_fsi_p_map']['crc32']} | "
            f"decoupled crc32 = {map_validation['decoupled_union_p_map']['crc32']}",
            "",
            "## Pressure-dominated physical-FSI modes in band (MAC vs decoupled 244.3916 Hz)",
            "",
            "| f (Hz) | MAC raw | MAC s_pp | MAC unscaled | p_frac energy | file |",
            "|--------|---------|----------|--------------|---------------|------|",
        ]
        for row in band_mac_rows:
            ov = row["pressure_overlap"]
            lines.append(
                f"| {row['frequency_hz']:.6f} | {ov['mac_pressure_raw']:.4f} | "
                f"{ov['mac_pressure_gnhep_undo_s_pp']:.4f} | "
                f"{ov['mac_pressure_fully_unscaled']:.4f} | "
                f"{row['p_frac_energy_phys']:.4e} | {row['vector_path']} |"
            )
        lines.extend(
            [
                "",
                "## Wood participation",
                "",
                WOOD_PARTICIPATION_NOTE,
                "",
                "## Metric conflict",
                "",
                report["metric_conflict_explanation"],
                "",
            ]
        )
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[physical_fsi_participation_audit] candidate f={cand_f:.6f} Hz")
        print(
            f"  p_frac prod={cand_audit['p_frac_production']:.4e} "
            f"gnhep_undo={cand_audit['p_frac_phys_gnhep']:.4e} "
            f"energy={cand_audit['p_frac_energy_phys']:.4e}"
        )
        print(f"  pressure_overlap={mac_report}")
        print(f"  conclusions={conclusions}")
        for row in band_mac_rows:
            ov = row["pressure_overlap"]
            print(
                f"  band f={row['frequency_hz']:.6f} Hz MAC_s_pp="
                f"{ov['mac_pressure_gnhep_undo_s_pp']:.4f} file={row['vector_path']}"
            )
        print(f"[physical_fsi_participation_audit] wrote {out_json} and {out_md}")

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    if not args.skip_summary_refresh:
        _refresh_physical_fsi_summary(audit=report, ref_hz=ref_hz, freq_tol_hz=freq_tol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
