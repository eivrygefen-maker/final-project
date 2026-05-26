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
from typing import Any, Dict, List, Optional, Tuple

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
from v2_mapping_fixed_candidate_persistence import (
    VERDICT_PERSISTENCE_FAILURE,
    bank_records_from_eigvecs,
    check_persistence_gate,
    load_preserve_all_bank_from_config,
    persist_candidate_bank,
    pressure_block_mapping_metadata,
    write_eps_candidate_bank_json,
)
from v2_sensitivity_mesh import sample_geometry
from wood_library import apply_wood_ids_to_config

SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
DEFAULT_BAND_LO = 220.0
DEFAULT_BAND_HI = 265.0
ENERGY_ACOUSTIC_THRESHOLD = 0.85
SEED_BRANCH_DIAG_SIGMA_RETRY_OFFSETS_HZ = (
    2.0,
    -2.0,
    5.0,
    -5.0,
    8.0,
    -8.0,
    12.0,
    -12.0,
    15.0,
    -15.0,
)
# In-band nonzero-offset ladder for unregularized diagnostic rerun (never σ = seed f).
SEED_BRANCH_UNREG_OFFSET_SIGMA_HZ = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pick_acoustic_branch(
    in_band: List[Dict[str, Any]],
    *,
    select_by_energy: bool,
    reference_f_hz: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (branch_row, selection_method)."""
    if not in_band:
        return None, "none"
    if select_by_energy:
        acoustic = [
            m
            for m in in_band
            if m.get("mode_class_physical_energy") == "acoustic_dominated"
            or float(m.get("p_frac_energy_phys", 0.0)) >= ENERGY_ACOUSTIC_THRESHOLD
        ]
        if acoustic:
            best = max(acoustic, key=lambda m: float(m["p_frac_energy_phys"]))
            return best, "max_p_frac_energy_phys_acoustic_dominated"
        ranked = sorted(in_band, key=lambda m: float(m["p_frac_energy_phys"]), reverse=True)
        if ranked and float(ranked[0]["p_frac_energy_phys"]) >= 0.35:
            return ranked[0], "max_p_frac_energy_phys_relaxed"
        return None, "no_acoustic_candidate_in_band"
    pool = [
        m
        for m in in_band
        if m.get("mode_class_physical_energy") == "acoustic_dominated"
        or float(m.get("p_frac_energy_phys", 0.0)) >= 0.35
    ]
    use = pool if pool else in_band
    best = min(use, key=lambda m: abs(float(m["frequency_hz"]) - reference_f_hz))
    return best, "nearest_frequency_to_reference"


def _apply_seed_branch_recovery_diagnostic_solver_cfg(
    sc: Dict[str, Any], target_hz: float
) -> Dict[str, Any]:
    """
    Experiment-only: ST sigma ladder centered on seed Rayleigh frequency.
    Does not use production offset-above-harvest-window sigma policy.
    """
    f0 = float(target_hz)
    local_half = max(0.5, f0 * 0.01)
    local_band = [f0 - local_half, f0 + local_half]
    sc["seeded_branch_recovery_diagnostic"] = True
    sc["shift_invert_target_hz"] = f0
    sc["_worker_target_hz"] = f0
    sc["eps_st_sigma_try_target_first"] = True
    sc["eps_st_sigma_include_target_in_ladder"] = False
    sc["eps_st_sigma_primary_offset_hz"] = 0.0
    sc["eps_st_sigma_min_offset_hz"] = 0.0
    sc["eps_st_sigma_frac_offset"] = 0.0
    sc["eps_st_sigma_retry_offsets_hz"] = list(SEED_BRANCH_DIAG_SIGMA_RETRY_OFFSETS_HZ)
    sc["eps_st_sigma_ladder_max"] = 12
    sc["eps_broad_search_hz"] = 0.0
    sc["eps_harvest_sigma_margin_hz"] = 3.0
    sc["eps_interval_half_width_hz"] = max(local_half, 8.0)
    harvest_half = max(12.0, local_half * 4.0)
    sc["_worker_harvest_lo_hz"] = f0 - harvest_half
    sc["_worker_harvest_hi_hz"] = f0 + harvest_half
    sc["eps_reject_sigma_spurious"] = False
    sc["eps_reject_target_locked"] = False
    ladder = fem3d._slepc_st_sigma_hz_candidates(sc, f0)
    return _seed_branch_diag_meta(f0, ladder, local_band, harvest_half)


def _seed_branch_diag_meta(
    f0: float,
    ladder: List[float],
    local_band: List[float],
    harvest_half: float,
) -> Dict[str, Any]:
    return {
        "solver_mode": "seeded_branch_recovery_diagnostic",
        "standard_harvest_sigma_policy_unchanged": True,
        "standard_policy_not_used_for_this_diagnostic": True,
        "seed_rayleigh_f_hz": f0,
        "diagnostic_sigma_hz": float(ladder[0]) if ladder else f0,
        "diagnostic_sigma_retry_ladder_hz": [float(x) for x in ladder],
        "diagnostic_local_band_hz": local_band,
        "diagnostic_harvest_window_hz": [f0 - harvest_half, f0 + harvest_half],
    }


def _apply_mapping_fixed_unregularized_baseline_diagnostic_solver_cfg(
    sc: Dict[str, Any], target_hz: float
) -> Dict[str, Any]:
    """
    Experiment-only: mapping-corrected (lam_phys=mu) unregularized ST baseline.
    Preserves every converged EPS candidate before post-save physical replay.
    """
    meta = _apply_seed_branch_unregularized_offset_diagnostic_solver_cfg(sc, target_hz)
    sc["eps_eigenvalue_semantics"] = "slepc_backtransformed"
    sc["legacy_double_shift_mapping_disabled"] = True
    sc["eps_diagnostic_preserve_all_nconv_candidates"] = True
    sc["eigenpair_batch_size"] = max(int(sc.get("eigenpair_batch_size", 100)), 128)
    sc["eps_reject_sigma_spurious"] = False
    sc["eps_reject_target_locked"] = False
    sc["eps_reject_decoupled_u_only"] = False
    sc["eps_harvest_allow_weak_coupling"] = True
    meta.update(
        {
            "solver_mode": "seed_branch_recovery_diagnostic_mapping_fixed_unregularized",
            "mapping_corrected_baseline_diagnostic": True,
            "eps_eigenvalue_semantics": "slepc_backtransformed",
            "legacy_double_shift_mapping_disabled": True,
            "eps_diagnostic_preserve_all_nconv_candidates": True,
            "save_all_converged_candidates_before_physical_filter": True,
            "prior_seven_mode_harvest_not_evidence_of_mapping_failure": True,
            "supersedes_pre_mapping_fix_unregularized_offset_solve": True,
        }
    )
    return meta


def _apply_seed_branch_unregularized_offset_diagnostic_solver_cfg(
    sc: Dict[str, Any], target_hz: float
) -> Dict[str, Any]:
    """
    Experiment-only: in-band sigma ladder with nonzero offsets only; ST must remain
    unregularized (no A-shift / mass-reg ST copies). Fail-closed vs replay operator.
    """
    f0 = float(target_hz)
    local_half = max(0.5, f0 * 0.01)
    local_band = [f0 - local_half, f0 + local_half]
    sc["seeded_branch_recovery_diagnostic"] = True
    sc["diagnostic_requires_unregularized_ST"] = True
    sc["eps_st_unregularized_only"] = True
    sc["shift_invert_target_hz"] = f0
    sc["_worker_target_hz"] = f0
    sc["eps_st_sigma_try_target_first"] = False
    sc["eps_st_sigma_include_target_in_ladder"] = False
    sc["eps_st_sigma_primary_offset_hz"] = float(SEED_BRANCH_UNREG_OFFSET_SIGMA_HZ[0])
    sc["eps_st_sigma_min_offset_hz"] = 0.0
    sc["eps_st_sigma_frac_offset"] = 0.0
    sc["eps_st_sigma_retry_offsets_hz"] = list(SEED_BRANCH_UNREG_OFFSET_SIGMA_HZ)
    sc["eps_st_sigma_ladder_max"] = max(8, len(SEED_BRANCH_UNREG_OFFSET_SIGMA_HZ) + 2)
    sc["eps_st_a_diagonal_shift_frac_ladder"] = (0.0,)
    sc["eps_st_mass_reg_frac_ladder"] = (0.0,)
    sc["eps_st_mass_reg_frac"] = 0.0
    sc["eps_broad_search_hz"] = 0.0
    sc["eps_harvest_sigma_margin_hz"] = 3.0
    sc["eps_interval_half_width_hz"] = max(local_half, 8.0)
    harvest_half = max(12.0, local_half * 4.0)
    sc["_worker_harvest_lo_hz"] = f0 - harvest_half
    sc["_worker_harvest_hi_hz"] = f0 + harvest_half
    sc["eps_reject_sigma_spurious"] = True
    sc["eps_reject_target_locked"] = True
    ladder = fem3d._slepc_st_sigma_hz_candidates(sc, f0)
    meta = _seed_branch_diag_meta(f0, ladder, local_band, harvest_half)
    meta.update(
        {
            "solver_mode": "seed_branch_recovery_diagnostic_unregularized_offset",
            "diagnostic_requires_unregularized_ST": True,
            "diagnostic_sigma_offset_ladder_hz": [float(x) for x in ladder],
            "diagnostic_sigma_offset_from_seed_hz": [
                float(x) - f0 for x in ladder
            ],
            "never_sigma_at_seed_rayleigh_frequency": True,
            "eps_st_unregularized_only": True,
            "post_evaluate_physical_filter": True,
        }
    )
    return meta


def _apply_seed_branch_filtered_diagnostic_solver_cfg(
    sc: Dict[str, Any], target_hz: float
) -> Dict[str, Any]:
    """
    Legacy filtered diagnostic (superseded): targeted exact seed sigma with optional ST
    regularization retry. Retained for log compatibility; new reruns should use
    unregularized-offset diagnostic instead.
    """
    meta = _apply_seed_branch_recovery_diagnostic_solver_cfg(sc, target_hz)
    sc["eps_reject_sigma_spurious"] = True
    sc["eps_reject_target_locked"] = True
    meta["eps_reject_sigma_spurious_enabled"] = True
    meta["eps_reject_target_locked_enabled"] = True
    meta["filtered_diagnostic_rerun"] = True
    meta["post_evaluate_physical_filter"] = True
    meta["superseded_by"] = "seed_branch_recovery_diagnostic_unregularized_offset"
    return meta


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
    parser.add_argument("--harvest-lo-hz", type=float, default=None)
    parser.add_argument("--harvest-hi-hz", type=float, default=None)
    parser.add_argument(
        "--select-by-energy",
        action="store_true",
        help="Pick acoustic branch by max p_frac_energy_phys (not nearest f to reference)",
    )
    parser.add_argument(
        "--reference-f-hz",
        type=float,
        default=244.394153389752,
        help="Reference frequency for nearest-f selection (ignored if --select-by-energy)",
    )
    parser.add_argument("--num-modes", type=int, default=0, help="Override num_modes (0=cfg default)")
    parser.add_argument(
        "--structural-spectrum-harvest",
        action="store_true",
        help="Structural validation harvest: success if v2_converged even without acoustic branch in band",
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Parent directory for sample case folders (default: v2_sensitivity_validation/samples)",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Explicit case output directory (overrides --case-root / sample-id layout).",
    )
    parser.add_argument(
        "--eps-seed-npy",
        type=Path,
        default=None,
        help="Experiment-only: full W-space vector for EPS initial space (branch-tracking)",
    )
    parser.add_argument(
        "--seed-branch-recovery-diagnostic",
        action="store_true",
        help=(
            "Experiment-only: center ST sigma on seed/target Hz for known-branch recovery; "
            "does not use production above-window sigma/harvest policy."
        ),
    )
    parser.add_argument(
        "--seed-branch-filtered-diagnostic",
        action="store_true",
        help=(
            "Legacy filtered diagnostic (superseded). Prefer "
            "--seed-branch-unregularized-offset-diagnostic."
        ),
    )
    parser.add_argument(
        "--seed-branch-unregularized-offset-diagnostic",
        action="store_true",
        help=(
            "Experiment-only: in-band nonzero-offset sigma ladder; ST must stay "
            "unregularized; fail-closed if ST regularization would be required."
        ),
    )
    parser.add_argument(
        "--seed-branch-mapping-fixed-unregularized-diagnostic",
        action="store_true",
        help=(
            "Experiment-only: mapping-corrected unregularized ST baseline; save all "
            "converged EPS candidates before physical replay eligibility."
        ),
    )
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
    reference_f_hz = float(args.reference_f_hz)
    seed_branch_diag_meta: Optional[Dict[str, Any]] = None
    if args.seed_branch_recovery_diagnostic:
        target_hz = reference_f_hz
    band_lo = float(args.harvest_lo_hz if args.harvest_lo_hz is not None else DEFAULT_BAND_LO)
    band_hi = float(args.harvest_hi_hz if args.harvest_hi_hz is not None else DEFAULT_BAND_HI)
    if args.case_dir is not None:
        case_dir = args.case_dir.resolve()
    else:
        case_parent = (
            args.case_root.resolve()
            if args.case_root is not None
            else SENS_ROOT / "samples"
        )
        case_dir = case_parent / sample_id
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
    sc["_worker_harvest_lo_hz"] = band_lo
    sc["_worker_harvest_hi_hz"] = band_hi
    sc["shift_invert_target_hz"] = target_hz
    if args.seed_branch_recovery_diagnostic:
        if args.seed_branch_mapping_fixed_unregularized_diagnostic:
            seed_branch_diag_meta = (
                _apply_mapping_fixed_unregularized_baseline_diagnostic_solver_cfg(
                    sc, target_hz
                )
            )
        elif args.seed_branch_unregularized_offset_diagnostic:
            seed_branch_diag_meta = _apply_seed_branch_unregularized_offset_diagnostic_solver_cfg(
                sc, target_hz
            )
        elif args.seed_branch_filtered_diagnostic:
            seed_branch_diag_meta = _apply_seed_branch_filtered_diagnostic_solver_cfg(
                sc, target_hz
            )
        else:
            seed_branch_diag_meta = _apply_seed_branch_recovery_diagnostic_solver_cfg(
                sc, target_hz
            )
        band_lo = float(seed_branch_diag_meta["diagnostic_local_band_hz"][0])
        band_hi = float(seed_branch_diag_meta["diagnostic_local_band_hz"][1])
        sc["_worker_harvest_lo_hz"] = float(seed_branch_diag_meta["diagnostic_harvest_window_hz"][0])
        sc["_worker_harvest_hi_hz"] = float(seed_branch_diag_meta["diagnostic_harvest_window_hz"][1])
    cfg["geometry"] = sample_geometry(sample)
    mats = sample.get("materials") or {}
    if mats.get("top_wood_id") or mats.get("back_wood_id"):
        apply_wood_ids_to_config(
            cfg,
            top_wood_id=mats.get("top_wood_id"),
            back_wood_id=mats.get("back_wood_id"),
        )
    mo = sample.get("materials_override") or {}
    top = mo.get("top") or {}
    scale = float(top.get("E_L_scale", 1.0))
    if abs(scale - 1.0) > 1.0e-12:
        mat = cfg.setdefault("materials", {}).setdefault("top", {})
        mat["E_L"] = float(mat.get("E_L", 0.0)) * scale

    eps_band_solver = str(sc.get("eps_band_solver", "shift_invert")).strip() or "shift_invert"
    nm_req = int(args.num_modes) if int(args.num_modes) > 0 else int(sc.get("num_modes", 12))
    nm = _apply_master_worker_solver_profile(
        cfg,
        num_modes=nm_req,
        structural_only=False,
        eps_band_solver=eps_band_solver,
    )
    lam_t = (2.0 * math.pi * target_hz) ** 2
    sc["_worker_eps_target_lambda"] = lam_t
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_num_modes"] = nm
    eps_seed_info: Dict[str, Any] = {
        "seed_file_used": None,
        "seed_vector_length": None,
        "seed_layout_valid": None,
        "eps_initial_space_set": False,
        "eps_initial_space_norm": None,
    }
    if args.eps_seed_npy is not None and args.eps_seed_npy.is_file():
        seed = np.asarray(np.load(str(args.eps_seed_npy.resolve())), dtype=np.float64).ravel()
        seed_norm = float(np.linalg.norm(seed))
        sc["_continuation_eps_seed_vector"] = seed
        sc["_continuation_eps_seed_metadata"] = {
            "continuation_seed_source": "acoustic_locator_coupled_embedding",
            "seed_f_hz": target_hz,
            "seed_vector_length": int(seed.size),
            "seed_npy": str(args.eps_seed_npy.resolve()),
        }
        eps_seed_info = {
            "seed_file_used": str(args.eps_seed_npy.resolve()),
            "seed_vector_length": int(seed.size),
            "seed_layout_valid": bool(seed.size > 0 and math.isfinite(seed_norm) and seed_norm > 0),
            "eps_initial_space_set": True,
            "eps_initial_space_norm": seed_norm,
        }

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
    save_errors: List[str] = []
    eps_diag_pre = dict(cfg.get("_eps_batch_diagnostics") or {})
    preserve_all_bank = bool(
        (seed_branch_diag_meta or {}).get("eps_diagnostic_preserve_all_nconv_candidates")
        or sc.get("eps_diagnostic_preserve_all_nconv_candidates")
    )
    bank_records = load_preserve_all_bank_from_config(cfg)
    if preserve_all_bank and not bank_records:
        bank_records = bank_records_from_eigvecs(
            list(freqs_hz),
            eigvecs,
            eps_diag=eps_diag_pre,
        )
        if bank_records:
            save_errors.append("bank_loaded_from_eigvec_fallback")

    def _save_one(vec: np.ndarray, mode_path: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
        save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
        f_hz = rec.get("reported_frequency_hz")
        if f_hz is None or not math.isfinite(float(f_hz)):
            f_hz = float("nan")
        diag = diagnose_mixed_mode(
            vec,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            gnhep=gnhep,
            frequency_hz=float(f_hz),
        )
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        row = {
            **diag,
            **{
                k: energy[k]
                for k in energy
                if k.endswith("_phys") or k == "p_frac_energy_phys"
            },
            "frequency_hz": float(f_hz),
            "mode_class_physical_energy": _classify_phys_energy(
                float(energy["p_frac_energy_phys"])
            ),
        }
        return row

    if preserve_all_bank and bank_records:
        n_saved, mode_rows, save_errors_persist = persist_candidate_bank(
            case_dir,
            bank_records,
            save_vector_fn=_save_one,
        )
        save_errors.extend(save_errors_persist)
        nconv_bank = int(
            eps_diag_pre.get(
                "eps_diagnostic_candidate_bank_count",
                len(bank_records),
            )
        )
        write_eps_candidate_bank_json(
            case_dir,
            bank_records=bank_records,
            saved_rows=mode_rows,
            nconv_marked=int(eps_diag_pre.get("nconv_marked", nconv_bank)),
            save_errors=save_errors,
            pressure_block_mapping=pressure_block_mapping_metadata(
                p_to_W=p_to_W,
                source="v2_sensitivity_solve_result_p_to_W",
            ),
        )
        for row in mode_rows:
            f_hz = float(row.get("frequency_hz", float("nan")))
            if math.isfinite(f_hz) and band_lo <= f_hz <= band_hi:
                in_band.append(row)
    else:
        n_col = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
        for j in range(n_col):
            vec = eigvecs[:, j]
            mode_path = (
                case_dir
                / "modes"
                / f"candidate_eps_slot_{j:04d}{MODE_VECTOR_FILE_SUFFIX}"
            )
            row = _save_one(
                vec,
                mode_path,
                {"reported_frequency_hz": float(freqs_hz[j])},
            )
            row["candidate_index"] = j
            row["mode_index"] = j
            row["vector_file"] = str(mode_path.relative_to(case_dir)).replace("\\", "/")
            row["vector_path"] = row["vector_file"]
            mode_rows.append(row)
            if band_lo <= float(freqs_hz[j]) <= band_hi:
                in_band.append(row)

    n_modes = len(mode_rows)

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    eps_diag = dict(cfg.get("_eps_batch_diagnostics") or sc.get("_eps_batch_diagnostics") or {})
    persistence_failure: Optional[Dict[str, Any]] = None
    if preserve_all_bank and args.seed_branch_mapping_fixed_unregularized_diagnostic:
        persistence_failure = check_persistence_gate(
            nconv_marked=int(eps_diag.get("nconv_marked", 0)),
            bank_count=int(
                eps_diag.get("eps_diagnostic_candidate_bank_count", len(bank_records))
            ),
            num_vectors_saved=int(n_modes),
            save_errors=save_errors,
        )
        if persistence_failure is not None:
            persistence_failure["candidate_output_dir"] = str(case_dir)
    continuation_seed_applied = bool(eps_diag.get("continuation_seed_applied"))
    st_a_used = float(eps_diag.get("st_a_shift_frac_used", 0.0) or 0.0)
    st_m_used = float(eps_diag.get("st_mass_reg_frac_used", 0.0) or 0.0)
    op_consistent = bool(
        eps_diag.get(
            "diagnostic_operator_consistent_with_replay",
            abs(st_a_used) <= 1.0e-15 and abs(st_m_used) <= 1.0e-15,
        )
    )
    if eps_seed_info.get("eps_initial_space_set"):
        continuation_seed_applied = True

    if args.seed_branch_recovery_diagnostic:
        branch, sel_method = None, "seed_branch_recovery_diagnostic_all_modes"
    else:
        branch, sel_method = _pick_acoustic_branch(
            in_band,
            select_by_energy=bool(args.select_by_energy),
            reference_f_hz=reference_f_hz,
        )

    result = {
        "sample_id": sample_id,
        "elapsed_s": elapsed,
        "mesh_file": str(mesh_path),
        "n_reduced_W": int(restr.get("n_reduced_W", -1)),
        "n_u_active": int(restr.get("n_u_active", u_to_W.size)),
        "n_p_active": int(restr.get("n_p_active", p_to_W.size)),
        "p_to_W": p_to_W.tolist(),
        "u_to_W": u_to_W.tolist(),
        "eps_batch_diagnostics": eps_diag,
        "eps_seed": eps_seed_info,
        "target_hz": float(target_hz),
        "nconv": int(eps_diag.get("nconv_marked", -1)),
        "nconv_marked": int(eps_diag.get("nconv_marked", -1)),
        "v2_converged": int(eps_diag.get("nconv_marked", -1)) > 0,
        "harvest_band_hz": [band_lo, band_hi],
        "shift_invert_target_hz": target_hz,
        "acoustic_branch_selection": sel_method,
        "in_band_modes": in_band,
        "acoustic_branch_by_energy": branch if args.select_by_energy else None,
        "nearest_acoustic_branch": branch,
        "num_modes_saved": n_modes,
        "gnhep_scales": {k: float(gnhep.get(k, 1.0)) for k in ("s_uu", "s_pp", "s_couple")},
        "continuation_seed_applied": continuation_seed_applied,
        "st_sigma_hz_used": float(
            eps_diag.get("st_sigma_hz_used", cfg.get("_worker_st_sigma_hz", float("nan")))
        ),
        "actual_st_a_shift_frac": st_a_used,
        "actual_st_mass_reg_frac": st_m_used,
        "diagnostic_operator_consistent_with_replay": op_consistent,
    }
    if seed_branch_diag_meta is not None:
        requires_unreg = bool(
            seed_branch_diag_meta.get("diagnostic_requires_unregularized_ST")
            or args.seed_branch_unregularized_offset_diagnostic
            or args.seed_branch_mapping_fixed_unregularized_diagnostic
        )
        diag_verdict: Optional[str] = None
        if persistence_failure is not None:
            diag_verdict = str(persistence_failure["diagnostic_verdict"])
        elif requires_unreg and not op_consistent:
            diag_verdict = (
                "DIAGNOSTIC_ST_REGULARIZATION_REQUIRED_NO_PHYSICAL_VERDICT"
            )
        result["seed_branch_recovery_diagnostic"] = {
            **seed_branch_diag_meta,
            "diagnostic_requires_unregularized_ST": requires_unreg,
            "continuation_seed_applied": continuation_seed_applied,
            "seed_file_used": eps_seed_info.get("seed_file_used"),
            "seed_layout_valid": eps_seed_info.get("seed_layout_valid"),
            "seed_vector_length": eps_seed_info.get("seed_vector_length"),
            "eps_initial_space_set": eps_seed_info.get("eps_initial_space_set"),
            "eps_initial_space_norm": eps_seed_info.get("eps_initial_space_norm"),
            "actual_sigma_hz": float(
                eps_diag.get("st_sigma_hz_used", result["st_sigma_hz_used"])
            ),
            "actual_st_a_shift_frac": st_a_used,
            "actual_st_mass_reg_frac": st_m_used,
            "diagnostic_operator_consistent_with_replay": op_consistent,
            "diagnostic_verdict": diag_verdict,
        }
        if persistence_failure is not None:
            result["seed_branch_recovery_diagnostic"]["persistence_failure"] = (
                persistence_failure
            )
            result["num_modes_saved"] = int(n_modes)
        if diag_verdict is not None:
            result["diagnostic_verdict"] = diag_verdict
        result["solver_mode"] = seed_branch_diag_meta["solver_mode"]
    else:
        result["solver_mode"] = "standard_v2_sensitivity"
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    _write_json(case_dir / "diagnostics" / "mode_energy_summary.json", {"modes": mode_rows})
    if MPI.COMM_WORLD.rank == 0:
        f_br = float((branch or {}).get("frequency_hz", float("nan")))
        p_br = float((branch or {}).get("p_frac_energy_phys", float("nan")))
        print(
            f"[v2_sensitivity_solve] sample={sample_id} v2_converged={result['v2_converged']} "
            f"band=[{band_lo},{band_hi}] target={target_hz} selection={sel_method} "
            f"f_branch={f_br:.6f} p_frac_energy={p_br:.4f}",
            flush=True,
        )
    if args.seed_branch_recovery_diagnostic:
        ok = bool(result["v2_converged"] and continuation_seed_applied)
    else:
        ok = bool(
            result["v2_converged"]
            and (branch is not None or args.structural_spectrum_harvest)
        )
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
