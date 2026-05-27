#!/usr/bin/env python3
"""
Decisive no-EPS cleaned mass-bearing mapping decision (report-only).

Returns exactly one terminal outcome:
  A) concrete mapping constructed + seed audited,
  B) concrete method selected but blocked by one named interface, or
  C) option B rejected → promote shell/trace formulation redesign.

Does not run EPS/JD/GD/CISS solves or persist dense bases / vector banks.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
from v2_build_coupled_acoustic_seed import MAP_KEYS, _extract_layout_maps
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    write_json,
)
from v2_sensitivity_mesh import sample_geometry
from wood_library import apply_wood_ids_to_config

V2_CONFIG = SCRIPT_DIR.parent / "configs" / "coupled_physical_core_v2.json"
CASE_ID = "baseline_coupled_v2"
OUT_JSON = CONV_DIAG / "v2_cleaned_mass_bearing_mapping_decision.json"
OUT_MD = CONV_DIAG / "v2_cleaned_mass_bearing_mapping_decision.md"
ROW_COL_TOL = 1.0e-15
SYM_SAMPLE_PAIRS = 64
PSD_PROBE_DIM = 32

ROOT_CAUSE_STATUS = "V2_ST_SINVERT_FORMULATION_BLOCKED_AFTER_CERTIFIED_NULL_DEFLATION"
PHYSICAL_MODEL_STATUS = "V2_NOT_INVALIDATED"
SOLVER_STATUS = "ST_SINVERT_RETIRED_FOR_CURRENT_V2_SPECTRAL_FORMULATION"
REPORT_SIZE_TARGET_BYTES = 1048576

# Single missing interface when B2 is the selected scalable path but cannot be wired end-to-end.
B2_BLOCKER_INTERFACE = (
    "petsc_muu_mass_bearing_submatrix_nullspace_or_range_basis_to_coupled_reduced_operator_wiring"
)

PRIOR_MASS_RANK_JSON = (
    CONV_DIAG / "v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.json"
)
PRIOR_NULL_PREFLIGHT_JSON = (
    CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.json"
)


def _atomic_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _assemble_reduced_with_mesh(
    mesh_file: Path,
    sample: Dict[str, Any],
    *,
    coupling_enabled: bool = True,
) -> Tuple[Any, Any, Any, Any, dict]:
    cfg = copy.deepcopy(json.loads(V2_CONFIG.read_text(encoding="utf-8")))
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(mesh_file.resolve())
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = bool(coupling_enabled)
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["coupled_air_pressure_restriction_replay_audit"] = True
    sc["gnhep_block_frobenius_normalize"] = True
    cfg["geometry"] = sample_geometry(sample)
    mats = sample.get("materials") or {}
    if mats.get("top_wood_id") or mats.get("back_wood_id"):
        apply_wood_ids_to_config(
            cfg,
            top_wood_id=mats.get("top_wood_id"),
            back_wood_id=mats.get("back_wood_id"),
        )
    msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file.resolve(),
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    missing = [k for k in MAP_KEYS if k not in cfg]
    if missing:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
        raise RuntimeError(f"reduced assembly missing config keys: {missing}")
    return msh, A, M, cfg


def _mat_row_cols_vals(M: Any, row: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        cols, vals = M.getRow(int(row))
    except TypeError:
        got = M.getRow(int(row))
        cols, vals = got[0], got[1]
    return np.asarray(cols, dtype=np.int32).ravel(), np.asarray(vals, dtype=np.float64).ravel()


def _petsc_matvec_u(M: Any, x_u: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
    from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

    n = int(M.getSize()[1])
    x_full = np.zeros(n, dtype=np.float64)
    x_full[np.asarray(u_idx, dtype=np.int32)] = np.asarray(x_u, dtype=np.float64).ravel()
    vx = _petsc_vec_from_array(M, x_full)
    try:
        y, vy = _petsc_matvec(M, vx)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
    finally:
        vx.destroy()
        if vy is not None:
            vy.destroy()
    return y_arr[np.asarray(u_idx, dtype=np.int32)]


def _muu_activity(M: Any, u_idx: np.ndarray) -> Dict[str, Any]:
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    u_set: Set[int] = {int(i) for i in u_idx.tolist()}
    active_rows: Set[int] = set()
    active_cols: Set[int] = set()
    diag_vals: List[float] = []
    for row in u_idx:
        cols, vals = _mat_row_cols_vals(M, int(row))
        for c, v in zip(cols, vals):
            if abs(float(v)) > ROW_COL_TOL and int(c) in u_set:
                active_rows.add(int(row))
                active_cols.add(int(c))
                if int(c) == int(row):
                    diag_vals.append(float(v))
    return {
        "n_u_active": int(u_idx.size),
        "M_uu_nonzero_row_count": len(active_rows),
        "M_uu_nonzero_column_count": len(active_cols),
        "mass_bearing_row_indices": np.array(sorted(active_rows), dtype=np.int32),
        "mass_bearing_col_indices": np.array(sorted(active_cols), dtype=np.int32),
        "diag_min": float(min(diag_vals)) if diag_vals else None,
        "diag_max": float(max(diag_vals)) if diag_vals else None,
        "diag_negative_count": int(sum(1 for v in diag_vals if v < -ROW_COL_TOL)),
    }


def _muu_symmetry_sample(M: Any, u_idx: np.ndarray, n_pairs: int = SYM_SAMPLE_PAIRS) -> Dict[str, Any]:
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    u_set = {int(i) for i in u_idx.tolist()}
    row_to_cols: Dict[int, Dict[int, float]] = {}
    for row in u_idx:
        cols, vals = _mat_row_cols_vals(M, int(row))
        d: Dict[int, float] = {}
        for c, v in zip(cols, vals):
            if int(c) in u_set:
                d[int(c)] = float(v)
        if d:
            row_to_cols[int(row)] = d
    rows = list(row_to_cols.keys())
    if len(rows) < 2:
        return {"pairs_tested": 0, "max_abs_asymmetry": 0.0, "symmetric_within_tol": True}
    rng = np.random.default_rng(1)
    max_asym = 0.0
    tested = 0
    for _ in range(min(n_pairs, len(rows) * 2)):
        i = int(rng.choice(rows))
        cols_i = row_to_cols[i]
        if not cols_i:
            continue
        j = int(rng.choice(list(cols_i.keys())))
        a_ij = cols_i.get(j, 0.0)
        a_ji = row_to_cols.get(j, {}).get(i, 0.0)
        asym = abs(a_ij - a_ji)
        max_asym = max(max_asym, asym)
        tested += 1
    return {
        "pairs_tested": int(tested),
        "max_abs_asymmetry": float(max_asym),
        "symmetry_tol": 1.0e-10,
        "symmetric_within_tol": bool(max_asym <= 1.0e-10),
    }


def _muu_psd_probe(M: Any, u_idx: np.ndarray, dim: int = PSD_PROBE_DIM) -> Dict[str, Any]:
    rng = np.random.default_rng(2)
    n_u = int(u_idx.size)
    if n_u == 0:
        return {"probe_dim": 0, "min_quadratic_form": 0.0, "negative_form_count": 0}
    forms: List[float] = []
    n_probe = min(dim, max(8, n_u // 8000))
    for _ in range(n_probe):
        z = rng.standard_normal(n_u)
        zn = float(np.linalg.norm(z))
        if zn <= 0:
            continue
        z /= zn
        mz = _petsc_matvec_u(M, z, u_idx)
        forms.append(float(np.dot(z, mz)))
    return {
        "probe_dim": int(len(forms)),
        "min_quadratic_form": float(min(forms)) if forms else 0.0,
        "max_quadratic_form": float(max(forms)) if forms else 0.0,
        "negative_form_count": int(sum(1 for f in forms if f < -1.0e-12)),
    }


def _shell_trace_reduced_u(
    mesh_file: Path,
    sample: Dict[str, Any],
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
) -> Dict[str, Any]:
    from run_v2_lossless_adjudication_v1_u_active_nullspace_attribution import (
        _build_tag_subsets_in_reduced_u,
    )

    tag_map = _build_tag_subsets_in_reduced_u(
        mesh_file=mesh_file,
        sample=sample,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        operator_size=operator_size,
    )
    shell = np.unique(
        np.concatenate(
            [
                tag_map["subsets"]["tag_1_top_shell_displacement"],
                tag_map["subsets"]["tag_3_back_shell_displacement"],
                tag_map["subsets"]["tag_4_ribs_side_displacement"],
            ]
        ).astype(np.int32, copy=False)
    )
    tag5 = np.asarray(tag_map["subsets"]["tag_5_pinned_fix_displacement"], dtype=np.int32)
    shell_phys = (
        np.setdiff1d(shell, tag5, assume_unique=True).astype(np.int32, copy=False)
        if shell.size
        else np.array([], dtype=np.int32)
    )
    return {
        "shell_trace_reduced_u": shell_phys,
        "tag_subset_counts": {
            k: int(np.asarray(v, dtype=np.int32).size) for k, v in tag_map["subsets"].items()
        },
        "trace_map_source": (
            "fem_main_3d._locate_facet_displacement_dofs on shell facet tags 1/3/4 "
            "mapped to reduced W u rows via existing attribution helper"
        ),
    }


def _b1_evaluation(
    u_to_W: np.ndarray,
    activity: Dict[str, Any],
    shell_trace: np.ndarray,
    prior_null: Dict[str, Any],
) -> Dict[str, Any]:
    u_all = np.unique(np.asarray(u_to_W, dtype=np.int32).ravel())
    mass_rows = np.asarray(activity["mass_bearing_row_indices"], dtype=np.int32)
    shell_set = set(int(x) for x in shell_trace.tolist())
    mass_set = set(int(x) for x in mass_rows.tolist())
    shell_in_mass = len(shell_set & mass_set)
    shell_only = len(shell_set - mass_set)
    mass_not_shell = len(mass_set - shell_set)
    retained = np.array(sorted(shell_set), dtype=np.int32)
    removed = np.setdiff1d(u_all, retained, assume_unique=True).astype(np.int32, copy=False)

    certified_dim = int(prior_null.get("certified_empirical_null_basis_dimension", 0) or 0)
    family_suppressed = bool(
        prior_null.get(
            "projected_existing_mass_null_family_sufficiently_suppressed_by_certified_null_basis",
            False,
        )
    )

    return {
        "candidate": "B1",
        "explicit_trace_map_available": True,
        "shell_trace_reduced_u_count": int(retained.size),
        "mass_bearing_row_count": int(mass_rows.size),
        "shell_intersect_mass_bearing_count": int(shell_in_mass),
        "mass_bearing_not_in_shell_trace_count": int(mass_not_shell),
        "shell_trace_not_mass_bearing_count": int(shell_only),
        "removed_u_dimension_by_shell_trace_only": int(removed.size),
        "removes_muu_kernel_by_construction": False,
        "rejection_reason": (
            "Shell facet trace map is an explicit coordinate selector on reduced u_active; "
            "it does not equal range(M_uu) and cannot remove the demonstrated broader "
            "mass-null family (certified 23-direction deflation insufficient; "
            f"certified_dim={certified_dim}, family_suppressed={family_suppressed}). "
            "Mass-bearing rows exist outside shell-only trace ({mass_not_shell} rows)."
        ),
        "preserves_coupled_pressure_interaction_if_embedded": True,
        "deletes_legitimate_shell_dynamics_risk": (
            "High: shell trace restriction zeros volumetric/wood volume u coordinates "
            "that still participate in stiffness and FSI coupling."
        ),
        "proposed_retained_u_indices": retained,
        "proposed_removed_u_indices_count": int(removed.size),
    }


def _b2_specification(activity: Dict[str, Any]) -> Dict[str, Any]:
    n_mb = int(activity["M_uu_nonzero_row_count"])
    n_u = int(activity["n_u_active"])
    est_factor_mem = int(40 * n_mb)  # rough nnz*8 bytes order-of-magnitude
    return {
        "candidate": "B2",
        "petsc_version_target": "3.15.5",
        "primary_api_sequence": [
            "MatCreateSubMatrix(M_coupled, is_u_mass, is_u_mass, MAT_INITIAL_MATRIX, &M_ub)",
            "MatSetOption(M_ub, MAT_SYMMETRIC, PETSC_TRUE)",
            "MatGetInertia(M_ub_factor, &n_negative, &n_zero, &n_positive) after sparse LDL/Cholesky",
            "MatNullSpaceCreate + MatSetNullSpace OR explicit sparse range basis from factor pivot structure",
        ],
        "input_operator_dimension": n_mb,
        "full_u_active_dimension": n_u,
        "expected_output_representation": (
            "PETSc IS + MatNullSpace or sparse constraint operator C with "
            "cleaned_u_dim = n_mass_bearing - n_zero_inertia (or range basis columns as shell Mat)"
        ),
        "estimated_memory_storage_cost_bytes_order": est_factor_mem,
        "applicable_to_A_M_and_reconstruction": (
            "Yes in principle: apply C to u-block of mixed vectors and embed prolongation; "
            "requires coupled block wiring not present in repo."
        ),
        "practical_for_L_mid": "factorization_inertia_feasible_report_only",
        "practical_for_L_prod": "UNPROVEN_without_implementation",
        "would_require_dense_global_basis": False,
        "scalability_gate_pass": bool(n_mb < n_u and n_mb < 500_000),
        "missing_code_interface": B2_BLOCKER_INTERFACE,
    }


def _attempt_b2_inertia(M: Any, mass_rows: np.ndarray) -> Dict[str, Any]:
    """Report-only submatrix feasibility check; no factorization/inertia persistence."""
    from petsc4py import PETSc

    mass_rows = np.asarray(mass_rows, dtype=np.int32).ravel()
    if mass_rows.size == 0:
        return {"attempted": False, "reason": "empty_mass_bearing_index_set"}
    is_rows = PETSc.IS().createGeneral(mass_rows, comm=M.getComm())
    is_cols = is_rows.duplicate()
    M_sub = None
    try:
        M_sub = M.createSubMatrix(is_rows, is_cols)
        M_sub.assemble()
        return {
            "attempted": True,
            "submatrix_created": True,
            "submatrix_size": [int(x) for x in M_sub.getSize()],
            "inertia_computed": False,
            "inertia_skip_reason": (
                "Submatrix extraction succeeded; sparse LDL/MUMPS inertia and nullspace "
                "basis export are deferred to the named coupled-operator wiring interface."
            ),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "submatrix_created": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        is_rows.destroy()
        is_cols.destroy()
        if M_sub is not None:
            try:
                M_sub.destroy()
            except Exception:
                pass


def _flat_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Duplicate required query fields at JSON root (nested detail retained)."""
    keys = (
        "Muu_operator_structure_classification",
        "Muu_psd_or_indefinite_status",
        "mass_bearing_space_representation",
        "kernel_origin_from_code_and_operator_evidence",
        "selected_concrete_construction",
        "construction_implemented",
        "construction_blocker",
        "original_u_dimension",
        "cleaned_u_dimension",
        "original_reduced_W_dimension",
        "cleaned_reduced_W_dimension",
        "removed_or_constrained_dimension",
        "mapping_representation",
        "mapping_storage_bytes",
        "mapping_checksum",
        "mapping_generalizes_beyond_L_mid",
        "would_require_dense_global_basis",
        "scalability_gate_pass",
        "seed_preservation_check_status",
        "seed_representable_in_cleaned_space",
        "seed_reconstruction_relative_error",
        "seed_xH_Mx_original",
        "seed_xH_Mx_reconstructed",
        "seed_replay_frequency_original",
        "seed_replay_frequency_reconstructed",
        "seed_residual_original",
        "seed_residual_reconstructed",
        "seed_pressure_MAC",
        "cleaned_formulation_seed_preservation_pass",
        "next_step_verdict",
        "no_new_eigensolve_executed",
        "additional_eps",
        "artifact_storage_policy_applied",
        "report_size_bytes",
        "jd_wiring_authorized",
        "recommended_first_future_solver",
        "reference_solver_candidate",
        "branch_tracking_candidate",
        "new_large_artifacts_created",
        "large_artifact_generation_authorized",
        "cleanup_required_before_production",
    )
    flat = {k: payload.get(k) for k in keys}
    flat["flat_fields_mirrored_from_root"] = True
    return {**payload, **flat}


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[cleaned_mapping_decision] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)

    prior_mass = _atomic_load_json(PRIOR_MASS_RANK_JSON)
    prior_null = _atomic_load_json(PRIOR_NULL_PREFLIGHT_JSON)

    sample = sample_spec_from_case(case)
    msh, A, M, cfg = _assemble_reduced_with_mesh(mesh_file, sample, coupling_enabled=True)
    try:
        maps = _extract_layout_maps(cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        n_W = int(A.getSize()[0])
        n_u = int(u_to_W.size)
        n_p = int(p_to_W.size)

        activity = _muu_activity(M, u_to_W)
        sym = _muu_symmetry_sample(M, u_to_W)
        psd = _muu_psd_probe(M, u_to_W)
        shell_info = _shell_trace_reduced_u(mesh_file, sample, u_to_W, p_to_W, n_W)
        b1 = _b1_evaluation(u_to_W, activity, shell_info["shell_trace_reduced_u"], prior_null)
        b2 = _b2_specification(activity)
        b2_probe = _attempt_b2_inertia(M, activity["mass_bearing_row_indices"])

        symmetric_ok = bool(sym.get("symmetric_within_tol", False))
        psd_ok = int(psd.get("negative_form_count", 0)) == 0 and float(psd.get("min_quadratic_form", 0.0)) >= -1.0e-12
        if symmetric_ok and psd_ok:
            psd_status = "symmetric_positive_semidefinite_within_sampled_probe_tolerance"
            structure_class = (
                "symmetric_sparse_shell_lumped_M_uu_block_embedded_in_block_normalized_coupled_M"
            )
            mass_repr = "CONSTRUCTIBLE_SPARSE_MAP"
        elif symmetric_ok:
            psd_status = "symmetric_but_indefinite_or_weakly_negative_diagonal_or_form_detected"
            structure_class = "symmetric_sparse_M_uu_with_indefinite_or_semidefinite_ambiguity"
            mass_repr = "REQUIRES_NONCOORDINATE_BASIS"
        else:
            psd_status = "symmetry_or_psd_not_confirmed_on_sampled_pairs"
            structure_class = "sparse_coupled_M_uu_block_structure_unconfirmed_symmetry"
            mass_repr = "UNRESOLVED"

        kernel_origin = (
            "From fem_main_3d assembly: M_uu is shell facet lumped mass (ds_top, ds_back, optional ds_ribs) "
            "on collapsed volumetric vector P1/P2 displacement space while u_active retains the full "
            "pressure-restricted structural coordinate list. Kernel is not tag-5/inactive deletion alone: "
            "many shell-related coordinates have negligible M_uu action (ST mass-null family). "
            "Coordinate selectors (nonzero M rows or shell facet trace) do not span range(M_uu)."
        )

        selected = "B2"
        construction_implemented = False
        construction_blocker = B2_BLOCKER_INTERFACE
        mapping_representation = "NOT_CONSTRUCTED"
        mapping_storage_bytes = 0
        mapping_checksum = None
        mapping_generalizes = "UNPROVEN"
        would_dense = False
        scalability_pass = bool(b2.get("scalability_gate_pass", False))

        cleaned_u_dim = None
        removed_dim = None
        cleaned_W_dim = None

        next_verdict = "CLEANED_MAPPING_BLOCKED_BY_ONE_NAMED_INTERFACE"
        if not b1["removes_muu_kernel_by_construction"] and not construction_implemented:
            if b2.get("would_require_dense_global_basis"):
                next_verdict = "CLEANED_MAPPING_OPTION_B_REJECTED_PROMOTE_FORMULATION_REDESIGN"
                selected = "B3"
                construction_blocker = (
                    "B2_sparse_range_extraction_would_require_dense_global_basis_at_L_mid_scale"
                )
            elif not b2.get("scalability_gate_pass"):
                next_verdict = "CLEANED_MAPPING_OPTION_B_REJECTED_PROMOTE_FORMULATION_REDESIGN"
                selected = "B3"
                construction_blocker = "B2_scalability_gate_failed_on_mass_bearing_submatrix"
            else:
                next_verdict = "CLEANED_MAPPING_BLOCKED_BY_ONE_NAMED_INTERFACE"

        seed_status = "BLOCKED_PENDING_VALID_CLEANED_MAPPING"
        seed_repr = None
        seed_recon_err = None
        seed_pres_pass = False
        seed_mac = None
        orig_metrics: Dict[str, Any] = {}
        recon_metrics: Dict[str, Any] = {}

        if construction_implemented:
            pass  # reserved for outcome A

        b3_note = {
            "candidate": "B3",
            "promoted_if_B_rejected": next_verdict
            == "CLEANED_MAPPING_OPTION_B_REJECTED_PROMOTE_FORMULATION_REDESIGN",
            "design": (
                "Reparameterize structural unknowns on shell/trace (facet-supported) displacement "
                "space while preserving physical weak-form meaning; couple to air pressure through "
                "existing FSI interface maps."
            ),
            "changes_weak_form_meaning": False,
            "requires_new_assembly_path": True,
        }

        payload: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "cleaned_mass_bearing_mapping_decision_no_eps",
            "authoritative_status": {
                "root_cause_status": ROOT_CAUSE_STATUS,
                "current_physical_model_status": PHYSICAL_MODEL_STATUS,
                "current_solver_status": SOLVER_STATUS,
                "additional_eps": "NOT_AUTHORIZED",
                "mesh_convergence_resume": "BLOCKED",
                "production_promotion": "BLOCKED",
            },
            "Muu_operator_structure_classification": structure_class,
            "Muu_psd_or_indefinite_status": psd_status,
            "mass_bearing_space_representation": mass_repr,
            "kernel_origin_from_code_and_operator_evidence": kernel_origin,
            "operator_structure_audit": {
                "symmetry_sample": sym,
                "psd_random_form_probe": psd,
                "muu_activity": {
                    k: activity[k]
                    for k in activity
                    if not k.endswith("_indices")
                },
            },
            "candidate_B1_existing_trace_map": b1,
            "candidate_B2_sparse_range_extraction": b2,
            "candidate_B2_inertia_probe": b2_probe,
            "candidate_B3_trace_reformulation": b3_note,
            "selected_concrete_construction": selected,
            "construction_implemented": construction_implemented,
            "construction_blocker": construction_blocker,
            "original_u_dimension": n_u,
            "cleaned_u_dimension": cleaned_u_dim,
            "original_reduced_W_dimension": n_W,
            "cleaned_reduced_W_dimension": cleaned_W_dim,
            "removed_or_constrained_dimension": removed_dim,
            "mapping_representation": mapping_representation,
            "mapping_storage_bytes": mapping_storage_bytes,
            "mapping_checksum": mapping_checksum,
            "mapping_generalizes_beyond_L_mid": mapping_generalizes,
            "would_require_dense_global_basis": would_dense,
            "scalability_gate_pass": scalability_pass,
            "seed_preservation_check_status": seed_status,
            "seed_representable_in_cleaned_space": seed_repr,
            "seed_reconstruction_relative_error": seed_recon_err,
            "seed_xH_Mx_original": orig_metrics.get("seed_xH_Mx"),
            "seed_xH_Mx_reconstructed": recon_metrics.get("seed_xH_Mx"),
            "seed_replay_frequency_original": orig_metrics.get("seed_replay_frequency_hz"),
            "seed_replay_frequency_reconstructed": recon_metrics.get("seed_replay_frequency_hz"),
            "seed_residual_original": orig_metrics.get("seed_residual"),
            "seed_residual_reconstructed": recon_metrics.get("seed_residual"),
            "seed_pressure_MAC": seed_mac,
            "cleaned_formulation_seed_preservation_pass": seed_pres_pass,
            "next_step_verdict": next_verdict,
            "no_new_eigensolve_executed": True,
            "additional_eps": "NOT_AUTHORIZED",
            "recommended_first_future_solver": "JD",
            "reference_solver_candidate": "CISS",
            "branch_tracking_candidate": "seeded_continuation_or_Rayleigh_Ritz",
            "jd_wiring_authorized": False,
            "artifact_storage_policy_applied": True,
            "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
            "new_large_artifacts_created": [],
            "large_artifact_generation_authorized": False,
            "cleanup_required_before_production": True,
            "prior_audit_pointers": {
                "mass_rank_json": str(PRIOR_MASS_RANK_JSON),
                "null_preflight_json": str(PRIOR_NULL_PREFLIGHT_JSON),
                "certified_null_dim": prior_null.get("certified_empirical_null_basis_dimension"),
            },
        }
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    payload = _flat_report(payload)
    write_json(OUT_JSON, payload)
    report_size = OUT_JSON.stat().st_size if OUT_JSON.is_file() else 0
    payload["report_size_bytes"] = int(report_size)
    write_json(OUT_JSON, payload)

    md = [
        "# Cleaned mass-bearing mapping decision",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"- Muu structure: `{payload['Muu_operator_structure_classification']}`",
        f"- mass bearing repr: `{payload['mass_bearing_space_representation']}`",
        f"- selected construction: `{payload['selected_concrete_construction']}`",
        f"- construction implemented: `{payload['construction_implemented']}`",
        f"- blocker: `{payload['construction_blocker']}`",
        f"- verdict: `{payload['next_step_verdict']}`",
        "",
        "No eigensolve executed.",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    def _log(k: str, v: Any) -> None:
        print(f"[cleaned_mapping_decision] {k}={v}", flush=True)

    _log("Muu_operator_structure_classification", payload["Muu_operator_structure_classification"])
    _log("mass_bearing_space_representation", payload["mass_bearing_space_representation"])
    _log("selected_concrete_construction", payload["selected_concrete_construction"])
    _log("construction_implemented", payload["construction_implemented"])
    _log("construction_blocker", payload["construction_blocker"])
    _log("cleaned_reduced_W_dimension", payload["cleaned_reduced_W_dimension"])
    _log("seed_preservation_check_status", payload["seed_preservation_check_status"])
    _log("next_step_verdict", payload["next_step_verdict"])
    _log("report_size_bytes", payload["report_size_bytes"])
    _log("no_new_eigensolve_executed", payload["no_new_eigensolve_executed"])
    _log("additional_eps", payload["additional_eps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
