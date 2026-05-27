#!/usr/bin/env python3
"""
Phase-1 bounded no-EPS preflight: cleaned mass-bearing formulation design + seed mapping scaffold.

Does not call eps.solve() or persist vector banks.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    write_json,
)
from v2_slepc_api_preflight_lib import slepc_eps_api_probe

OUT_JSON = CONV_DIAG / "v2_cleaned_mass_bearing_formulation_phase1_preflight.json"
OUT_MD = CONV_DIAG / "v2_cleaned_mass_bearing_formulation_phase1_preflight.md"

ROOT_CAUSE_STATUS = "V2_ST_SINVERT_FORMULATION_BLOCKED_AFTER_CERTIFIED_NULL_DEFLATION"
PHYSICAL_MODEL_STATUS = "V2_NOT_INVALIDATED"
SOLVER_STATUS = "ST_SINVERT_RETIRED_FOR_CURRENT_V2_SPECTRAL_FORMULATION"

CASE_ID = "baseline_coupled_v2"
LEVEL_ID = "L_mid"
SEED_F_HZ = 243.0754171175576
M_UU_SVD_RCOND = 1e-10
NULL_PROBE_TOL = 1e-10
NULL_PROBE_COUNT = 48
SEED_REL_TOL = 1e-8
SEED_FREQ_REL_TOL = 1e-6
SEED_XH_REL_TOL = 1e-6

RECOMMENDED_FORMULATION = (
    "algebraic_mass_bearing_range_constrained_gnhep_from_assembled_M_uu_structure"
)
N_U_ACTIVE_REPORTED = 253587


def _formulation_cause_from_code() -> Dict[str, Any]:
    return {
        "displacement_space_model": {
            "classification": "CONFIRMED_FROM_CODE",
            "summary": (
                "Reduced coupled EVP uses full active structural displacement coordinates "
                "(u_active via _coupled_air_u_to_W_map) together with active pressure DOFs. "
                "Shell stiffness and structural mass M_uu integrate only on tagged shell facets "
                "(structural_shell_facet_tags, typically tags 1/3/4), while u_active spans "
                "all retained structural DOFs including non-shell and pinned subsets."
            ),
            "code_refs": [
                "FEM/scripts/fem_main_3d.py: shell forms on facet tags; u_to_W_map for all active u",
                "run_v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.py",
            ],
        },
        "why_empirical_23_deflation_failed": {
            "classification": "CONFIRMED_FROM_CODE_AND_PRIOR_VM_EVIDENCE",
            "summary": (
                "Certified empirical null basis (dim=23) removed ~99.999999999989257 of prior "
                "candidate energy but EPS still returned 56/56 mass-null u-dominated modes. "
                "Numerical rank probes estimated a larger null family (~38) than the certified "
                "empirical subspace; deflation inside EPS did not change the solver-visible "
                "contaminated search family on the raw pencil."
            ),
            "not_a_tag5_only_problem": True,
            "attribution": "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL",
        },
        "general_cleaned_formulation_principle": (
            "Remove or constrain the structural M_uu kernel by operator-derived range/null "
            "structure at assembly time, preserving V2 weak forms and u/p coupling blocks."
        ),
    }


def _compare_formulation_options() -> Dict[str, Any]:
    return {
        "option_A_structural_exact_coordinate_restriction": {
            "description": (
                "Structural exact coordinate restriction derived from shell/coupling support only"
            ),
            "mathematical_validity_for_proven_kernel_class": (
                "insufficient_unless_proof_shows_kernel_removed_without_deleting_physical_shell_dynamics"
            ),
            "expected_scalability_L_mid_to_L_prod": "high_if_simple_index_restriction",
            "memory_risk": "low",
            "mapping_back_replay_implications": "medium (coordinate bookkeeping required)",
            "generalizes_beyond_empirical_23": True,
            "no_eps_seed_testability": True,
            "changes_physical_weak_forms": False,
            "generalizes_beyond_L_mid": "partial_mesh_specific",
            "uses_empirical_23_vector_basis": False,
            "risk": "high_false_cleaning_risk_for_kernel_through_linear_combinations",
        },
        "option_B_rank_revealing_or_nullspace_range_construction": {
            "description": (
                "Rank-revealing nullspace/range construction on physically supported restricted M_uu "
                "using scalable PETSc/SLEPc-compatible algebraic tooling"
            ),
            "mathematical_validity_for_proven_kernel_class": "high",
            "expected_scalability_L_mid_to_L_prod": "medium_pending_benchmark_and_solver_memory_profile",
            "memory_risk": "medium_high_for_large_factorizations_or_dense_basis_ops",
            "mapping_back_replay_implications": "low_medium_if_projector_metadata_persisted",
            "generalizes_beyond_empirical_23": True,
            "no_eps_seed_testability": True,
            "changes_physical_weak_forms": False,
            "generalizes_beyond_L_mid": "unproven_until_implemented",
            "uses_empirical_23_vector_basis": False,
            "risk": "implementation_complexity_and_memory_tuning_required",
        },
        "option_C_constraint_based_admissible_mass_bearing_coordinates": {
            "description": (
                "Constraint-based cleaned formulation representing only admissible mass-bearing "
                "structural coordinates with preserved p_active coupling"
            ),
            "mathematical_validity_for_proven_kernel_class": "medium_high_if_constraints_are_operator_derived",
            "expected_scalability_L_mid_to_L_prod": "medium",
            "memory_risk": "medium",
            "mapping_back_replay_implications": "medium_high_due_to_constraint_maps",
            "generalizes_beyond_empirical_23": True,
            "no_eps_seed_testability": True,
            "changes_physical_weak_forms": False,
            "generalizes_beyond_L_mid": "unproven_until_constraints_defined",
            "uses_empirical_23_vector_basis": False,
            "risk": "constraint_definition_and_debugability_risk",
        },
        "option_D_schur_pressure_led_reformulation": {
            "description": "Deep Schur/PEP/NEP reformulation",
            "mathematical_validity_for_proven_kernel_class": "potentially_high",
            "expected_scalability_L_mid_to_L_prod": "unknown_design_dependent",
            "memory_risk": "unknown",
            "mapping_back_replay_implications": "high",
            "generalizes_beyond_empirical_23": True,
            "no_eps_seed_testability": "partial",
            "changes_physical_weak_forms": True,
            "generalizes_beyond_L_mid": "unknown",
            "uses_empirical_23_vector_basis": False,
            "risk": "High implementation/recertification burden; fallback only",
            "selected": False,
        },
        "recommended_cleaned_formulation": RECOMMENDED_FORMULATION,
        "recommended_phase1_mapping_construction_method": (
            "option_B_rank_revealing_or_nullspace_range_construction_on_restricted_Muu"
        ),
        "recommendation_reason": (
            "Operator-derived rank-revealing/nullspace construction is the first method that can "
            "directly target the proven kernel class without relying on empirical 23-vector deflation "
            "or random-probe absence tests. Structural tag-only restriction is insufficient, while Schur "
            "reformulation is deep fallback."
        ),
        "recommended_cleaned_formulation_status": "DESIGN_SELECTED_IMPLEMENTATION_NOT_YET_CERTIFIED",
        "generalization_design_goal": True,
        "changes_physical_weak_forms": False,
        "generalizes_beyond_L_mid": (
            "UNPROVEN_PENDING_OPERATOR_DERIVED_RANGE_OR_NULLSPACE_CONSTRUCTION"
        ),
        "uses_empirical_23_vector_basis_as_production_mechanism": False,
        "problem_scale_context": {"n_u_active_reported": N_U_ACTIVE_REPORTED},
    }


def _authoritative_vm_api_facts() -> Dict[str, Any]:
    return {
        "vm_slepc_import_pass": True,
        "petsc_version": "3.15.5",
        "vm_slepc_version": "3.15.2",
        "jd_api_available": True,
        "gd_api_available": True,
        "ciss_api_available": True,
        "krylovschur_api_available": True,
        "ciss_region_api_available": True,
        "new_dependency_required": False,
        "recommended_primary_solver_api_status": (
            "AVAILABLE_REQUIRES_CLEANED_FORMULATION_AND_DISPATCH_INTEGRATION"
        ),
        "source": "authoritative_vm_no_eps_setType_getType_probe",
    }


def _random_probe_diagnostic_context() -> Dict[str, Any]:
    return {
        "random_probe_informational_only": True,
        "random_probe_can_certify_nullspace_absence": False,
        "prior_random_probe_false_negative_demonstrated": True,
        "prior_random_probe_evidence": {
            "probe_median_norm": 6.017e-02,
            "estimated_nullity_upper_bound": 0,
            "contradicted_by_authoritative_lossless_evidence": True,
        },
    }


def _seed_scaffold() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cleaned_formulation_mapping_constructed": False,
        "original_reduced_dimension": None,
        "cleaned_reduced_dimension": None,
        "removed_or_constrained_dimension": None,
        "seed_representable_in_cleaned_space": False,
        "seed_reconstruction_relative_error": None,
        "seed_xH_Mx_original": None,
        "seed_xH_Mx_reconstructed": None,
        "seed_replay_frequency_original": None,
        "seed_replay_frequency_reconstructed": None,
        "seed_residual_original": None,
        "seed_residual_reconstructed": None,
        "seed_pressure_support_preserved": None,
        "seed_MAC_preserved": None,
        "cleaned_formulation_seed_preservation_pass": False,
        "seed_preservation_check_status": "BLOCKED_PENDING_VALID_CLEANED_MAPPING",
        "status": "NOT_YET_IMPLEMENTED",
        "exact_missing_code_path": (
            "implement operator-derived rank-revealing/nullspace-range construction module "
            "(non-random), then wire cleaned->W reconstruction + seed replay gate evaluator"
        ),
    }
    return out


def _future_solver_wiring() -> Dict[str, Any]:
    return {
        "jd_gd": {
            "role": "primary_production_candidate_after_cleaned_formulation",
            "repo_status": "NOT_WIRED_IN_DISPATCH",
            "integration_points": [
                "FEM/scripts/fem_main_3d.py:_slepc_shift_invert_batch (new eps_band_solver branch)",
                "v2_sensitivity_solve.py (solver cfg flags)",
                "cleaned formulation projector module (pre-EPS assembly hook)",
            ],
            "recommended_first_future_solver": "JD",
            "recommended_first_future_solver_reason": (
                "Interior target near 243 Hz with validated acoustic seed; JD is the natural "
                "SLEPc choice for a few targeted modes with correction/subspace iteration. "
                "GD is a reasonable fallback if JD preconditioner tuning is unstable."
            ),
        },
        "ciss": {
            "role": "narrow_band_reference_certification_on_cleaned_formulation",
            "repo_status": "WIRED_FOR_SHIFT_INVERT_BATCH",
            "note": "Use only after cleaned formulation passes seed gates; not full-band production default",
        },
        "seeded_continuation": {
            "role": "branch_tracking_mesh_validation_companion",
            "repo_status": "PARTIAL_INFRASTRUCTURE_EXISTS",
            "note": "MAC/replay/mass-norm gates; not sole production harvest engine",
        },
    }


def _unresolved_production_facts() -> Dict[str, Any]:
    return {
        "production_band_60_550_status": "NOT_FOUND_IN_CODE_USER_DECISION_REQUIRED",
        "production_mode_requirement": "USER_DECISION_REQUIRED",
        "worker_model": "independent_process_per_window",
        "parallel_wall_clock_and_ram": "UNKNOWN_UNTIL_VALID_SOLVER_BENCHMARK",
    }


def main() -> int:
    api_runtime = slepc_eps_api_probe()
    api = _authoritative_vm_api_facts()
    seed = _seed_scaffold()
    random_probe_context = _random_probe_diagnostic_context()
    formulation = _compare_formulation_options()
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "phase_1_cleaned_formulation_no_eps_preflight",
        "authoritative_status": {
            "root_cause_status": ROOT_CAUSE_STATUS,
            "current_physical_model_status": PHYSICAL_MODEL_STATUS,
            "current_solver_status": SOLVER_STATUS,
            "additional_eps": "NOT_AUTHORIZED",
            "mesh_convergence_resume": "BLOCKED",
            "production_promotion": "BLOCKED",
        },
        "solver_api_preflight": api,
        "solver_api_preflight_runtime_observation_local": api_runtime,
        "formulation_cause_analysis": _formulation_cause_from_code(),
        "formulation_option_comparison": formulation,
        "random_probe_diagnostic_context": random_probe_context,
        "seed_preservation_preflight": seed,
        "future_solver_wiring_plan": _future_solver_wiring(),
        "unresolved_production_facts": _unresolved_production_facts(),
        "runtime_evidence_framing": {
            "MEASURED": {"valid_production_equivalent_solver_runtime": "none"},
            "CONFIRMED_FROM_CODE": {
                "configured_windows_and_worker_model": True,
                "slepc_dispatch_wiring_status": "KRYLOVSCHUR_and_CISS_wired; JD_GD_not_wired",
            },
            "ENGINEERING_EXPECTATION_PENDING_BENCHMARK": {
                "jd_gd_primary_after_cleaned_formulation": True,
                "ciss_narrow_band_reference": True,
                "continuation_branch_tracker": True,
                "frequency_response_not_full_band_production": True,
                "numeric_runtime_percentages": "not_reported",
            },
            "conditional_60_550_scaling": {
                "jd_gd": "window/target scaling; parallelizable in principle after benchmark",
                "ciss": "region/contour cost grows with band width; reference role",
                "continuation": "branch-based; not full-bank production",
                "frequency_response": "dense frequency samples; impractical as full-band primary",
                "schur_pep_nep": "design-dependent deep fallback",
            },
        },
        "cleaned_formulation_design_ready": False,
        "artifact_storage_policy": {
            "artifact_storage_policy_applied": True,
            "new_large_artifacts_created": [],
            "cleanup_required_before_production": True,
            "policy": (
                "No vector banks, meshes, solve trees, or raw matrices persisted in Phase 1; "
                "compact JSON/MD summary only."
            ),
        },
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
    }
    write_json(OUT_JSON, payload)
    size_b = OUT_JSON.stat().st_size if OUT_JSON.is_file() else 0
    payload["artifact_storage_policy"]["report_size_bytes"] = int(size_b)

    rec = formulation["recommended_cleaned_formulation"]
    md = [
        "# v2 cleaned mass-bearing formulation — Phase 1 preflight",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "## Status",
        "",
        f"- `root_cause_status`: `{ROOT_CAUSE_STATUS}`",
        f"- `additional_eps`: `NOT_AUTHORIZED`",
        "",
        "## VM API (no solve)",
        "",
        f"- `vm_slepc_import_pass`: `{api.get('vm_slepc_import_pass')}`",
        f"- `vm_slepc_version`: `{api.get('vm_slepc_version')}`",
        f"- `jd_api_available`: `{api.get('jd_api_available')}`",
        f"- `gd_api_available`: `{api.get('gd_api_available')}`",
        f"- `ciss_api_available`: `{api.get('ciss_api_available')}`",
        "",
        "## Recommended cleaned formulation",
        "",
        f"`{rec}`",
        "",
        f"- `recommended_cleaned_formulation_status`: `{formulation.get('recommended_cleaned_formulation_status')}`",
        f"- `random_probe_informational_only`: `{random_probe_context.get('random_probe_informational_only')}`",
        f"- `seed_preservation_check_status`: `{seed.get('seed_preservation_check_status')}`",
        f"- First future solver: `{payload['future_solver_wiring_plan']['jd_gd']['recommended_first_future_solver']}`",
        "",
        "No eigensolve executed.",
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    write_json(OUT_JSON, payload)

    print(f"[cleaned_phase1] jd_api_available={api.get('jd_api_available')}", flush=True)
    print(f"[cleaned_phase1] gd_api_available={api.get('gd_api_available')}", flush=True)
    print(f"[cleaned_phase1] ciss_api_available={api.get('ciss_api_available')}", flush=True)
    print(
        "[cleaned_phase1] random_probe_informational_only="
        f"{random_probe_context.get('random_probe_informational_only')}",
        flush=True,
    )
    print(
        "[cleaned_phase1] prior_random_probe_false_negative_demonstrated="
        f"{random_probe_context.get('prior_random_probe_false_negative_demonstrated')}",
        flush=True,
    )
    print(f"[cleaned_phase1] recommended_cleaned_formulation={rec}", flush=True)
    print(
        "[cleaned_phase1] recommended_phase1_mapping_construction_method="
        f"{formulation.get('recommended_phase1_mapping_construction_method')}",
        flush=True,
    )
    print(
        "[cleaned_phase1] cleaned_formulation_mapping_constructed="
        f"{seed.get('cleaned_formulation_mapping_constructed')}",
        flush=True,
    )
    print(
        "[cleaned_phase1] seed_preservation_check_status="
        f"{seed.get('seed_preservation_check_status')}",
        flush=True,
    )
    print(
        "[cleaned_phase1] recommended_first_future_solver="
        f"{payload['future_solver_wiring_plan']['jd_gd']['recommended_first_future_solver']}",
        flush=True,
    )
    print("[cleaned_phase1] artifact_storage_policy_applied=True", flush=True)
    print(f"[cleaned_phase1] report_size_bytes={size_b}", flush=True)
    print("[cleaned_phase1] no_new_eigensolve_executed=True", flush=True)
    print("[cleaned_phase1] additional_eps=NOT_AUTHORIZED", flush=True)
    print(f"[cleaned_phase1] wrote {OUT_JSON} ({size_b} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
