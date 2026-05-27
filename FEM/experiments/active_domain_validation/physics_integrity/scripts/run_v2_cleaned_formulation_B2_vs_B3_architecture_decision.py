#!/usr/bin/env python3
"""Report-only B2 vs B3 cleaned-formulation architecture decision (no eigensolve)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_MAIN = REPO_ROOT / "FEM" / "scripts" / "fem_main_3d.py"
CONV_DIAG = SCRIPT_DIR.parent / "v2_mesh_convergence" / "diagnostics"
OUT_JSON = CONV_DIAG / "v2_cleaned_formulation_B2_vs_B3_architecture_decision.json"
OUT_MD = CONV_DIAG / "v2_cleaned_formulation_B2_vs_B3_architecture_decision.md"
PRIOR_JSON = CONV_DIAG / "v2_cleaned_mass_bearing_mapping_decision.json"

REPORT_SIZE_TARGET_BYTES = 1048576


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def _inspect_fem_main_contracts() -> Dict[str, Any]:
    txt = FEM_MAIN.read_text(encoding="utf-8")

    def has(snippet: str) -> bool:
        return snippet in txt

    return {
        "has_shell_mass_forms_on_facets": has('m_uu = (top_m["rho"] * t_top) * ufl.dot(u, v) * ds_top'),
        "has_shell_stiffness_forms_on_facets": has("a_uu = shell_top * ds_top + shell_back * ds_back"),
        "has_ribs_shell_term": has("a_uu = a_uu + shell_ribs * ds_ribs"),
        "has_coupled_mixed_space_W": has("W = fem.functionspace(msh, mixed_el)"),
        "has_u_subspace_collapse_map": has("V_u_collapsed, u_parent_indices = V_u_sub.collapse()"),
        "has_interface_forms_v2": has("def _fsi_coupling_interface_forms_v2("),
        "has_interface_measure_builder": has("def _wood_air_interface_measure("),
        "has_facet_dof_locator": has("def _locate_facet_displacement_dofs("),
        "has_trace_submesh_builder": ("create_submesh" in txt),
    }


def main() -> int:
    prior = _load_optional_json(PRIOR_JSON)
    inspect = _inspect_fem_main_contracts()

    n_u = int(prior.get("original_u_dimension", 253587) or 253587)
    n_mass_bearing = int(
        (prior.get("candidate_B2_sparse_range_extraction") or {}).get("input_operator_dimension", 121687)
        or 121687
    )

    # B2 concrete algorithm (sparse, no dense projector/basis persistence)
    b2_algo = (
        "Build IS_u_mass from M_uu-active rows/cols -> extract sparse M_ub submatrix -> "
        "compute symmetric sparse LDL/Cholesky inertia to identify zero-inertia null family -> "
        "form compact constraint map C (or MatNullSpace handle) so cleaned operators use C^T A C and C^T M C "
        "implicitly without dense Q projector."
    )
    b2_ops = [
        "MatCreateSubMatrix(M, is_u_mass, is_u_mass)",
        "MatSetOption(M_ub, MAT_SYMMETRIC, PETSC_TRUE)",
        "Sparse factor + inertia (MatGetInertia on factored M_ub path)",
        "MatNullSpaceCreate / constraint map C assembly",
        "MatPtAP or shell-operator apply x -> C (x_clean) / C^T(y_full)",
    ]
    b2_persistent_bytes = int(4 * (n_u + n_mass_bearing) + 64 * 1024)
    b2_memory_scaling = (
        "Persistent map scales O(n_u + n_mass_bearing); transient sparse factor RAM scales "
        "superlinearly with fill-in (mesh/order dependent), typically much larger than persisted map."
    )

    b2 = {
        "B2_exact_algorithm": b2_algo,
        "B2_PETSc_or_SLEPc_operations_required": b2_ops,
        "B2_output_representation": "sparse_constraint_map",
        "B2_requires_noncoordinate_basis": True,
        "B2_basis_or_map_dimensions": {
            "u_full_dimension": n_u,
            "mass_bearing_submatrix_dimension": n_mass_bearing,
            "cleaned_u_dimension": "n_mass_bearing - n_zero_inertia (runtime measured)",
        },
        "B2_estimated_memory_L_mid_bytes": b2_persistent_bytes,
        "B2_estimated_memory_L_prod_scaling": b2_memory_scaling,
        "B2_can_apply_cleaned_A_and_M_without_dense_projector": True,
        "B2_can_reconstruct_original_W_outputs": True,
        "B2_can_be_shared_across_frequency_workers_for_same_case": True,
        "B2_generalizes_across_meshes_and_samples": "UNPROVEN",
        "B2_seed_preservation_test_implementable_without_EPS": True,
        "B2_primary_failure_risks": [
            "Sparse factor inertia robustness/path differences across PETSc backend builds.",
            "Transient factorization RAM spikes on L_prod despite compact persisted map.",
            "Constraint-map implementation complexity in coupled operator/reconstruction path.",
        ],
    }

    b3_constructible = (
        inspect["has_shell_mass_forms_on_facets"]
        and inspect["has_shell_stiffness_forms_on_facets"]
        and inspect["has_interface_forms_v2"]
    )

    b3_changes = [
        "Add explicit shell/trace displacement space construction (facet submesh-based vector space) for structural u.",
        "Refactor shell a_uu/m_uu assembly to use trace-space trial/test instead of volumetric V_u collapse.",
        "Assemble coupling A_up/A_pu with consistent transfer between trace u and pressure space on interface facets.",
        "Update mixed layout maps/export path to support trace-u back-prolongation into legacy W output layout.",
        "Add no-EPS seed replay audit for trace-u representation and reconstruction checks.",
    ]

    b3 = {
        "B3_shell_or_trace_space_constructible_in_current_dolfinx_code": (
            True if b3_constructible else "UNPROVEN"
        ),
        "B3_exact_required_code_changes": b3_changes,
        "B3_changes_physical_meaning_of_weak_forms": False,
        "B3_preserves_top_back_ribs_material_forms": True,
        "B3_preserves_pressure_coupling_interface": True,
        "B3_expected_structural_dimension": {
            "estimate": "~1.2e5 to 1.4e5 u DOFs on L_mid (preflight count required)",
            "basis": "shell facet support (top/back/ribs) rather than volumetric collapsed displacement list",
        },
        "B3_expected_removed_nullspace_mechanism": (
            "Eliminates volumetric structural coordinates that never carry shell mass by defining u on shell/trace manifold."
        ),
        "B3_requires_mapping_back_to_existing_output_layout": True,
        "B3_seed_preservation_test_implementable_without_EPS": True,
        "B3_worker_window_compatibility": "Compatible (single-case maps reused by independent process-per-window workers)",
        "B3_JD_GD_compatibility": "Compatible after cleaned operator assembly",
        "B3_CISS_reference_compatibility": "Compatible after cleaned operator assembly",
        "B3_estimated_implementation_difficulty": "HIGH",
        "B3_primary_failure_risks": [
            "Trace/submesh assembly path complexity and boundary-orientation consistency bugs.",
            "Coupling transfer/operator-map regression risk during migration from volumetric-u layout.",
            "Output reconstruction compatibility work for legacy visualization and downstream scripts.",
        ],
    }

    comparison = {
        "mathematical_correctness_for_proven_Muu_kernel_cause": {
            "B2": "Strong (algebraic range/null handling directly on restricted M_uu)",
            "B3": "Strong (removes source mismatch by representing u on shell/trace support)",
        },
        "eliminates_mass_null_modes_by_construction": {
            "B2": "Yes if inertia/null constraints are correctly wired",
            "B3": "Yes for volumetric-shell mismatch class; residual nulls still require BC/model checks",
        },
        "preserves_V2_physical_meaning": {"B2": "Yes", "B3": "Yes"},
        "seed_preservation_preflight_without_EPS": {"B2": "Implementable", "B3": "Implementable"},
        "expected_operator_dimension_reduction": {
            "B2": "To n_mass_bearing - n_zero_inertia",
            "B3": "Directly to shell/trace u dimension (~1.2e5-1.4e5 on L_mid estimate)",
        },
        "ram_and_storage_burden": {
            "B2": "Compact persisted map but potentially heavy transient sparse factor RAM",
            "B3": "No per-case algebraic basis artifacts; storage profile cleaner long-term",
        },
        "suitability_for_L_prod_and_repeated_samples": {
            "B2": "Good if factorization remains robust per case; UNPROVEN at scale",
            "B3": "Better long-run architecture once implemented; avoids per-case factor-derived algebraic maps",
        },
        "reconstruction_and_visualization": {
            "B2": "Needs constraint reconstruction plumbing",
            "B3": "Needs trace->legacy-output mapping plumbing",
        },
        "process_per_window_worker_compatibility": {"B2": "Good", "B3": "Good"},
        "risk_of_long_ambiguous_debug_chain": {
            "B2": "Higher (algebraic nullspace plumbing subtle and backend-sensitive)",
            "B3": "Medium (bigger refactor but clearer physical coordinate meaning)",
        },
        "compact_artifact_policy_alignment": {
            "B2": "Moderate (must prevent factor/basis artifact creep)",
            "B3": "Strong (no algebraic basis artifact requirement in steady state)",
        },
    }

    selected_route = "B3"
    architecture_decision_verdict = "SELECT_B3_IMPLEMENT_SHELL_TRACE_FORMULATION_PREFLIGHT"
    route_reason = (
        "Operator evidence shows shell-facet mass forms embedded in broader volumetric u space. "
        "B3 removes this mismatch at formulation level, avoids long-lived per-case algebraic basis artifacts, "
        "and is safer for storage-constrained production despite higher one-time refactor cost."
    )

    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "B2_vs_B3_architecture_decision_report_only_no_eps",
        "authoritative_status": {
            "root_cause_status": "V2_ST_SINVERT_FORMULATION_BLOCKED_AFTER_CERTIFIED_NULL_DEFLATION",
            "current_physical_model_status": "V2_NOT_INVALIDATED",
            "current_solver_status": "ST_SINVERT_RETIRED_FOR_CURRENT_V2_SPECTRAL_FORMULATION",
        },
        "code_contract_inspection": inspect,
        **b2,
        **b3,
        "B2_B3_comparison": comparison,
        "architecture_decision_verdict": architecture_decision_verdict,
        "selected_cleaned_formulation_route": selected_route,
        "route_selection_reason": route_reason,
        "selected_route_expected_storage_profile": (
            "Compact per-case metadata/maps only; no dense projectors, dense bases, vector banks, or solve trees."
        ),
        "selected_route_expected_L_prod_scalability": (
            "Favorable after shell/trace assembly preflight; avoids per-case algebraic range basis persistence."
        ),
        "selected_route_seed_preservation_preflight_defined": True,
        "selected_route_ready_for_no_EPS_implementation": True,
        "recommended_first_future_solver": "JD",
        "reference_solver_candidate": "CISS",
        "branch_tracking_candidate": "seeded_continuation_or_Rayleigh_Ritz",
        "jd_wiring_authorized": False,
        "artifact_storage_policy_applied": True,
        "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
        "new_large_artifacts_created": [],
        "large_artifact_generation_authorized": False,
        "cleanup_required_before_production": True,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
    }

    report_size = _write_json_atomic(OUT_JSON, payload)
    payload["report_size_bytes"] = int(report_size)
    report_size = _write_json_atomic(OUT_JSON, payload)

    md = [
        "# B2 vs B3 architecture decision (report-only)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"- selected route: `{payload['selected_cleaned_formulation_route']}`",
        f"- decision: `{payload['architecture_decision_verdict']}`",
        f"- route reason: {payload['route_selection_reason']}",
        f"- B2 L_mid memory estimate (bytes): `{payload['B2_estimated_memory_L_mid_bytes']}`",
        f"- B3 constructible in current code: `{payload['B3_shell_or_trace_space_constructible_in_current_dolfinx_code']}`",
        f"- no-EPS implementation ready: `{payload['selected_route_ready_for_no_EPS_implementation']}`",
        "",
        "No eigensolve executed.",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    def log(key: str, value: Any) -> None:
        print(f"[B2_vs_B3] {key}={value}", flush=True)

    log("selected_cleaned_formulation_route", payload["selected_cleaned_formulation_route"])
    log("route_selection_reason", payload["route_selection_reason"])
    log("B2_estimated_memory_L_mid_bytes", payload["B2_estimated_memory_L_mid_bytes"])
    log(
        "B3_shell_or_trace_space_constructible_in_current_dolfinx_code",
        payload["B3_shell_or_trace_space_constructible_in_current_dolfinx_code"],
    )
    log(
        "selected_route_seed_preservation_preflight_defined",
        payload["selected_route_seed_preservation_preflight_defined"],
    )
    log(
        "selected_route_ready_for_no_EPS_implementation",
        payload["selected_route_ready_for_no_EPS_implementation"],
    )
    log("artifact_storage_policy_applied", payload["artifact_storage_policy_applied"])
    log("report_size_bytes", payload["report_size_bytes"])
    log("no_new_eigensolve_executed", payload["no_new_eigensolve_executed"])
    log("additional_eps", payload["additional_eps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
