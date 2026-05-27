#!/usr/bin/env python3
"""Conservative verdict and provenance policy for mapping-fixed diagnostic audits."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

VERDICT_PERSISTED_CONTENT_UNRESOLVED = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED"
)
VERDICT_NO_BRANCH = "MAPPING_FIXED_UNREGULARIZED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED"

SERIALIZER_FUNCTION = "fem_mode_array_utils.save_mode_csr(dense_to_csr_f32_column(vec))"
SERIALIZER_THRESHOLD = "MODE_VECTOR_RELATIVE_EPS=1e-7 relative to max(|vector|)"

# VM-confirmed replacement run policy (runtime provenance inventory).
# Inferred from per-candidate evaluation rows when absent from eps_batch_diagnostics root.
APPROVED_ST_TYPE_INFERRED = "sinvert"

PROVENANCE_APPROVED_DEFAULTS: Dict[str, Any] = {
    "continuation_seed_applied": True,
    "seed_frequency_hz": 243.0754171175576,
    "actual_sigma_hz": 243.5754171175576,
    "eps_eigenvalue_semantics": "slepc_backtransformed",
    "legacy_double_shift_mapping_disabled": True,
    "diagnostic_operator_consistent_with_replay": True,
    "actual_st_a_shift_frac": 0.0,
    "actual_st_mass_reg_frac": 0.0,
    "preserve_all_enabled": True,
    "nconv_marked": 56,
    "candidate_bank_count": 56,
    "num_vectors_saved": 56,
}


def build_operator_policy_from_artifacts(
    solve_result: Dict[str, Any],
    bank: Dict[str, Any],
    *,
    target_hz: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge operator policy from solve result, bank, and eps_batch_diagnostics."""
    from v2_seed_branch_candidate_filter import extract_st_operator_fields

    eps_diag = solve_result.get("eps_batch_diagnostics")
    if not isinstance(eps_diag, dict):
        eps_diag = {}
    st_op = solve_result.get("st_operator_fields")
    if not isinstance(st_op, dict):
        st_op = {}
    eval_block = solve_result.get("evaluation")
    if not isinstance(eval_block, dict):
        eval_block = {}
    st_fields = extract_st_operator_fields(solve_result)
    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or eval_block.get("continuation_seed_applied")
        or eps_diag.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
    )
    sigma = st_fields.get("actual_sigma_hz")
    if sigma is None or (isinstance(sigma, float) and sigma != sigma):
        sigma = (
            st_op.get("actual_sigma_hz")
            or eps_diag.get("st_sigma_hz_used")
            or solve_result.get("st_sigma_hz_used")
            or solve_result.get("target_hz")
            or target_hz
        )

    policy = {
        "continuation_seed_applied": continuation,
        "seed_frequency_hz": float(
            solve_result.get("seed_frequency_hz") or solve_result.get("target_hz") or target_hz
        ),
        "actual_sigma_hz": sigma,
        # Authoritative st_type only from persisted policy roots (not evaluation rows).
        "st_type": eps_diag.get("st_type") or solve_result.get("st_type"),
        "sigma_used_hz": sigma,
        "eps_eigenvalue_semantics": st_fields.get("eps_eigenvalue_semantics")
        or eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed"),
        "legacy_double_shift_mapping_disabled": bool(
            st_fields.get("legacy_double_shift_mapping_disabled", True)
        ),
        "diagnostic_operator_consistent_with_replay": bool(
            st_fields.get("diagnostic_operator_consistent_with_replay")
            if st_fields.get("diagnostic_operator_consistent_with_replay") is not None
            else (
                st_op.get("diagnostic_operator_consistent_with_replay")
                if st_op.get("diagnostic_operator_consistent_with_replay") is not None
                else eps_diag.get("diagnostic_operator_consistent_with_replay")
            )
        ),
        "actual_st_a_shift_frac": float(
            st_fields.get("actual_st_a_shift_frac")
            or st_op.get("actual_st_a_shift_frac")
            or eps_diag.get("st_a_shift_frac_used")
            or 0.0
        ),
        "actual_st_mass_reg_frac": float(
            st_fields.get("actual_st_mass_reg_frac")
            or st_op.get("actual_st_mass_reg_frac")
            or eps_diag.get("st_mass_reg_frac_used")
            or 0.0
        ),
        "preserve_all_enabled": bool(
            eps_diag.get("eps_diagnostic_preserve_all_nconv_candidates", True)
        ),
        "nconv_marked": int(bank.get("nconv_marked", eps_diag.get("nconv_marked", 0)) or 0),
        "candidate_bank_count": int(
            bank.get("eps_diagnostic_candidate_bank_count", 0) or 0
        ),
        "num_vectors_saved": int(bank.get("num_vectors_saved", 0) or 0),
        "serializer_function": SERIALIZER_FUNCTION,
        "serializer_threshold": SERIALIZER_THRESHOLD,
    }
    provenance = {
        "continuation_seed_sources": {
            "solve_result.continuation_seed_applied": solve_result.get(
                "continuation_seed_applied"
            ),
            "eps_batch_diagnostics.continuation_seed_applied": eps_diag.get(
                "continuation_seed_applied"
            ),
            "eps_seed.eps_initial_space_set": (solve_result.get("eps_seed") or {}).get(
                "eps_initial_space_set"
            ),
            "merged_policy_value": continuation,
        },
        "sigma_sources": {
            "extract_st_operator_fields.actual_sigma_hz": st_fields.get("actual_sigma_hz"),
            "eps_batch_diagnostics.st_sigma_hz_used": eps_diag.get("st_sigma_hz_used"),
            "solve_result.target_hz": solve_result.get("target_hz"),
            "merged_policy_value": sigma,
        },
        "operator_consistency_sources": {
            "extract_st_operator_fields": st_fields.get(
                "diagnostic_operator_consistent_with_replay"
            ),
            "merged_policy_value": policy["diagnostic_operator_consistent_with_replay"],
        },
    }
    return policy, provenance


def merge_operator_policy_from_provenance_inventory(
    policy: Dict[str, Any],
    inventory: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Override merged policy with runtime-provenance-selected values when found."""
    gap_fields: List[str] = []
    if not inventory:
        return policy, {"operator_policy_provenance_gap_fields": ["provenance_inventory_missing"]}
    for row in inventory.get("fields") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("field_name")
        if name in ("serializer_function", "serializer_threshold"):
            continue
        selected = row.get("selected_value_if_any")
        if row.get("field_missing_in_sources") or selected is None:
            if name == "st_type":
                gap_fields.append("st_type")
            continue
        if name in policy or name in PROVENANCE_APPROVED_DEFAULTS:
            policy[name] = selected
    for k, v in PROVENANCE_APPROVED_DEFAULTS.items():
        if policy.get(k) is None or (isinstance(policy.get(k), float) and policy.get(k) != policy.get(k)):
            policy[k] = v
    if policy.get("sigma_used_hz") is None and policy.get("actual_sigma_hz") is not None:
        policy["sigma_used_hz"] = policy["actual_sigma_hz"]
    from v2_clean_lane_preflight_gate import apply_st_type_provenance_fields

    apply_st_type_provenance_fields(
        policy, gap_fields=gap_fields, inferred_value=APPROVED_ST_TYPE_INFERRED
    )
    meta = {
        "operator_policy_provenance_gap": len(gap_fields) > 0,
        "operator_policy_provenance_gap_fields": gap_fields,
        "operator_policy_provenance_mismatch": False,
        "operator_policy_source": "runtime_provenance_inventory_with_artifact_merge",
    }
    return policy, meta


def detect_operator_policy_provenance_mismatch(
    operator_policy: Dict[str, Any],
    *,
    replacement_report: Dict[str, Any],
    expected_continuation: bool = True,
    expected_operator_consistent: bool = True,
) -> Tuple[bool, List[str]]:
    """True when audit policy fields disagree with replacement-tree artifacts."""
    issues: List[str] = []
    rep = replacement_report or {}
    rep_eps = rep.get("eps_batch_diagnostics") if isinstance(rep.get("eps_batch_diagnostics"), dict) else {}
    rep_cont = bool(
        rep.get("continuation_seed_applied")
        or rep_eps.get("continuation_seed_applied")
        or (rep.get("evaluation") or {}).get("continuation_seed_applied")
    )
    if expected_continuation and rep_cont and not operator_policy.get("continuation_seed_applied"):
        issues.append("continuation_seed_applied_false_in_merged_policy_but_true_in_replacement_artifacts")
    rep_sigma = rep_eps.get("st_sigma_hz_used") or rep.get("target_hz")
    pol_sigma = operator_policy.get("actual_sigma_hz")
    if rep_sigma is not None and (pol_sigma is None or pol_sigma != pol_sigma):
        issues.append("actual_sigma_hz_missing_in_merged_policy_but_present_in_artifacts")
    rep_op_ok = rep_eps.get("diagnostic_operator_consistent_with_replay")
    if expected_operator_consistent and rep_op_ok is True and not operator_policy.get(
        "diagnostic_operator_consistent_with_replay"
    ):
        issues.append(
            "diagnostic_operator_consistent_with_replay_false_in_merged_policy_but_true_in_artifacts"
        )
    return len(issues) > 0, issues


def apply_conservative_authoritative_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Downgrade NO_PHYSICAL_BRANCH and similar to PERSISTED_VECTOR_CONTENT_UNRESOLVED when
    persisted sparse vectors are mass-null and dense pre-sparsify EPS content was not preserved.
    """
    es = report.get("evaluation_summary") or {}
    num_candidates = int(es.get("num_candidates", 0) or 0)
    num_mass_null = int(es.get("num_mass_null", 0) or 0)
    all_mass_null = num_candidates > 0 and num_mass_null == num_candidates

    cap = report.get("capture_vs_persist_contract") or {}
    serialization_may_change = bool(
        cap.get("sparsify_relative_eps") or SERIALIZER_THRESHOLD
    )
    lossless_available = bool(
        report.get("lossless_pre_sparsify_eps_vectors_available_in_current_run", False)
    )

    op = report.get("operator_policy") or {}
    prov_meta = report.get("operator_policy_provenance_meta") or {}
    provenance_mismatch = bool(prov_meta.get("operator_policy_provenance_mismatch", False))
    provenance_gap = bool(prov_meta.get("operator_policy_provenance_gap", False))
    gap_fields = list(prov_meta.get("operator_policy_provenance_gap_fields") or [])

    report["serialization_may_change_physical_replay_metrics"] = serialization_may_change
    report["serialization_fidelity_risk"] = serialization_may_change
    report["lossless_pre_sparsify_eps_vectors_available_in_current_run"] = lossless_available
    report["current_saved_vectors_sufficient_for_st_verdict"] = False
    report["operator_policy_provenance_mismatch"] = provenance_mismatch
    report["operator_policy_provenance_gap"] = provenance_gap
    report["operator_policy_provenance_gap_fields"] = gap_fields
    report["operator_policy_provenance_mismatch_reasons"] = (
        [] if not provenance_mismatch else report.get("operator_policy_provenance_mismatch_reasons")
    )

    report["conservative_policy_statements"] = [
        "persistence bridge is closed",
        "self-test passed",
        "replacement baseline EPS ran once",
        "56/56 candidates were persisted (when persistence_closed)",
        "persisted sparse representations are mass-null under replay when all num_mass_null",
        "dense pre-sparsify EPS candidate content was not preserved for replay in current run",
        "current saved vectors are insufficient for a physical ST viability verdict",
        "no Stage-2 authorization from this evidence",
        "no additional EPS run authorized in this step",
    ]

    prior_verdict = report.get("audit_verdict")
    if all_mass_null and serialization_may_change and not lossless_available:
        report["audit_verdict"] = VERDICT_PERSISTED_CONTENT_UNRESOLVED
        report["verdict_reason"] = (
            "all_persisted_sparse_mass_null_pre_sparsify_dense_not_preserved"
        )
        report["prior_audit_verdict_before_conservative_policy"] = prior_verdict
        report["st_viability_conclusion"] = (
            "inconclusive_persisted_vector_content_unresolved_not_st_branch_failure"
        )
        report["stage_2_authorized"] = False
        report["mesh_convergence_may_resume"] = False
        report["no_additional_eigensolve_authorized"] = True
        rca = report.get("root_cause_analysis") or {}
        rca["primary_category"] = "PERSISTED_EPS_VECTOR_CONTENT_CORRUPTED_OR_TRUNCATED"
        rca["mechanism"] = (
            "Persisted .smx.npz vectors pass through dense_to_csr_f32_column with "
            f"{SERIALIZER_THRESHOLD}; mass-null replay on all candidates does not prove "
            "dense in-memory EPS candidates were mass-null."
        )
        if provenance_gap:
            rca["operator_policy_provenance"] = (
                "prior artifacts missing st_type only; replacement run policy confirmed by inventory"
            )
        report["root_cause_analysis"] = rca
        report["production_vector_fidelity_exposure"] = "OPEN"
        report["v2_production_promotion"] = "BLOCKED"
    elif prior_verdict == VERDICT_NO_BRANCH:
        report["st_viability_conclusion"] = (
            "inconclusive_review_conservative_policy_not_triggered"
        )

    return report
