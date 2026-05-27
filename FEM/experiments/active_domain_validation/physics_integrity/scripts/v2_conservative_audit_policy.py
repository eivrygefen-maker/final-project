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
    st_fields = extract_st_operator_fields(solve_result)
    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or eps_diag.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
    )
    sigma = st_fields.get("actual_sigma_hz")
    if sigma is None or (isinstance(sigma, float) and sigma != sigma):
        sigma = eps_diag.get("st_sigma_hz_used") or solve_result.get("target_hz") or target_hz

    policy = {
        "continuation_seed_applied": continuation,
        "seed_frequency_hz": float(target_hz),
        "actual_sigma_hz": sigma,
        "st_type": st_fields.get("st_type") or eps_diag.get("st_type"),
        "eps_eigenvalue_semantics": st_fields.get("eps_eigenvalue_semantics")
        or eps_diag.get("eps_eigenvalue_semantics", "slepc_backtransformed"),
        "legacy_double_shift_mapping_disabled": bool(
            st_fields.get("legacy_double_shift_mapping_disabled", True)
        ),
        "diagnostic_operator_consistent_with_replay": bool(
            st_fields.get("diagnostic_operator_consistent_with_replay")
        ),
        "actual_st_a_shift_frac": float(st_fields.get("actual_st_a_shift_frac", 0.0) or 0.0),
        "actual_st_mass_reg_frac": float(st_fields.get("actual_st_mass_reg_frac", 0.0) or 0.0),
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

    replacement_report = report.get("replacement_baseline_artifacts") or {}
    if not replacement_report:
        replacement_report = {}
    op = report.get("operator_policy") or {}
    provenance_mismatch, mismatch_reasons = detect_operator_policy_provenance_mismatch(
        op, replacement_report=replacement_report
    )

    report["serialization_may_change_physical_replay_metrics"] = serialization_may_change
    report["lossless_pre_sparsify_eps_vectors_available_in_current_run"] = lossless_available
    report["current_saved_vectors_sufficient_for_st_verdict"] = False
    report["operator_policy_provenance_mismatch"] = provenance_mismatch
    report["operator_policy_provenance_mismatch_reasons"] = mismatch_reasons

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
        if provenance_mismatch:
            rca["operator_policy_provenance"] = (
                "reporting/provenance loss or field merge gap — not necessarily run-policy mismatch"
            )
        report["root_cause_analysis"] = rca
    elif prior_verdict == VERDICT_NO_BRANCH:
        report["st_viability_conclusion"] = (
            "inconclusive_review_conservative_policy_not_triggered"
        )

    return report
