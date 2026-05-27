#!/usr/bin/env python3
"""Machine-checkable clean-lane preflight gate contract (report-only; no EPS)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

VERDICT_PERSISTED_CONTENT_UNRESOLVED = (
    "MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED"
)

ST_TYPE_PROVENANCE_INFERRED = "inferred_from_evaluation_rows_not_authoritative_prior_policy"
ST_TYPE_FUTURE_REQUIRED = "sinvert"
PRODUCTION_VECTOR_FIDELITY_EXPOSURE = "OPEN"


def apply_st_type_provenance_fields(
    operator_policy: Dict[str, Any],
    *,
    gap_fields: Optional[List[str]] = None,
    inferred_value: str = ST_TYPE_FUTURE_REQUIRED,
) -> Dict[str, Any]:
    """
    Distinguish prior authoritative persistence gap from evaluation-row inference.
    Does not represent inferred sinvert as closing the authoritative gap.
    """
    gap_fields = list(gap_fields or [])
    authoritative = operator_policy.get("st_type")
    persisted = authoritative not in (
        None,
        "",
        "missing",
        "missing_in_prior_artifacts",
    )
    if not persisted and "st_type" not in gap_fields:
        gap_fields.append("st_type")

    operator_policy["prior_authoritative_persisted_st_type"] = (
        authoritative if persisted else "missing"
    )
    operator_policy["st_type_value"] = inferred_value
    operator_policy["st_type_provenance"] = ST_TYPE_PROVENANCE_INFERRED
    operator_policy["st_type_persisted_in_prior_authoritative_artifacts"] = persisted
    operator_policy["future_adjudication_required_persisted_st_type"] = ST_TYPE_FUTURE_REQUIRED
    # Authoritative gap remains until future run persists st_type at eps_batch root.
    operator_policy["st_type"] = "missing_in_prior_artifacts"
    return operator_policy


def comparisons_authorized_differences_only(comparisons: List[Dict[str, Any]]) -> bool:
    for row in comparisons:
        if not row.get("match") and not row.get("authorized_evidence_only_difference"):
            return False
    return True


def build_clean_lane_gate_contract(
    *,
    pipeline_audit: Dict[str, Any],
    filter_classification: Dict[str, Any],
    lossless_self_test: Dict[str, Any],
    policy_equivalence_pass: bool,
    policy_equivalence_failures: List[str],
    mesh_file_exists: bool,
    comparisons: List[Dict[str, Any]],
    single_lossless_adjudication_run_ready: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Top-level gate fields for preflight JSON and stdout contract."""
    clf_summary = filter_classification.get("summary") or {}
    filter_complete = bool(clf_summary.get("filter_classification_complete", False))
    lossless_pass = bool(lossless_self_test.get("self_test_pass", False))

    failure_reasons: List[str] = []
    if not mesh_file_exists:
        failure_reasons.append("mesh_file_missing")
    if policy_equivalence_failures:
        failure_reasons.extend(
            [f"policy_equivalence:{f}" for f in policy_equivalence_failures]
        )
    if not lossless_pass:
        failure_reasons.append("lossless_self_test_failed")
    if not filter_complete:
        failure_reasons.append("filter_classification_incomplete")
    if not pipeline_audit:
        failure_reasons.append("pipeline_audit_missing")
    else:
        if pipeline_audit.get("audit_verdict") != VERDICT_PERSISTED_CONTENT_UNRESOLVED:
            failure_reasons.append("unexpected_authoritative_verdict")
        if pipeline_audit.get("operator_policy_provenance_mismatch") is True:
            failure_reasons.append("operator_policy_provenance_mismatch")

    auth_diff_only = comparisons_authorized_differences_only(comparisons)
    if not auth_diff_only:
        failure_reasons.append("unauthorized_policy_difference")

    pipe = pipeline_audit or {}
    gate_ready = (
        policy_equivalence_pass
        and lossless_pass
        and filter_complete
        and mesh_file_exists
        and bool(pipe)
        and not failure_reasons
    )
    if not gate_ready and single_lossless_adjudication_run_ready:
        failure_reasons.append("preflight_subchecks_incomplete")
    gap_fields = list(pipe.get("operator_policy_provenance_gap_fields") or ["st_type"])

    contract: Dict[str, Any] = {
        "authoritative_current_verdict": pipe.get(
            "audit_verdict", VERDICT_PERSISTED_CONTENT_UNRESOLVED
        ),
        "operator_policy_provenance_mismatch": bool(
            pipe.get("operator_policy_provenance_mismatch", False)
        ),
        "operator_policy_provenance_gap": bool(
            pipe.get("operator_policy_provenance_gap", True)
        ),
        "operator_policy_provenance_gap_fields": gap_fields,
        "serialization_fidelity_risk": bool(
            pipe.get("serialization_fidelity_risk", True)
        ),
        "production_vector_fidelity_exposure": PRODUCTION_VECTOR_FIDELITY_EXPOSURE,
        "lossless_self_test_pass": lossless_pass,
        "policy_equivalence_pass": bool(policy_equivalence_pass),
        "filter_classification_complete": filter_complete,
        "single_lossless_adjudication_run_ready": bool(gate_ready),
        "single_lossless_adjudication_run_authorized": False,
        "no_new_eigensolve_executed": True,
        "authorized_differences_only": auth_diff_only,
        "failure_reasons": failure_reasons,
    }
    return contract, failure_reasons


def validate_gate_contract_for_eps_authorization(
    gate: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """All top-level gate fields must match expected values before EPS may run."""
    expected = {
        "authoritative_current_verdict": VERDICT_PERSISTED_CONTENT_UNRESOLVED,
        "operator_policy_provenance_mismatch": False,
        "operator_policy_provenance_gap": True,
        "operator_policy_provenance_gap_fields": ["st_type"],
        "serialization_fidelity_risk": True,
        "production_vector_fidelity_exposure": PRODUCTION_VECTOR_FIDELITY_EXPOSURE,
        "lossless_self_test_pass": True,
        "policy_equivalence_pass": True,
        "filter_classification_complete": True,
        "single_lossless_adjudication_run_ready": True,
        "single_lossless_adjudication_run_authorized": False,
        "no_new_eigensolve_executed": True,
        "authorized_differences_only": True,
        "failure_reasons": [],
    }
    issues: List[str] = []
    for key, exp in expected.items():
        got = gate.get(key)
        if got != exp:
            issues.append(f"{key}: got {got!r}, expected {exp!r}")
    return len(issues) == 0, issues


def print_clean_lane_preflight_lines(gate: Dict[str, Any]) -> None:
    """Stdout contract for VM report-only bundle."""
    for key in (
        "authoritative_current_verdict",
        "operator_policy_provenance_mismatch",
        "operator_policy_provenance_gap",
        "operator_policy_provenance_gap_fields",
        "serialization_fidelity_risk",
        "production_vector_fidelity_exposure",
        "lossless_self_test_pass",
        "policy_equivalence_pass",
        "filter_classification_complete",
        "authorized_differences_only",
        "single_lossless_adjudication_run_ready",
        "single_lossless_adjudication_run_authorized",
        "no_new_eigensolve_executed",
    ):
        print(f"[clean_lane_preflight] {key}={gate.get(key)!r}", flush=True)
    reasons = gate.get("failure_reasons")
    if reasons:
        print(f"[clean_lane_preflight] failure_reasons={reasons!r}", flush=True)
