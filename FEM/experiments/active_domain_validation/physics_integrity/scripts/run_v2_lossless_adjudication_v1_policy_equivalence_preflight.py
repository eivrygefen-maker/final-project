#!/usr/bin/env python3
"""No-EPS preflight: policy equivalence + machine-checkable top-level gate contract."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import (
    APPROVED_SIGMA_HZ,
    AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES,
    CASE_ID,
    OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
    SEED_F_HZ,
    approved_replacement_policy_from_provenance,
    planned_clean_lane_policy,
)
from v2_clean_lane_preflight_gate import build_clean_lane_gate_contract
from v2_mesh_convergence_common import CONV_DIAG, mesh_path

OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.md"
PROV_JSON = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.json"
LOSSLESS_TEST_JSON = CONV_DIAG / "v2_lossless_candidate_persistence_self_test.json"
PIPELINE_JSON = CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
CLASS_JSON = CONV_DIAG / "v2_clean_adjudication_filter_and_policy_classification.json"
BUNDLE_DIR = CONV_DIAG / "cursor_runtime_evidence_bundle_lossless_adjudication_preflight"


def _load(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    bundle_alt = BUNDLE_DIR / path.name
    if bundle_alt.is_file():
        return json.loads(bundle_alt.read_text(encoding="utf-8"))
    return {}


def _compare_field(
    name: str,
    approved: Any,
    planned: Any,
    *,
    authorized_difference: bool = False,
) -> Dict[str, Any]:
    match = approved == planned or (
        approved is not None
        and planned is not None
        and str(approved) == str(planned)
    )
    return {
        "field": name,
        "approved_replacement_value": approved,
        "planned_clean_lane_value": planned,
        "match": bool(match),
        "authorized_evidence_only_difference": authorized_difference,
        "fail_if_mismatch": not match and not authorized_difference,
    }


def _md_gate_table(gate: Dict[str, Any]) -> List[str]:
    lines = ["## Top-level gate contract", ""]
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
        "single_lossless_adjudication_run_ready",
        "single_lossless_adjudication_run_authorized",
        "no_new_eigensolve_executed",
        "authorized_differences_only",
        "failure_reasons",
    ):
        lines.append(f"- **{key}:** `{gate.get(key)!r}`")
    return lines


def main() -> int:
    prov = _load(PROV_JSON)
    lossless_test = _load(LOSSLESS_TEST_JSON)
    pipeline = _load(PIPELINE_JSON)
    classification = _load(CLASS_JSON)
    approved = approved_replacement_policy_from_provenance(prov)
    planned = planned_clean_lane_policy()

    comparisons: List[Dict[str, Any]] = []
    comparisons.append(_compare_field("case_id", CASE_ID, planned["case_id"]))
    comparisons.append(
        _compare_field(
            "seed_frequency_hz",
            approved.get("seed_frequency_hz", SEED_F_HZ),
            planned["seed_frequency_hz"],
        )
    )
    comparisons.append(
        _compare_field(
            "continuation_seed_applied",
            approved.get("continuation_seed_applied"),
            planned["continuation_seed_applied"],
        )
    )
    comparisons.append(
        _compare_field(
            "actual_sigma_hz",
            approved.get("actual_sigma_hz", approved.get("sigma_used_hz")),
            planned["actual_sigma_hz_target"],
        )
    )
    comparisons.append(
        _compare_field(
            "sigma_used_hz",
            approved.get("sigma_used_hz"),
            planned["actual_sigma_hz_target"],
        )
    )
    comparisons.append(
        _compare_field(
            "actual_st_a_shift_frac",
            approved.get("actual_st_a_shift_frac"),
            planned["actual_st_a_shift_frac_target"],
        )
    )
    comparisons.append(
        _compare_field(
            "actual_st_mass_reg_frac",
            approved.get("actual_st_mass_reg_frac"),
            planned["actual_st_mass_reg_frac_target"],
        )
    )
    comparisons.append(
        _compare_field(
            "eps_eigenvalue_semantics",
            approved.get("eps_eigenvalue_semantics"),
            planned["eps_eigenvalue_semantics"],
        )
    )
    comparisons.append(
        _compare_field(
            "legacy_double_shift_mapping_disabled",
            approved.get("legacy_double_shift_mapping_disabled"),
            planned["legacy_double_shift_mapping_disabled"],
        )
    )
    comparisons.append(
        _compare_field(
            "diagnostic_operator_consistent_with_replay",
            approved.get("diagnostic_operator_consistent_with_replay"),
            planned["diagnostic_operator_consistent_with_replay"],
        )
    )
    comparisons.append(
        _compare_field(
            "preserve_all_enabled",
            approved.get("preserve_all_enabled"),
            planned["preserve_all_enabled"],
        )
    )
    comparisons.append(
        _compare_field(
            "output_subdir",
            "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed",
            OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
            authorized_difference=True,
        )
    )
    comparisons.append(
        _compare_field("lossless_save_enabled", False, True, authorized_difference=True)
    )
    comparisons.append(
        _compare_field(
            "lossless_replay_authoritative", False, True, authorized_difference=True
        )
    )
    comparisons.append(
        _compare_field(
            "pre_replay_candidate_filtering_at_capture",
            "implicit_sparse_only",
            False,
            authorized_difference=True,
        )
    )

    mesh_file = mesh_path("L_mid", CASE_ID)
    mesh_ok = mesh_file.is_file()
    failures = [c["field"] for c in comparisons if c.get("fail_if_mismatch")]
    policy_equivalence_pass = len(failures) == 0 and mesh_ok
    lossless_self_test_pass = bool(lossless_test.get("self_test_pass"))
    preflight_ready = policy_equivalence_pass and lossless_self_test_pass

    gate, _ = build_clean_lane_gate_contract(
        pipeline_audit=pipeline,
        filter_classification=classification,
        lossless_self_test=lossless_test,
        policy_equivalence_pass=policy_equivalence_pass,
        policy_equivalence_failures=failures,
        mesh_file_exists=mesh_ok,
        comparisons=comparisons,
        single_lossless_adjudication_run_ready=preflight_ready,
    )

    op_pol = pipeline.get("operator_policy") or {}
    st_type_semantics = {
        "prior_authoritative_persisted_st_type": op_pol.get(
            "prior_authoritative_persisted_st_type", "missing"
        ),
        "st_type_value": op_pol.get("st_type_value", "sinvert"),
        "st_type_provenance": op_pol.get("st_type_provenance"),
        "st_type_persisted_in_prior_authoritative_artifacts": op_pol.get(
            "st_type_persisted_in_prior_authoritative_artifacts", False
        ),
        "future_adjudication_required_persisted_st_type": op_pol.get(
            "future_adjudication_required_persisted_st_type", "sinvert"
        ),
    }

    report: Dict[str, Any] = {
        **gate,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "report_only_no_eps",
        "st_type_semantics": st_type_semantics,
        "approved_replacement_provenance": approved,
        "planned_clean_lane_policy": planned,
        "comparisons": comparisons,
        "authorized_evidence_layer_differences_only": AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES,
        "mesh_file": str(mesh_file),
        "mesh_file_exists": mesh_ok,
        "policy_equivalence_failures": failures,
        "authorization_note": "Preflight only; human review required before EPS.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Lossless adjudication v1 policy equivalence preflight",
        "",
        f"Generated: {report['generated_utc']}",
        "",
    ]
    lines.extend(_md_gate_table(gate))
    lines.extend(
        [
            "",
            "## st_type semantics (gap not closed by inference)",
            "",
            f"- prior authoritative persisted: `{st_type_semantics['prior_authoritative_persisted_st_type']}`",
            f"- inferred from evaluation rows: `{st_type_semantics['st_type_value']}`",
            f"- provenance: `{st_type_semantics['st_type_provenance']}`",
            f"- persisted in prior authoritative artifacts: `{st_type_semantics['st_type_persisted_in_prior_authoritative_artifacts']}`",
            f"- future run must persist: `{st_type_semantics['future_adjudication_required_persisted_st_type']}`",
            "",
            "## Authorized differences only",
            "",
        ]
    )
    for d in AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES:
        lines.append(f"- {d}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[policy_equivalence_preflight] gate_ready={gate['single_lossless_adjudication_run_ready']} "
        f"wrote {OUT_JSON}",
        flush=True,
    )
    return 0 if gate.get("failure_reasons") == [] and policy_equivalence_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
