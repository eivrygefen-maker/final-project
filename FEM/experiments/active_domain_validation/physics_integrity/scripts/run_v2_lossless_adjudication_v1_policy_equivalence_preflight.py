#!/usr/bin/env python3
"""No-EPS preflight: planned clean adjudication lane vs approved replacement provenance."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from v2_mesh_convergence_common import CONV_DIAG, mesh_path, solve_case_dir

OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_policy_equivalence_preflight.md"
PROV_JSON = CONV_DIAG / "v2_mapping_fixed_replacement_runtime_provenance_inventory.json"
LOSSLESS_TEST_JSON = CONV_DIAG / "v2_lossless_candidate_persistence_self_test.json"
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


def main() -> int:
    prov = _load(PROV_JSON)
    lossless_test = _load(LOSSLESS_TEST_JSON)
    approved = approved_replacement_policy_from_provenance(prov)
    planned = planned_clean_lane_policy()

    comparisons: List[Dict[str, Any]] = []
    comparisons.append(
        _compare_field("case_id", CASE_ID, planned["case_id"])
    )
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
            authorized_difference=False,
        )
    )
    comparisons.append(
        _compare_field(
            "sigma_used_hz",
            approved.get("sigma_used_hz"),
            planned["actual_sigma_hz_target"],
            authorized_difference=False,
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
        _compare_field(
            "lossless_save_enabled",
            False,
            True,
            authorized_difference=True,
        )
    )
    comparisons.append(
        _compare_field(
            "lossless_replay_authoritative",
            False,
            True,
            authorized_difference=True,
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
    failures = [c for c in comparisons if c.get("fail_if_mismatch")]
    policy_equivalence_pass = len(failures) == 0 and mesh_ok
    lossless_self_test_pass = bool(lossless_test.get("self_test_pass"))
    st_type_gap = approved.get("st_type") in (None, "missing", "missing_in_prior_artifacts")
    st_type_inferred = "sinvert"  # from runtime_policy_field_occurrences / evaluation rows

    single_run_ready = policy_equivalence_pass and lossless_self_test_pass

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "report_only_no_eps",
        "approved_replacement_provenance": approved,
        "planned_clean_lane_policy": planned,
        "comparisons": comparisons,
        "authorized_evidence_layer_differences_only": AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES,
        "mesh_file": str(mesh_file),
        "mesh_file_exists": mesh_ok,
        "policy_equivalence_pass": policy_equivalence_pass,
        "policy_equivalence_failures": [f["field"] for f in failures],
        "lossless_self_test_pass": lossless_self_test_pass,
        "st_type_missing_in_prior_artifacts": st_type_gap,
        "st_type_inferred_from_runtime_rows": st_type_inferred,
        "st_type_persistence_required_in_future_run": True,
        "st_type_prior_artifact_gap_only_at_eps_batch_root": True,
        "single_lossless_adjudication_run_ready": single_run_ready,
        "single_lossless_adjudication_run_authorized": False,
        "authorization_note": "Preflight only; human review required before EPS.",
        "no_eigensolve_executed": True,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Lossless adjudication v1 policy equivalence preflight",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**policy_equivalence_pass:** `{policy_equivalence_pass}`",
        f"**single_lossless_adjudication_run_ready:** `{single_run_ready}` (not authorized here)",
        "",
        "## Authorized differences only",
        "",
    ]
    for d in AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES:
        lines.append(f"- {d}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[policy_equivalence_preflight] pass={policy_equivalence_pass} "
        f"ready={single_run_ready} wrote {OUT_JSON}",
        flush=True,
    )
    return 0 if policy_equivalence_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
