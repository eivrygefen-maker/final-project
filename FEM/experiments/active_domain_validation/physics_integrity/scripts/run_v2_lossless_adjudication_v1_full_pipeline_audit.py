#!/usr/bin/env python3
"""Report-only audit for lossless adjudication v1 isolated tree (no new EPS)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
from v2_clean_lane_preflight_gate import apply_st_type_provenance_fields
from v2_conservative_audit_policy import (
    VERDICT_PERSISTED_CONTENT_UNRESOLVED,
    build_operator_policy_from_artifacts,
)
from v2_mesh_convergence_common import CONV_DIAG, solve_case_dir, write_json

CASE_ID = "baseline_coupled_v2"
OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_full_pipeline_audit.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_full_pipeline_audit.md"
DIAG_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json"


def main() -> int:
    if not DIAG_JSON.is_file():
        print("[lossless_audit] diagnostic JSON missing", file=sys.stderr)
        return 2

    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    ev = diag.get("evaluation") or {}
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    bank_path = out_dir / "diagnostics" / "eps_candidate_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.is_file() else {}
    results = sorted((out_dir / "results").glob("result_*.json"))
    solve_result = json.loads(results[-1].read_text(encoding="utf-8")) if results else {}

    policy, _ = build_operator_policy_from_artifacts(solve_result, bank, target_hz=243.0754171175576)
    eps_st = (solve_result.get("eps_batch_diagnostics") or {}).get("st_type")
    if eps_st:
        policy["st_type"] = eps_st
        policy["st_type_authoritative_persisted"] = True
        policy["st_type_persisted_in_prior_authoritative_artifacts"] = True
        policy["st_type_provenance"] = "authoritative_eps_batch_diagnostics"
        policy["prior_authoritative_persisted_st_type"] = str(eps_st)
    else:
        apply_st_type_provenance_fields(policy, gap_fields=[])

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "lossless_adjudication_v1_post_eps_audit",
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "preflight_gate_pass": bool(diag.get("preflight_gate_pass")),
        "single_lossless_adjudication_run_authorized": bool(
            diag.get("single_lossless_adjudication_run_authorized")
        ),
        "eps_run_count_for_this_lane": int(diag.get("eps_run_count_for_this_lane", 0)),
        "no_additional_eps_run_authorized": True,
        "prior_baseline_verdict": VERDICT_PERSISTED_CONTENT_UNRESOLVED,
        "lossless_adjudication_verdict": ev.get("diagnostic_verdict"),
        "nconv_marked": ev.get("eps_nconv_marked"),
        "lossless_candidate_count": ev.get("lossless_candidate_count"),
        "lossless_vectors_saved": ev.get("lossless_vectors_saved"),
        "lossless_roundtrip_failures": ev.get("lossless_roundtrip_failures", 0),
        "legacy_sparse_comparison_saved": ev.get("legacy_sparse_comparison_saved"),
        "st_type_authoritative_provenance": ev.get("st_type_authoritative_provenance"),
        "operator_policy": policy,
        "lossless_pre_sparsify_eps_vectors_available_in_current_run": True,
        "current_saved_vectors_sufficient_for_st_verdict": ev.get("diagnostic_verdict")
        in (
            "MAPPING_FIXED_UNREGULARIZED_BASELINE_BRANCH_RECOVERED",
            "MAPPING_FIXED_UNREGULARIZED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED",
        ),
        "serialization_may_change_physical_replay_metrics": True,
        "serialization_fidelity_risk": True,
        "evaluation_summary": ev.get("summary"),
        "any_branch_recovery_pass": ev.get("any_branch_recovery_pass"),
        "counterfactual_filters_reported": ev.get("counterfactual_filters_reported"),
        "not_evidence_for_production_promotion": True,
        "mesh_convergence_may_resume": False,
    }
    write_json(OUT_JSON, report)
    OUT_MD.write_text(
        "\n".join(
            [
                "# Lossless adjudication v1 full pipeline audit",
                "",
                f"Generated: {report['generated_utc']}",
                f"**lossless_adjudication_verdict:** `{report['lossless_adjudication_verdict']}`",
                f"**lossless_vectors_saved:** `{report['lossless_vectors_saved']}`",
                f"**st_type (authoritative):** `{policy.get('st_type')}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[lossless_audit] verdict={report['lossless_adjudication_verdict']} "
        f"lossless_saved={report['lossless_vectors_saved']} wrote {OUT_JSON}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
