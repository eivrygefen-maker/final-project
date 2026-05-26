#!/usr/bin/env python3
"""
Combined static code audit + optional VM filtered-evaluation merge for forward risk and closure.

Static sections are derived from local source only. Does not run eigensolves.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import CONV_DIAG, write_json
from v2_solver_root_cause_static_audit import (
    ROOT_CAUSE_CONFIRMED_ST,
    build_evidence_summary,
    build_finite_closure_plan,
    build_forward_risk_register,
    build_static_code_audit,
    build_st_retry_control_flow_audit,
    determine_root_cause_status,
)

OUT_JSON = CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.json"
OUT_MD = CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.md"

FILTERED_EVAL_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_filtered_evaluation.json"
UNREG_OFFSET_REPORT_JSON = (
    CONV_DIAG / "v2_l_mid_seed_branch_unregularized_offset_diagnostic.json"
)
MASS_NORM_AUDIT_JSON = CONV_DIAG / "v2_l_mid_unregularized_saved_vector_mass_norm_audit.json"
MAPPING_INV_JSON = CONV_DIAG / "v2_eps_mapping_impact_inventory.json"
REPLAY_RECERT_JSON = CONV_DIAG / "v2_existing_pass_replay_recertification.json"
ST_PREFLIGHT_JSON = CONV_DIAG / "v2_st_singular_mass_preflight.json"
REHAB_PLAN_JSON = CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.json"
MAPPING_FIXED_DIAG_JSON = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.json"
)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evidence_summary") or {}
    closure = report.get("finite_closure_plan") or {}
    fe = report.get("vm_runtime_filtered_evaluation") or {}
    st_flow = report.get("st_retry_control_flow") or {}
    st_nested = st_flow.get("st_retry_nested_loop_order") or {}
    exposure = report.get("production_exposure_classification") or {}

    lines = [
        "# v2 solver root-cause and forward risk audit",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        f"**root_cause_status:** `{closure.get('root_cause_status')}`",
        "",
        "## Leading confirmed diagnosis",
        "",
    ]
    for item in (report.get("static_code_audit") or {}).get("leading_confirmed_diagnosis", {}).get(
        "summary", []
    ):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## ST retry control flow (from code)",
            "",
            f"**Key question answer:** {st_nested.get('answers_key_question', '')}",
            "",
            "### Default production retry order",
            "",
        ]
    )
    for step in st_nested.get("default_production_order") or []:
        lines.append(f"- {step}")
    lines.extend(["", "### Unregularized-offset diagnostic order", ""])
    for step in st_nested.get("diagnostic_unregularized_order_when_flag_set") or []:
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "## Invalid audit method (rejected)",
            "",
            f"- **Method:** {(report.get('static_code_audit') or {}).get('invalid_audit_method_rejected', {}).get('method')}",
            f"- **Reason:** {(report.get('static_code_audit') or {}).get('invalid_audit_method_rejected', {}).get('reason')}",
            "",
            "## Evidence scope",
            "",
            "### Confirmed from local code",
            "",
        ]
    )
    for item in ev.get("confirmed_from_local_code") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### Reported from VM operator evidence", ""])
    for item in ev.get("reported_from_VM_operator_evidence") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### Requires VM runtime artifact evaluation", ""])
    for item in ev.get("requires_VM_runtime_artifact_evaluation") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Production exposure",
            "",
            f"- **replay_only_true_seed_audits:** {exposure.get('replay_only_true_seed_audits')}",
            f"- **paths_using_eps_with_default_st_ladder:** {exposure.get('paths_using_eps_with_default_st_ladder')}",
            f"- **prior_PASS_L0_material_mesh_convergence:** {exposure.get('prior_PASS_L0_material_mesh_convergence')}",
            f"- **standard_seeded_retrieval_with_st_a_shift_frac_0.001:** {exposure.get('standard_seeded_retrieval_with_st_a_shift_frac_0.001')}",
            "",
            "## VM filtered evaluation (runtime, optional)",
            "",
        ]
    )
    if fe.get("loaded"):
        lines.append(f"**Loaded:** verdict `{fe.get('verdict')}`")
    else:
        lines.append("**Not loaded** — optional; prior filtered run evidence embedded as operator constants.")
    lines.extend(
        [
            "",
            "## Finite closure plan",
            "",
            f"- **next_allowed_action_after_VM_report:** {closure.get('next_allowed_action_after_VM_report')}",
            f"- **maximum_additional_baseline_solves_before_escalation:** "
            f"{closure.get('maximum_additional_baseline_solves_before_escalation')}",
            f"- **maximum_additional_code_fix_cycles:** "
            f"{closure.get('maximum_additional_code_fix_cycles_before_reconsidering_solver_architecture')}",
            "",
            "**Blocked actions:**",
        ]
    )
    for b in closure.get("blocked_actions") or []:
        lines.append(f"- {b}")
    mass = report.get("saved_vector_mass_norm_audit") or {}
    lines.extend(
        [
            "",
            f"**single_permitted_next_action:** `{ev.get('single_permitted_next_action')}`",
            f"**unregularized_offset_solve_completed:** `{ev.get('unregularized_offset_solve_completed')}`",
            f"**unregularized_offset_evaluation_verdict:** `{ev.get('unregularized_offset_evaluation_verdict')}`",
            f"**saved_vector_mass_norm_classification:** `{mass.get('classification_verdict')}`",
            "",
            "## Forward Risk Register",
            "",
        ]
    )

    header = [
        "risk_or_mismatch",
        "where_in_code",
        "evidence_source",
        "already_triggered_or_only_possible",
        "impact_if_unfixed",
        "how_to_detect_before_next_solve",
        "minimal_fix_required",
        "blocks_hole_radius_large?",
        "blocks_mesh_convergence_resume?",
        "blocks_v2_production_promotion?",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in report.get("forward_risk_register") or []:

        def _cell(v: Any) -> str:
            s = str(v) if v is not None else ""
            return s.replace("\n", " ").replace("|", "\\|")

        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("risk_or_mismatch")),
                    _cell(row.get("where_in_code")),
                    _cell(row.get("evidence_source")),
                    _cell(row.get("already_triggered_or_only_possible")),
                    _cell(row.get("impact_if_unfixed")),
                    _cell(row.get("how_to_detect_before_next_solve")),
                    _cell(row.get("minimal_fix_required")),
                    _cell(row.get("blocks_hole_radius_large")),
                    _cell(row.get("blocks_mesh_convergence_resume")),
                    _cell(row.get("blocks_v2_production_promotion")),
                ]
            )
            + " |"
        )
    staged = report.get("staged_status") or {}
    lines.extend(
        [
            "",
            f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
            f"**mesh_convergence_pass:** `{staged.get('mesh_convergence_pass')}`",
            f"**v2_production_promotion_ready:** `{staged.get('v2_production_promotion_ready')}`",
            f"**lhs_promotion_blocked:** `{staged.get('lhs_promotion_blocked')}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    static = build_static_code_audit()
    filtered_eval = _load_json(FILTERED_EVAL_JSON)
    unreg_eval = _load_json(UNREG_OFFSET_REPORT_JSON)
    mass_norm_audit = _load_json(MASS_NORM_AUDIT_JSON)
    mapping_inv = _load_json(MAPPING_INV_JSON)
    replay_recert = _load_json(REPLAY_RECERT_JSON)
    st_preflight = _load_json(ST_PREFLIGHT_JSON)
    rehab_plan = _load_json(REHAB_PLAN_JSON)
    mapping_fixed_eval = _load_json(MAPPING_FIXED_DIAG_JSON)
    root_cause_status = determine_root_cause_status(
        filtered_eval=filtered_eval, unreg_eval=unreg_eval
    )
    st_flow = build_st_retry_control_flow_audit()
    unreg_ev = (unreg_eval or {}).get("evaluation") or {}
    mass_verdict = (mass_norm_audit or {}).get("classification_verdict")

    vm_runtime = {
        "source_json": str(FILTERED_EVAL_JSON),
        "loaded": filtered_eval is not None,
        "verdict": None if not filtered_eval else filtered_eval.get("verdict"),
        "note": (
            "Prior filtered diagnostic on VM; superseded for verdict by "
            "unregularized-offset diagnostic rerun."
        ),
    }

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_cause_status": root_cause_status,
        "static_code_audit": static,
        "st_retry_control_flow": st_flow,
        "production_exposure_classification": static.get(
            "production_exposure_classification"
        ),
        "vm_runtime_filtered_evaluation": vm_runtime,
        "unregularized_offset_diagnostic": {
            "report_json": str(UNREG_OFFSET_REPORT_JSON),
            "loaded": unreg_eval is not None,
            "solve_completed_on_vm": True,
            "operator_consistency_confirmed": True,
            "evaluation_verdict": unreg_ev.get("diagnostic_verdict"),
            "candidates_unevaluable_all_xH_Mx_zero": True,
            "prior_regularized_diagnostics_superseded": True,
        },
        "saved_vector_mass_norm_audit": {
            "report_json": str(MASS_NORM_AUDIT_JSON),
            "loaded": mass_norm_audit is not None,
            "classification_verdict": mass_verdict,
        },
        "eps_mapping_impact_inventory": {
            "report_json": str(MAPPING_INV_JSON),
            "loaded": mapping_inv is not None,
        },
        "existing_pass_replay_recertification": {
            "report_json": str(REPLAY_RECERT_JSON),
            "loaded": replay_recert is not None,
        },
        "st_singular_mass_preflight": {
            "report_json": str(ST_PREFLIGHT_JSON),
            "loaded": st_preflight is not None,
            "PGNHEP_purification_applicability": (st_preflight or {}).get(
                "PGNHEP_purification_applicability"
            ),
        },
        "st_singular_mass_rehabilitation_plan": {
            "report_json": str(REHAB_PLAN_JSON),
            "loaded": rehab_plan is not None,
        },
        "mapping_corrected_baseline_diagnostic": {
            "report_json": str(MAPPING_FIXED_DIAG_JSON),
            "loaded": mapping_fixed_eval is not None,
            "evaluation_verdict": (mapping_fixed_eval or {})
            .get("evaluation", {})
            .get("diagnostic_verdict"),
            "recommended_vm_command": (
                "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "run_v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.sh"
            ),
        },
        "recommended_vm_command_report_only": None,
        "recommended_vm_command_baseline_solve": (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_mapping_fixed_persistence_fixed_baseline_vm.sh"
        ),
        "evidence_summary": build_evidence_summary(
            filtered_eval=filtered_eval,
            unreg_eval=unreg_eval,
            mass_norm_audit=mass_norm_audit,
            mapping_fixed_eval=mapping_fixed_eval,
            static_audit=static,
        ),
        "forward_risk_register": build_forward_risk_register(
            filtered_eval=filtered_eval, static_audit=static
        ),
        "finite_closure_plan": build_finite_closure_plan(
            filtered_eval=filtered_eval,
            unreg_eval=unreg_eval,
            mass_norm_audit=mass_norm_audit,
            mapping_fixed_eval=mapping_fixed_eval,
            root_cause_status=root_cause_status,
        ),
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "diagnostic_exposure_conclusion": {
            "status": "confirmed_from_local_code_with_VM_operator_evidence",
            "summary": (
                "Eigenvalue mapping fix implemented (lam_phys=mu, slepc_backtransformed). "
                "PGNHEP/purification ruled out on VM. Pre-mapping seven-mode harvest is not "
                "evidence against corrected mapping. Exactly one mapping-corrected unregularized "
                "baseline ST diagnostic authorized (preserve all nconv candidates). Stage-2 only "
                "if that run fails. hole_radius_large and mesh convergence remain blocked."
            ),
            "replay_only_validations_protected": True,
            "eps_default_st_ladder_potentially_exposed": True,
            "unregularized_offset_solve_not_exposed_to_prior_st_shift": True,
        },
        "code_fix_cycle": {
            "maximum_additional_code_fix_cycles_before_reconsidering_solver_architecture": 1,
            "this_implementation_counts_as": 1,
            "remaining_code_fix_cycles": 0,
        },
    }

    write_json(OUT_JSON, report)
    _write_md(report)
    print(f"[root_cause_audit] wrote {OUT_JSON}", flush=True)
    print(f"[root_cause_audit] root_cause_status={root_cause_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
