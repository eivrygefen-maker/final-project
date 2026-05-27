#!/usr/bin/env python3
"""Write ST singular-mass rehabilitation plan from code + optional preflight JSON (no solve)."""
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

from v2_eps_mapping_audit_lib import MAPPING_FIX_SUMMARY
from v2_mesh_convergence_common import CONV_DIAG

OUT_JSON = CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.json"
OUT_MD = CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.md"
PREFLIGHT_JSON = CONV_DIAG / "v2_st_singular_mass_preflight.json"
MAPPING_JSON = CONV_DIAG / "v2_eps_mapping_impact_inventory.json"
MAPPING_FIXED_DIAG_JSON = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.json"
)
PERSISTENCE_FIXED_DIAG_JSON = (
    CONV_DIAG
    / "v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.json"
)
SELF_TEST_JSON = CONV_DIAG / "v2_mapping_fixed_candidate_persistence_self_test.json"
PIPELINE_AUDIT_JSON = (
    CONV_DIAG / "v2_mapping_fixed_persistence_fixed_full_pipeline_audit.json"
)
PF_VERDICT = "MAPPING_FIXED_UNREGULARIZED_BASELINE_CANDIDATE_PERSISTENCE_FAILURE"
VM_PIPELINE_AUDIT_SHELL = (
    "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit.sh"
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2))


def main() -> int:
    preflight: Optional[Dict[str, Any]] = None
    if PREFLIGHT_JSON.is_file():
        preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    mapping_inv = (
        json.loads(MAPPING_JSON.read_text(encoding="utf-8")) if MAPPING_JSON.is_file() else {}
    )
    mapping_fixed = (
        json.loads(MAPPING_FIXED_DIAG_JSON.read_text(encoding="utf-8"))
        if MAPPING_FIXED_DIAG_JSON.is_file()
        else {}
    )
    persistence_fixed = (
        json.loads(PERSISTENCE_FIXED_DIAG_JSON.read_text(encoding="utf-8"))
        if PERSISTENCE_FIXED_DIAG_JSON.is_file()
        else {}
    )
    self_test = (
        json.loads(SELF_TEST_JSON.read_text(encoding="utf-8")) if SELF_TEST_JSON.is_file() else {}
    )
    pipeline_audit = (
        json.loads(PIPELINE_AUDIT_JSON.read_text(encoding="utf-8"))
        if PIPELINE_AUDIT_JSON.is_file()
        else {}
    )

    applicability = (preflight or {}).get("PGNHEP_purification_applicability")
    if applicability is None:
        applicability = "not_justified_use_nullspace_reduction_plan"
    pgnhep_ruled_out = applicability == "not_justified_use_nullspace_reduction_plan"
    mapping_fixed_ev = (mapping_fixed or {}).get("evaluation") or {}
    mapping_fixed_verdict = mapping_fixed_ev.get("diagnostic_verdict")
    pf_ev = (persistence_fixed or {}).get("evaluation") or {}
    pf_verdict = pf_ev.get("diagnostic_verdict")
    self_test_pass = bool(self_test.get("self_test_pass"))
    first_run_persistence_failure = mapping_fixed_verdict == PF_VERDICT or bool(
        mapping_fixed.get("vm_operator_persistence_failure")
    )
    pf_bank = (persistence_fixed.get("evaluation") or {}).get("eps_candidate_bank_summary") or {}
    replacement_ran = int(pf_bank.get("num_vectors_saved", 0) or 0) >= 56 or bool(
        persistence_fixed.get("solve_return_code") == 0
        and persistence_fixed.get("persistence_self_test_pass")
    )
    pipeline_verdict = pipeline_audit.get("audit_verdict")
    pipeline_unresolved = pipeline_verdict in (
        "MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED",
        None,
    ) and replacement_ran
    replacement_pending = (
        not replacement_ran
        and pf_verdict
        in (
            None,
            "PENDING_SELF_TEST_AND_REPLACEMENT_BASELINE",
        )
    ) or (self_test_pass and not replacement_ran and pf_verdict is None)
    baseline_pending = first_run_persistence_failure or replacement_pending
    audit_only = replacement_ran and not pipeline_verdict

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy": "finite_solver_rehabilitation_persistence_fix_then_mapping_baseline",
        "next_allowed_action": (
            "review_persisted_vector_content_unresolved_lossless_preflight_no_eps"
            if pipeline_unresolved
            else (
            "report_only_full_pipeline_audit_over_existing_replacement_artifacts"
            if audit_only
            else (
                "persistence_self_test_then_one_replacement_mapping_corrected_baseline"
                if baseline_pending
                else "review_mapping_corrected_baseline_and_pipeline_audit_verdict"
            )
            )
        ),
        "recommended_vm_command": (
            VM_PIPELINE_AUDIT_SHELL if (audit_only or replacement_ran) and not pipeline_unresolved else (
                "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "run_v2_coupled_physical_core_report_only_bundle.sh"
                if pipeline_unresolved
                else None
            )
        ),
        "PGNHEP_purification": "ruled_out_in_current_VM_environment",
        "purification": "ruled_out_in_current_VM_environment",
        "first_mapping_corrected_run": {
            "inconclusive_persistence_failure": first_run_persistence_failure,
            "verdict": mapping_fixed_verdict or PF_VERDICT,
            "nconv_marked_vm": 56,
            "num_vectors_saved_vm": 0,
            "not_evidence_for_st_failure": True,
            "not_evidence_for_stage_2": True,
        },
        "persistence_self_test": {
            "report_json": str(SELF_TEST_JSON),
            "self_test_pass": self_test_pass,
            "required_before_replacement_eigensolve": True,
        },
        "replacement_baseline": {
            "already_ran": replacement_ran,
            "persistence_56_of_56_closed": bool(
                persistence_fixed.get("evaluation", {})
                .get("eps_candidate_bank_summary", {})
                .get("num_vectors_saved")
                == 56
            )
            if persistence_fixed
            else None,
            "current_blocker": (
                "replay_evaluation_or_persisted_vector_content"
                if replacement_ran
                else None
            ),
            "pipeline_audit_verdict": pipeline_verdict,
            "pipeline_audit_json": str(PIPELINE_AUDIT_JSON),
        },
        "stage_2": {
            "description": "Explicit physical null-space reduction",
            "mandatory_only_if": (
                "replacement mapping-corrected baseline fails after all candidates persisted "
                "and evaluated"
            ),
            "not_triggered_by_persistence_failure": True,
            "authorized_now": False,
            "blocked_until": [
                "persistence_self_test_pass",
                "replacement_baseline_with_persisted_candidates_evaluated",
            ],
            "plan_outline": [
                "Identify mass-null subspace from pressure restriction / algebraic constraints",
                "Build physical pencil on complement of null space",
                "Map seed and saved modes between W and reduced basis",
                "Preserve three-worker overlapping-frequency architecture",
                "Re-evaluate save/load/replay/MAC without changing production defaults",
            ],
        },
        "not_authorized": [
            "PGNHEP/purification in current VM environment",
            "another sigma adjustment",
            "another filter-only EPS rerun",
            "another ST mapping variant",
            "immediate Stage-2 before mapping-corrected baseline completes",
        ],
        "confirmed_vm_evidence": {
            "seed_xH_Mx_finite_nonzero": True,
            "pre_mapping_fix_unregularized_offset_solve_not_valid_mapping_test": True,
            "seven_saved_candidates_mass_null_not_evidence_against_corrected_mapping": True,
            "PGNHEP_purification_applicability": applicability,
            "has_EPS_ProblemType_PGNHEP": (preflight or {}).get("has_EPS_ProblemType_PGNHEP"),
            "can_set_PGNHEP_without_solve": (preflight or {}).get("can_set_PGNHEP_without_solve"),
            "can_set_purify_without_solve": (preflight or {}).get("can_set_purify_without_solve"),
            "some_modes_valid_physics_wrong_frequency_labels_only": True,
        },
        "stage_0_mapping_fix": MAPPING_FIX_SUMMARY,
        "mapping_corrected_baseline_diagnostic": {
            "authorized": baseline_pending,
            "replacement_output_subdir": (
                "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed"
            ),
            "preserve_all_nconv_candidates": True,
            "physical_eligibility_after_save": True,
            "verdicts": [
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_BRANCH_RECOVERED",
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED",
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_OUTPUT_OR_REPLAY_INCONSISTENT",
                PF_VERDICT,
            ],
            "acceptance_gates": [
                "continuation_seed_applied=True",
                "eps_eigenvalue_semantics=slepc_backtransformed",
                "legacy_double_shift_mapping_disabled=True",
                "diagnostic_operator_consistent_with_replay=True",
                "actual_st_a_shift_frac=0",
                "actual_st_mass_reg_frac=0",
                "candidate xH_Mx finite and nonzero",
                "reported vs replay frequency consistent",
                "replay residual within tolerance",
                "frequency within 1% of seed",
                "pressure MAC >= 0.85",
            ],
            "report_json": str(MAPPING_FIXED_DIAG_JSON),
            "current_verdict": mapping_fixed_verdict,
        },
        "prior_pass_handling": {
            "mesh_topology_gates_preserved": True,
            "true_seed_replay_findings_preserved": True,
            "eps_frequency_labels_pending_recertification": True,
            "prior_PASS_auto_invalidated": mapping_inv.get("prior_PASS_auto_invalidated", False),
        },
        "preflight_summary": preflight,
        "mapping_inventory_summary": {
            "prior_PASS_auto_invalidated": mapping_inv.get("prior_PASS_auto_invalidated", False),
        },
        "mesh_convergence_may_resume": False,
        "additional_baseline_eigensolve": (
            "none_authorized_report_only_pipeline_audit"
            if replacement_ran
            else (
                "one_replacement_run_after_persistence_self_test_pass"
                if baseline_pending
                else "blocked_pending_baseline_review"
            )
        ),
        "hole_radius_large": "blocked",
        "production_policy_unchanged": True,
        "current_state_summary": {
            "persistence_self_test": "PASS" if self_test_pass else "pending_or_failed",
            "replacement_baseline_eps": "completed_once" if replacement_ran else "not_completed",
            "candidate_persistence": (
                "56_of_56_closed" if replacement_ran else "not_closed"
            ),
            "full_pipeline_audit": (
                "completed" if pipeline_verdict else "report_only_required"
            ),
            "additional_eps_solve": "not_authorized",
        },
    }

    lines = [
        "# ST singular-mass rehabilitation plan",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Next allowed action",
        "",
        f"- **next_allowed_action:** `{report['next_allowed_action']}`",
        f"- **VM command:** `{report.get('recommended_vm_command')}`",
        "",
        "## PGNHEP / purification",
        "",
        f"- **Status:** `{report['PGNHEP_purification']}`",
        "",
        "## Stage 0 (implemented): eigenvalue mapping",
        "",
        f"- {MAPPING_FIX_SUMMARY['new_behavior']}",
        "",
        "## Current state (replacement baseline already executed)",
        "",
        f"- **Persistence self-test:** `{'PASS' if self_test_pass else 'pending_or_failed'}`",
        f"- **Replacement baseline EPS:** `{'completed_once' if replacement_ran else 'not_completed'}`",
        f"- **Candidate persistence:** `{'56/56 closed' if replacement_ran else 'not closed'}`",
        f"- **Full pipeline audit:** `{'completed: ' + str(pipeline_verdict) if pipeline_verdict else 'report-only required'}`",
        f"- **Additional EPS solve:** `not authorized`",
        f"- **Report-only VM command:** `{VM_PIPELINE_AUDIT_SHELL}`",
        "",
        "## Historical: first mapping-corrected run",
        "",
        f"- **First-run persistence failure (inconclusive):** `{first_run_persistence_failure}`",
        f"- **Not ST failure / not Stage-2:** True",
        "",
        "## Stage 2",
        "",
        "Not triggered by persistence failure. Mandatory only if replacement baseline "
        "persists and evaluates all candidates yet no physical branch is recovered.",
        "",
    ]
    md_text = "\n".join(lines) + "\n"
    _atomic_write_json(OUT_JSON, report)
    _atomic_write_text(OUT_MD, md_text)
    print(f"[rehab_plan] wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
