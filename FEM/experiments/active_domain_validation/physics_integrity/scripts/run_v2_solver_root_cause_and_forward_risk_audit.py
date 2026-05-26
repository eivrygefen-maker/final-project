#!/usr/bin/env python3
"""
Combined static code audit + VM runtime merge for forward risk and closure plan.

Static sections are derived from local source. VM runtime sections remain pending until
run_v2_l_mid_seed_branch_filtered_evaluation.py completes on the VM with artifacts present.
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
    build_evidence_summary,
    build_finite_closure_plan,
    build_forward_risk_register,
    build_static_code_audit,
)

OUT_JSON = CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.json"
OUT_MD = CONV_DIAG / "v2_solver_root_cause_and_forward_risk_audit.md"

FILTERED_EVAL_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_filtered_evaluation.json"


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
    pending = fe.get("verdict_pending_until_vm_run", True)

    lines = [
        "# v2 solver root-cause and forward risk audit",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "## Evidence scope",
        "",
        "### Confirmed from local code",
        "",
    ]
    for item in ev.get("confirmed_from_local_code") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### Reported from VM operator evidence", ""])
    for item in ev.get("reported_from_VM_operator_evidence") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### Requires VM runtime artifact evaluation", ""])
    for item in ev.get("requires_VM_runtime_artifact_evaluation") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### Combined future checks (single VM run)", ""])
    for item in ev.get("single_vm_run_combines") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## VM filtered evaluation (runtime)", ""])
    if pending:
        lines.append("**Status: PENDING** — run consolidated VM bundle after code sync.")
    else:
        lines.append(f"**Status: COMPLETED** — verdict `{fe.get('verdict')}`")
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
    lines.extend(["", "## Forward Risk Register and Prevention Plan", ""])

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
    lines.append("**mesh_convergence_may_resume:** `False`\n")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    static = build_static_code_audit()
    filtered_eval = _load_json(FILTERED_EVAL_JSON)

    vm_runtime = {
        "source_json": str(FILTERED_EVAL_JSON),
        "loaded": filtered_eval is not None,
        "verdict_pending_until_vm_run": True,
        "runtime_evaluation_completed": False,
    }
    if filtered_eval:
        vm_runtime.update(
            {
                "verdict": filtered_eval.get("verdict"),
                "verdict_pending_until_vm_run": filtered_eval.get(
                    "verdict_pending_until_vm_run", True
                ),
                "runtime_evaluation_completed": filtered_eval.get(
                    "runtime_evaluation_completed", False
                ),
                "summary": filtered_eval.get("summary"),
                "candidates": filtered_eval.get("candidates"),
            }
        )

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%d-%mT%H:%M:%SZ", time.gmtime()),
        "static_code_audit": static,
        "vm_runtime_filtered_evaluation": vm_runtime,
        "evidence_summary": build_evidence_summary(
            filtered_eval=filtered_eval, static_audit=static
        ),
        "forward_risk_register": build_forward_risk_register(
            filtered_eval=filtered_eval, static_audit=static
        ),
        "finite_closure_plan": build_finite_closure_plan(filtered_eval=filtered_eval),
        "stale_reports_superseded_by_this_audit": static.get(
            "stale_report_paths_to_supersede"
        ),
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "diagnostic_exposure_conclusion": {
            "status": "not_yet_verified",
            "note": (
                "Do not claim earlier PASS artifacts are safe or contaminated until VM "
                "filtered_evaluation and targeted recertification (if needed) complete."
            ),
        },
    }

    write_json(OUT_JSON, report)
    _write_md(report)
    print(f"[root_cause_audit] wrote {OUT_JSON}", flush=True)
    print(
        f"[root_cause_audit] vm_runtime_pending={vm_runtime.get('verdict_pending_until_vm_run')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
