#!/usr/bin/env python3
"""
L_mid baseline: unregularized-offset seed-branch recovery diagnostic (experiment-only).

Modes:
  --evaluate-only   Report-only replay/MAC evaluation of existing VM solve artifacts (no EPS).
  --prepare-only    Write launch stub only.
  (default)         Run mpiexec solve then evaluate (not authorized; baseline solve completed).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_sensitivity_common import REPO_ROOT, hz_result_tag
from v2_unreg_offset_report_evaluator import (
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    evaluate_unregularized_offset_artifacts,
)

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 48
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SIGMA_OFFSETS_HZ = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0)
SEED_F_HZ = 243.0754171175576

REPORT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_unregularized_offset_diagnostic.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_seed_branch_unregularized_offset_diagnostic.md"
OUT_SUBDIR = "seed_branch_recovery_diagnostic_unregularized_offset"


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR


def _artifact_status(out_dir: Path, seed_npy: Path) -> Dict[str, Any]:
    summary = out_dir / "diagnostics" / "mode_energy_summary.json"
    modes_dir = out_dir / "modes"
    result_glob = list((out_dir / "results").glob("result_*.json"))
    return {
        "output_dir": str(out_dir),
        "output_dir_exists": out_dir.is_dir(),
        "mode_energy_summary_exists": summary.is_file(),
        "modes_dir_exists": modes_dir.is_dir(),
        "result_json_count": len(result_glob),
        "seed_npy_exists": seed_npy.is_file(),
        "artifacts_ok": bool(
            out_dir.is_dir() and summary.is_file() and seed_npy.is_file() and modes_dir.is_dir()
        ),
    }


def _load_solve_result(out_dir: Path, target_hz: float) -> Dict[str, Any]:
    result_path = out_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        results = sorted((out_dir / "results").glob("result_*.json"))
        if not results:
            return {}
        result_path = results[-1]
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evaluation") or {}
    st = ev.get("st_operator_fields") or {}
    lines = [
        "# L_mid unregularized-offset seed-branch recovery diagnostic",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        f"**Verdict:** `{ev.get('diagnostic_verdict')}`",
        "",
        f"- continuation_seed_applied: {ev.get('continuation_seed_applied')}",
        f"- eps_nconv_marked: {ev.get('eps_nconv_marked')}",
        f"- diagnostic_operator_consistent_with_replay: {st.get('diagnostic_operator_consistent_with_replay')}",
        f"- actual_sigma_hz: {st.get('actual_sigma_hz')}",
        f"- actual_st_a_shift_frac: {st.get('actual_st_a_shift_frac')}",
        f"- actual_st_mass_reg_frac: {st.get('actual_st_mass_reg_frac')}",
        "",
        f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
        "",
    ]
    summary = ev.get("summary") or {}
    if summary:
        lines.extend(
            [
                "## Summary",
                "",
                f"- candidates evaluated: {summary.get('num_candidates_evaluated')}",
                f"- metrics_computation_ok: {summary.get('num_metrics_computation_ok')}",
                f"- branch_recovery_pass: {summary.get('num_branch_recovery_pass')}",
                f"- physically_eligible: {summary.get('num_physically_eligible')}",
                "",
            ]
        )
    inp = ev.get("input_paths") or {}
    if inp:
        lines.extend(["## Input paths (exclusive tree)", ""])
        for k, v in inp.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    lines.append("## Candidates (instrumented)")
    lines.append("")
    for c in ev.get("candidates") or []:
        lines.append(
            f"- idx={c.get('candidate_index')} "
            f"metrics={c.get('metrics_computation_status')} "
            f"f={c.get('reported_frequency_hz')} "
            f"MAC={c.get('pressure_MAC_to_true_acoustic_seed')} "
            f"replay_f={c.get('replay_rayleigh_frequency_hz')} "
            f"xH_Mx={c.get('rayleigh_denominator')} "
            f"branch_pass={c.get('branch_recovery_pass')}"
        )
        if c.get("nonfinite_reason"):
            lines.append(f"  - nonfinite_reason: {c.get('nonfinite_reason')}")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Report-only evaluation of completed solve artifacts (no EPS).",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    out_dir = _out_dir(case_dir)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = (
        json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    )
    target_hz = float(seed_meta.get("locator_frequency_hz", SEED_F_HZ))
    fallback_sample = sample_spec_from_case(case)

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR,
        "seed_frequency_hz": target_hz,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "baseline_eigensolve_budget_exhausted": True,
        "prior_regularized_diagnostics_superseded_for_branch_verdict": True,
        "vm_operator_solve_evidence": {
            "solve_completed": True,
            "continuation_seed_applied": True,
            "eps_nconv_marked": 56,
            "actual_sigma_hz": 243.5754171175576,
            "st_a_shift_frac_used": 0.0,
            "st_mass_reg_frac_used": 0.0,
            "diagnostic_operator_consistent_with_replay": True,
            "saved_candidates": 7,
            "source": "reported_from_VM_operator_evidence",
        },
    }

    if args.prepare_only:
        local_half = max(0.5, target_hz * 0.01)
        ladder = [target_hz + float(o) for o in SIGMA_OFFSETS_HZ]
        report["prepare_only"] = True
        report["recommended_vm_command"] = (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_l_mid_seed_branch_unregularized_offset_evaluation.sh"
        )
        report["sigma_ladder_policy"] = {
            "seed_rayleigh_f_hz": target_hz,
            "offsets_hz_from_seed": list(SIGMA_OFFSETS_HZ),
            "diagnostic_sigma_hz_ladder": ladder,
        }
        report["evaluation"] = {
            "diagnostic_verdict": "PENDING_VM_EVALUATION",
            "note": "Solve completed; run recommended_vm_command for report-only evaluation.",
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        return 0

    if args.evaluate_only:
        report["evaluate_only"] = True
        report["recommended_vm_command"] = (
            "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_l_mid_seed_branch_unregularized_offset_evaluation.sh"
        )
        status = _artifact_status(out_dir, seed_npy)
        solve_result = _load_solve_result(out_dir, target_hz)
        if not status["artifacts_ok"] or not solve_result:
            report["evaluation"] = {
                "artifact_status": status,
                "diagnostic_verdict": VERDICT_INCONSISTENT,
                "verdict_reason": "missing_artifacts_or_result_json",
            }
        else:
            report["evaluation"] = evaluate_unregularized_offset_artifacts(
                out_dir=out_dir,
                case_dir=case_dir,
                mesh_file=mesh_file,
                seed_npy=seed_npy,
                seed_meta=seed_meta,
                fallback_sample=fallback_sample,
                solve_result=solve_result,
                target_hz=target_hz,
            )
            report["evaluation"]["artifact_status"] = status
        write_json(REPORT_JSON, report)
        _write_md(report)
        verdict = (report.get("evaluation") or {}).get("diagnostic_verdict", VERDICT_INCONSISTENT)
        print(f"[unreg_offset_eval] verdict={verdict}", flush=True)
        return 0 if verdict == VERDICT_BRANCH_RECOVERED else 1

    print(
        "[unreg_offset_diag] ERROR: baseline eigensolve already completed; use --evaluate-only",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
