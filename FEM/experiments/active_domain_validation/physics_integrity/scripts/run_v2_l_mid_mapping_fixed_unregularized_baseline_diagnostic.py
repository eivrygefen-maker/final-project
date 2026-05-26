#!/usr/bin/env python3
"""
L_mid baseline: mapping-corrected unregularized ST diagnostic (experiment-only).

Authorized exactly one final baseline eigensolve with full nconv candidate preservation.
Modes:
  --evaluate-only   Report-only replay/MAC on existing artifacts (no EPS).
  --prepare-only    Write launch metadata and update static reports (no EPS).
  (default)         mpiexec solve then immediate --evaluate-only evaluation.
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
from typing import Any, Dict, List, Tuple

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
from v2_mapping_fixed_baseline_evaluator import (
    OUT_SUBDIR,
    VERDICT_BRANCH_RECOVERED,
    VERDICT_INCONSISTENT,
    VERDICT_PERSISTENCE_FAILURE,
    evaluate_mapping_fixed_baseline_artifacts,
)
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

CASE_ID = "baseline_coupled_v2"
NUM_MODES = 64
SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
SIGMA_OFFSETS_HZ = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0)
SEED_F_HZ = 243.0754171175576

REPORT_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.md"
VM_SHELL = (
    "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "run_v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.sh"
)


def _out_dir(case_dir: Path) -> Path:
    return case_dir / OUT_SUBDIR


def _artifact_status(out_dir: Path, seed_npy: Path) -> Dict[str, Any]:
    summary = out_dir / "diagnostics" / "mode_energy_summary.json"
    bank = out_dir / "diagnostics" / "eps_candidate_bank.json"
    modes_dir = out_dir / "modes"
    result_glob = list((out_dir / "results").glob("result_*.json"))
    return {
        "output_dir": str(out_dir),
        "output_dir_exists": out_dir.is_dir(),
        "mode_energy_summary_exists": summary.is_file(),
        "eps_candidate_bank_exists": bank.is_file(),
        "modes_dir_exists": modes_dir.is_dir(),
        "result_json_count": len(result_glob),
        "seed_npy_exists": seed_npy.is_file(),
        "artifacts_ok": bool(
            out_dir.is_dir()
            and summary.is_file()
            and bank.is_file()
            and seed_npy.is_file()
            and modes_dir.is_dir()
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


def _validate_prelaunch(
    mesh_file: Path,
    seed_npy: Path,
    seed_meta: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    n_w = int(seed_meta.get("n_reduced_W", 0))
    if seed_npy.is_file():
        seed_len = int(np.load(str(seed_npy)).size)
        if n_w > 0 and seed_len != n_w:
            errors.append(f"seed length {seed_len} != meta n_reduced_W {n_w}")
    else:
        errors.append(f"missing seed: {seed_npy}")
    if not mesh_file.is_file():
        errors.append(f"missing mesh: {mesh_file}")
    cli = subprocess.run(
        [sys.executable, str(SOLVE_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    help_text = (cli.stdout or "") + (cli.stderr or "")
    if "--seed-branch-mapping-fixed-unregularized-diagnostic" not in help_text:
        errors.append(
            "v2_sensitivity_solve missing --seed-branch-mapping-fixed-unregularized-diagnostic"
        )
    return {"valid": len(errors) == 0, "errors": errors}


def _run_solve(
    sample: Dict[str, Any],
    mesh_file: Path,
    *,
    target_hz: float,
    seed_npy: Path,
    out_dir: Path,
) -> Tuple[int, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_json = out_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path = out_dir / "logs" / "mapping_fixed_baseline_diagnostic.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        "-u",
        str(SOLVE_SCRIPT.resolve()),
        "--sample-id",
        str(sample["id"]),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--case-dir",
        str(out_dir.resolve()),
        "--target-hz",
        str(float(target_hz)),
        "--reference-f-hz",
        str(float(target_hz)),
        "--num-modes",
        str(int(NUM_MODES)),
        "--eps-seed-npy",
        str(seed_npy.resolve()),
        "--seed-branch-recovery-diagnostic",
        "--seed-branch-mapping-fixed-unregularized-diagnostic",
    ]
    launch_record = {
        "exact_command_argv": cmd,
        "shell_command": shlex.join(cmd),
        "working_directory": str(REPO_ROOT.resolve()),
        "output_case_dir": str(out_dir),
        "log_path": str(log_path),
        "mapping_corrected_baseline": True,
        "eps_diagnostic_preserve_all_nconv_candidates": True,
    }
    write_json(out_dir / "launch_record.json", launch_record)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env, check=False
    )
    log_path.write_text(
        "\n".join(
            [
                "# mapping_fixed_unregularized_baseline_diagnostic",
                f"return_code={completed.returncode}",
                f"command={shlex.join(cmd)}",
                "",
                "=== stdout ===",
                completed.stdout or "",
                "",
                "=== stderr ===",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    launch_record["return_code"] = int(completed.returncode)
    write_json(out_dir / "launch_record.json", launch_record)
    result = _load_solve_result(out_dir, target_hz)
    return int(completed.returncode), result


def _write_md(report: Dict[str, Any]) -> None:
    ev = report.get("evaluation") or {}
    st = ev.get("st_operator_fields") or {}
    lines = [
        "# L_mid mapping-corrected unregularized baseline diagnostic",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        f"**Verdict:** `{ev.get('diagnostic_verdict')}`",
        "",
        f"- continuation_seed_applied: {ev.get('continuation_seed_applied')}",
        f"- eps_nconv_marked: {ev.get('eps_nconv_marked')}",
        f"- eps_eigenvalue_semantics: {st.get('eps_eigenvalue_semantics')}",
        f"- legacy_double_shift_mapping_disabled: {st.get('legacy_double_shift_mapping_disabled')}",
        f"- diagnostic_operator_consistent_with_replay: {st.get('diagnostic_operator_consistent_with_replay')}",
        f"- actual_sigma_hz: {st.get('actual_sigma_hz')}",
        "",
        f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
        "",
    ]
    bank = ev.get("eps_candidate_bank_summary") or {}
    if bank:
        lines.extend(
            [
                "## Candidate bank",
                "",
                f"- num_vectors_saved: {bank.get('num_vectors_saved')}",
                f"- nconv_marked: {bank.get('nconv_marked')}",
                "",
            ]
        )
    summary = ev.get("summary") or {}
    if summary:
        lines.extend(
            [
                "## Summary",
                "",
                f"- candidates evaluated: {summary.get('num_candidates_evaluated')}",
                f"- branch_recovery_pass: {summary.get('num_branch_recovery_pass')}",
                "",
            ]
        )
    for c in ev.get("candidates") or []:
        lines.append(
            f"- slot={c.get('candidate_index')} f={c.get('reported_frequency_hz')} "
            f"MAC={c.get('pressure_MAC_to_true_acoustic_seed')} "
            f"xH_Mx={c.get('rayleigh_denominator')} pass={c.get('branch_recovery_pass')}"
        )
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_static_reports() -> None:
    from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main
    from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main

    rehab_main()
    audit_main()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--record-vm-persistence-failure",
        action="store_true",
        help="Write operator VM persistence-failure evidence into report (no EPS).",
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
    ladder = [target_hz + float(o) for o in SIGMA_OFFSETS_HZ]

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR,
        "seed_frequency_hz": target_hz,
        "mesh_convergence_may_resume": False,
        "recommended_vm_command": VM_SHELL,
        "solver_requirements": {
            "lam_phys_equals_mu": True,
            "eps_eigenvalue_semantics": "slepc_backtransformed",
            "legacy_double_shift_mapping_disabled": True,
            "st_a_shift_frac": 0.0,
            "st_mass_reg_frac": 0.0,
            "no_PGNHEP": True,
            "no_purification": True,
            "no_nullspace_reduction": True,
            "preserve_all_nconv_candidates": True,
        },
        "sigma_ladder_policy": {
            "never_sigma_at_seed_rayleigh_frequency": True,
            "offsets_hz_from_seed": list(SIGMA_OFFSETS_HZ),
            "diagnostic_sigma_hz_ladder": ladder,
        },
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "prior_pass_handling": {
            "mesh_topology_gates_preserved": True,
            "true_seed_replay_findings_preserved": True,
            "eps_frequency_labels_pending_recertification": True,
        },
        "PGNHEP_purification": "ruled_out_in_current_VM_environment",
        "stage_2_trigger": "mandatory_only_if_mapping_corrected_baseline_fails",
    }

    pre = _validate_prelaunch(mesh_file, seed_npy, seed_meta)
    report["prelaunch"] = pre

    if args.record_vm_persistence_failure:
        report["vm_operator_persistence_failure"] = {
            "nconv_marked": 56,
            "eps_diagnostic_candidate_bank_count": 56,
            "num_vectors_saved": 0,
            "preserve_all_nconv_kept": 56,
            "worker_usable_rows": 56,
            "worker_rows_after_filter": 0,
            "root_cause": "bank_not_on_config_and_worker_filtered_rt_rb_none",
        }
        report["evaluation"] = {
            "diagnostic_verdict": VERDICT_PERSISTENCE_FAILURE,
            "interpretation": (
                "Corrected unregularized EPS produced 56 converged candidates in memory, but "
                "diagnostic preserve-all vectors were not persisted. No physical branch verdict."
            ),
            "not_evidence_for_st_failure": True,
            "not_evidence_for_stage_2": True,
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        _refresh_static_reports()
        return 0

    if args.prepare_only:
        report["prepare_only"] = True
        report["evaluation"] = {
            "diagnostic_verdict": "PENDING_VM_SOLVE_AND_EVALUATION",
            "note": f"Run {VM_SHELL} on VM when authorized.",
        }
        write_json(REPORT_JSON, report)
        _write_md(report)
        _refresh_static_reports()
        print(f"[mapping_fixed_baseline] prepared; VM command: {VM_SHELL}", flush=True)
        return 0 if pre["valid"] else 2

    if args.evaluate_only:
        report["evaluate_only"] = True
        status = _artifact_status(out_dir, seed_npy)
        solve_result = _load_solve_result(out_dir, target_hz)
        if not status["artifacts_ok"] or not solve_result:
            report["evaluation"] = {
                "artifact_status": status,
                "diagnostic_verdict": VERDICT_INCONSISTENT,
                "verdict_reason": "missing_artifacts_or_result_json",
            }
        else:
            report["evaluation"] = evaluate_mapping_fixed_baseline_artifacts(
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
        _refresh_static_reports()
        verdict = (report.get("evaluation") or {}).get("diagnostic_verdict", VERDICT_INCONSISTENT)
        print(f"[mapping_fixed_baseline_eval] verdict={verdict}", flush=True)
        if verdict == VERDICT_BRANCH_RECOVERED:
            return 0
        if verdict in (VERDICT_INCONSISTENT, VERDICT_PERSISTENCE_FAILURE):
            return 2
        return 1

    if not pre["valid"]:
        print(f"[mapping_fixed_baseline] prelaunch failed: {pre['errors']}", file=sys.stderr)
        return 2

    rc, solve_result = _run_solve(
        fallback_sample,
        mesh_file,
        target_hz=target_hz,
        seed_npy=seed_npy,
        out_dir=out_dir,
    )
    report["solve_return_code"] = rc
    report["solve_result_summary"] = {
        "nconv_marked": (solve_result.get("eps_batch_diagnostics") or {}).get("nconv_marked"),
        "num_modes_saved": solve_result.get("num_modes_saved"),
        "continuation_seed_applied": solve_result.get("continuation_seed_applied"),
    }
    status = _artifact_status(out_dir, seed_npy)
    if rc != 0 or not status["artifacts_ok"] or not solve_result:
        report["evaluation"] = {
            "artifact_status": status,
            "diagnostic_verdict": VERDICT_INCONSISTENT,
            "verdict_reason": "solve_failed_or_incomplete_artifacts",
        }
    else:
        report["evaluation"] = evaluate_mapping_fixed_baseline_artifacts(
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
    _refresh_static_reports()
    verdict = (report.get("evaluation") or {}).get("diagnostic_verdict", VERDICT_INCONSISTENT)
    print(f"[mapping_fixed_baseline] solve_rc={rc} verdict={verdict}", flush=True)
    if verdict == VERDICT_BRANCH_RECOVERED:
        return 0
    if verdict == VERDICT_INCONSISTENT:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
